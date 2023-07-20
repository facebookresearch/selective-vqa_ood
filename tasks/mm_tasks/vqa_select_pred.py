# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import glob
import json
import logging
import math
import os
import pickle
from argparse import Namespace
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

from data import data_utils

from fairseq.tasks import register_task
from scoring.risk_coverage import RiskCoverage

from tasks.mm_tasks.vqa_gen import VqaGenConfig, VqaGenTask

logger = logging.getLogger(__name__)


def get_arg(args, key, default=None):
    # args = vars(args)
    if isinstance(args, Namespace):
        args = vars(args)

    return args.get(key, default)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(
        x.int().cpu(),
        extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
    )
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


@dataclass
class VqaConfidenceConfig(VqaGenConfig):
    pass


@register_task("vqa_select_pred", dataclass=VqaConfidenceConfig)
class VQASelectPredTask(VqaGenTask):
    def __init__(self, cfg: VqaConfidenceConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.scorer = RiskCoverage(cfg)

    def beamsearch(self, model, sample):
        raw_hyps = self.inference_step(
            self.generator, [model], sample, prefix_tokens=sample["prefix_tokens"]
        )
        hyps = []
        for i, sample_id in enumerate(sample["id"].tolist()):
            prefix_len = sample["prefix_tokens"][i].ne(1).sum().item()
            prob = raw_hyps[i][0]["positional_scores"][prefix_len:].sum().exp()
            all_tokens = raw_hyps[i][0]["tokens"]
            answer_tokens = all_tokens[prefix_len:]
            detok_hypo_str = decode_fn(
                answer_tokens, self.tgt_dict, self.bpe, self.generator
            )
            hyps.append((detok_hypo_str.strip(), prob, answer_tokens, all_tokens))
        return hyps

    def valid_step(self, sample, model, criterion, **extra_kwargs):
        loss, sample_size, logging_output = super().valid_step(
            sample, model, criterion, **extra_kwargs
        )
        self.scorer.add(
            logging_output["confidences"].cpu().numpy(),
            logging_output["scores"].cpu().numpy(),
        )
        return loss, sample_size, logging_output

    def get_valid_stats(self, cfg, trainer, agg):
        # This is called at the end of the epoch.
        results = self.scorer.compute()
        self.scorer.reset()
        print("RANK:", dist.get_rank(), "res:", results)
        for key in results:
            agg[key] = results[key]

        return agg

    def conf_step(self, model, sample, criterion=None, training=False):
        # The training arg is used for mixup training. We don't want mixup in evaluation.

        src_tokens = sample["net_input"]["src_tokens"]
        # with torch.no_grad() # TODO do we need this? it should not matter
        hypos = self.beamsearch(model, sample)
        scores = torch.tensor(
            [
                ref_dict.get(hyp, 0)
                for ref_dict, (hyp, prob, tok, all_tok) in zip(
                    sample["ref_dict"], hypos
                )
            ]
        ).to(src_tokens.device)
        probs = torch.tensor([prob for (hyp, prob, tok, all_tok) in hypos]).to(
            device=src_tokens.device
        )

        all_ans_tokens = [tok for (hyp, prob, tok, all_tok) in hypos]
        token_ids = [tok[:-1] for (hyp, prob, tok, all_tok) in hypos]
        token_ids_collate = data_utils.collate_tokens(
            token_ids,
            pad_idx=model.encoder.padding_idx,
        )

        embed_tokens = model.encoder.embed_tokens(token_ids_collate)
        answer_emb = embed_tokens.mean(dim=1)
        prefix_len = [
            sample["prefix_tokens"][i].ne(1).sum().item()
            for i in range(len(sample["prefix_tokens"]))
        ]
        selector_input = {
            "prob": probs,
            "answer_emb": answer_emb,
            "prefix_len": prefix_len,
            "token_ids": all_ans_tokens,
        }

        # copy sample
        new_net_input = deepcopy(sample["net_input"])

        new_prev_output_tokens = []
        bos_token = torch.tensor([self.generator.bos]).to(
            device=new_net_input["prev_output_tokens"].device
        )

        for i in range(len(prefix_len)):
            (hyp, prob, tok, all_tok) = hypos[i]
            # Here check if we have a labeled predicted answer

            # Add BOS and remove EOS for input to the generator
            all_tok = torch.cat([bos_token, all_tok[:-1]])
            if "prediction" in sample and "score" in sample:
                prediction = sample["prediction"][i]
                if prediction is not None:
                    new_ans = (
                        self.tgt_dict.encode_line(
                            line=self.bpe.encode(prediction),
                            add_if_not_exist=False,
                            append_eos=False,
                        )
                        .long()
                        .to(device=all_tok.device)
                    )
                all_tok = torch.cat([all_tok[: prefix_len[i] + 1], new_ans])
            new_prev_output_tokens.append(all_tok)

        new_prev_output_tokens = data_utils.collate_tokens(
            new_prev_output_tokens, pad_idx=model.encoder.padding_idx
        )
        new_net_input["prev_output_tokens"] = new_prev_output_tokens.detach()

        x, extras = model(**new_net_input, selector_input=selector_input)
        confidence_output = extras["confidence"]
        confidences = confidence_output.squeeze(1)

        return x, extras, hypos, confidences, scores, probs

    def compute_probability(self, ans_logits: list, hypos):
        if self.constraint_trie is not None:
            for i in range(len(ans_logits)):
                logits = ans_logits[i]
                constraint_masks = logits.new_zeros(logits.size()).bool()
                hyp, prob, ans_tok, all_tok = hypos[i]
                for j in range(len(ans_tok)):
                    possible_ans = self.constraint_trie.get_next_layer(
                        [0] + ans_tok[:j].tolist()
                    )
                    constraint_masks[j, possible_ans] = True
                logits.masked_fill_(~constraint_masks, -math.inf)
        log_softmax = [F.log_softmax(log, dim=-1) for log in ans_logits]
        return log_softmax

    def inference(self, model, sample):
        x, extras, hypos, confidences, scores, probs = self.conf_step(model, sample)

        return [
            (hyp, prob, conf, score)
            for ((hyp, _, ans, all_tok), conf, prob, score) in zip(
                hypos, confidences, probs, scores
            )
        ]
