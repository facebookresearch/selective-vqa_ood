# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from email.policy import default
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture

from models.ofa.ofa import OFAClassificationHead, OFAModel


def get_arg(args, key, default=None):
    args = vars(args)
    return args.get(key, default)


@register_model("ofa_select")
class OFASelect(OFAModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        freeze_vqa = not get_arg(args, "no_freeze_vqa", False)

        if freeze_vqa:
            for module in self.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()
                    module.track_running_stats = False
                if isinstance(module, torch.nn.modules.BatchNorm2d):
                    module.eval()
                    module.track_running_stats = False
                if isinstance(module, torch.nn.modules.BatchNorm3d):
                    module.eval()
                    module.track_running_stats = False

            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False

        token_dim = get_arg(args, "decoder_embed_dim", 768)
        hidden_1 = get_arg(args, "selector_hidden_1", 768)
        hidden_2 = get_arg(args, "selector_hidden_2", 768)
        select_dropout = get_arg(args, "select_dropout", 0.1)

        self.selector_features = get_arg(
            args, "selector_features", "pooled_text"
        ).split("+")

        input_dim = 0
        if "pooled_text" in self.selector_features:
            input_dim += token_dim
        if "original_pooled_text" in self.selector_features:
            input_dim += token_dim

        if "pooled_img" in self.selector_features:
            input_dim += token_dim
        if "original_pooled_img" in self.selector_features:
            input_dim += token_dim
        if "first_logits" in self.selector_features:
            input_dim += len(self.encoder.dictionary)
        if "first_output_emb" in self.selector_features:
            input_dim += token_dim
        if "mean_output_emb" in self.selector_features:
            input_dim += token_dim
        if "original_mean_output_emb" in self.selector_features:
            input_dim += token_dim
        if "answer_emb" in self.selector_features:
            input_dim += token_dim

        if "model_prob" in self.selector_features:
            input_dim += 1
        if "prob" in self.selector_features:
            input_dim += 1

        self.selective_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.Dropout(p=select_dropout),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.Dropout(p=select_dropout),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
        )

    @staticmethod
    def add_args(parser):
        super(OFASelect, OFASelect).add_args(parser)
        parser.add_argument(
            "--selector-features", type=str, default="pooled_text+pooled_img+prob"
        )
        parser.add_argument("--selector-hidden-1", default=768, type=int)
        parser.add_argument("--selector-hidden-2", default=768, type=int)
        parser.add_argument("--selector-dropout", default=0.1, type=float)
        parser.add_argument("--no-freeze-vqa", action="store_true", default=False)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        patch_images: Optional[torch.Tensor] = None,
        patch_images_2: Optional[torch.Tensor] = None,
        patch_masks: Optional[torch.Tensor] = None,
        code_masks: Optional[torch.Tensor] = None,
        sample_patch_num: Optional[int] = None,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        selector_input=None,
    ):
        # Freeze batchnorm during fine-tuning
        for module in self.modules():
            if isinstance(module, torch.nn.modules.BatchNorm1d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm2d):
                module.eval()
            if isinstance(module, torch.nn.modules.BatchNorm3d):
                module.eval()

        if classification_head_name is not None:
            features_only = True

        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            patch_images=patch_images,
            patch_masks=patch_masks,
            patch_images_2=patch_images_2,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens,
            sample_patch_num=sample_patch_num,
        )
        text_embed_len = encoder_out["src_tokens_len"]
        img_embed_len = encoder_out["image_embed_len"]
        encoder_token_embeddings = encoder_out["token_embedding"][0]

        img_embed = encoder_out["encoder_out"][0][:img_embed_len]
        text_embed = encoder_out["encoder_out"][0][
            img_embed_len : img_embed_len + text_embed_len
        ]

        text_embed_pooled = text_embed.max(dim=0).values
        img_embed_pooled = img_embed.max(dim=0).values

        x, extra = self.decoder(
            prev_output_tokens,
            code_masks=code_masks,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        extra["encoder_token_embeddings"] = encoder_token_embeddings
        extra["pooled_text"] = text_embed_pooled
        extra["pooled_img"] = img_embed_pooled
        extra["text_embed"] = text_embed
        extra["img_embed"] = img_embed
        extra["encoder_out"] = encoder_out["encoder_out"][0]

        extra["all_text_embed"] = [
            text_embed[:k, i] for (k, i) in zip(src_lengths, range(len(src_lengths)))
        ]
        extra["text_embed_mean"] = torch.stack(
            [txt.mean(dim=0) for txt in extra["all_text_embed"]]
        )
        extra["img_embed_mean"] = img_embed.mean(dim=0)

        if selector_input is not None:
            # first output token
            prev_length = prev_output_tokens.ne(self.encoder.padding_idx).sum(1)

            if "prefix_len" in selector_input:
                prefix_len = selector_input["prefix_len"]

            extra["all_logits"] = [
                x[i][k:end]
                for (i, k, end) in zip(range(len(x)), prefix_len, prev_length)
            ]
            extra["first_logits"] = torch.stack(
                [x[i][k] for (i, k) in zip(range(len(x)), prefix_len)]
            )
            extra["first_output_emb"] = torch.stack(
                [extra["features"][i][k] for (i, k) in zip(range(len(x)), prefix_len)]
            )
            # output emb
            extra["mean_output_emb"] = torch.stack(
                [
                    extra["features"][i][k:end].mean(0)
                    for (i, k, end) in zip(range(len(x)), prefix_len, prev_length)
                ]
            )

            if "constraint_trie" in selector_input and "hypos" in selector_input:
                probs = self.compute_probability(
                    extra["all_logits"],
                    selector_input["hypos"],
                    selector_input["constraint_trie"],
                )
                extra["model_prob"] = probs.unsqueeze(1)

            # This will be overriden by selector_input if necessary.
            extra["original_pooled_text"] = extra["pooled_text"]
            extra["original_pooled_img"] = extra["pooled_img"]
            extra["original_mean_output_emb"] = extra["mean_output_emb"]

            for key in selector_input:
                extra[key] = selector_input[key]

            extra["confidence"] = self.forward_confidence(extra)

        pad = self.encoder.padding_idx
        if classification_head_name is not None:
            prev_lengths = prev_output_tokens.ne(pad).sum(1)
            gather_index = (
                prev_lengths[:, None, None].expand(x.size(0), 1, x.size(2)) - 1
            )
            sentence_representation = x.gather(1, gather_index).squeeze()
            if self.classification_heads[classification_head_name].use_two_images:
                hidden_size = sentence_representation.size(1)
                sentence_representation = sentence_representation.view(
                    -1, hidden_size * 2
                )
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break

        return x, extra

    def compute_probability(self, ans_logits: list, hypos, constraint_trie):
        if constraint_trie is not None:
            for i in range(len(ans_logits)):
                logits = ans_logits[i]
                constraint_masks = logits.new_zeros(logits.size()).bool()
                hyp, prob, ans_tok, all_tok = hypos[i]
                for j in range(len(ans_tok)):
                    possible_ans = constraint_trie.get_next_layer(
                        [0] + ans_tok[:j].tolist()
                    )
                    constraint_masks[j, possible_ans] = True
                logits.masked_fill_(~constraint_masks, -math.inf)
        log_softmax = [F.log_softmax(log, dim=-1) for log in ans_logits]
        all_ans_tokens = [tok for (hyp, prob, tok, all_tok) in hypos]
        logprobs = [
            torch.gather(lg, 1, tk_ids[:, None]).sum()
            for (lg, tk_ids) in zip(log_softmax, all_ans_tokens)
        ]
        return torch.stack(logprobs)

    def forward_confidence(self, inputs):
        selector_inputs = []
        if "pooled_text" in self.selector_features:
            selector_inputs.append(inputs["pooled_text"])
        if "text_encoder_out" in self.selector_features:
            selector_inputs.append(inputs["text_encoder_out"])

        if "pooled_img" in self.selector_features:
            selector_inputs.append(inputs["pooled_img"])
        if "img_encoder_out" in self.selector_features:
            selector_inputs.append(inputs["img_encoder_out"])

        if "prob" in self.selector_features:
            # (batch, 1)
            probs = inputs["prob"].unsqueeze(1).to(dtype=inputs["pooled_img"].dtype)
            selector_inputs.append(probs)
        if "answer_emb" in self.selector_features:
            selector_inputs.append(inputs["answer_emb"])
        if "first_logits" in self.selector_features:
            selector_inputs.append(inputs["first_logits"])
        if "first_output_emb" in self.selector_features:
            selector_inputs.append(inputs["first_output_emb"])
        if "mean_output_emb" in self.selector_features:
            selector_inputs.append(inputs["mean_output_emb"])
        if "original_mean_output_emb" in self.selector_features:
            selector_inputs.append(inputs["original_mean_output_emb"])

        if "original_pooled_text" in self.selector_features:
            selector_inputs.append(inputs["original_pooled_text"])

        if "original_pooled_img" in self.selector_features:
            selector_inputs.append(inputs["original_pooled_img"])

        if "model_prob" in self.selector_features:
            selector_inputs.append(inputs["model_prob"])

        sel_input = torch.cat(selector_inputs, dim=1)

        if "mixup_pairs" in inputs:
            print("mixup in model")
            new_inputs = []
            # add new items to batch
            for pair, lambd in zip(inputs["mixup_pairs"], inputs["mixup_lambdas"]):
                new_input = sel_input[pair[0]] * lambd + sel_input[pair[1]] * (
                    1 - lambd
                )
                new_inputs.append(new_input)

            new_inputs = torch.stack(new_inputs, dim=0)
            sel_input = torch.cat([sel_input, new_inputs], dim=0)

        if "detach" not in inputs:
            sel_input = sel_input.detach()
        elif "detach" in inputs and inputs["detach"] == True:
            sel_input = sel_input.detach()
        return self.selective_predictor(sel_input)


@register_model_architecture("ofa_select", "ofa_select_large")
def ofa_large_architecture(args):
    args.no_strict_load = getattr(args, "no_strict_load", True)
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.0)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.max_source_positions = getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.pooler_classifier = getattr(args, "pooler_classifier", "mlp")

    args.resnet_drop_path_rate = getattr(args, "resnet_drop_path_rate", 0.0)
    args.encoder_drop_path_rate = getattr(args, "encoder_drop_path_rate", 0.0)
    args.decoder_drop_path_rate = getattr(args, "decoder_drop_path_rate", 0.0)

    args.resnet_type = getattr(args, "resnet_type", "resnet152")
    args.token_bucket_size = getattr(args, "token_bucket_size", 256)
    args.image_bucket_size = getattr(args, "image_bucket_size", 42)

    args.freeze_encoder_embedding = getattr(args, "freeze_encoder_embedding", False)
    args.freeze_decoder_embedding = getattr(args, "freeze_decoder_embedding", False)
    args.add_type_embedding = getattr(args, "add_type_embedding", True)
    args.attn_scale_factor = getattr(args, "attn_scale_factor", 2)

    args.code_image_size = getattr(args, "code_image_size", 128)
    args.patch_layernorm_embedding = getattr(args, "patch_layernorm_embedding", True)
    args.code_layernorm_embedding = getattr(args, "code_layernorm_embedding", True)
    args.entangle_position_embedding = getattr(
        args, "entangle_position_embedding", False
    )
    args.disable_entangle = getattr(args, "disable_entangle", False)
    args.sync_bn = getattr(args, "sync_bn", False)

    args.scale_attn = getattr(args, "scale_attn", False)
    args.scale_fc = getattr(args, "scale_fc", False)
    args.scale_heads = getattr(args, "scale_heads", False)
    args.scale_resids = getattr(args, "scale_resids", False)

    # TODO selective predictor architecture


@register_model_architecture("ofa_select", "ofa_select_base")
def ofa_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.resnet_type = getattr(args, "resnet_type", "resnet101")
    ofa_large_architecture(args)


@register_model_architecture("ofa_select", "ofa_select_medium")
def ofa_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 512)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.resnet_type = getattr(args, "resnet_type", "resnet101")
    ofa_large_architecture(args)
