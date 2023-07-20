# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data_utils import collate_tokens
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

# from scr


@dataclass
class SelectivePredLossConfig(FairseqDataclass):
    pass


@register_criterion("selective_prediction", dataclass=SelectivePredLossConfig)
class SelectivePredictionCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
    ):
        super().__init__(task)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, model, sample, update_num=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # forward model once to get the features

        # Run beamsearch to get an answer with its probability
        x, extras, hypos, confidences, scores, probs = self.task.conf_step(
            model, sample, training=(update_num is not None)
        )

        if "prediction" in sample and "score" in sample:
            saved_scores = sample["score"]
            for i in range(len(scores)):
                if saved_scores[i] is None:
                    saved_scores[i] = scores[i]
            saved_scores = saved_scores.astype(np.float32)
            scores = torch.tensor(saved_scores).to(
                device=scores.device, dtype=scores.dtype
            )

        if scores.dtype == torch.long:
            scores = scores.float()

        loss = self.loss(confidences, scores)
        sample_size = len(scores)
        mse = F.mse_loss(confidences.sigmoid(), scores)

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            # "nll_loss": loss.data,
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "mse": mse,
            "rmse": torch.sqrt(mse),
            "scores": scores,
            "confidences": confidences,
            "acc": scores.mean(),
        }

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        mse = sum(log.get("mse", 0) for log in logging_outputs)
        mse = mse / len(logging_outputs)
        rmse = torch.sqrt(mse)
        metrics.log_scalar(
            "loss", loss_sum / len(logging_outputs), sample_size, round=3
        )

        metrics.log_scalar(
            "acc",
            sum(log.get("acc", 0) for log in logging_outputs) / len(logging_outputs),
            round=3,
        )

        metrics.log_scalar("ntokens", ntokens, 1, round=3)
        metrics.log_scalar("nsentences", nsentences, 1, round=3)
        metrics.log_scalar("sample_size", sample_size, 1, round=3)
        metrics.log_scalar("rmse", rmse, 1, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
