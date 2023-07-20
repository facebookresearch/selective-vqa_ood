# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sklearn.metrics import auc
import torch.distributed as dist
import numpy as np

class RiskCoverage:

    def __init__(self, cfg=None, gather_dist=True):
        self.cfg = cfg
        self.confidences = []
        self.scores = []
        self.gather_dist = gather_dist

    def add(self, confidences, scores):
        # should be numpy arrays
        self.confidences.append(confidences)
        self.scores.append(scores)

    def compute_cov_at_risk(self, risk_level, cumulative_risks, sorted_conf):
        
        n = len(cumulative_risks)
        index = n

        while index > 0 and cumulative_risks[index - 1] >= risk_level:
            index -= 1
        index -= 1

        if -1 < index < (n - 1) and sorted_conf[index] == sorted_conf[index + 1]:
            while index > -1 and (
                sorted_conf[index] == sorted_conf[index + 1]
                    or cumulative_risks[index] >= risk_level
                ):
                index -= 1

        threshold = sorted_conf[index] if index > -1 else 0.0

        if index > -1:
            cov = (index+1) / n
        else:
            cov = 0

        return cov, threshold

    def compute(self):
        confidences = np.concatenate(self.confidences)
        scores = np.concatenate(self.scores)
        print("confidences", len(confidences))
        print("scores", len(scores))

        # gather results
        if self.gather_dist and dist.get_world_size() > 1:
            all_confidences = [None for _ in range(dist.get_world_size())]
            all_score = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_confidences, confidences)
            dist.all_gather_object(all_score, scores)
            confidences = np.concatenate(all_confidences)
            scores = np.concatenate(all_score)

        print("confidences after gather", len(confidences))
        print("scores after gather", len(scores))

        # if dist.get_world_size() == 1 or dist.get_rank() == 0:
        n = len(confidences)
        risks = 1 - scores
        num_elements = np.arange(1, n+1)
        coverage = num_elements / n
        # coverage = np.linspace(0, 1, num=n+1)[1:]  # remove the first element (zero)
        argsort = confidences.argsort()[::-1] # sort from high to low confidence
        sorted_conf = confidences[argsort]
        # sorted_conf = all_confidences[argsort]
        sorted_risks = risks[argsort]
        cumulative_risks = sorted_risks.cumsum() / num_elements
        auc_score = auc(coverage, cumulative_risks)

        res = {"auc": auc_score}

        for risk_level in [0.01, 0.05, 0.1, 0.2]:
            cov, threshold = self.compute_cov_at_risk(risk_level, cumulative_risks, sorted_conf)
            res[f"cov@{risk_level}"] = cov
            res[f"thresh@{risk_level}"] = threshold

        return res

    def reset(self):
        self.confidences = []
        self.scores = []
