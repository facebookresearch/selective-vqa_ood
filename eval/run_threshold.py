# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json

from run import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run reliable VQA evaluations.")
    parser.add_argument(
        "-q", "--questions", required=True, help="Path to question VQA2 json file"
    )
    parser.add_argument(
        "-a", "--annotations", required=True, help="Path to VQA2 annotation json file"
    )
    parser.add_argument(
        "--advqa-questions", required=False, help="Path to AdVQA annotation json file"
    )
    parser.add_argument(
        "--advqa-annots", required=False, help="Path to AdVQA annotation json file"
    )
    parser.add_argument(
        "-p",
        "--predictions-vqa",
        required=True,
        help="Path to VQA2 prediction json file",
    )
    parser.add_argument(
        "--predictions-val",
        required=True,
        help="Path to VQA2 prediction json file for validation",
    )
    parser.add_argument(
        "--predictions-advqa", required=False, help="Path to AdVQA prediction json file"
    )
    parser.add_argument(
        "--mixture-qids", required=False, help="Json file containing qids for mixture"
    )
    parser.add_argument(
        "-r", "--risk_tols", nargs="*", type=float, default=[0.01, 0.05, 0.1, 0.2]
    )
    parser.add_argument("-t", "--thresholds", nargs="*", type=float, default=None)
    parser.add_argument(
        "-c", "--costs", nargs="*", type=float, default=[1.0, 10.0, 100.0]
    )
    parser.add_argument(
        "--use-prob",
        action="store_true",
        help="use 'prob' instead of 'confidence' key from json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    full_ques_file = args.questions
    full_ann_file = args.annotations
    advqa_ques_file = args.advqa_questions
    advqa_ann_file = args.advqa_annots
    result_file = args.predictions_vqa
    advqa_res_file = args.predictions_advqa
    mixture_qids = args.mixture_qids
    result_val_file = args.predictions_val
    risk_tols = args.risk_tols

    use_prob = args.use_prob

    gt_data, pred_data, evaluator = load_data(
        full_ques_file, full_ann_file, result_val_file, risk_tols
    )

    # thresholds
    qids = pred_data.getQuesIds()
    evaluator.evaluate(gt_data, pred_data, quesIds=qids, use_prob=use_prob)
    evaluator.accuracy

    keys = ["cov@" + str(r) for r in risk_tols]

    thresholds = [evaluator.accuracy[k]["threshold"] for k in keys]

    keys_eff = ["thresholds_phi@" + str(r) for r in args.costs]
    thresholds_eff = [evaluator.accuracy[k] for k in keys_eff]

    # Test
    # for result_file_test in result_files_test:
    gt_data, pred_data, evaluator = load_data(
        full_ques_file,
        full_ann_file,
        result_file,
        risk_tols,
        advqa_ques_file=advqa_ques_file,
        advqa_annot_file=advqa_ann_file,
        advqa_res_file=advqa_res_file,
        thresholds=thresholds,
        mixture_qids=mixture_qids,
        thresholds_eff=thresholds_eff,
    )

    qids = list(set(pred_data.getQuesIds()))

    evaluator.evaluate(gt_data, pred_data, quesIds=qids, use_prob=use_prob)

    print(json.dumps(evaluator.accuracy, sort_keys=True, indent=4))
