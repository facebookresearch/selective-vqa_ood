# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os

from reliable_vqa_eval import ReliabilityEval
from vqa import VQA


def load_json(fname):
    with open(fname, "r") as f:
        data_ = json.load(f)
    return data_


def load_data(
    ques_file,
    ann_file,
    res_file,
    risk_tolerances,
    advqa_ques_file=None,
    advqa_annot_file=None,
    advqa_res_file=None,
    thresholds=None,
    thresholds_eff=None,
    mixture_qids=None,
):
    questions = load_json(ques_file)
    annotations = load_json(ann_file)

    if mixture_qids is not None:
        mixture_qids = load_json(mixture_qids)

    shift_qid = None
    if advqa_annot_file is not None and advqa_ques_file is not None:
        # Merge all the annotations
        advqa_questions = load_json(advqa_ques_file)
        advqa_annotations = load_json(advqa_annot_file)

        # First, find the maximum qid in the original questions
        max_qid = max([q["question_id"] for q in questions["questions"]])
        shift_qid = max_qid + 1
        # Add this to all the AdVQA questions and annotations, and predictions
        for q in advqa_questions["questions"]:
            q["question_id"] += shift_qid
        for a in advqa_annotations["annotations"]:
            a["question_id"] += shift_qid
        print(len(questions["questions"]))
        # concatenate the questions and annotations
        questions["questions"] += advqa_questions["questions"]
        print(len(questions["questions"]))
        annotations["annotations"] += advqa_annotations["annotations"]

    ann_vqa = VQA(annotations=annotations, questions=questions)
    all_qids = ann_vqa.getQuesIds()

    vqa_eval = ReliabilityEval(
        all_qids,
        risk_tolerances=risk_tolerances,
        n=2,
        thresholds=thresholds,
        thresholds_eff=thresholds_eff,
    )
    res_vqa = ann_vqa.loadRes(
        VQA(),
        res_file,
        resFileAdVQA=advqa_res_file,
        shift_qid=shift_qid,
        mixture_qids=mixture_qids,
    )

    return ann_vqa, res_vqa, vqa_eval


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run reliable VQA evaluations.")
    parser.add_argument(
        "-q", "--questions", required=True, help="Path to question VQA2 json file"
    )
    parser.add_argument(
        "-a", "--annotations", required=True, help="Path to VQA2 annotation json file"
    )
    parser.add_argument("--advqa-questions", help="Path to AdVQA annotation json file")
    parser.add_argument("--advqa-annots", help="Path to AdVQA annotation json file")
    parser.add_argument(
        "-p",
        "--predictions-vqa",
        required=True,
        help="Path to VQA2 prediction json file",
    )
    parser.add_argument(
        "--predictions-advqa", help="Path to AdVQA prediction json file"
    )
    parser.add_argument("--mixture-qids", help="Json file containing qids for mixture")
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
    if not result_file.endswith(".json"):
        result_file = os.path.join(result_file, "val_predict.json")
    risk_tols = args.risk_tols

    thresholds = args.thresholds
    use_prob = args.use_prob

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
    )

    qids = list(set(pred_data.getQuesIds()))

    preds = load_json(result_file)

    evaluator.evaluate(gt_data, pred_data, quesIds=qids, use_prob=use_prob)

    print(json.dumps(evaluator.accuracy, sort_keys=True, indent=4))
