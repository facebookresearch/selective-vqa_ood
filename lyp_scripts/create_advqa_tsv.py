# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import base64
import csv
import json
import os
from collections import Counter
from io import BytesIO

import numpy as np

from answer_processor import EvalAIAnswerProcessor

from PIL import Image
from tqdm import tqdm

processor = EvalAIAnswerProcessor()


def get_scores_vqa(answers):
    for i in range(len(answers)):
        answers[i] = answers[i].replace("\n", " ")
        answers[i] = answers[i].replace("\t", " ")
        answers[i] = answers[i].strip()

    res = {}
    for resAns in set(answers):
        gtAcc = []
        for i in range(len(answers)):
            otherGTAns = answers[:i] + answers[i + 1 :]
            # print(len(otherGTAns))
            matchingAns = [ans for ans in otherGTAns if ans == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)

        avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
        res[resAns] = avgGTAcc
    return res


def get_ans_str(answers):
    ans_str = []
    answers = [processor(ans) for ans in answers]

    scores = get_scores_vqa(answers)

    for ans in scores:
        score = scores[ans]

        ans_str.append(f"{score}|!+{ans}")
    ans_str = "&&".join(ans_str)
    return ans_str


def encode_question(q):
    q = q.replace("  ", " ")
    q = q.replace("  ", " ")
    q = q.replace("  ", " ")
    q = q.lower().strip()
    return q


def main(questions_path, annot_path, output_path, question_ids=None, img_dir=None):
    with open(questions_path, "r") as f:
        questions = json.load(f)["questions"]

    with open(annot_path, "r") as f:
        annots = json.load(f)["annotations"]

    # Filter by qid
    if question_ids is not None:
        with open(question_ids, "r") as f:
            question_ids = [int(qid) for qid in json.load(f)]
        breakpoint()
        questions = [q for q in questions if q["question_id"] in question_ids]
        annots = [a for a in annots if a["question_id"] in question_ids]

    with open(output_path, "w", newline="") as tsvfile:
        writer = csv.writer(
            tsvfile,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            quotechar="",
        )

        for i, item in enumerate(tqdm(questions)):
            annot = annots[i]
            image_id = item["image_id"]
            image_name = f"COCO_val2014_{image_id:012d}.jpg"
            answers = [a["answer"] for a in annot["answers"]]

            ans_str = get_ans_str(answers)
            if img_dir is not None:
                img_path = os.path.join(img_dir, image_name)
            else:
                img_path = image_name

            img_data = "path://" + img_path

            line = [
                item["question_id"],
                image_id,
                encode_question(item["question"]),
                ans_str,
                "",
                img_data,
            ]
            writer.writerow(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("questions_path")
    parser.add_argument("annot_path")
    parser.add_argument("output_path")
    parser.add_argument("--question-ids")
    parser.add_argument("--img-directory", required=True)
    args = parser.parse_args()
    main(
        args.questions_path,
        args.annot_path,
        args.output_path,
        args.question_ids,
        args.img_directory,
    )
