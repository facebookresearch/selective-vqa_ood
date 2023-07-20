# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import json
import sys

import tqdm

csv.field_size_limit(sys.maxsize)


def main(original_train, out, predictions_path):
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    qid_to_pred = {pred["question_id"]: pred for pred in predictions}

    with open(original_train) as fd:
        rd = csv.reader(fd, delimiter="\t")

        with open(out, "w") as outfile:
            writer = csv.writer(outfile, delimiter="\t", lineterminator="\n")
            for row in tqdm.tqdm(rd):
                qid = int(row[0])
                if qid in qid_to_pred:
                    pred = qid_to_pred[qid]
                    ans = pred["answer"]
                    ref = row[3]
                    answers = {
                        item.split("|!+")[1]: float(item.split("|!+")[0])
                        for item in ref.rsplit("&&")
                    }
                    score = answers.get(ans, 0.0)
                    row.append(ans)
                    row.append(score)
                    writer.writerow(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--original_train", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--predictions_path", type=str, required=True)
    args = parser.parse_args()
    main(args.original_train, args.out, args.predictions_path)
