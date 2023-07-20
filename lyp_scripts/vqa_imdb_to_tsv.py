# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import base64
import csv
import os
from collections import Counter
from io import BytesIO

import numpy as np
from answer_processor import EvalAIAnswerProcessor

from PIL import Image
from tqdm import tqdm

processor = EvalAIAnswerProcessor()
import pickle
import re


def encode_question(q):
    q = q.replace("  ", " ")
    q = q.replace("  ", " ")
    q = q.replace("  ", " ")
    q = q.lower().strip()
    return q


contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manualMap = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile("(?<=\d)(\,)+(?=\d)")
puncStrip = re.compile(
    r"(?<=[ \\;/\"`\[\](){}<>@=+_\-,?!])([\\;/\"`\[\](){}<>@=+_\-,?!])|([\\;/\"`\[\](){}<>@=+_\-,?!])(?=[ \\;/\"`\[\](){}<>@=+_\-,?!])"
)
puncStrip2 = re.compile(r"(?<=[a-zA-Z])([\\;/\"`\[\](){}<>@=+_\-,?!])(?=[a-zA-Z])")
puncStripBegin = re.compile(r"\A([ \\;/\"`\[\](){}<>@=+_\-,?!]+)(?=[a-zA-Z0-9 ])")
puncStripEnd = re.compile(r"(?<=[a-zA-Z0-9 ])([ \\;/\"`\[\](){}<>@=+_\-,?!]+)\Z")
spaceCleanup = re.compile(r"([ ]+)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def processPunctuation(inText):
    outText = puncStripBegin.sub("", inText)
    outText = puncStripEnd.sub("", outText)
    outText = commaStrip.sub("", outText)
    outText = puncStrip.sub(" ", outText)
    outText = spaceCleanup.sub(" ", outText)
    outText = puncStrip2.sub(" ", outText)
    outText = puncStrip2.sub("", outText)
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = " ".join(outText)
    return outText


def get_scores_vqa(answers):
    for i in range(len(answers)):
        answers[i] = answers[i].replace("\n", " ")
        answers[i] = answers[i].replace("\t", " ")
        answers[i] = answers[i].strip()

    # resAns = res['answer']
    # resAns = resAns.replace('\n', ' ')
    # resAns = resAns.replace('\t', ' ')
    # resAns = resAns.strip()
    if len(set(answers)) > 1:
        for i in range(len(answers)):
            answers[i] = processPunctuation(answers[i])
            answers[i] = processDigitArticle(answers[i])

    # resConf = 0.
    # resConf = res["confidence"]
    # resAnswered = -1
    # resAnswered = res.get("answered", None)

    # gtAnswers = [ans['answer'] for ans in gt['answers']]
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


def get_score_from_answers_train(answers):
    all_ans_and_scores = []
    ans_str = []
    answers = [processor(ans) for ans in answers]

    scores = get_scores_vqa(answers)

    for ans in scores:
        score = scores[ans]
        all_ans_and_scores.append((ans, f"{score}|!+{ans}"))

    return all_ans_and_scores


def get_score_from_answers_val(answers):
    ans_str = []
    answers = [processor(ans) for ans in answers]

    scores = get_scores_vqa(answers)

    for ans in scores:
        score = scores[ans]

        ans_str.append(f"{score}|!+{ans}")
    ans_str = "&&".join(ans_str)
    return ans_str


def main(imdb_path, output_path, img_root=None, ans2label=None, format="train"):
    print(f"Converting {imdb_path} to {output_path}")

    with open(ans2label, "rb") as f:
        ans2label = pickle.load(f)

    dataset = np.load(imdb_path, allow_pickle=True)
    val_coco_path = f"{img_root}/val2014/"
    train_coco_path = f"{img_root}/train2014/"
    error_printed = False
    if "dataset_name" in dataset[0]:
        dataset = dataset[1:]

    with open(output_path, "w", newline="") as tsvfile:
        writer = csv.writer(
            tsvfile,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            quotechar="",
        )

        for item in tqdm(dataset):
            image_name = item["feature_path"].replace(".npy", ".jpg")
            image_name = image_name.replace("vqa2/", "")

            if "val" in image_name:
                img_path = os.path.join(val_coco_path, image_name)
            elif "train" in image_name:
                img_path = os.path.join(train_coco_path, image_name)

            img_data = "path://" + img_path

            if format == "train":
                ans_and_scores = get_score_from_answers_train(
                    item["all_answers"] if "all_answers" in item else item["answers"]
                )
                for ans, ans_str in ans_and_scores:
                    if ans not in ans2label:
                        continue
                    line = [
                        item["question_id"],
                        item["image_id"],
                        encode_question(item["question_str"]),
                        ans_str,
                        "",  # objects
                        img_data,
                    ]
                    writer.writerow(line)
            elif format == "val":
                ans_str = get_score_from_answers_val(
                    item["all_answers"] if "all_answers" in item else item["answers"]
                )
                line = [
                    item["question_id"],
                    item["image_id"],
                    encode_question(item["question_str"]),
                    ans_str,
                    "",  # objects
                    img_data,
                ]
                writer.writerow(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("imdb_path")
    parser.add_argument("output_path")
    parser.add_argument("--ans2label", default="datasets/vqa2/trainval_ans2label.pkl")
    parser.add_argument("--img-root", default="/datasets01/COCO/060817/val2014")
    parser.add_argument("--format", default="val", choices=["train", "val"])
    args = parser.parse_args()
    main(
        args.imdb_path,
        args.output_path,
        img_root=args.img_root,
        ans2label=args.ans2label,
        format=args.format,
    )
