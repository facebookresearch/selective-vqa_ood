# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#### LICENSE from https://github.com/GT-Vision-Lab/VQA
# Copyright (c) 2014, Aishwarya Agrawal
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The views and conclusions contained in the software and documentation are
# those
# of the authors and should not be interpreted as representing official
# policies,
# either expressed or implied, of the FreeBSD Project.

from __future__ import print_function, unicode_literals

__author__ = "aagrawal"
__version__ = "0.9"

import copy
import datetime

# Interface for accessing the VQA dataset.

# This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
# (https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py).

# The following functions are defined:
#  VQA        - VQA class that loads VQA annotation file and prepares data structures.
#  getQuesIds - Get question ids that satisfy given filter conditions.
#  getImgIds  - Get image ids that satisfy given filter conditions.
#  loadQA     - Load questions and answers with the specified question ids.
#  showQA     - Display the specified questions and answers.
#  loadRes    - Load result file and create result object.

# Help on each function can be accessed by: "help(COCO.function)"

import json
import sys


def log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


class VQA:
    def __init__(self, annotations=None, questions=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if not annotations == None and not questions == None:
            log("loading VQA annotations and questions into memory...")
            time_t = datetime.datetime.utcnow()
            # dataset = json.load(open(annotation_file, 'r'))
            # questions = json.load(open(question_file, 'r'))
            log(datetime.datetime.utcnow() - time_t)
            self.dataset = annotations
            self.questions = questions
            self.createIndex()

    def createIndex(self):
        # create index
        log("creating index...")
        imgToQA = {ann["image_id"]: [] for ann in self.dataset["annotations"]}
        qa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        qqa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        for ann in self.dataset["annotations"]:
            imgToQA[ann["image_id"]] += [ann]
            qa[ann["question_id"]] = ann
        for ques in self.questions["questions"]:
            qqa[ques["question_id"]] = ques
        log("index created!")

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.datset["info"].items():
            log("%s: %s" % (key, value))

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param  imgIds    (int array)   : get question ids for given imgs
                quesTypes (str array)   : get question ids for given question types
                ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                anns = sum(
                    [self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA],
                    [],
                )
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(quesTypes) == 0
                else [ann for ann in anns if ann["question_type"] in quesTypes]
            )
            anns = (
                anns
                if len(ansTypes) == 0
                else [ann for ann in anns if ann["answer_type"] in ansTypes]
            )
        ids = [ann["question_id"] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
        Get image ids that satisfy given filter conditions. default skips that filter
        :param quesIds   (int array)   : get image ids for given question ids
               quesTypes (str array)   : get image ids for given question types
               ansTypes  (str array)   : get image ids for given answer types
        :return: ids     (int array)   : integer array of image ids
        """
        quesIds = quesIds if type(quesIds) == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(quesIds) == 0:
                anns = sum(
                    [self.qa[quesId] for quesId in quesIds if quesId in self.qa], []
                )
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(quesTypes) == 0
                else [ann for ann in anns if ann["question_type"] in quesTypes]
            )
            anns = (
                anns
                if len(ansTypes) == 0
                else [ann for ann in anns if ann["answer_type"] in ansTypes]
            )
        ids = [ann["image_id"] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann["question_id"]
            log("Question: %s" % (self.qqa[quesId]["question"]))
            for ans in ann["answers"]:
                log("Answer %d: %s" % (ans["answer_id"], ans["answer"]))

    def loadRes(
        self, res, resFile, resFileAdVQA=None, shift_qid=None, mixture_qids=None
    ):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        # res = VQA()
        # res.questions = json.load(open(quesFile))
        res.questions = self.questions
        res.dataset["info"] = copy.deepcopy(self.questions["info"])
        res.dataset["task_type"] = copy.deepcopy(self.questions["task_type"])
        res.dataset["data_type"] = copy.deepcopy(self.questions["data_type"])
        res.dataset["data_subtype"] = copy.deepcopy(self.questions["data_subtype"])
        res.dataset["license"] = copy.deepcopy(self.questions["license"])

        log("Loading and preparing results...     ")
        time_t = datetime.datetime.utcnow()
        anns = json.load(open(resFile))

        # filter mixture qids from result file
        if mixture_qids is not None:
            mixture_vqa = set(mixture_qids["vqa2"])
            anns = [ann for ann in anns if ann["question_id"] in mixture_vqa]

        assert type(anns) == list, "results is not an array of objects"
        # annsQuesIds = [ann['question_id'] for ann in anns]
        # assert set(annsQuesIds) == set(self.getQuesIds()), \
        # 'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file. Please note that this year, you need to upload predictions on ALL test questions for test-dev evaluation unlike previous years when you needed to upload predictions on test-dev questions only.'
        for ann in anns:
            quesId = int(ann["question_id"])
            ann["question_id"] = quesId
            if res.dataset["task_type"] == "Multiple Choice":
                assert (
                    ann["answer"] in self.qqa[quesId]["multiple_choices"]
                ), "predicted answer is not one of the multiple choices"
            # breakpoint()
            qaAnn = self.qa[quesId]
            ann["image_id"] = qaAnn["image_id"]
            ann["question_type"] = qaAnn["question_type"]
            ann["answer_type"] = qaAnn["answer_type"]
        log("DONE (t=%0.2fs)" % ((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset["annotations"] = anns

        if resFileAdVQA is not None:
            annsAdVQ = json.load(open(resFileAdVQA))
            if mixture_qids is not None:
                mixture_advqa = set(mixture_qids["advqa"])
                annsAdVQ = [
                    ann for ann in annsAdVQ if ann["question_id"] in mixture_advqa
                ]

            for ann in annsAdVQ:
                ann["question_id"] = ann["question_id"] + shift_qid
                qaAnn = self.qa[ann["question_id"]]
                ann["image_id"] = qaAnn["image_id"]
                ann["question_type"] = qaAnn.get("question_type", "none")
                ann["answer_type"] = qaAnn.get("answer_type", "none")
                anns.append(ann)

            log(
                "DONE AdVQA (t=%0.2fs)"
                % ((datetime.datetime.utcnow() - time_t).total_seconds())
            )

        res.createIndex()
        return res
