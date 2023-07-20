# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

mixture_file=$1     # for instance, datasets/mixtures/test-5000-vqa2_5000-advqa.json
val_file=$2         # val_predict.json for VQA v2
advqa_preds=$3

python eval/run.py \
-q datasets/vqa2/v2_OpenEnded_mscoco_val2014_questions.json \
-a datasets/vqa2/v2_mscoco_val2014_annotations.json \
--advqa-questions datasets/advqa/v1_OpenEnded_mscoco_val2017_advqa_questions.json \
--advqa-annots datasets/advqa/v1_mscoco_val2017_advqa_annotations.json \
-p $val_file \
--predictions-advqa $advqa_preds \
--mixture-qids $mixture_file
