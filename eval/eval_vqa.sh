# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

val_file=$1     # val_predict.json for VQA v2

python eval/run.py \
-q datasets/vqa2/v2_OpenEnded_mscoco_val2014_questions.json \
-a datasets/vqa2/v2_mscoco_val2014_annotations.json \
-p $val_file
