# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

COCO_PATH=$1

python lyp_scripts/vqa_imdb_to_tsv.py datasets/reliable_vqa/annotations/imdb_val2014-dev.npy datasets/vqa2/imdb_val2014-dev.tsv --img-root ${COCO_PATH}
python lyp_scripts/vqa_imdb_to_tsv.py datasets/reliable_vqa/annotations/imdb_val2014-dev.npy datasets/vqa2/imdb_val2014-dev.trainformat.tsv  --format train --img-root ${COCO_PATH}

python lyp_scripts/vqa_imdb_to_tsv.py datasets/reliable_vqa/annotations/imdb_val2014-val.npy datasets/vqa2/imdb_val2014-predval.tsv --img-root ${COCO_PATH}

python lyp_scripts/vqa_imdb_to_tsv.py datasets/reliable_vqa/annotations/imdb_val2014-test.npy datasets/vqa2/imdb_val2014-test.tsv --img-root ${COCO_PATH}

python lyp_scripts/vqa_json_to_tsv.py datasets/vqa2/v2_OpenEnded_mscoco_train2014_questions.json datasets/vqa2/v2_mscoco_train2014_annotations.json datasets/vqa2/imdb_train2014.trainformat.tsv --img-root ${COCO_PATH}

python lyp_scripts/vqa_json_to_tsv.py datasets/vqa2/v2_OpenEnded_mscoco_train2014_questions.json datasets/vqa2/v2_mscoco_train2014_annotations.json datasets/vqa2/imdb_train2014.valformat.tsv --format val --img-root ${COCO_PATH}

cat datasets/vqa2/imdb_train2014.trainformat.tsv | shuf > datasets/vqa2/imdb_train2014.trainformat.shuf.tsv
cat datasets/vqa2/imdb_train2014.trainformat.tsv datasets/vqa2/imdb_val2014-dev.trainformat.tsv | shuf > datasets/vqa2/imdb_val2014-traindev.trainformat.tsv

cat datasets/vqa2/imdb_train2014.valformat.tsv datasets/vqa2/imdb_val2014-dev.tsv | shuf > datasets/vqa2/imdb_val2014-traindev.valformat.tsv

python lyp_scripts/create_advqa_tsv.py  datasets/advqa/v1_OpenEnded_mscoco_val2017_advqa_questions.json datasets/advqa/v1_mscoco_val2017_advqa_annotations.json datasets/advqa/advqa.tsv --img-directory $COCO_PATH/val2014