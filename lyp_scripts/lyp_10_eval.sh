# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

cd run_scripts/vqa

for i in 0 1 2 3 4 5 6 7 8 9
do
bash eval_ema_dataset.sh vqa2-heldout \
vqa_checkpoints/base_trainval_10models_90pct_subset/subset_${i}_20_5e-5_480/checkpoint_best.pt \
/checkpoint/mrf/projects/22rmm/cdancette/new/ofa-data/vqa_data/trainval-10-subsets/held-out-${i}.held-out.tsv
done
