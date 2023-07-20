# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for i in 0 1 2 3 4 5 6 7 8 9
do
python lyp_scripts/add_conf_labels.py \
    --original_train datasets/vqa2/trainval-10-subsets/held-out-$subset.held-out.tsv \
    --predictions_path vqa_checkpoints/base_trainval_10models_90pct_subset/subset_${i}_20_0.04_5e-5_480/vqa2-heldout/val_predict.json \
    --out vqa_checkpoints/base_trainval_10models_90pct_subset/subset_${i}_20_0.04_5e-5_480/vqa2-heldout/conf-labeled.valformat.tsv
done

## Concatenate all labeled held-out sets.
cat vqa_checkpoints/base_trainval_10models_90pct_subset/subset_*_20_0.04_5e-5_480/vqa2-heldout/conf-labeled.valformat.tsv \
> datasets/vqa2/trainval-10-subsets/trainval-conflabeled.valformat.tsv
