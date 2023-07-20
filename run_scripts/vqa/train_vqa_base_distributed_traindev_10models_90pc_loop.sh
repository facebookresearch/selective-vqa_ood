# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

for id in 0 1 2 3 4 5 6 7 8 9
do
sbatch -J "train-models-trainval-10models-subset-$id" \
train_model.slurm \
bash train_vqa_base_distributed_trainval_10models_90pc.sh $id
done
