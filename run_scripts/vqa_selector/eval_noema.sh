#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
# export MASTER_PORT=8182
name=$1
path=$2
GPUS_PER_NODE=2
# MEM_PER_NODE=$((64*$GPUS_PER_NODE))G

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# val or test
split="val"

data=$3
ans2label_file=../../datasets/vqa2/trainval_ans2label.pkl
# result_path=../../results/$name
path_dir=$(dirname $path)
filename=$(basename $path)
# check if filename is equal to checkpoint_best.pt
# if not, then append the checkpoint_name to $name
if [ "$filename" != "checkpoint_best.pt" ]; then
    name=$name"_"$filename
fi

result_path=$path_dir/${name}/
# if val_predict.json exists in result_path, exit
if [ -f $result_path/val_predict.json ]; then
    echo "val_predict.json already exists in $result_path"
    exit
fi
selected_cols=0,5,2,3,4
valid_batch_size=64

# -m torch.distributed.launch --nproc_per_node=8 
port=`shuf -i 2000-65000 -n 1`

python3 -m torch.distributed.launch  \
--nproc_per_node=$GPUS_PER_NODE \
--master_port $port \
../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen \
    --batch-size=${valid_batch_size} \
    --log-format=simple \
    --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --beam-search-vqa-eval \
    --beam=5 \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=1 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\",\"valid_batch_size\":\"${valid_batch_size}\",\"add_object\":False}"
