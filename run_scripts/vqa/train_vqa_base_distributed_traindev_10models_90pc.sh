#!/usr/bin/env

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Number of GPUs per GPU worker
GPUS_PER_NODE=4
# Number of GPU workers, for single-worker training, please set to 1
# WORKER_CNT=$SLURM_NNODES
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=localhost

# The port for communication
export MASTER_PORT=12350
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=$SLURM_NODEID

echo "MASTER_ADDR: $MASTER_ADDR"
echo "RANK :$RANK"

subset=$1 # Which subset to train on

ans2label_file=../../datasets/vqa2/trainval_ans2label.pkl
restore_file=../../checkpoints/ofa_base.pt
predval=../../datasets/vqa2/imdb_val2014-predval.tsv

selected_cols=0,5,2,3,4

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

task=vqa_gen
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
batch_size=16
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_object_length=30
max_tgt_length=30
num_bins=1000
patch_image_size=480

uses_ema="--uses-ema"
store_ema="--store-ema"
ema_fp32="--ema-fp32"
ema_decay=0.9999
ema_start_update=0

# Specify the inference type in validation after each fine-tuning epoch
# As mentioned in the readme, you can choose from allcand or beamsearch evaluation, default to allcand
val_inference_type=beamsearch


max_epoch=20
warmup_ratio=0.04
lr=5e-5
patch_image_size=480
train_data=../../datasets/vqa2/trainval-10-subsets/trainval-$subset.train.tsv

data=$train_data,$predval

name=base_trainval_10models_90pct_subset
save_dir=./vqa_checkpoints/${name}

mkdir -p $save_dir

# data=${data_dir}/vqa_train_corrected_subset-${subset}.tsv,${data_dir}/vqa_val.tsv
name_i="subset_${subset}"_"${max_epoch}"_"${lr}"_"${patch_image_size}"

save_path=${save_dir}/$name_i
mkdir -p $save_path

log_file=${save_path}/logs.txt

tensorboard_logdir=$save_path/tensorboard
mkdir -p $tensorboard_logdir


# sbatch -J "train-"${name}"_"${name_i} \
# --gpus-per-node ${GPUS_PER_NODE} --partition learnlab --time 4200 \
# train_selector.slurm \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE}  \
        --nnodes=${WORKER_CNT} \
        --node_rank=${RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
../../train.py \
    ${data} \
    --seed=${subset} \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --user-dir=${user_dir} \
    --finetune-from-model=${restore_file} \
    --restore-file="checkpoint_last.pt" \
    --save-dir=${save_path} \
    --task=${task} \
    --arch=${arch} \
    --criterion=${criterion} \
    --label-smoothing=${label_smoothing} \
    --batch-size=${batch_size} \
    --update-freq=${update_freq} \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --layernorm-embedding \
    --patch-layernorm-embedding \
    --code-layernorm-embedding \
    --resnet-drop-path-rate=${resnet_drop_path_rate} \
    --encoder-drop-path-rate=${encoder_drop_path_rate} \
    --decoder-drop-path-rate=${decoder_drop_path_rate} \
    --dropout=${dropout} \
    --attention-dropout=${attention_dropout} \
    --weight-decay=0.01 \
    --optimizer=adam \
    --adam-betas="(0.9,0.999)" \
    --adam-eps=1e-08 \
    --clip-norm=1.0 \
    --lr-scheduler=polynomial_decay \
    --lr=${lr} \
    --max-epoch=${max_epoch} \
    --warmup-ratio=${warmup_ratio} \
    --log-format=simple \
    --log-interval=10 \
    --fixed-validation-seed=7 \
    --keep-last-epochs=30 \
    --save-interval=1 --validate-interval=1 \
    --best-checkpoint-metric=vqa_score \
    --maximize-best-checkpoint-metric \
    --max-src-length=${max_src_length} \
    --max-object-length=${max_object_length} \
    --max-tgt-length=${max_tgt_length} \
    --find-unused-parameters \
    --freeze-encoder-embedding \
    --freeze-decoder-embedding \
    --ans2label-file=${ans2label_file} \
    --valid-batch-size=20 \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --scale-heads \
    --disable-entangle \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --prompt-type=prev_output \
    --fp16 \
    --fp16-scale-window=512 \
    ${uses_ema} \
    ${store_ema} \
    ${ema_fp32} \
    --ema-decay=${ema_decay} \
    --ema-start-update=${ema_start_update} \
    --val-inference-type=${val_inference_type} \
    --tensorboard-logdir=${tensorboard_logdir} \
    --num-workers=1  2>&1 | tee ${log_file} 
