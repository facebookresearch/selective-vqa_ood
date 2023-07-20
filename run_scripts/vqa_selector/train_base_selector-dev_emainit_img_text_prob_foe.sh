#!/usr/bin/env

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Number of GPUs per GPU worker
GPUS_PER_NODE=2
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

data_dir=../../datasets/vqa2/

data=${data_dir}/imdb_val2014-dev.tsv,${data_dir}/imdb_val2014-predval.tsv

ans2label_file=../../datasets/vqa2/trainval_ans2label.pkl
finetune_from_model=../../../ofa/run_scripts/vqa/vqa_checkpoints/base_no_obj_trainonly/15_0.04_5e-5_480/checkpoint_best.pt

selected_cols=0,5,2,3,4

name=base_no_obj_trainonly_selector_emainit_feats-img-text-prob-first_output_emb
log_dir=./vqa_logs/${name}
save_dir=./vqa_checkpoints/${name}
tensorboard_logdir=tensorboard/${name}
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

task=vqa_select_pred
arch=ofa_select_base
criterion=selective_prediction
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
selector_features="pooled_text+pooled_img+prob+first_output_emb"

uses_ema=""
store_ema=""
ema_fp32="--ema-fp32"
ema_decay=0.9999
ema_start_update=0

val_inference_type=beamsearch
max_epoch=128
lr=1e-4
patch_image_size=480

port=`shuf -i 2000-65000 -n 1`
log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}"_rank"${RANK}".log"
save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}
tensorboard_logfile=${save_path}
mkdir -p $save_path

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE}  \
  --nnodes=1 \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  ../../train.py \
      ${data} \
      --selected-cols=${selected_cols} \
      --selector-features=${selector_features}  \
      --bpe-dir=${bpe_dir} \
      --user-dir=${user_dir} \
      --finetune-from-model=${finetune_from_model} \
      --restore-file="checkpoint_last.pt" \
      --save-dir=${save_path} \
      --task=${task} \
      --arch=${arch} \
      --criterion=${criterion} \
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
      --log-format=simple \
      --log-interval=10 \
      --fixed-validation-seed=7 \
      --keep-last-epochs=15 \
      --save-interval=1 --validate-interval=1 \
      --best-checkpoint-metric=auc \
      --max-src-length=${max_src_length} \
      --max-object-length=${max_object_length} \
      --max-tgt-length=${max_tgt_length} \
      `# --find-unused-parameters` \
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
      --use-ema-weights-to-init-param \
      --ema-decay=${ema_decay} \
      --ema-start-update=${ema_start_update} \
      --val-inference-type=${val_inference_type} \
      --tensorboard-logdir=${tensorboard_logfile} \
      --all-gather-list-size 32768 \
      --num-workers=1  2>&1 | tee ${log_file}
