#!/bin/bash
base_dir_0='/mnt/pub0'
base_dir_1='/mnt/pub1'
log_dir="${base_dir_1}/ssl-pretraining/logs"

set -e
set -x

data_dir="${base_dir_1}/ssl-pretraining/data/polyp"
run_id="$(date +"%y%m%d%H%M%S")-PixPro"
log_dir="${base_dir_1}/ssl-pretraining/logs/${run_id}"

# --zip  
# python main_pretrain.py \
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 12348 --nproc_per_node=1 \
    main_pretrain.py \
    --seed 0 \
    --run_id $run_id \
    --data-dir ${data_dir} \
    --output-dir ${log_dir} \
    --local_rank 0 \
    \
    --cache-mode no \
    --crop 0.08 \
    --aug BYOL \
    --dataset ImageNet \
    --batch-size 32 \
    --num_workers 32 \
    \
    --model PixPro \
    --arch resnet50 \
    --head-type early_return \
    \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 100 \
    --amp-opt-level O1 \
    \
    --save-freq 10 \
    --auto-resume \
    \
    --pixpro-p 2 \
    --pixpro-momentum 0.99 \
    --pixpro-pos-ratio 0.7 \
    --pixpro-transform-layer 1 \
    --pixpro-ins-loss-weight 1 \
