#!/bin/bash

torchrun --nproc_per_node=4 ./src/main.py \
    --config_format base --model mup_llama --distributed_backend nccl \
    --n_embd 1536 --n_head 24 --n_layer 48 \
    --batch_size 13 --sequence_length 512 --acc_steps 160 \
    --dataset finewebedu --iterations 8000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.1 --seed 0 \
    --opt d-muon --lr 2e-3 --weight_decay 0.1 \
    --beta1 0.9 --beta2 0.99 \
    --momentum 0.95 --nesterov True \
    --scheduler wsd --decay_type sqrt --wsd_fract_decay 0.2 \
    --eval_interval 115 --latest_ckpt_interval 200 \
    --log_interval 1 \
    --untied_embeds \
    --permanent_ckpt_interval 400 \
    --log_parameter_norms \