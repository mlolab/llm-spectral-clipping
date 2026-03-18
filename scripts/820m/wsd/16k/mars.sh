
torchrun --nproc_per_node=4 your_path/src/main.py \
    --config_format base --model llama --distributed_backend nccl \
    --n_embd 2048 --n_head 16 --n_layer 12 \
    --batch_size 31 --sequence_length 1024 --acc_steps 32 \
    --dataset finewebedu --iterations 16000 --datasets_dir 'your_path/data/' \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.1 --seed 0 \
    --opt mars --lr 1e-3 --mars_lr 3e-3 --weight_decay 0.1 \
    --scheduler wsd --decay_type sqrt --wsd_fract_decay 0.2 \
    --beta1 0.8 --mars_beta1 0.95 --beta2 0.999 --mars_beta2 0.99 --mars_vr_gamma 0.025 \
    --wandb --wandb_project your_project --wandb_entity your_entity \
    --eval_interval 115 --latest_ckpt_interval 1000 \
    --untied_embeds \
    --log_parameter_norms \
    --shared_memory \
    --results_base_folder your_path/exp/820M/16k/ \
