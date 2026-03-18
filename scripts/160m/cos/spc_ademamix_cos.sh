
torchrun --nproc_per_node=4 your_path/src/main.py \
    --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 12 \
    --batch_size 64 --sequence_length 512 --acc_steps 4 \
    --dataset finewebedu --iterations 128000 --datasets_dir 'your_path/data/' \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt ademamix --lr 2e-3 --weight_decay 0.1 --scheduler cos \
    --spectral_post_process clip --spectral_clip_c 5.0 \
    --spectral_ns_steps 10 --spectral_apply_to 'all' \
    --beta1 0.9 --beta2 0.999 \
    --adema_beta3 0.999 --adema_alpha 8.0 \
    --adema_beta3_warmup 128000 --adema_alpha_warmup 128000 \
    --wandb --wandb_project your_project --wandb_entity your_entity \
    --eval_interval 115 --latest_ckpt_interval 1000 \
    --untied_embeds \
    --log_parameter_norms \
    --shared_memory \
    --results_base_folder your_path/exp/160M/
