
torchrun --nproc_per_node=4 your_path/src/main.py \
    --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 12 \
    --batch_size 64 --sequence_length 512 --acc_steps 4 \
    --dataset slimpajama --iterations 16000 --datasets_dir 'your_path/data/' \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt adamw --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.8 --beta2 0.999 \
    --wandb --wandb_project your_project --wandb_entity your_entity \
    --eval_interval 115 --latest_ckpt_interval 1000 \
    --results_base_folder your_path/results/noise/ \
    --record_noise_structure --noise_num_samples 4096 --noise_top_k 5 \
    --noise_num_repeats 50 --noise_batch_size 1 \
    --svd_record_steps 0.0 0.05 0.5 0.99 \
    --svd_layers embedding early middle late \