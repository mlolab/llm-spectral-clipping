import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import wandb
import yaml

from logger.logger import DynamicsLogger
from optim.svd_recorder import (create_noise_recorder, create_svd_recorder,
                                create_update_noise_recorder)
from optim.weight_averaging import (ExponentialWeightAverager, WeightAverager,
                                    eval_ewa, eval_wa)

from .utils import (eval, get_batch, get_parameter_norms,
                    get_sum_of_linf_norms, load_checkpoint, load_worker_state,
                    log_prodigy_lr, save_checkpoint, save_worker_state,
                    visualize_routing)


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        load_worker_state(ckpt_dir)
    else:
        curr_iter = 0

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )
    if cfg.exponential_weight_average:
        ewa = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ewa_interval,
            decay=cfg.ewa_decay,
            warmup=cfg.warmup_steps if cfg.ewa_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter

    # Initialize SVD recorder for spectral analysis
    # This uses a general method that works for ANY optimizer by computing
    # U_k = (X_k - X_{k+1}) / (-lr) from weight differences before/after step
    svd_recorder = None
    if distributed_backend.is_master_process():
        svd_recorder = create_svd_recorder(not_compiled_model, cfg, exp_dir)

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]

    # Initialize noise structure recorder for analyzing gradient noise
    # This computes the "true" gradient G by averaging over many batches,
    # then analyzes how stochastic gradient noise N = g - G relates to G
    # NOTE: Must be after train_reader is extracted from datareaders
    noise_recorder = None
    if distributed_backend.is_master_process():
        # Get data source for the noise recorder's independent DataReader
        data_src = (
            train_reader.data
            if train_reader.data is not None
            else train_reader.data_path
        )
        noise_recorder = create_noise_recorder(
            not_compiled_model, cfg, data_src, exp_dir
        )

    # Initialize update noise recorder for analyzing optimizer update noise
    # This computes the "true" update U by doing optimizer.step() with large-batch gradient,
    # then analyzes how stochastic update noise N = u - U relates to U
    update_noise_recorder = None
    if distributed_backend.is_master_process():
        data_src = (
            train_reader.data
            if train_reader.data is not None
            else train_reader.data_path
        )
        update_noise_recorder = create_update_noise_recorder(
            not_compiled_model, opt, cfg, data_src, exp_dir
        )

    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    grad_norms = []
    model.train()

    # Timing stats for --log_step_timing (cleared every log_interval)
    timing_stats = {
        "forward": [],
        "backward": [],
        "spectral": [],
        "optimizer": [],
        "other": [],
    }
    # Full-run timing accumulator (never cleared, saved at end of training)
    timing_stats_all = {
        "forward": [],
        "backward": [],
        "spectral": [],
        "optimizer": [],
        "other": [],
    }

    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_and_log(
                tokens,
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

            if cfg.exponential_weight_average:
                eval_ewa(
                    curr_iter,
                    not_compiled_model,
                    ewa,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()

        # Initialize timing accumulators for this step
        if cfg.log_step_timing:
            t_forward_total = 0.0
            t_backward_total = 0.0
            if "cuda" in cfg.device:
                torch.cuda.synchronize()

        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)

            # Forward pass timing
            if cfg.log_step_timing:
                if "cuda" in cfg.device:
                    torch.cuda.synchronize()
                t_forward_start = time.perf_counter()

            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y, moe=cfg.moe)

            if cfg.log_step_timing:
                if "cuda" in cfg.device:
                    torch.cuda.synchronize()
                t_forward_total += time.perf_counter() - t_forward_start

            loss = outputs["loss"] / cfg.acc_steps

            # Backward pass timing
            if cfg.log_step_timing:
                if "cuda" in cfg.device:
                    torch.cuda.synchronize()
                t_backward_start = time.perf_counter()

            loss.backward()

            if cfg.log_step_timing:
                if "cuda" in cfg.device:
                    torch.cuda.synchronize()
                t_backward_total += time.perf_counter() - t_backward_start

            substep += 1

        # Record SVD of raw gradients (before clipping and optimizer processing)
        if svd_recorder is not None and svd_recorder.should_record(curr_iter + 1):
            # Use curr_iter + 1 because we record after backward but before step
            # The model used here should be the non-compiled version for parameter access
            raw_model = not_compiled_model
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                raw_model = model.module
            svd_recorder.record_gradients(raw_model, curr_iter + 1)

        # Spectral gradient clipping (per-parameter, 2D matrices only)
        # Initialize spectral timing
        if cfg.log_step_timing:
            t_spectral_total = 0.0

        if cfg.spectral_grad_clip == "clip":
            from optim.post_process import clip_sigvals

            raw_model = not_compiled_model
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                raw_model = model.module

            # Pre-clip threshold schedule
            if (
                cfg.spectral_grad_clip_c_end is not None
                and cfg.spectral_grad_clip_schedule != "constant"
            ):
                c_start = cfg.spectral_grad_clip_c
                c_end = cfg.spectral_grad_clip_c_end
                t = min(curr_iter / max(cfg.iterations, 1), 1.0)

                if cfg.spectral_grad_clip_schedule == "linear":
                    clip_c = c_start + (c_end - c_start) * t
                elif cfg.spectral_grad_clip_schedule == "cos":
                    clip_c = (
                        c_start + (c_end - c_start) * (1 - math.cos(math.pi * t)) * 0.5
                    )
                elif cfg.spectral_grad_clip_schedule == "sqrt":
                    clip_c = c_start + (c_end - c_start) * math.sqrt(t)
                elif cfg.spectral_grad_clip_schedule == "exp":
                    clip_c = c_start * (c_end / max(c_start, 1e-10)) ** t
                elif cfg.spectral_grad_clip_schedule == "square":
                    clip_c = c_start + (c_end - c_start) * t**2
                else:
                    clip_c = cfg.spectral_grad_clip_c
            else:
                # Existing behavior: constant c (with optional warmup dynamic clip)
                current_lr = opt.param_groups[0]["lr"]
                if (
                    not cfg.disable_dynamic_clip
                    and cfg.warmup_steps > 0
                    and curr_iter < cfg.warmup_steps
                ):
                    # During warmup: c * lr = prod_lr_c (constant)
                    prod_lr_c = cfg.spectral_grad_clip_c * cfg.lr
                    clip_c = prod_lr_c / max(current_lr, 1e-10)
                else:
                    clip_c = cfg.spectral_grad_clip_c

            # Time spectral gradient clipping
            if cfg.log_step_timing:
                if "cuda" in cfg.device:
                    torch.cuda.synchronize()
                t_spectral_start = time.perf_counter()

            for p in raw_model.parameters():
                if p.grad is not None and p.ndim == 2:
                    # Use copy_ to preserve tensor properties for fused optimizers
                    p.grad.data.copy_(
                        clip_sigvals(
                            p.grad.data,
                            clip_c=clip_c,
                            ns_iter=cfg.spectral_ns_steps,
                            use_float32=cfg.ns_float32,
                        )
                    )

            if cfg.log_step_timing:
                if "cuda" in cfg.device:
                    torch.cuda.synchronize()
                t_spectral_total += time.perf_counter() - t_spectral_start

        if cfg.grad_clip != 0.0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), cfg.grad_clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
            grad_norms.append(grad_norm)

        if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
            opt.train()

        # Store weights before optimizer step for general update SVD recording
        if svd_recorder is not None and svd_recorder.should_record(curr_iter + 1):
            raw_model = not_compiled_model
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                raw_model = model.module
            svd_recorder.store_weights_before_step(raw_model, curr_iter + 1)

        # Optimizer step timing
        if cfg.log_step_timing:
            if "cuda" in cfg.device:
                torch.cuda.synchronize()
            t_optimizer_start = time.perf_counter()

        (
            opt.step()
            if cfg.opt != "sophiag"
            else opt.step(bs=cfg.sophia_bs * cfg.sequence_length)
        )

        if cfg.log_step_timing:
            if "cuda" in cfg.device:
                torch.cuda.synchronize()
            t_optimizer_total = time.perf_counter() - t_optimizer_start

        # Compute and record update SVD from weight differences (works for ALL optimizers)
        if svd_recorder is not None and svd_recorder.should_record(curr_iter + 1):
            raw_model = not_compiled_model
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                raw_model = model.module
            svd_recorder.compute_and_record_updates(raw_model, opt, curr_iter + 1)
        if cfg.scheduler != "none":
            scheduler.step()
        if cfg.opt == "sophiag":
            opt.zero_grad(set_to_none=True)
            if curr_iter % cfg.precondition_frequency == cfg.precondition_frequency - 1:
                sample_again = model(x, targets=y, get_logits=True)
                samp_dist = torch.distributions.Categorical(
                    logits=sample_again["logits"]
                )
                y_sample = samp_dist.sample()
                loss_sampled = torch.nn.functional.cross_entropy(
                    sample_again["logits"].view(-1, sample_again["logits"].size(-1)),
                    y_sample.view(-1),
                    ignore_index=-1,
                )
                (loss_sampled / cfg.acc_steps).backward()
                opt.update_hessian()
                opt.zero_grad(set_to_none=True)
                model.zero_grad()
        elif cfg.opt == "mars":
            opt.zero_grad(set_to_none=True)
            opt.update_last_grad()
        else:
            opt.zero_grad(set_to_none=True)

        # Save SVD records after optimizer step (updates have been recorded by optimizer)
        if svd_recorder is not None and svd_recorder.should_record(curr_iter + 1):
            svd_recorder.save_records(curr_iter + 1)

        # Record noise structure (analyzes true gradient vs stochastic gradient noise)
        # This is done after SVD recording and optimizer step to not interfere with training
        if noise_recorder is not None and noise_recorder.should_record(curr_iter + 1):
            raw_model = not_compiled_model
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                raw_model = model.module
            noise_recorder.record_noise_structure(
                raw_model, curr_iter + 1, type_ctx, cfg.device
            )

        # Record update noise structure (analyzes optimizer update noise)
        # This computes true update U from large-batch gradient, then analyzes stochastic update noise
        # IMPORTANT: This only runs on master (rank 0), but modifies optimizer state.
        # We need to sync optimizer state across all ranks after recording.
        should_record_update_noise = (
            update_noise_recorder is not None
            and update_noise_recorder.should_record(curr_iter + 1)
        )
        if should_record_update_noise:
            raw_model = not_compiled_model
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                raw_model = model.module
            update_noise_recorder.record_update_noise_structure(
                raw_model, opt, curr_iter + 1, type_ctx, cfg.device
            )

        # Sync model weights across all ranks after update noise recording
        # This is only needed for weights since the optimizer is never modified
        # (we use a temporary optimizer for recording that is discarded)
        if cfg.distributed_backend == "nccl" and getattr(
            cfg, "record_update_noise", False
        ):
            import torch.distributed as dist

            # Check if this step was a recording step (use same logic as recorder)
            svd_fractions = getattr(cfg, "svd_record_steps", [0.0, 0.05, 0.5, 0.99])
            record_steps = set(
                max(1, int(frac * cfg.iterations)) for frac in svd_fractions
            )
            if (curr_iter + 1) in record_steps:
                # Ensure all ranks reach this point before syncing
                dist.barrier()

                # Sync model weights using broadcast
                # Only rank 0 did the recording, so weights need to be synced
                for param in not_compiled_model.parameters():
                    dist.broadcast(param.data, src=0)

                if dist.get_rank() == 0:
                    print(
                        f"[UpdateNoiseRecorder] Synced weights across all {dist.get_world_size()} ranks"
                    )

        if cfg.weight_average:
            weight_averager.step(
                not_compiled_model, distributed_backend.is_master_process()
            )
        if cfg.exponential_weight_average:
            ewa.step(not_compiled_model, distributed_backend.is_master_process())

        dt = (time.perf_counter_ns() - t_start) / 1e9

        # Collect timing stats for this step
        if cfg.log_step_timing:
            timing_stats["forward"].append(t_forward_total * 1000)  # Convert to ms
            timing_stats["backward"].append(t_backward_total * 1000)
            timing_stats["spectral"].append(t_spectral_total * 1000)
            timing_stats["optimizer"].append(t_optimizer_total * 1000)
            t_other = (
                dt * 1000
                - (
                    t_forward_total
                    + t_backward_total
                    + t_spectral_total
                    + t_optimizer_total
                )
                * 1000
            )
            timing_stats["other"].append(t_other)
            # Also accumulate for full-run summary
            timing_stats_all["forward"].append(t_forward_total * 1000)
            timing_stats_all["backward"].append(t_backward_total * 1000)
            timing_stats_all["spectral"].append(t_spectral_total * 1000)
            timing_stats_all["optimizer"].append(t_optimizer_total * 1000)
            timing_stats_all["other"].append(t_other)

        curr_iter += 1

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps
            train_aux_losses = {
                f"train/{k}": v for k, v in outputs["aux_losses"].items()
            }

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            if cfg.opt == "prodigy":
                prodigy_efective_lrs = log_prodigy_lr(opt)

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e}"
            )
            if cfg.opt == "prodigy":
                print(f"effective_lr={prodigy_efective_lrs[0]:.2e}")

            if cfg.wandb:
                wandb_logs = {
                    "tokens": tokens,
                    "iter": curr_iter,
                    "train/loss": train_loss,
                    "train/perplexity": 2.71828**train_loss,
                    "lr": current_lrs[0],
                    "iter_dt": dt,
                    "max_grad_norm": max(grad_norms).item() if grad_norms else 0,
                    "mean_grad_norm": (
                        torch.tensor(grad_norms).mean().item() if grad_norms else 0
                    ),
                    **train_aux_losses,
                }

                if cfg.opt == "prodigy":
                    wandb_logs["effective_lr"] = prodigy_efective_lrs[0]

                if cfg.log_parameter_norms:
                    raw_model = distributed_backend.get_raw_model(model)
                    total_norm = get_parameter_norms(
                        raw_model, order=cfg.norm_order, only_2d=False
                    )
                    matrix_norm = get_parameter_norms(
                        raw_model, order=cfg.norm_order, only_2d=True
                    )
                    wandb_logs["model_norm/total"] = total_norm
                    wandb_logs["model_norm/matrices"] = matrix_norm
                    # Sum of L∞-norms
                    sum_linf_total = get_sum_of_linf_norms(raw_model, only_2d=False)
                    sum_linf_matrices = get_sum_of_linf_norms(raw_model, only_2d=True)
                    wandb_logs["model_norm/sum_linf_total"] = sum_linf_total
                    wandb_logs["model_norm/sum_linf_matrices"] = sum_linf_matrices

                if cfg.log_step_timing and timing_stats["forward"]:
                    # Compute averages over the logging interval
                    n = len(timing_stats["forward"])
                    t_forward_avg = sum(timing_stats["forward"]) / n
                    t_backward_avg = sum(timing_stats["backward"]) / n
                    t_spectral_avg = sum(timing_stats["spectral"]) / n
                    t_optimizer_avg = sum(timing_stats["optimizer"]) / n
                    t_other_avg = sum(timing_stats["other"]) / n
                    t_total_avg = (
                        t_forward_avg
                        + t_backward_avg
                        + t_spectral_avg
                        + t_optimizer_avg
                        + t_other_avg
                    )

                    wandb_logs["timing/forward_ms"] = t_forward_avg
                    wandb_logs["timing/backward_ms"] = t_backward_avg
                    wandb_logs["timing/spectral_ms"] = t_spectral_avg
                    wandb_logs["timing/optimizer_ms"] = t_optimizer_avg
                    wandb_logs["timing/other_ms"] = t_other_avg
                    wandb_logs["timing/total_ms"] = t_total_avg

                    # Compute fractions
                    if t_total_avg > 0:
                        wandb_logs["timing/forward_frac"] = t_forward_avg / t_total_avg
                        wandb_logs["timing/backward_frac"] = (
                            t_backward_avg / t_total_avg
                        )
                        wandb_logs["timing/spectral_frac"] = (
                            t_spectral_avg / t_total_avg
                        )
                        wandb_logs["timing/optimizer_frac"] = (
                            t_optimizer_avg / t_total_avg
                        )
                        wandb_logs["timing/other_frac"] = t_other_avg / t_total_avg

                    # Clear timing stats for next interval
                    for key in timing_stats:
                        timing_stats[key] = []

                wandb.log(wandb_logs)

            grad_norms = []

    # Save all SVD records at the end of training
    if svd_recorder is not None:
        svd_recorder.save_all_records()

    # Save all noise structure records at the end of training
    if noise_recorder is not None:
        noise_recorder.save_all_records()

    # Save all update noise structure records at the end of training
    if update_noise_recorder is not None:
        update_noise_recorder.save_all_records()

    # Save timing summary to results folder
    if (
        cfg.log_step_timing
        and timing_stats_all["forward"]
        and distributed_backend.is_master_process()
    ):
        import json as json_mod

        warmup_skip = max(10, len(timing_stats_all["forward"]) // 10)
        summary = {}
        for key in timing_stats_all:
            values = timing_stats_all[key][warmup_skip:]
            if values:
                mean = sum(values) / len(values)
                summary[f"{key}_ms_mean"] = round(mean, 3)
                summary[f"{key}_ms_std"] = round(
                    (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5, 3
                )

        total = sum(
            summary.get(f"{k}_ms_mean", 0)
            for k in ["forward", "backward", "spectral", "optimizer", "other"]
        )
        summary["total_ms_mean"] = round(total, 3)
        if total > 0:
            for key in ["forward", "backward", "spectral", "optimizer", "other"]:
                summary[f"{key}_frac"] = round(
                    summary.get(f"{key}_ms_mean", 0) / total, 4
                )

        summary["num_steps_total"] = len(timing_stats_all["forward"])
        summary["num_steps_measured"] = len(timing_stats_all["forward"]) - warmup_skip

        timing_path = exp_dir / "timing_summary.json"
        with open(timing_path, "w") as f:
            json_mod.dump(summary, f, indent=2)

    return stats


def eval_and_log(
    tokens,
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()
    if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity, val_aux_losses, router_logits = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        moe=cfg.moe,
        get_router_logits=cfg.moe and cfg.plot_router_logits,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "tokens": tokens,
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
                **val_aux_losses,
            }
        else:
            logs = {
                "tokens": tokens,
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
                **val_aux_losses,
            }
        if cfg.moe and cfg.plot_router_logits:
            routing_logs = visualize_routing(router_logits, cfg)
            logs = {**logs, **routing_logs}

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
