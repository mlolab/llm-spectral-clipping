"""
Standalone downstream evaluation script using lm-evaluation-harness.

Evaluates trained model checkpoints on standard benchmarks (HellaSwag, ARC, PIQA, etc.)
and saves results as JSON files.

Usage:
    # Single checkpoint
    python src/eval/run_eval.py \
        --checkpoint_path /path/to/ckpts/16000/main.pt \
        --config_path /path/to/exp/summary.json \
        --tasks hellaswag,arc_easy,piqa \
        --num_fewshot 0 --batch_size 32 --device cuda:0 \
        --output_dir ./eval_results/

    # Multiple checkpoints from one run
    python src/eval/run_eval.py \
        --checkpoint_dir /path/to/ckpts/ \
        --config_path /path/to/exp/summary.json \
        --iterations 4000,8000,16000 \
        --tasks hellaswag,arc_easy,piqa

    # Without summary.json (specify model args directly)
    python src/eval/run_eval.py \
        --checkpoint_path /path/to/main.pt \
        --model_type llama --n_embd 768 --n_head 12 --n_layer 12 \
        --tasks hellaswag
"""

import argparse
import json
import sys
from pathlib import Path

# Add src/ to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import lm_eval
import torch
from lm_eval import evaluator

from eval.lm_eval_wrapper import CustomLM
from models.utils import get_model


def load_config_from_file(config_path):
    """Load training config from summary.json or config.json."""
    with open(config_path) as f:
        data = json.load(f)
    # summary.json wraps args under "args" key; config.json is flat
    if "args" in data:
        return data["args"]
    return data


def build_model_args(args_dict):
    """Build an argparse.Namespace with model-relevant fields from a config dict."""
    ns = argparse.Namespace(**args_dict)
    # Ensure required fields have defaults
    for attr, default in [
        ("model", "llama"),
        ("n_embd", 768),
        ("n_head", 12),
        ("n_layer", 12),
        ("sequence_length", 512),
        ("vocab_size", 50304),
        ("dropout", 0.0),
        ("bias", False),
        ("untied_embeds", False),
        ("init_std", 0.02),
        ("multiple_of", 256),
        ("n_kv_head", None),
        ("rmsnorm_eps", 1e-5),
        ("use_pretrained", "none"),
        ("moe", False),
        ("moe_num_experts", 8),
        ("moe_routing", "standard_gating"),
        ("capacity_factor", 2.0),
        ("moe_num_shared_experts", 0),
        ("mlp_dim_exp_factor", 1.0),
        ("parallel_block", False),
        ("from_dense", False),
        ("device", "cuda:0"),
    ]:
        if not hasattr(ns, attr):
            setattr(ns, attr, default)
    return ns


def load_model_from_checkpoint(checkpoint_path, config_path, device, cli_args=None):
    """Load model from checkpoint + config."""
    if config_path:
        args_dict = load_config_from_file(config_path)
    else:
        # Build from CLI args
        args_dict = {
            "model": cli_args.model_type,
            "n_embd": cli_args.n_embd,
            "n_head": cli_args.n_head,
            "n_layer": cli_args.n_layer,
            "sequence_length": cli_args.sequence_length,
            "vocab_size": cli_args.vocab_size,
            "dropout": 0.0,
            "bias": False,
            "untied_embeds": False,
            "init_std": 0.02,
            "multiple_of": 256,
            "n_kv_head": None,
            "rmsnorm_eps": 1e-5,
            "use_pretrained": "none",
            "moe": False,
        }

    args_dict["device"] = device
    model_args = build_model_args(args_dict)
    model = get_model(model_args).to(device)

    # Load checkpoint weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]

    # Handle compiled model checkpoints (_orig_mod prefix)
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned[k[len("_orig_mod.") :]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned)
    model.eval()

    return model, model_args


def main():
    parser = argparse.ArgumentParser(description="Downstream task evaluation")

    # Checkpoint source (one of these is required)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to a single checkpoint (main.pt)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing iteration subdirs with main.pt",
    )
    parser.add_argument(
        "--iterations",
        type=str,
        default=None,
        help="Comma-separated iteration numbers (used with --checkpoint_dir)",
    )

    # Config source
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to summary.json or config.json",
    )

    # Fallback model args (when no config file available)
    parser.add_argument(
        "--model_type", type=str, default="llama", choices=["base", "llama"]
    )
    parser.add_argument("--n_embd", type=int, default=768)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=50304)

    # Eval params
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,arc_easy,arc_challenge,piqa,winogrande,boolq,lambada_openai",
        help="Comma-separated lm-eval task names",
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--output_dir", type=str, default="./eval_results/")
    parser.add_argument(
        "--log_samples", action="store_true", help="Log individual sample predictions"
    )

    args = parser.parse_args()

    # Determine checkpoints to evaluate
    checkpoints = []
    if args.checkpoint_path:
        checkpoints.append(args.checkpoint_path)
    elif args.checkpoint_dir and args.iterations:
        for itr in args.iterations.split(","):
            ckpt_path = Path(args.checkpoint_dir) / itr.strip() / "main.pt"
            if ckpt_path.exists():
                checkpoints.append(str(ckpt_path))
            else:
                print(f"Warning: checkpoint not found: {ckpt_path}")
    elif args.checkpoint_dir:
        # Auto-discover iteration subdirs
        ckpt_dir = Path(args.checkpoint_dir)
        for subdir in sorted(ckpt_dir.iterdir()):
            if subdir.is_dir() and (subdir / "main.pt").exists():
                if subdir.name != "latest":
                    checkpoints.append(str(subdir / "main.pt"))
    else:
        parser.error("Must provide --checkpoint_path or --checkpoint_dir")

    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return

    tasks = args.tasks.split(",")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for ckpt_path in checkpoints:
        # Extract iteration number from path
        itr = Path(ckpt_path).parent.name

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {ckpt_path}")
        print(f"Tasks: {', '.join(tasks)}")
        print(f"{'=' * 60}")

        model, model_args = load_model_from_checkpoint(
            ckpt_path, args.config_path, args.device, cli_args=args
        )

        lm_wrapper = CustomLM(
            model=model,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
        )

        results = evaluator.simple_evaluate(
            model=lm_wrapper,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            log_samples=args.log_samples,
        )

        task_results = results["results"]

        # Save per-checkpoint results
        result_file = output_dir / f"eval_iter_{itr}.json"
        with open(result_file, "w") as f:
            json.dump(task_results, f, indent=2, default=str)

        # Print summary
        print(f"\nResults for iteration {itr}:")

        # Separate MMLU subtasks from other tasks
        mmlu_scores = []
        other_tasks = {}
        for task_name, task_metrics in task_results.items():
            if task_name.startswith("mmlu_") and task_name != "mmlu":
                acc = task_metrics.get("acc,none", None)
                if acc is not None:
                    mmlu_scores.append(acc)
            elif task_name == "mmlu":
                # Skip the group-level entry (we compute our own average)
                continue
            else:
                other_tasks[task_name] = task_metrics

        # Print non-MMLU tasks
        all_scores = []
        for task_name, task_metrics in other_tasks.items():
            acc_norm = task_metrics.get("acc_norm,none", None)
            acc = task_metrics.get("acc,none", None)
            metric = acc_norm if acc_norm is not None else acc
            metric_name = "acc_norm" if acc_norm is not None else "acc"
            if metric is not None:
                print(f"  {task_name:20s}: {metric_name}={metric:.4f}")
                all_scores.append(metric)
            else:
                print(f"  {task_name:20s}: {task_metrics}")

        # Print MMLU average
        if mmlu_scores:
            mmlu_avg = sum(mmlu_scores) / len(mmlu_scores)
            print(
                f"  {'mmlu (avg)':20s}: acc={mmlu_avg:.4f}  ({len(mmlu_scores)} subjects)"
            )
            all_scores.append(mmlu_avg)

        # Print overall average
        if all_scores:
            overall_avg = sum(all_scores) / len(all_scores)
            print(f"\n  {'AVERAGE':20s}: {overall_avg:.4f}  ({len(all_scores)} tasks)")

        all_results[itr] = task_results

        # Free GPU memory before next checkpoint
        del model, lm_wrapper
        torch.cuda.empty_cache()

    # Save combined results
    combined_file = output_dir / "eval_all.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {combined_file}")


if __name__ == "__main__":
    main()
