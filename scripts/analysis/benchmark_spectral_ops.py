"""
Microbenchmark for spectral clipping operations.
Compares clip_sigvals vs Muon's normalize_sigvals (zeropower_via_newtonschulz5).

Usage:
    python scripts/analysis/benchmark_spectral_ops.py
    python scripts/analysis/benchmark_spectral_ops.py --output_dir ./plots
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from src.optim.post_process import clip_sigvals, _normalize_sigvals_matrix


def benchmark_function(fn, *args, warmup=20, repeats=100, **kwargs):
    """Benchmark a function with proper GPU synchronization."""
    device = args[0].device if hasattr(args[0], "device") else "cpu"

    # Warmup - ensure JIT compilation completes
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    if device != "cpu":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeats):
        if device != "cpu":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fn(*args, **kwargs)
        if device != "cpu":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        "mean": times.mean(),
        "std": times.std(),
        "min": times.min(),
        "max": times.max(),
        "median": np.median(times),
    }


def print_table(results, sizes, ns_iters):
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print("MICROBENCHMARK RESULTS: soft spectral clipping (SSC) vs orthogonalization (Muon)")
    print("=" * 100)

    for m, n, desc in sizes:
        print(f"\n{desc}: {m} x {n}")
        print("-" * 80)
        print(f"{'ns_iter':>8} | {'soft spectral clipping (ms)':>25} | {'orthogonalization (ms)':>25} | {'ratio':>10}")
        print("-" * 80)

        for ns_iter in ns_iters:
            key = (m, n, ns_iter)
            clip_stats = results["clip"][key]
            norm_stats = results["norm"][key]
            ratio = clip_stats["mean"] / norm_stats["mean"]

            print(
                f"{ns_iter:>8} | "
                f"{clip_stats['mean']:>7.3f} +/- {clip_stats['std']:>5.3f} | "
                f"{norm_stats['mean']:>7.3f} +/- {norm_stats['std']:>5.3f} | "
                f"{ratio:>10.2f}x"
            )


def plot_results(results, sizes, ns_iters, output_dir):
    """Generate plots for paper."""
    os.makedirs(output_dir, exist_ok=True)

    # Set up matplotlib for paper-quality figures
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.figsize": (8, 5),
            "figure.dpi": 150,
        }
    )

    # Plot 1: Time vs ns_iter for each matrix size
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    for idx, (m, n, desc) in enumerate(sizes):
        ax = axes[idx]

        clip_times = [results["clip"][(m, n, ns)]["mean"] for ns in ns_iters]
        clip_stds = [results["clip"][(m, n, ns)]["std"] for ns in ns_iters]
        norm_times = [results["norm"][(m, n, ns)]["mean"] for ns in ns_iters]
        norm_stds = [results["norm"][(m, n, ns)]["std"] for ns in ns_iters]

        ax.errorbar(
            ns_iters,
            clip_times,
            yerr=clip_stds,
            marker="o",
            label="soft spectral clipping (SSC)",
            color="#c842cb",
            capsize=3,
        )
        ax.errorbar(
            ns_iters,
            norm_times,
            yerr=norm_stds,
            marker="s",
            label="orthogonalization (Muon)",
            color="#6246a3",
            capsize=3,
        )

        ax.set_xlabel("Newton-Schulz iterations")
        ax.set_ylabel("Time (ms)")
        ax.set_title(f"{desc}\n({m} x {n})")
        ax.set_xticks(ns_iters)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_time_vs_iter.pdf"))
    plt.savefig(os.path.join(output_dir, "benchmark_time_vs_iter.png"))
    plt.close()

    # Plot 2: Ratio comparison (bar chart) at ns_iter=10
    fig, ax = plt.subplots(figsize=(8, 5))

    ns_iter_fixed = 10
    labels = [f"{m}x{n}\n({desc})" for m, n, desc in sizes]
    clip_times = [results["clip"][(m, n, ns_iter_fixed)]["mean"] for m, n, _ in sizes]
    norm_times = [results["norm"][(m, n, ns_iter_fixed)]["mean"] for m, n, _ in sizes]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, clip_times, width, label="soft spectral clipping (SSC)", color="#c842cb")
    bars2 = ax.bar(x + width / 2, norm_times, width, label="orthogonalization (Muon)", color="#6246a3")

    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Per-Matrix Time Comparison (ns_iter={ns_iter_fixed})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add ratio labels on top of bars
    for i, (c, n) in enumerate(zip(clip_times, norm_times)):
        ratio = c / n
        max_height = max(c, n)
        ax.annotate(
            f"{ratio:.2f}x",
            xy=(i, max_height),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_comparison.pdf"))
    plt.savefig(os.path.join(output_dir, "benchmark_comparison.png"))
    plt.close()

    # Plot 3: Scaling with matrix size (fixed ns_iter=10)
    fig, ax = plt.subplots(figsize=(8, 5))

    # Sort by total elements m*n
    sorted_sizes = sorted(sizes, key=lambda x: x[0] * x[1])
    total_elements = [m * n / 1e6 for m, n, _ in sorted_sizes]  # in millions

    clip_times = [
        results["clip"][(m, n, ns_iter_fixed)]["mean"] for m, n, _ in sorted_sizes
    ]
    norm_times = [
        results["norm"][(m, n, ns_iter_fixed)]["mean"] for m, n, _ in sorted_sizes
    ]

    ax.plot(total_elements, clip_times, "o-", label="soft spectral clipping (SSC)", color="#c842cb", markersize=8)
    ax.plot(total_elements, norm_times, "s-", label="orthogonalization (Muon)", color="#6246a3", markersize=8)

    ax.set_xlabel("Matrix Size (millions of elements)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Scaling with Matrix Size (ns_iter={ns_iter_fixed})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_scaling.pdf"))
    plt.savefig(os.path.join(output_dir, "benchmark_scaling.png"))
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Benchmark spectral operations")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./benchmark_plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--repeats", type=int, default=100, help="Number of timed iterations"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run benchmark on"
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Matrix sizes from typical transformer models
    # Format: (m, n, description) where m = min(rows, cols), n = max
    sizes = [
        (768, 2304, "160M attn"),
        (768, 3072, "160M mlp"),
        (1024, 3072, "250M attn"),
        (1024, 4096, "250M mlp"),
        (1536, 4608, "720M attn"),
        (1536, 6144, "720M mlp"),
    ]

    ns_iters = [5, 10, 15, 20]

    print(f"Running benchmark on {args.device}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}")

    results = {"clip": {}, "norm": {}}

    for m, n, desc in sizes:
        print(f"\nBenchmarking {desc}: {m} x {n}...")

        # Create random matrix
        X = torch.randn(m, n, device=args.device, dtype=torch.bfloat16)

        for ns_iter in ns_iters:
            # Benchmark clip_sigvals
            clip_stats = benchmark_function(
                clip_sigvals,
                X,
                clip_c=1.0,
                ns_iter=ns_iter,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            results["clip"][(m, n, ns_iter)] = clip_stats

            # Benchmark normalize_sigvals (Muon's method)
            norm_stats = benchmark_function(
                _normalize_sigvals_matrix,
                X,
                steps=ns_iter,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            results["norm"][(m, n, ns_iter)] = norm_stats

            print(f"  ns_iter={ns_iter}: clip={clip_stats['mean']:.3f}ms, norm={norm_stats['mean']:.3f}ms")

    # Print table
    print_table(results, sizes, ns_iters)

    # Generate plots
    plot_results(results, sizes, ns_iters, args.output_dir)


if __name__ == "__main__":
    main()
