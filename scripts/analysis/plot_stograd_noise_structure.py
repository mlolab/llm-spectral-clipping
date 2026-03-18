"""
Plot noise structure analysis results.

This script visualizes the relationship between "true" gradients G and
stochastic gradient noise N = g - G at different training steps.

Output plots:
1. Grid plot with:
   - Columns: training steps (0%, 5%, 50%, 99%)
   - Rows: layers (2 rows per layer)
     - Row 1: Histogram of singular values for G and top-k noise SVs
     - Row 2: Histogram of subspace distances (top-k N vs top-k G)

2. Distance evolution plot: subspace distance over training steps

3. Spectral norm comparison: ||G||_2 vs ||N||_2 across training

Usage:
    python plot_noise_structure.py --noise_dir /path/to/noise_records --output_dir /path/to/output
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

# Disable scientific notation globally
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['axes.formatter.use_mathtext'] = False


# Define canonical layer ordering (early → middle → late)
LAYER_ORDER = [
    "embedding",
    "early_attn",
    "early_mlp",
    "middle_attn",
    "middle_mlp",
    "late_attn",
    "late_mlp",
]


def sort_layers(layer_names: List[str]) -> List[str]:
    """Sort layer names in canonical order (early → middle → late)."""
    priority = {name: idx for idx, name in enumerate(LAYER_ORDER)}
    return sorted(layer_names, key=lambda x: (priority.get(x, len(LAYER_ORDER)), x))


def format_number(x: float) -> str:
    """Format number for axis labels - use integers when possible, otherwise short decimals."""
    if x == 0:
        return "0"
    abs_x = abs(x)
    if abs_x >= 1e9:
        return f"{x/1e9:.1f}B"
    elif abs_x >= 1e6:
        return f"{x/1e6:.0f}M" if x == int(x/1e6)*1e6 else f"{x/1e6:.1f}M"
    elif abs_x >= 1000:
        return f"{x/1e3:.0f}K" if x == int(x/1e3)*1e3 else f"{x/1e3:.1f}K"
    elif abs_x >= 1:
        # Use integer if close to integer, otherwise 1 decimal
        if abs(x - round(x)) < 0.01:
            return f"{int(round(x))}"
        else:
            return f"{x:.1f}"
    elif abs_x >= 0.01:
        return f"{x:.2f}"
    elif abs_x >= 0.001:
        return f"{x:.3f}"
    elif abs_x >= 0.0001:
        return f"{x:.4f}"
    elif abs_x >= 0.00001:
        return f"{x:.5f}"
    elif abs_x >= 0.000001:
        return f"{x:.6f}"
    else:
        # For extremely small numbers, use milli/micro notation
        if abs_x >= 1e-9:
            return f"{x*1e6:.2f}μ"
        else:
            return f"{x*1e9:.2f}n"


def setup_clean_axis(ax, axis='x', log_scale=False, max_ticks=6):
    """Set up clean axis with controlled tick count and formatting."""
    formatter = ticker.FuncFormatter(lambda x, p: format_number(x) if x > 0 else "")

    if axis == 'x':
        # For log scale, check if data spans enough range for LogLocator
        # Otherwise use MaxNLocator which works better for narrow ranges
        if log_scale:
            xlim = ax.get_xlim()
            if xlim[1] > 0 and xlim[0] > 0 and xlim[1] / xlim[0] > 10:
                # Data spans more than 1 order of magnitude - use LogLocator
                ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=max_ticks))
            else:
                # Narrow range - use MaxNLocator even on log scale
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_ticks))
        else:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_ticks, integer=True))
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(axis='x', labelsize=7, rotation=0)
        ax.xaxis.get_offset_text().set_visible(False)
    else:
        # For log scale, check if data spans enough range for LogLocator
        if log_scale:
            ylim = ax.get_ylim()
            if ylim[1] > 0 and ylim[0] > 0 and ylim[1] / ylim[0] > 10:
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=max_ticks))
            else:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_ticks))
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_ticks, integer=True))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.tick_params(axis='y', labelsize=7)
        ax.yaxis.get_offset_text().set_visible(False)


def load_noise_records(noise_dir: Path) -> Dict[int, Dict]:
    """Load all noise structure records from a directory."""
    all_records_path = noise_dir / "noise_all_records.pt"

    if all_records_path.exists():
        print(f"Loading from {all_records_path}")
        return torch.load(all_records_path, map_location="cpu", weights_only=False)

    # Otherwise, load individual step files
    records = {}
    for step_file in sorted(noise_dir.glob("noise_step_*.pt")):
        step = int(step_file.stem.split("_")[-1])
        print(f"Loading {step_file}")
        records[step] = torch.load(step_file, map_location="cpu", weights_only=False)

    return records


def _plot_noise_grid_for_layers(
    records: Dict[int, Dict],
    layer_names: List[str],
    steps: List[int],
    output_path: Path,
    n_bins: int = 30,
    figsize_per_subplot: tuple = (4, 2.5),
    is_last_part: bool = True,
):
    """
    Helper function to plot noise structure grid for a subset of layers.
    """
    n_rows = len(layer_names) * 2
    n_cols = len(steps)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
        squeeze=False
    )

    # Plot each cell
    for layer_idx, layer in enumerate(layer_names):
        row_sv = layer_idx * 2  # Singular value histogram row
        row_align = layer_idx * 2 + 1  # Alignment histogram row

        for col_idx, step in enumerate(steps):
            ax_sv = axes[row_sv, col_idx]
            ax_align = axes[row_align, col_idx]

            # Check if data exists
            if "layers" not in records[step] or layer not in records[step]["layers"]:
                ax_sv.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax_sv.transAxes)
                ax_sv.set_xticks([])
                ax_sv.set_yticks([])
                ax_align.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax_align.transAxes)
                ax_align.set_xticks([])
                ax_align.set_yticks([])
                continue

            layer_data = records[step]["layers"][layer]
            noise_samples = layer_data["noise_samples"]

            if not noise_samples:
                ax_sv.text(0.5, 0.5, "No noise samples", ha='center', va='center', transform=ax_sv.transAxes)
                ax_align.text(0.5, 0.5, "No noise samples", ha='center', va='center', transform=ax_align.transAxes)
                continue

            # --- Row 1: Histogram of singular values ---
            # Get G's full singular values distribution
            sv_G = layer_data["true_grad"]["singular_values"].numpy()

            # Collect all top-k singular values from noise samples (flattened)
            all_topk_sv_N = np.concatenate([ns["top_k_singular_values"].numpy() for ns in noise_samples])
            max_noise_sv = layer_data.get("max_noise_spectral_norm", all_topk_sv_N.max())

            # Determine bin range that covers both G's SVs and N's top-k SVs
            all_values = np.concatenate([sv_G, all_topk_sv_N])
            sv_min = all_values.min()
            sv_max = all_values.max()

            # Create log-spaced bins if values span multiple orders of magnitude
            use_log_scale = sv_max / max(sv_min, 1e-10) > 100
            if use_log_scale:
                bins = np.logspace(np.log10(max(sv_min, 1e-10)), np.log10(sv_max), n_bins)
                ax_sv.set_xscale('log')
            else:
                bins = np.linspace(sv_min, sv_max, n_bins)

            # Plot G's singular values distribution (all singular values)
            ax_sv.hist(sv_G, bins=bins, color='steelblue', alpha=0.7,
                      edgecolor='black', linewidth=0.5, label='G all SVs')

            # Plot N's top-k singular values distribution (k per noise sample)
            k = len(noise_samples[0]["top_k_singular_values"])
            ax_sv.hist(all_topk_sv_N, bins=bins, color='coral', alpha=0.6,
                      edgecolor='darkred', linewidth=0.5, label=f'N top-{k} SVs (n={len(noise_samples)})')

            # Add vertical dotted lines for largest singular values
            ax_sv.axvline(x=sv_G[0], color='steelblue', linestyle=':', linewidth=2,
                         label=f'||G||₂={format_number(sv_G[0])}')
            ax_sv.axvline(x=max_noise_sv, color='coral', linestyle=':', linewidth=2,
                         label=f'max||N||₂={format_number(max_noise_sv)}')

            # Stats text
            ratio = max_noise_sv / sv_G[0] if sv_G[0] > 0 else 0
            stats_text = f"ratio={ratio:.2f}"
            ax_sv.text(0.98, 0.98, stats_text, transform=ax_sv.transAxes,
                      fontsize=7, verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax_sv.grid(True, alpha=0.3)
            ax_sv.legend(fontsize=5, loc='upper left')
            setup_clean_axis(ax_sv, axis='x', log_scale=use_log_scale)
            setup_clean_axis(ax_sv, axis='y', log_scale=False)

            # --- Row 2: Histogram of subspace distance values (spectral & chordal) ---
            all_spectral = np.array([ns["spectral_distance"] for ns in noise_samples])
            all_chordal = np.array([ns["chordal_distance"] for ns in noise_samples])

            ax_align.hist(all_spectral, bins=n_bins, color='coral', alpha=0.6,
                         edgecolor='darkred', linewidth=0.5, range=(0, 1), label='spectral')
            ax_align.hist(all_chordal, bins=n_bins, color='seagreen', alpha=0.6,
                         edgecolor='darkgreen', linewidth=0.5, range=(0, 1), label='chordal')

            # Stats text (show both means)
            dist_text = f"spec={np.mean(all_spectral):.3f}\nchord={np.mean(all_chordal):.3f}"
            ax_align.text(0.98, 0.98, dist_text, transform=ax_align.transAxes,
                         fontsize=7, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

            ax_align.set_xlim(0, 1)
            ax_align.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax_align.tick_params(axis='x', labelsize=7)
            ax_align.grid(True, alpha=0.3)
            ax_align.legend(fontsize=6, loc='upper left')
            setup_clean_axis(ax_align, axis='y', log_scale=False)

            # Labels
            if layer_idx == 0:
                ax_sv.set_title(f"Step {step}", fontsize=10)
            if col_idx == 0:
                ax_sv.set_ylabel(f"{layer}\n(SV dist)", fontsize=8)
                ax_align.set_ylabel(f"(subspace dist)", fontsize=8)
            if layer_idx == len(layer_names) - 1:
                ax_sv.set_xlabel("Singular Value", fontsize=8)
                ax_align.set_xlabel("Subspace Distance", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved noise structure plot to {output_path}")
    plt.close()


def plot_noise_structure_grid(
    records: Dict[int, Dict],
    output_path: Path,
    n_bins: int = 30,
    figsize_per_subplot: tuple = (4, 2.5),
):
    """
    Plot noise structure analysis in a grid format.
    Splits into two figures if there are more than 4 layers.

    Layout:
    - Columns: training steps (0%, 5%, 50%, 99%)
    - Rows: 2 rows per layer
      - Row 1: Histogram of top-1 singular values of noise N (distribution across samples)
              with vertical lines for ||G||_2 and max ||N||_2
      - Row 2: Histogram of alignment values (distribution across samples)

    Args:
        records: Noise structure records {step: {"layers": {...}, ...}}
        output_path: Path to save the plot
        n_bins: Number of bins for histogram
        figsize_per_subplot: Size of each subplot
    """
    # Get sorted steps
    steps = sorted(records.keys())

    # Get all layer names from the first step that has data
    layer_names = []
    for step in steps:
        if "layers" in records[step] and records[step]["layers"]:
            layer_names = sort_layers(list(records[step]["layers"].keys()))
            break

    if not layer_names:
        print("No layer data found in records!")
        return

    # Split layers into two parts if more than 4 layers
    if len(layer_names) > 4:
        mid = (len(layer_names) + 1) // 2  # First half gets more if odd number
        layers_part1 = layer_names[:mid]
        layers_part2 = layer_names[mid:]

        # Generate output paths for two parts
        output_stem = output_path.stem
        output_suffix = output_path.suffix
        output_dir = output_path.parent

        output_path1 = output_dir / f"{output_stem}_part1{output_suffix}"
        output_path2 = output_dir / f"{output_stem}_part2{output_suffix}"

        # Plot both parts
        _plot_noise_grid_for_layers(
            records, layers_part1, steps, output_path1,
            n_bins, figsize_per_subplot, is_last_part=False
        )
        _plot_noise_grid_for_layers(
            records, layers_part2, steps, output_path2,
            n_bins, figsize_per_subplot, is_last_part=True
        )
    else:
        # Single figure for 4 or fewer layers
        _plot_noise_grid_for_layers(
            records, layer_names, steps, output_path,
            n_bins, figsize_per_subplot, is_last_part=True
        )


def plot_distance_evolution(
    records: Dict[int, Dict],
    output_path: Path,
    figsize: tuple = (12, 8),
):
    """
    Plot how subspace distance evolves across training steps for each layer.

    One subplot per layer, showing mean subspace distance ± std across noise samples at each step.
    """
    steps = sorted(records.keys())

    # Get layer names
    layer_names = []
    for step in steps:
        if "layers" in records[step] and records[step]["layers"]:
            layer_names = sort_layers(list(records[step]["layers"].keys()))
            break

    if not layer_names:
        print("No layer data found!")
        return

    n_layers = len(layer_names)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))

    for layer_idx, layer in enumerate(layer_names):
        row = layer_idx // n_cols
        col = layer_idx % n_cols
        ax = axes[row, col]

        step_values = []
        mean_spectral = []
        std_spectral = []
        mean_chordal = []
        std_chordal = []

        for step in steps:
            if "layers" not in records[step] or layer not in records[step]["layers"]:
                continue
            noise_samples = records[step]["layers"][layer]["noise_samples"]
            if not noise_samples:
                continue

            # Each noise sample has "spectral_distance" and "chordal_distance"
            spectral_dists = [ns["spectral_distance"] for ns in noise_samples]
            chordal_dists = [ns["chordal_distance"] for ns in noise_samples]

            if spectral_dists:
                step_values.append(step)
                mean_spectral.append(np.mean(spectral_dists))
                std_spectral.append(np.std(spectral_dists))
                mean_chordal.append(np.mean(chordal_dists))
                std_chordal.append(np.std(chordal_dists))

        if not step_values:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(layer, fontsize=10)
            continue

        # Plot spectral distance (coral)
        ax.errorbar(step_values, mean_spectral, yerr=std_spectral,
                   color='coral', marker='o', markersize=6, linewidth=2,
                   capsize=4, label='spectral')
        # Plot chordal distance (green)
        ax.errorbar(step_values, mean_chordal, yerr=std_chordal,
                   color='seagreen', marker='s', markersize=5, linewidth=2,
                   capsize=4, label='chordal')

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Training Step", fontsize=9)
        ax.set_ylabel("Subspace Distance", fontsize=9)
        ax.set_title(layer, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')
        setup_clean_axis(ax, axis='x')
        setup_clean_axis(ax, axis='y')

    # Hide unused subplots
    for idx in range(n_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved subspace distance evolution plot to {output_path}")
    plt.close()


def plot_spectral_norm_comparison(
    records: Dict[int, Dict],
    output_path: Path,
    figsize: tuple = (10, 6),
):
    """
    Plot spectral norm comparison: ||G||_2 vs ||N||_2 across layers and steps.
    """
    steps = sorted(records.keys())

    # Get layer names
    layer_names = []
    for step in steps:
        if "layers" in records[step] and records[step]["layers"]:
            layer_names = sort_layers(list(records[step]["layers"].keys()))
            break

    if not layer_names:
        print("No layer data found!")
        return

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Spectral norm of G across layers and steps
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))

    for layer_idx, layer in enumerate(layer_names):
        valid_steps = []
        spectral_G = []

        for step in steps:
            if "layers" in records[step] and layer in records[step]["layers"]:
                valid_steps.append(step)
                spectral_G.append(records[step]["layers"][layer]["true_grad"]["spectral_norm"])

        if valid_steps:
            ax1.plot(valid_steps, spectral_G, 'o-', color=colors[layer_idx],
                    linewidth=2, markersize=8, label=layer)

    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("Spectral Norm ||G||₂", fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')
    ax1.set_title("True Gradient Spectral Norm", fontsize=12)
    setup_clean_axis(ax1, axis='x')
    setup_clean_axis(ax1, axis='y', log_scale=True)

    # Plot 2: Ratio max||N||_2 / ||G||_2
    ax2 = axes[1]

    for layer_idx, layer in enumerate(layer_names):
        valid_steps = []
        ratios_max = []
        ratios_mean = []

        for step in steps:
            if "layers" not in records[step] or layer not in records[step]["layers"]:
                continue
            layer_data = records[step]["layers"][layer]
            spectral_G = layer_data["true_grad"]["spectral_norm"]

            # Get noise spectral norms (top-1 from top_k_singular_values)
            if layer_data["noise_samples"]:
                all_sv_N = [ns["top_k_singular_values"][0].item() for ns in layer_data["noise_samples"]]
                max_spectral_N = layer_data.get("max_noise_spectral_norm", max(all_sv_N))
                mean_spectral_N = np.mean(all_sv_N)
                valid_steps.append(step)
                ratios_max.append(max_spectral_N / spectral_G if spectral_G > 0 else 0)
                ratios_mean.append(mean_spectral_N / spectral_G if spectral_G > 0 else 0)

        if valid_steps:
            ax2.plot(valid_steps, ratios_max, 'o-', color=colors[layer_idx],
                    linewidth=2, markersize=8, label=f'{layer} (max)')
            ax2.plot(valid_steps, ratios_mean, 's--', color=colors[layer_idx],
                    linewidth=1.5, markersize=6, alpha=0.6)

    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("||N||₂ / ||G||₂", fontsize=11)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, loc='best')
    ax2.set_title("Noise-to-Signal Ratio (max solid, mean dashed)", fontsize=11)
    setup_clean_axis(ax2, axis='x')
    setup_clean_axis(ax2, axis='y', log_scale=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved spectral norm comparison plot to {output_path}")
    plt.close()


def print_summary_statistics(records: Dict[int, Dict]):
    """Print summary statistics for noise structure analysis."""
    steps = sorted(records.keys())

    print("\n" + "=" * 80)
    print("NOISE STRUCTURE SUMMARY STATISTICS")
    print("=" * 80)

    for step in steps:
        print(f"\n--- Step {step} ---")

        if "layers" not in records[step]:
            print("  No layer data")
            continue

        layer_names = sort_layers(list(records[step]["layers"].keys()))

        for layer in layer_names:
            layer_data = records[step]["layers"][layer]
            print(f"\n  {layer}:")

            # True gradient stats
            sv_G = layer_data["true_grad"]["singular_values"].numpy()
            print(f"    True Gradient G:")
            print(f"      ||G||_2 (spectral norm): {sv_G[0]:.4e}")
            print(f"      min SV: {sv_G[-1]:.4e}")
            print(f"      condition number: {sv_G[0]/sv_G[-1]:.4e}")

            # Noise stats
            noise_samples = layer_data["noise_samples"]
            if noise_samples:
                # Each sample has "top_k_singular_values", "spectral_distance", "chordal_distance"
                spectral_norms_N = [ns["top_k_singular_values"][0].item() for ns in noise_samples]
                max_sv_N = layer_data.get("max_noise_spectral_norm", max(spectral_norms_N))
                print(f"    Noise N (across {len(noise_samples)} samples):")
                print(f"      max ||N||_2: {max_sv_N:.4e}")
                print(f"      mean ||N||_2: {np.mean(spectral_norms_N):.4e}")
                print(f"      std ||N||_2: {np.std(spectral_norms_N):.4e}")
                print(f"      max ||N||_2 / ||G||_2: {max_sv_N/sv_G[0]:.4f}")
                print(f"      mean ||N||_2 / ||G||_2: {np.mean(spectral_norms_N)/sv_G[0]:.4f}")

                # Spectral distance stats (worst-case)
                all_spectral = [ns["spectral_distance"] for ns in noise_samples]
                print(f"    Spectral Distance (worst-case, max|sin θ|):")
                print(f"      mean: {np.mean(all_spectral):.4f}")
                print(f"      std: {np.std(all_spectral):.4f}")
                print(f"      min: {np.min(all_spectral):.4f}")
                print(f"      max: {np.max(all_spectral):.4f}")

                # Chordal distance stats (average-case)
                all_chordal = [ns["chordal_distance"] for ns in noise_samples]
                print(f"    Chordal Distance (average-case, RMS of sin θ):")
                print(f"      mean: {np.mean(all_chordal):.4f}")
                print(f"      std: {np.std(all_chordal):.4f}")
                print(f"      min: {np.min(all_chordal):.4f}")
                print(f"      max: {np.max(all_chordal):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot noise structure analysis results")
    parser.add_argument("--noise_dir", type=str, required=True,
                       help="Directory containing noise_records")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for plots (default: same as noise_dir)")
    parser.add_argument("--prefix", type=str, default="noise",
                       help="Prefix for output files")
    parser.add_argument("--format", type=str, default="pdf",
                       choices=["pdf", "png", "both"],
                       help="Output format: pdf, png, or both")

    args = parser.parse_args()

    noise_dir = Path(args.noise_dir)
    output_dir = Path(args.output_dir) if args.output_dir else noise_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load records
    records = load_noise_records(noise_dir)

    if not records:
        print("No noise structure records found!")
        return

    print(f"Loaded records for steps: {sorted(records.keys())}")

    # Print summary statistics
    print_summary_statistics(records)

    # Determine output formats
    formats = ["pdf", "png"] if args.format == "both" else [args.format]

    # Generate plots
    print("\nGenerating plots...")

    for fmt in formats:
        # 1. Main grid plot (histogram + subspace distance)
        plot_noise_structure_grid(
            records,
            output_dir / f"{args.prefix}_grid.{fmt}",
        )

        # 2. Subspace distance evolution across training
        plot_distance_evolution(
            records,
            output_dir / f"{args.prefix}_distance_evolution.{fmt}"
        )

        # 3. Spectral norm comparison
        plot_spectral_norm_comparison(
            records,
            output_dir / f"{args.prefix}_spectral_comparison.{fmt}"
        )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
