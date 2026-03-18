"""
Plot singular value distributions from SVD recordings.

This script visualizes the spectral structure of gradients and updates
across different layers and training steps.

Usage:
    python plot_singular_values.py --svd_dir /path/to/svd_records --output plot.pdf
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
    # Create a priority map based on LAYER_ORDER
    priority = {name: idx for idx, name in enumerate(LAYER_ORDER)}
    # Sort: known layers by priority, unknown layers alphabetically at the end
    return sorted(layer_names, key=lambda x: (priority.get(x, len(LAYER_ORDER)), x))


def format_number(x: float) -> str:
    """Format number without scientific notation."""
    if x == 0:
        return "0"
    abs_x = abs(x)
    # Large: K/M/B suffixes
    if abs_x >= 1e9:
        return f"{x/1e9:.1f}B"
    elif abs_x >= 1e6:
        return f"{x/1e6:.0f}M" if x == int(x/1e6)*1e6 else f"{x/1e6:.1f}M"
    elif abs_x >= 1000:
        return f"{x/1e3:.0f}K" if x == int(x/1e3)*1e3 else f"{x/1e3:.1f}K"
    # Medium: integers or 1 decimal
    elif abs_x >= 1:
        if abs(x - round(x)) < 0.1:
            return f"{int(round(x))}"
        return f"{x:.1f}"
    # Small: appropriate decimals
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
        # For extremely small numbers, use micro notation
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


def load_svd_records(svd_dir: Path) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
    """Load all SVD records from a directory."""
    all_records_path = svd_dir / "svd_all_records.pt"

    if all_records_path.exists():
        print(f"Loading from {all_records_path}")
        return torch.load(all_records_path, map_location="cpu")

    # Otherwise, load individual step files
    records = {}
    for step_file in sorted(svd_dir.glob("svd_step_*.pt")):
        step = int(step_file.stem.split("_")[-1])
        print(f"Loading {step_file}")
        records[step] = torch.load(step_file, map_location="cpu")

    return records


def plot_singular_values_grid(
    records: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    output_path: Path,
    record_type: str = "grad_sv",  # "grad_sv" or "update_sv"
    log_scale: bool = True,
    figsize_per_subplot: tuple = (4, 3),
):
    """
    Plot singular values in a grid format (line plot showing SV vs index).

    Rows: different layers (embedding, early_attn, early_mlp, middle_attn, etc.)
    Columns: different training steps (0%, 5%, 50%, 99%)

    Args:
        records: SVD records dictionary {step: {layer: {"grad_sv": tensor, "update_sv": tensor}}}
        output_path: Path to save the plot
        record_type: Which type of singular values to plot ("grad_sv" or "update_sv")
        log_scale: Whether to use log scale for y-axis
        figsize_per_subplot: Size of each subplot
    """
    # Get sorted steps and layer names
    steps = sorted(records.keys())

    # Get all layer names from the first step that has data
    layer_names = []
    for step in steps:
        if records[step]:
            layer_names = sort_layers(list(records[step].keys()))
            break

    if not layer_names:
        print("No layer data found in records!")
        return

    # Filter layers that have the requested record type (preserving order)
    valid_layers = []
    for layer in layer_names:
        for step in steps:
            if layer in records[step] and record_type in records[step][layer]:
                valid_layers.append(layer)
                break

    if not valid_layers:
        print(f"No {record_type} data found in records!")
        return

    layer_names = valid_layers

    # Create figure
    n_rows = len(layer_names)
    n_cols = len(steps)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
        squeeze=False
    )

    # Title for the entire figure
    title_map = {
        "grad_sv": "Gradient Singular Values",
        "update_sv": "Update Singular Values"
    }
    #fig.suptitle(title_map.get(record_type, record_type), fontsize=14, fontweight='bold')

    # Plot each cell
    for row_idx, layer in enumerate(layer_names):
        for col_idx, step in enumerate(steps):
            ax = axes[row_idx, col_idx]

            # Check if data exists
            if layer not in records[step] or record_type not in records[step][layer]:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                sv = records[step][layer][record_type].numpy()
                indices = np.arange(1, len(sv) + 1)

                ax.plot(indices, sv, 'b-', linewidth=1.5, alpha=0.8)
                ax.fill_between(indices, sv, alpha=0.3)

                if log_scale:
                    ax.set_yscale('log')

                # Add statistics as text
                ratio_val = sv[0]/sv[-1] if sv[-1] > 0 else 0
                stats_text = f"max: {format_number(sv[0])}\nmin: {format_number(sv[-1])}\nratio: {format_number(ratio_val)}"
                ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=7, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.set_xlim(1, len(sv))
                ax.grid(True, alpha=0.3)
                setup_clean_axis(ax, axis='x', log_scale=False)
                setup_clean_axis(ax, axis='y', log_scale=log_scale)

            # Labels
            if row_idx == 0:
                ax.set_title(f"Step {step}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(layer, fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Index", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_singular_values_histogram_grid(
    records: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    output_path: Path,
    record_type: str = "grad_sv",  # "grad_sv" or "update_sv"
    n_bins: int = 50,
    log_scale_x: bool = True,
    figsize_per_subplot: tuple = (4, 3),
):
    """
    Plot singular value distribution histograms in a grid format.

    Rows: different layers (embedding, early_attn, early_mlp, middle_attn, etc.)
    Columns: different training steps (0%, 5%, 50%, 99%)

    Each subplot shows a histogram of singular values with a vertical dashed line
    at the maximum singular value (spectral norm).

    Args:
        records: SVD records dictionary {step: {layer: {"grad_sv": tensor, "update_sv": tensor}}}
        output_path: Path to save the plot
        record_type: Which type of singular values to plot ("grad_sv" or "update_sv")
        n_bins: Number of bins for histogram
        log_scale_x: Whether to use log scale for x-axis (singular values)
        figsize_per_subplot: Size of each subplot
    """
    # Get sorted steps and layer names
    steps = sorted(records.keys())

    # Get all layer names from the first step that has data
    layer_names = []
    for step in steps:
        if records[step]:
            layer_names = sort_layers(list(records[step].keys()))
            break

    if not layer_names:
        print("No layer data found in records!")
        return

    # Filter layers that have the requested record type (preserving order)
    valid_layers = []
    for layer in layer_names:
        for step in steps:
            if layer in records[step] and record_type in records[step][layer]:
                valid_layers.append(layer)
                break

    if not valid_layers:
        print(f"No {record_type} data found in records!")
        return

    layer_names = valid_layers

    # Create figure
    n_rows = len(layer_names)
    n_cols = len(steps)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_subplot[0] * n_cols, figsize_per_subplot[1] * n_rows),
        squeeze=False
    )

    # Title for the entire figure
    title_map = {
        "grad_sv": "Gradient Singular Value Distribution",
        "update_sv": "Update Singular Value Distribution"
    }
    #fig.suptitle(title_map.get(record_type, record_type), fontsize=14, fontweight='bold')

    # Plot each cell
    for row_idx, layer in enumerate(layer_names):
        for col_idx, step in enumerate(steps):
            ax = axes[row_idx, col_idx]

            # Check if data exists
            if layer not in records[step] or record_type not in records[step][layer]:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                sv = records[step][layer][record_type].numpy()
                sv_max = sv[0]
                sv_min = sv[-1]

                # Use log-spaced bins if log scale
                if log_scale_x and sv_min > 0:
                    bins = np.logspace(np.log10(sv_min), np.log10(sv_max), n_bins)
                else:
                    bins = np.linspace(sv_min, sv_max, n_bins)

                # Plot histogram
                ax.hist(sv, bins=bins, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)

                # Add vertical dashed line at maximum singular value
                ax.axvline(x=sv_max, color='red', linestyle='--', linewidth=2, label=f'max={format_number(sv_max)}')

                use_log = log_scale_x and sv_min > 0
                if use_log:
                    ax.set_xscale('log')

                # Add statistics as text
                ratio_val = sv_max/sv_min if sv_min > 0 else 0
                stats_text = f"max: {format_number(sv_max)}\nmin: {format_number(sv_min)}\nratio: {format_number(ratio_val)}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=7, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.grid(True, alpha=0.3)
                setup_clean_axis(ax, axis='x', log_scale=use_log)
                setup_clean_axis(ax, axis='y', log_scale=False)

            # Labels
            if row_idx == 0:
                ax.set_title(f"Step {step}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(layer, fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Singular Value", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved histogram plot to {output_path}")
    plt.close()


def plot_singular_values_comparison(
    records: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    output_path: Path,
    layers_to_plot: Optional[List[str]] = None,
    figsize: tuple = (12, 8),
):
    """
    Plot singular values comparing different steps for each layer.

    Each subplot shows one layer with different steps as different colored lines.

    Args:
        records: SVD records dictionary
        output_path: Path to save the plot
        layers_to_plot: List of layers to plot (if None, plot all)
        figsize: Figure size
    """
    steps = sorted(records.keys())

    # Get all layer names
    layer_names = []
    for step in steps:
        if records[step]:
            layer_names = sort_layers(list(records[step].keys()))
            break

    if layers_to_plot:
        # Filter while preserving sort order
        layer_names = [l for l in layer_names if l in layers_to_plot]

    if not layer_names:
        print("No layers to plot!")
        return

    # Create subplots: 2 columns (grad_sv, update_sv), rows for each layer
    n_rows = len(layer_names)
    fig, axes = plt.subplots(n_rows, 2, figsize=(figsize[0], figsize[1] * n_rows / 4), squeeze=False)

    colors = plt.cm.viridis(np.linspace(0, 1, len(steps)))

    for row_idx, layer in enumerate(layer_names):
        for col_idx, record_type in enumerate(["grad_sv", "update_sv"]):
            ax = axes[row_idx, col_idx]

            has_data = False
            for step_idx, step in enumerate(steps):
                if layer in records[step] and record_type in records[step][layer]:
                    sv = records[step][layer][record_type].numpy()
                    indices = np.arange(1, len(sv) + 1)
                    ax.plot(indices, sv, color=colors[step_idx],
                           label=f"Step {step}", linewidth=1.5, alpha=0.8)
                    has_data = True

            if has_data:
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=7, loc='upper right')
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)

            if row_idx == 0:
                title = "Gradient SV" if record_type == "grad_sv" else "Update SV"
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(layer, fontsize=9)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Singular Value Index", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_spectral_norm_evolution(
    records: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    output_path: Path,
    figsize: tuple = (10, 6),
):
    """
    Plot the evolution of spectral norm (max singular value) across training steps.

    Args:
        records: SVD records dictionary
        output_path: Path to save the plot
        figsize: Figure size
    """
    steps = sorted(records.keys())

    # Get all layer names
    layer_names = []
    for step in steps:
        if records[step]:
            layer_names = sort_layers(list(records[step].keys()))
            break

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for col_idx, record_type in enumerate(["grad_sv", "update_sv"]):
        ax = axes[col_idx]

        for layer in layer_names:
            spectral_norms = []
            valid_steps = []

            for step in steps:
                if layer in records[step] and record_type in records[step][layer]:
                    sv = records[step][layer][record_type].numpy()
                    spectral_norms.append(sv[0])  # Max singular value
                    valid_steps.append(step)

            if spectral_norms:
                ax.plot(valid_steps, spectral_norms, 'o-', label=layer, linewidth=2, markersize=8)

        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Spectral Norm (Max SV)", fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

        title = "Gradient Spectral Norm" if record_type == "grad_sv" else "Update Spectral Norm"
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved spectral norm evolution plot to {output_path}")
    plt.close()


def plot_condition_number_evolution(
    records: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    output_path: Path,
    figsize: tuple = (10, 6),
):
    """
    Plot the evolution of condition number (max/min singular value ratio) across training steps.

    Args:
        records: SVD records dictionary
        output_path: Path to save the plot
        figsize: Figure size
    """
    steps = sorted(records.keys())

    # Get all layer names
    layer_names = []
    for step in steps:
        if records[step]:
            layer_names = sort_layers(list(records[step].keys()))
            break

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for col_idx, record_type in enumerate(["grad_sv", "update_sv"]):
        ax = axes[col_idx]

        for layer in layer_names:
            condition_numbers = []
            valid_steps = []

            for step in steps:
                if layer in records[step] and record_type in records[step][layer]:
                    sv = records[step][layer][record_type].numpy()
                    # Avoid division by zero
                    min_sv = sv[-1] if sv[-1] > 1e-10 else 1e-10
                    condition_numbers.append(sv[0] / min_sv)
                    valid_steps.append(step)

            if condition_numbers:
                ax.plot(valid_steps, condition_numbers, 'o-', label=layer, linewidth=2, markersize=8)

        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Condition Number (Max/Min SV)", fontsize=11)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

        title = "Gradient Condition Number" if record_type == "grad_sv" else "Update Condition Number"
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved condition number evolution plot to {output_path}")
    plt.close()


def print_summary_statistics(records: Dict[int, Dict[str, Dict[str, torch.Tensor]]]):
    """Print summary statistics for all recorded singular values."""
    steps = sorted(records.keys())

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for step in steps:
        print(f"\n--- Step {step} ---")
        layer_names = sort_layers(list(records[step].keys()))
        for layer in layer_names:
            data = records[step][layer]
            print(f"\n  {layer}:")
            for record_type, sv in data.items():
                sv_np = sv.numpy()
                print(f"    {record_type}:")
                print(f"      shape: {sv_np.shape}")
                print(f"      max (spectral norm): {sv_np[0]:.4e}")
                print(f"      min: {sv_np[-1]:.4e}")
                print(f"      ratio (condition): {sv_np[0]/sv_np[-1]:.4e}")
                print(f"      mean: {sv_np.mean():.4e}")
                print(f"      median: {np.median(sv_np):.4e}")


def main():
    parser = argparse.ArgumentParser(description="Plot singular value distributions from SVD recordings")
    parser.add_argument("--svd_dir", type=str, required=True, help="Directory containing SVD records")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots (default: same as svd_dir)")
    parser.add_argument("--prefix", type=str, default="svd", help="Prefix for output files")
    parser.add_argument("--no_log_scale", action="store_true", help="Disable log scale for y-axis")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "both"], help="Output format: pdf, png, or both")

    args = parser.parse_args()

    svd_dir = Path(args.svd_dir)
    output_dir = Path(args.output_dir) if args.output_dir else svd_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load records
    records = load_svd_records(svd_dir)

    if not records:
        print("No SVD records found!")
        return

    print(f"Loaded records for steps: {sorted(records.keys())}")

    # Print summary statistics
    print_summary_statistics(records)

    # Determine output formats
    formats = ["pdf", "png"] if args.format == "both" else [args.format]

    # Generate plots
    print("\nGenerating plots...")

    for fmt in formats:
        # 1. Grid plot for gradients (line plot)
        plot_singular_values_grid(
            records,
            output_dir / f"{args.prefix}_grad_grid.{fmt}",
            record_type="grad_sv",
            log_scale=not args.no_log_scale
        )

        # 2. Grid plot for updates (line plot)
        plot_singular_values_grid(
            records,
            output_dir / f"{args.prefix}_update_grid.{fmt}",
            record_type="update_sv",
            log_scale=not args.no_log_scale
        )

        # 3. Histogram grid for gradients
        plot_singular_values_histogram_grid(
            records,
            output_dir / f"{args.prefix}_grad_hist.{fmt}",
            record_type="grad_sv",
            log_scale_x=not args.no_log_scale
        )

        # 4. Histogram grid for updates
        plot_singular_values_histogram_grid(
            records,
            output_dir / f"{args.prefix}_update_hist.{fmt}",
            record_type="update_sv",
            log_scale_x=not args.no_log_scale
        )

        # 5. Comparison plot (all steps on same axis)
        plot_singular_values_comparison(
            records,
            output_dir / f"{args.prefix}_comparison.{fmt}"
        )

        # 6. Spectral norm evolution
        plot_spectral_norm_evolution(
            records,
            output_dir / f"{args.prefix}_spectral_norm.{fmt}"
        )

        # 7. Condition number evolution
        plot_condition_number_evolution(
            records,
            output_dir / f"{args.prefix}_condition_number.{fmt}"
        )

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
