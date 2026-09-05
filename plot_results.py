"""
Script to plot results from results_*.json files.

Creates two plots:
1. Cosine similarity vs batch_size (for specified beta2 value(s))
2. Cosine similarity vs beta2 (for specified batch_size(s))

Each plot can compare:
- Multiple optimizers (e.g., EFAdam vs ReAdam)
- Multiple Fisher types (e.g., adam vs empirical)
- Multiple beta2 or batch_size values
- Or any combination!

Usage examples:
  # Compare two optimizers with adam Fisher
  python plot_results.py results_EFAdam.json results_ReAdam.json

  # Compare two Fisher types for one optimizer
  python plot_results.py results_EFAdam.json --fisher-types adam empirical

  # Compare across both optimizers and Fisher types
  python plot_results.py results_EFAdam.json results_ReAdam.json --fisher-types adam empirical

  # Plot multiple beta2 values on the batch_size plot
  python plot_results.py results_EFAdam.json --beta2 0.9 0.99 0.999

  # Plot multiple batch sizes on the beta2 plot
  python plot_results.py results_EFAdam.json --batch-size 10 50 100"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os


def load_results(json_file):
    """Load results from JSON file."""
    with open(json_file, "r") as f:
        return json.load(f)


def _display_fisher_label(fisher_type: str) -> str:
    """Return display label for fisher types."""
    if fisher_type == "empirical":
        return "eFIM"
    if fisher_type == "adam":
        return "AF"
    return fisher_type


def _display_optimizer_name(opt_name: str) -> str:
    """Map optimizer display names: show 'Adam' when optimizer is 'ReAdam'."""
    if opt_name == "ReAdam":
        return "Adam"
    return opt_name


def get_distinct_non_red_colors(n_lines):
    """Return a curated palette starting with the user's chosen colors.

    The palette avoids the reserved red (`#d62728`) and provides a few
    additional visually distinct colors that match the existing scheme.
    If more colors are requested than available, the palette cycles.
    """
    palette = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#9467bd",  # purple
       #"#ff7f0e",  # orange (user provided)
        "#8c564b",  # brown/olive
        "#17becf",  # cyan
        "#7f7f7f",  # gray
        "#e377c2",  # pink/magenta
        "#bcbd22",  # olive-yellow
        "#4c78a8",  # steel blue
        "#59a14f",  # medium green
        "#b07aa1",  # mauve
    ]

    if n_lines <= 0:
        return []

    if n_lines <= len(palette):
        return palette[:n_lines]

    return [palette[i % len(palette)] for i in range(n_lines)]


def apply_consistent_figure_layout(fig):
    """Legacy layout helper (kept for compatibility). No-op for internal legends."""
    return


def plot_vs_batch_size(all_data_list, beta2_values_list, output_file=None):
    """
    Plot cosine similarity vs batch_size.
    One line per (optimizer, fisher_type, beta2) combination, with std shaded around each line.
    """
    plt.figure(figsize=(10, 9))

    # Use a high-contrast non-red palette so red stays reserved for Emp vs Adam.
    total_lines = sum(
        len(item["fisher_types"]) * len(beta2_values_list) for item in all_data_list
    )
    colors = get_distinct_non_red_colors(total_lines)

    color_idx = 0
    for item in all_data_list:
        data = item["data"]
        fisher_types = item["fisher_types"]
        optimizer_name = data["optimizer"]
        display_optimizer = _display_optimizer_name(optimizer_name)

        batch_size_values = data["batch_size_values"]
        results = data["results"]

        for fisher_type in fisher_types:
            for beta2_fixed in beta2_values_list:
                means = []
                stds = []

                for batch_size in batch_size_values:
                    key = f"({beta2_fixed}, {batch_size})"
                    if key in results:
                        mean_sim = results[key]["mean_similarities"].get(
                            fisher_type, np.nan
                        )
                        std_sim = results[key]["std_similarities"].get(
                            fisher_type, np.nan
                        )
                        means.append(mean_sim)
                        stds.append(std_sim)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)

                means = np.array(means)
                stds = np.array(stds)

                # Plot line
                display_fisher = _display_fisher_label(fisher_type)
                label = f"{display_optimizer} ({display_fisher}, β₂={beta2_fixed})"
                line_style = ":" if fisher_type == "adam" else "-"
                plt.plot(
                    batch_size_values,
                    means,
                    marker="o",
                    label=label,
                    color=colors[color_idx],
                    linewidth=2,
                    linestyle=line_style,
                )

                # Plot shaded std
                plt.fill_between(
                    batch_size_values,
                    means - stds,
                    means + stds,
                    alpha=0.2,
                    color=colors[color_idx],
                )

                color_idx += 1

    # Add a single aggregated red dashed line for empirical-vs-adam similarity.
    ref_batch_sizes = all_data_list[0]["data"]["batch_size_values"]
    emp_adam_means = []
    emp_adam_stds = []
    for batch_size in ref_batch_sizes:
        mean_vals = []
        std_vals = []
        for item in all_data_list:
            results = item["data"]["results"]
            for beta2_fixed in beta2_values_list:
                key = f"({beta2_fixed}, {batch_size})"
                if key in results:
                    mean_vals.append(
                        results[key].get("mean_cosine_similarity_emp_with_adam", np.nan)
                    )
                    std_vals.append(
                        results[key].get("std_cosine_similarity_emp_with_adam", np.nan)
                    )

        mean_arr = np.array(mean_vals, dtype=float)
        std_arr = np.array(std_vals, dtype=float)
        emp_adam_means.append(
            np.nan if np.all(np.isnan(mean_arr)) else float(np.nanmean(mean_arr))
        )
        emp_adam_stds.append(
            np.nan if np.all(np.isnan(std_arr)) else float(np.nanmean(std_arr))
        )

    emp_adam_means = np.array(emp_adam_means)
    emp_adam_stds = np.array(emp_adam_stds)
    if not np.all(np.isnan(emp_adam_means)):
        # Aggregate label uses mapped names
        plt.plot(
            ref_batch_sizes,
            emp_adam_means,
            linestyle="--",
            color="red",
            linewidth=2.5,
            label=f"{_display_fisher_label('empirical')} vs {_display_fisher_label('adam')} (avg)",
        )
        plt.fill_between(
            ref_batch_sizes,
            emp_adam_means - emp_adam_stds,
            emp_adam_means + emp_adam_stds,
            alpha=0.12,
            color="red",
            edgecolor="red",
            hatch="///",
            linewidth=0.0,
        )
        print(f"Plot std for Emp vs Adam reference line: {emp_adam_stds}")

    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Cosine Similarity (second moment vs Fisher)", fontsize=12)
    beta_str = (
        ", ".join([str(b) for b in beta2_values_list])
        if len(beta2_values_list) > 1
        else str(beta2_values_list[0])
    )
    plt.title(
        f"Cosine Similarity vs Batch Size (β₂={beta_str})",
        fontsize=14,
        fontweight="bold",
    )
    plt.gca().set_box_aspect(1)
    plt.legend(fontsize=9, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_file}")
    else:
        plt.show()
    plt.close()


def plot_vs_beta2(all_data_list, batch_size_values_list, output_file=None):
    """
    Plot cosine similarity vs beta2.
    One line per (optimizer, fisher_type, batch_size) combination, with std shaded around each line.
    """
    plt.figure(figsize=(10, 9))

    # Use a high-contrast non-red palette so red stays reserved for Emp vs Adam.
    total_lines = sum(
        len(item["fisher_types"]) * len(batch_size_values_list)
        for item in all_data_list
    )
    colors = get_distinct_non_red_colors(total_lines)

    color_idx = 0
    has_valid_data = False
    all_one_minus_beta2_values = []

    for item in all_data_list:
        data = item["data"]
        fisher_types = item["fisher_types"]
        optimizer_name = data["optimizer"]
        display_optimizer = _display_optimizer_name(optimizer_name)

        beta2_values = data["beta2_values"]
        results = data["results"]

        one_minus_beta2 = np.array([1 - beta2 for beta2 in beta2_values])
        sort_idx = np.argsort(one_minus_beta2)
        x_values = one_minus_beta2[sort_idx]
        all_one_minus_beta2_values.extend(x_values.tolist())

        for fisher_type in fisher_types:
            for batch_size_fixed in batch_size_values_list:
                means = []
                stds = []

                for beta2 in beta2_values:
                    key = f"({beta2}, {batch_size_fixed})"
                    if key in results:
                        mean_sim = results[key]["mean_similarities"].get(
                            fisher_type, np.nan
                        )
                        std_sim = results[key]["std_similarities"].get(
                            fisher_type, np.nan
                        )
                        means.append(mean_sim)
                        stds.append(std_sim)
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)

                means = np.array(means)
                stds = np.array(stds)
                means = means[sort_idx]
                stds = stds[sort_idx]

                # Check if we have any valid data
                if not np.all(np.isnan(means)):
                    has_valid_data = True

                # Plot line
                display_fisher = _display_fisher_label(fisher_type)
                label = f"{display_optimizer} ({display_fisher}, bs={batch_size_fixed})"
                line_style = ":" if fisher_type == "adam" else "-"
                plt.plot(
                    x_values,
                    means,
                    marker="o",
                    label=label,
                    color=colors[color_idx],
                    linewidth=2,
                    linestyle=line_style,
                )

                # Plot shaded std
                plt.fill_between(
                    x_values,
                    means - stds,
                    means + stds,
                    alpha=0.2,
                    color=colors[color_idx],
                )

                color_idx += 1

    # Add a single aggregated red dashed line for empirical-vs-adam similarity.
    ref_beta2_values = all_data_list[0]["data"]["beta2_values"]
    ref_one_minus_beta2 = np.array([1 - beta2 for beta2 in ref_beta2_values])
    ref_sort_idx = np.argsort(ref_one_minus_beta2)
    ref_x_values = ref_one_minus_beta2[ref_sort_idx]

    emp_adam_means = []
    emp_adam_stds = []
    for beta2 in ref_beta2_values:
        mean_vals = []
        std_vals = []
        for item in all_data_list:
            results = item["data"]["results"]
            for batch_size_fixed in batch_size_values_list:
                key = f"({beta2}, {batch_size_fixed})"
                if key in results:
                    mean_vals.append(
                        results[key].get("mean_cosine_similarity_emp_with_adam", np.nan)
                    )
                    std_vals.append(
                        results[key].get("std_cosine_similarity_emp_with_adam", np.nan)
                    )

        mean_arr = np.array(mean_vals, dtype=float)
        std_arr = np.array(std_vals, dtype=float)
        emp_adam_means.append(
            np.nan if np.all(np.isnan(mean_arr)) else float(np.nanmean(mean_arr))
        )
        emp_adam_stds.append(
            np.nan if np.all(np.isnan(std_arr)) else float(np.nanmean(std_arr))
        )

    emp_adam_means = np.array(emp_adam_means)[ref_sort_idx]
    emp_adam_stds = np.array(emp_adam_stds)[ref_sort_idx]
    if not np.all(np.isnan(emp_adam_means)):
        plt.plot(
            ref_x_values,
            emp_adam_means,
            linestyle="--",
            color="red",
            linewidth=2.5,
            label="eFIM vs AF (avg)",
        )
        plt.fill_between(
            ref_x_values,
            emp_adam_means - emp_adam_stds,
            emp_adam_means + emp_adam_stds,
            alpha=0.12,
            color="red",
            edgecolor="red",
            hatch="///",
            linewidth=0.0,
        )

    # Only use log scale if we have valid positive data
    if has_valid_data and all(v > 0 for v in all_one_minus_beta2_values):
        plt.xscale("log")

    plt.xlabel("1 - β₂", fontsize=12)
    plt.ylabel("Cosine Similarity (second moment vs Fisher)", fontsize=12)
    bs_str = (
        ", ".join([str(bs) for bs in batch_size_values_list])
        if len(batch_size_values_list) > 1
        else str(batch_size_values_list[0])
    )
    plt.title(
        f"Cosine Similarity vs 1 - β₂ (batch_size={bs_str})",
        fontsize=14,
        fontweight="bold",
    )
    plt.gca().set_box_aspect(1)
    plt.legend(fontsize=9, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_file}")
    else:
        plt.show()
    plt.close()


def plot_loss_histogram_from_losses_json(
    losses_json_file, threshold=0.01, bins=20, output_file=None
):
    """
    Plot histogram of final losses from losses JSON file.

    Args:
        losses_json_file: Path to losses JSON file (from merge_run_jsons.py)
        threshold: Loss threshold to count losses above (default 0.01)
        bins: Number of histogram bins (default 20)
        output_file: Output path for saving the plot
    """
    with open(losses_json_file, "r") as f:
        data = json.load(f)

    optimizer = data.get("optimizer", "Unknown")
    losses = [record.get("final_loss") for record in data.get("losses", [])]

    # Filter out NaN values
    losses = [l for l in losses if not (isinstance(l, float) and np.isnan(l))]

    if not losses:
        print(f"Warning: No valid losses found in {losses_json_file}")
        return

    # Count losses above threshold
    above_threshold = sum(1 for l in losses if l > threshold)

    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(losses, bins=bins, edgecolor="black", alpha=0.7)

    # Add threshold line
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold})",
    )

    # Add text annotation
    ax.text(
        0.98,
        0.97,
        f"{optimizer}\nAbove threshold: {above_threshold}/{len(losses)}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Final Loss", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Distribution of Final Losses - {optimizer}", fontsize=14, fontweight="bold"
    )
    ax.set_box_aspect(1)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_file}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot results from experiment JSON files"
    )
    parser.add_argument(
        "json_files",
        type=str,
        nargs="+",
        help="Path(s) to results JSON file(s) (e.g., results_EFAdam.json results_ReAdam.json)",
    )
    parser.add_argument(
        "--fisher-types",
        type=str,
        nargs="+",
        default=["adam"],
        help="Which Fisher approximation(s) to plot (default: adam). Can specify multiple.",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        nargs="+",
        default=None,
        help="Beta2 value(s) for batch_size plot (default: use first available). Can specify multiple.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=None,
        help="Batch size(s) for beta2 plot (default: use first available). Can specify multiple.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Directory to save plots. A results folder will be made in this directory. If left empty the results directory will be made in the root directory.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Show plots instead of saving them",
    )
    parser.add_argument(
        "--loss-histogram-json",
        type=str,
        default=None,
        help="Path to losses JSON file to plot histogram from",
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=0.01,
        help="Threshold for counting losses above it in histogram (default: 0.01)",
    )
    parser.add_argument(
        "--loss-bins",
        type=int,
        default=20,
        help="Number of bins for loss histogram (default: 20)",
    )

    args = parser.parse_known_args()[0]

    # Handle loss histogram plotting if requested
    if args.loss_histogram_json:
        print(f"Plotting loss histogram from {args.loss_histogram_json}...")
        # Load losses JSON to get optimizer name
        with open(args.loss_histogram_json, "r") as f:
            losses_data = json.load(f)
        optimizer = losses_data.get("optimizer", "unknown")

        if not args.output_path == None:
            results_dir = Path(os.path.join(args.output_path, "results"))
        else:
            results_dir = Path("results")
        if not args.no_save:
            results_dir.mkdir(exist_ok=True)
            output_file = results_dir / f"loss_histogram_{optimizer}.pdf"
        else:
            output_file = None

        plot_loss_histogram_from_losses_json(
            args.loss_histogram_json,
            threshold=args.loss_threshold,
            bins=args.loss_bins,
            output_file=output_file,
        )

    # Load all data files
    all_data_list = []
    all_optimizer_names = []

    for json_file in args.json_files:
        data = load_results(json_file)
        optimizer_name = data.get("optimizer", "unknown")
        all_optimizer_names.append(optimizer_name)

        # Check which fisher types are available
        first_key = list(data["results"].keys())[0]
        available_types = list(data["results"][first_key]["mean_similarities"].keys())

        # Filter requested fisher types to only those available
        fisher_types = [ft for ft in args.fisher_types if ft in available_types]

        if not fisher_types:
            print(f"Warning: None of the requested Fisher types found in {json_file}")
            print(f"  Available types: {', '.join(available_types)}")
            continue

        missing = [ft for ft in args.fisher_types if ft not in available_types]
        if missing:
            print(
                f"Note: Fisher types {missing} not found in {json_file}, skipping them"
            )

        all_data_list.append(
            {
                "data": data,
                "fisher_types": fisher_types,
                "json_file": json_file,
            }
        )

    if not all_data_list:
        print("Error: No valid data to plot")
        return

    # Determine fixed values for plots
    first_data = all_data_list[0]["data"]

    if args.beta2 is None:
        beta2_values_list = [first_data["beta2_values"][0]]
        print(f"Using β₂={beta2_values_list[0]} for batch_size plot")
    else:
        beta2_values_list = args.beta2 if isinstance(args.beta2, list) else [args.beta2]
        print(f"Using β₂={beta2_values_list} for batch_size plot")

    if args.batch_size is None:
        batch_size_values_list = [first_data["batch_size_values"][0]]
        print(f"Using batch_size={batch_size_values_list[0]} for β₂ plot")
    else:
        batch_size_values_list = (
            args.batch_size if isinstance(args.batch_size, list) else [args.batch_size]
        )
        print(f"Using batch_size={batch_size_values_list} for β₂ plot")

    # Save plots in results directory
    if not args.output_path == None:
        results_dir = Path(os.path.join(args.output_path, "results"))
    else:
        results_dir = Path("results")
    if not args.no_save:
        results_dir.mkdir(exist_ok=True)
        opt_str = "_".join(all_optimizer_names)
        fisher_str = "_".join(args.fisher_types)
        beta_str = "_".join([str(b) for b in beta2_values_list])
        bs_str = "_".join([str(bs) for bs in batch_size_values_list])
        output_vs_batch = (
            results_dir
            / f"similarity_vs_batch_size_{opt_str}_{fisher_str}_beta{beta_str}.pdf"
        )
        output_vs_beta = (
            results_dir / f"similarity_vs_beta2_{opt_str}_{fisher_str}_bs{bs_str}.pdf"
        )
    else:
        output_vs_batch = None
        output_vs_beta = None

    # Generate plots
    print(f"Plotting comparison...")
    plot_vs_batch_size(all_data_list, beta2_values_list, output_vs_batch)
    plot_vs_beta2(all_data_list, batch_size_values_list, output_vs_beta)

    print("Done!")


if __name__ == "__main__":
    main()
