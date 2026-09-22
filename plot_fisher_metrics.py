"""
Plot Fisher information approximations over time from JSON results.

Combos (optimizer, beta2) are discovered directly from the filenames present
in each experiment's results folder -- never hardcoded -- so this stays
correct even if a future round adds/removes a beta2 value or optimizer.

Usage (single run, unchanged):
    python plot_fisher_metrics.py --results_folder linearregexperiment/results --beta2 0.9999 --optimizer EFAdam

Usage (full sweep across linearreg / sinexperiment / easy_classification,
every discovered optimizer and beta2, MNIST excluded):
    python plot_fisher_metrics.py --sweep

Usage (appendix figure: one big grid per experiment, cosine-similarity panels
only, one shared legend, rows = every combo, columns = every batch size):
    python plot_fisher_metrics.py --appendix
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tueplots import bundles

# Applied once, globally, for every plot this script makes.
plt.rcParams.update(bundles.icml2022())
plt.rcParams["text.usetex"] = False              # avoids requiring/using a LaTeX
                                                  # install; raw Unicode (beta, subscripts)
                                                  # in titles would otherwise fail to render.
plt.rcParams["figure.constrained_layout.use"] = False  # tueplots enables this by default,
                                                  # but it silently overrides
                                                  # apply_consistent_figure_layout()'s
                                                  # manual subplots_adjust() call below,
                                                  # which is what was causing titles to
                                                  # overlap each other.

# Which experiments to sweep, and their folders -- MNIST deliberately excluded
# per request. Each entry is (display_name, results_folder_path).
SWEEP_EXPERIMENTS = [
    ("linearregexperiment", "linearregexperiment/results"),
    ("sinexperiment", "sinexperiment/results"),
    ("easy_classification", "easy_classification/results"),
]

FILENAME_PATTERN = re.compile(r"^([A-Za-z0-9]+)_([0-9.]+)_(\d+)_(\d+)\.json$")


def discover_combos(results_folder):
    """Scans results_folder for files matching {optimizer}_{beta2}_{batch}_{seed}.json
    and returns the sorted, de-duplicated set of (optimizer, beta2) pairs actually
    present -- found from the data itself, not assumed from any external list."""
    results_folder = Path(results_folder)
    combos = set()
    if not results_folder.exists():
        return []
    for json_file in results_folder.glob("*.json"):
        match = FILENAME_PATTERN.match(json_file.name)
        if match:
            optimizer, beta2_str = match.group(1), match.group(2)
            combos.add((optimizer, float(beta2_str)))
    return sorted(combos, key=lambda c: (c[0], c[1]))


def apply_consistent_figure_layout(fig):
    """Keep the plotting box stable when legends are placed outside the axes."""
    fig.subplots_adjust(left=0.08, right=0.80, bottom=0.10, top=0.90)


def load_results(results_folder, optimizer, beta2):
    """
    Load all JSON files matching the pattern: {optimizer}_{beta2}_{batch_size}_{seed}.json

    Returns a dict: {batch_size: [{"epoch": int, "metrics": {...}}, ...]}
    """
    results_folder = Path(results_folder)
    data_by_batch_size = defaultdict(lambda: defaultdict(list))

    for json_file in sorted(results_folder.glob(f"{optimizer}_{beta2}_*.json")):
        try:
            parts = json_file.stem.split("_")
            if len(parts) >= 4:
                batch_size = int(parts[2])
                seed = int(parts[3])

                with open(json_file, "r") as f:
                    data = json.load(f)

                if "fisher_approximations_over_time" in data:
                    for entry in data["fisher_approximations_over_time"]:
                        epoch = entry.get("epoch")
                        metrics = entry.get("fisher_approximations_dist", {})

                        data_by_batch_size[batch_size][epoch].append(
                            {
                                "MA_adam": metrics.get("MA_adam"),
                                "MA_emp": metrics.get("MA_emp"),
                                "empirical": metrics.get("empirical"),
                                "adam": metrics.get("adam"),
                            }
                        )
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Could not parse {json_file}: {e}")

    return data_by_batch_size


def average_metrics(data_by_batch_size):
    """
    Average metrics across seeds for each batch size and epoch.

    Returns: {batch_size: {epoch: {metric_name: {mean: float, std: float}}}}
    """
    averaged = {}

    for batch_size, epoch_data in data_by_batch_size.items():
        averaged[batch_size] = {}

        for epoch, metrics_list in epoch_data.items():
            averaged_metrics = {}
            for metric_name in ["MA_adam", "MA_emp", "empirical", "adam"]:
                values = [
                    m[metric_name] for m in metrics_list if m[metric_name] is not None
                ]
                if values:
                    averaged_metrics[metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                    }

            if "empirical" in averaged_metrics and "adam" in averaged_metrics:
                emp_mean = averaged_metrics["empirical"]["mean"]
                adam_mean = averaged_metrics["adam"]["mean"]
                emp_std = averaged_metrics["empirical"]["std"]
                adam_std = averaged_metrics["adam"]["std"]
                dist_std = np.sqrt(emp_std**2 + adam_std**2)
                averaged_metrics["emp_adam_dist"] = {
                    "mean": abs(emp_mean - adam_mean),
                    "std": dist_std,
                }

            if "MA_adam" in averaged_metrics and "MA_emp" in averaged_metrics:
                ma_adam_mean = averaged_metrics["MA_adam"]["mean"]
                ma_emp_mean = averaged_metrics["MA_emp"]["mean"]
                ma_adam_std = averaged_metrics["MA_adam"]["std"]
                ma_emp_std = averaged_metrics["MA_emp"]["std"]
                dist_std = np.sqrt(ma_adam_std**2 + ma_emp_std**2)
                averaged_metrics["ma_adam_emp_dist"] = {
                    "mean": abs(ma_adam_mean - ma_emp_mean),
                    "std": dist_std,
                }

            if averaged_metrics:
                averaged[batch_size][epoch] = averaged_metrics

    return averaged


def plot_metrics(averaged_data, optimizer, beta2, output_path=None, plot_type="both"):
    """
    Create plots for batch sizes showing metrics over time.
    """
    batch_sizes = sorted(averaged_data.keys())

    if not batch_sizes:
        print("No data found to plot.")
        return

    if plot_type == "combined":
        plot_metrics_combined(averaged_data, optimizer, beta2, output_path, batch_sizes)
        return

    metrics = ["MA_adam", "MA_emp", "empirical", "adam"]
    distance_metrics = ["emp_adam_dist", "ma_adam_emp_dist"]
    colors = {
        "MA_adam": "#1f77b4",
        "MA_emp": "#ff7f0e",
        "empirical": "#2ca02c",
        "adam": "#d62728",
        "emp_adam_dist": "#9467bd",
        "ma_adam_emp_dist": "#8c564b",
    }
    labels = {
        "emp_adam_dist": "|empirical - adam|",
        "ma_adam_emp_dist": "|MA_adam - MA_emp|",
    }

    for batch_size in batch_sizes:
        epoch_data = averaged_data[batch_size]
        epochs = sorted(epoch_data.keys())

        if plot_type == "metrics":
            fig, ax_main = plt.subplots(1, 1, figsize=(8, 5))
            ax_dist = None
        elif plot_type == "distances":
            fig, ax_dist = plt.subplots(1, 1, figsize=(8, 5))
            ax_main = None
        else:  # "both"
            fig, (ax_main, ax_dist) = plt.subplots(1, 2, figsize=(14, 5))

        if plot_type in ["metrics", "both"]:
            for metric in metrics:
                means, stds, valid_epochs = [], [], []
                for e in epochs:
                    if metric in epoch_data[e] and epoch_data[e][metric] is not None:
                        means.append(epoch_data[e][metric]["mean"])
                        stds.append(epoch_data[e][metric]["std"])
                        valid_epochs.append(e)
                if means:
                    means, stds = np.array(means), np.array(stds)
                    ax_main.plot(valid_epochs, means, marker="o", label=metric,
                                color=colors.get(metric), linewidth=2, markersize=4)
                    ax_main.fill_between(valid_epochs, means - stds, means + stds,
                                        alpha=0.2, color=colors.get(metric))

            ax_main.set_xlabel("Epoch", fontsize=10)
            ax_main.set_ylabel("Cosine Similarity", fontsize=10)
            ax_main.set_title(f"BS={batch_size} - Metrics", fontsize=11, fontweight="bold")
            ax_main.grid(True, alpha=0.3)
            ax_main.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            ax_main.set_ylim([-0.05, 1.05])

        if plot_type in ["distances", "both"]:
            for dist_metric in distance_metrics:
                means, stds, valid_epochs = [], [], []
                for e in epochs:
                    if dist_metric in epoch_data[e] and epoch_data[e][dist_metric] is not None:
                        means.append(epoch_data[e][dist_metric]["mean"])
                        stds.append(epoch_data[e][dist_metric]["std"])
                        valid_epochs.append(e)
                if means:
                    means, stds = np.array(means), np.array(stds)
                    ax_dist.plot(valid_epochs, means, marker="s", label=labels.get(dist_metric, dist_metric),
                                color=colors.get(dist_metric), linewidth=2, markersize=4)
                    ax_dist.fill_between(valid_epochs, np.maximum(means - stds, 0), means + stds,
                                        alpha=0.2, color=colors.get(dist_metric))

            ax_dist.set_xlabel("Epoch", fontsize=10)
            ax_dist.set_ylabel("Absolute Distance", fontsize=10)
            ax_dist.set_title(f"BS={batch_size} - Distances", fontsize=11, fontweight="bold")
            ax_dist.grid(True, alpha=0.3)
            ax_dist.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            ax_dist.set_ylim(bottom=0)

        fig.suptitle(f"{optimizer} (beta2={beta2}, BS={batch_size}) - Fisher Metrics Over Time",
                    fontsize=14, fontweight="bold", y=1.00)
        apply_consistent_figure_layout(fig)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stem, suffix = output_path.stem, output_path.suffix
            batch_output_path = output_path.parent / f"{stem}_BS{batch_size}{suffix}"
            plt.savefig(batch_output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {batch_output_path}")
        else:
            plt.show()

        plt.close(fig)


def plot_metrics_combined(averaged_data, optimizer, beta2, output_path, batch_sizes):
    """
    Create a single figure with all batch sizes: metrics in top row, distances in bottom row.
    """
    metrics = ["MA_adam", "MA_emp", "empirical", "adam"]
    distance_metrics = ["emp_adam_dist", "ma_adam_emp_dist"]
    colors = {
        "MA_adam": "#1f77b4",
        "MA_emp": "#ff7f0e",
        "empirical": "#2ca02c",
        "adam": "#d62728",
        "emp_adam_dist": "#9467bd",
        "ma_adam_emp_dist": "#8c564b",
    }
    labels = {
        "emp_adam_dist": "|empirical - adam|",
        "ma_adam_emp_dist": "|MA_adam - MA_emp|",
    }

    n_batch_sizes = len(batch_sizes)
    fig_width = 5 * n_batch_sizes if n_batch_sizes > 1 else 8
    fig, axes = plt.subplots(2, n_batch_sizes, figsize=(fig_width, 10))

    if n_batch_sizes == 1:
        axes = axes.reshape(2, 1)

    for col_idx, batch_size in enumerate(batch_sizes):
        epoch_data = averaged_data[batch_size]
        epochs = sorted(epoch_data.keys())

        ax_main = axes[0, col_idx]
        for metric in metrics:
            means, stds, valid_epochs = [], [], []
            for e in epochs:
                if metric in epoch_data[e] and epoch_data[e][metric] is not None:
                    means.append(epoch_data[e][metric]["mean"])
                    stds.append(epoch_data[e][metric]["std"])
                    valid_epochs.append(e)
            if means:
                means, stds = np.array(means), np.array(stds)
                ax_main.plot(valid_epochs, means, marker="o", label=metric,
                            color=colors.get(metric), linewidth=2, markersize=4)
                ax_main.fill_between(valid_epochs, means - stds, means + stds,
                                    alpha=0.2, color=colors.get(metric))

        ax_main.set_xlabel("Epoch", fontsize=9)
        ax_main.set_ylabel("Cosine Similarity", fontsize=9)
        ax_main.set_title(f"BS={batch_size} - Metrics", fontsize=10, fontweight="bold")
        ax_main.grid(True, alpha=0.3)
        if col_idx == n_batch_sizes - 1:
            ax_main.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        ax_main.set_ylim([-0.05, 1.05])

        ax_dist = axes[1, col_idx]
        for dist_metric in distance_metrics:
            means, stds, valid_epochs = [], [], []
            for e in epochs:
                if dist_metric in epoch_data[e] and epoch_data[e][dist_metric] is not None:
                    means.append(epoch_data[e][dist_metric]["mean"])
                    stds.append(epoch_data[e][dist_metric]["std"])
                    valid_epochs.append(e)
            if means:
                means, stds = np.array(means), np.array(stds)
                ax_dist.plot(valid_epochs, means, marker="s", label=labels.get(dist_metric, dist_metric),
                            color=colors.get(dist_metric), linewidth=2, markersize=4)
                ax_dist.fill_between(valid_epochs, np.maximum(means - stds, 0), means + stds,
                                    alpha=0.2, color=colors.get(dist_metric))

        ax_dist.set_xlabel("Epoch", fontsize=9)
        ax_dist.set_ylabel("Absolute Distance", fontsize=9)
        ax_dist.set_title(f"BS={batch_size} - Distances", fontsize=10, fontweight="bold")
        ax_dist.grid(True, alpha=0.3)
        if col_idx == n_batch_sizes - 1:
            ax_dist.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        ax_dist.set_ylim(bottom=0)

    fig.suptitle(f"{optimizer} (beta2={beta2}) - Fisher Metrics Over Time", fontsize=14, fontweight="bold", y=0.995)
    apply_consistent_figure_layout(fig)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Combined plot saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_cosine_grid(exp_name, results_folder, output_path):
    """One appendix-style figure per experiment: rows = every discovered
    (optimizer, beta2) combo, columns = every discovered batch size, each
    cell showing ONLY the cosine-similarity panel (no distances). A single
    shared legend covers the whole figure. Batch size is shown once, as a
    column header on the top row; optimizer/beta2 is shown once per row, as
    a y-axis label on the leftmost column; "Epoch" only appears on the
    bottom row -- avoids repeating the same three labels in tiny text on
    every one of the 40+ panels."""
    combos = discover_combos(results_folder)
    if not combos:
        print(f"[skip] {exp_name}: no JSON files found in {results_folder}")
        return

    combo_data = {}
    all_batch_sizes = set()
    for optimizer, beta2 in combos:
        data_by_batch = load_results(results_folder, optimizer, beta2)
        averaged = average_metrics(data_by_batch)
        combo_data[(optimizer, beta2)] = averaged
        all_batch_sizes.update(averaged.keys())
    batch_sizes = sorted(all_batch_sizes)

    n_rows, n_cols = len(combos), len(batch_sizes)
    if n_rows == 0 or n_cols == 0:
        print(f"[skip] {exp_name}: nothing to plot")
        return

    metrics = ["MA_adam", "MA_emp", "empirical", "adam"]
    colors = {
        "MA_adam": "#1f77b4",
        "MA_emp": "#ff7f0e",
        "empirical": "#2ca02c",
        "adam": "#d62728",
    }
    metric_labels = {
        "MA_adam": "MA(Adam)",
        "MA_emp": "MA(Empirical)",
        "empirical": "Empirical",
        "adam": "Adam",
    }

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 2.4 * n_rows),
        squeeze=False,
        layout="constrained",
    )

    for row_idx, (optimizer, beta2) in enumerate(combos):
        averaged = combo_data[(optimizer, beta2)]
        for col_idx, batch_size in enumerate(batch_sizes):
            ax = axes[row_idx, col_idx]
            epoch_data = averaged.get(batch_size)
            if not epoch_data:
                ax.axis("off")
                continue

            epochs = sorted(epoch_data.keys())
            for metric in metrics:
                means, stds, valid_epochs = [], [], []
                for e in epochs:
                    if metric in epoch_data[e] and epoch_data[e][metric] is not None:
                        means.append(epoch_data[e][metric]["mean"])
                        stds.append(epoch_data[e][metric]["std"])
                        valid_epochs.append(e)
                if means:
                    means, stds = np.array(means), np.array(stds)
                    ax.plot(valid_epochs, means, color=colors[metric], linewidth=1.2)
                    ax.fill_between(valid_epochs, means - stds, means + stds,
                                    alpha=0.15, color=colors[metric])

            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=6)

            # Column header: batch size, top row only, larger/bolder than the
            # old per-panel titles since it now only has to appear once per column.
            if row_idx == 0:
                ax.set_title(f"BS={batch_size}", fontsize=10, fontweight="bold")

            # Row label: optimizer + beta2, left column only -- replaces the
            # old cramped per-panel title with a single, readable label per row.
            if col_idx == 0:
                ax.set_ylabel(f"{optimizer}, beta2={beta2}\nCosine sim.", fontsize=8)
            else:
                ax.set_yticklabels([])

            # "Epoch" only on the bottom row -- every panel already shares the
            # same x-axis meaning, so repeating it 40+ times added nothing.
            if row_idx == n_rows - 1:
                ax.set_xlabel("Epoch", fontsize=8)
            else:
                ax.set_xticklabels([])

    legend_handles = [
        Line2D([0], [0], color=colors[m], lw=2, label=metric_labels[m]) for m in metrics
    ]
    fig.legend(handles=legend_handles, loc="outside upper center",
            ncol=len(metrics), fontsize=9, frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Appendix grid saved to {output_path}")
    plt.close(fig)

# Config for the Figure-4-style reproduction: fixed optimizer and beta2,
# only these four batch sizes, rows = experiment rather than rows = combo.
FIG4_EXPERIMENTS = [
    ("Linear regression", "linearregexperiment/results"),
    ("Binary classification", "easy_classification/results"),
]
FIG4_BATCH_SIZES = [50, 100, 150, 300]
FIG4_OPTIMIZER = "ReAdam"
FIG4_BETA2 = 0.99


def plot_figure4_style(output_path):
    """Reproduces the reference paper figure's layout: one row per experiment
    (linear regression, binary classification), one column per batch size
    (50/100/150/300), a single fixed optimizer and beta2, cosine-similarity
    panels only, one shared legend for the whole figure. Same visual
    machinery as plot_cosine_grid() (tueplots style, constrained layout,
    outside-top legend), just with rows/columns meaning something different:
    here rows are experiments (a fixed axis), not every discovered combo."""
    metrics = ["MA_adam", "MA_emp", "empirical", "adam"]
    colors = {
        "MA_adam": "#1f77b4",
        "MA_emp": "#ff7f0e",
        "empirical": "#2ca02c",
        "adam": "#d62728",
    }
    metric_labels = {
        "MA_adam": "MA(Adam)",
        "MA_emp": "MA(Empirical)",
        "empirical": "Empirical",
        "adam": "Adam",
    }

    n_rows, n_cols = len(FIG4_EXPERIMENTS), len(FIG4_BATCH_SIZES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 2.4 * n_rows),
        squeeze=False,
        layout="constrained",
    )

    for row_idx, (exp_label, results_folder) in enumerate(FIG4_EXPERIMENTS):
        data_by_batch = load_results(results_folder, FIG4_OPTIMIZER, FIG4_BETA2)
        averaged = average_metrics(data_by_batch)

        for col_idx, batch_size in enumerate(FIG4_BATCH_SIZES):
            ax = axes[row_idx, col_idx]
            epoch_data = averaged.get(batch_size)
            if not epoch_data:
                print(f"[warn] {exp_label}: no data for BS={batch_size}, "
                      f"optimizer={FIG4_OPTIMIZER}, beta2={FIG4_BETA2}")
                ax.axis("off")
                continue

            epochs = sorted(epoch_data.keys())
            for metric in metrics:
                means, stds, valid_epochs = [], [], []
                for e in epochs:
                    if metric in epoch_data[e] and epoch_data[e][metric] is not None:
                        means.append(epoch_data[e][metric]["mean"])
                        stds.append(epoch_data[e][metric]["std"])
                        valid_epochs.append(e)
                if means:
                    means, stds = np.array(means), np.array(stds)
                    ax.plot(valid_epochs, means, color=colors[metric], linewidth=1.2)
                    ax.fill_between(valid_epochs, means - stds, means + stds,
                                    alpha=0.15, color=colors[metric])

            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=6)

            if row_idx == 0:
                ax.set_title(f"BS={batch_size}", fontsize=10, fontweight="bold")

            if col_idx == 0:
                ax.set_ylabel(f"{exp_label}\nCosine sim.", fontsize=8)
            else:
                ax.set_yticklabels([])

            if row_idx == n_rows - 1:
                ax.set_xlabel("Epoch", fontsize=8)
            else:
                ax.set_xticklabels([])

    legend_handles = [
        Line2D([0], [0], color=colors[m], lw=2, label=metric_labels[m]) for m in metrics
    ]
    fig.legend(handles=legend_handles, loc="outside upper center",
               ncol=len(metrics), fontsize=9, frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Figure-4-style plot saved to {output_path}")
    plt.close(fig)

# Config for the custom 2x3 figure. Each cell is independently specified
# (task, optimizer, beta2, batch_size) since -- unlike the other grids in
# this file -- the row meaning changes per column: column 1 varies beta2
# within one task, columns 2-3 vary task within one beta2.
FIG1_TASK_FOLDERS = {
    "sine": ("Sine regression", "sinexperiment/results"),
    "linreg": ("Linear regression", "linearregexperiment/results"),
    "binclass": ("Binary classification", "easy_classification/results"),
}

# (task_key, optimizer, beta2, batch_size) for each of the 6 cells, laid out
# [row][col] to match the reference figure exactly.
FIG1_GRID = [
    # col 1 (panel a): sine, ReAdam, batch=10, beta2 varies by row
    [("sine", "ReAdam", 0.9, 10), ("linreg", "EFAdam", 0.9999, 150), ("sine", "EFAdam", 0.9999, 300)],
    # row 2
    [("sine", "ReAdam", 0.9999, 10), ("binclass", "EFAdam", 0.9999, 150), ("binclass", "EFAdam", 0.9999, 300)],
]

FIG1_COL_TITLES = ["BS=10", "BS=150", "BS=300"]


def plot_figure1_style(output_path):
    """Reproduces the reference figure's 2x3 layout, where each column has
    DIFFERENT row semantics (col 1: fixed task/batch, beta2 varies by row;
    cols 2-3: fixed beta2/batch, task varies by row) -- so every cell's
    (task, optimizer, beta2, batch_size) is specified explicitly in
    FIG1_GRID rather than derived from a uniform rows x columns rule."""
    metrics = ["MA_adam", "MA_emp", "empirical", "adam"]
    colors = {
        "MA_adam": "#1f77b4",
        "MA_emp": "#ff7f0e",
        "empirical": "#2ca02c",
        "adam": "#d62728",
    }
    metric_labels = {
        "MA_adam": "MA(Adam)",
        "MA_emp": "MA(Empirical)",
        "empirical": "Empirical",
        "adam": "Adam",
    }

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.6 * n_cols, 2.6 * n_rows),
        squeeze=False,
        layout="constrained",
    )

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            task_key, optimizer, beta2, batch_size = FIG1_GRID[row_idx][col_idx]
            task_label, results_folder = FIG1_TASK_FOLDERS[task_key]
            ax = axes[row_idx, col_idx]

            data_by_batch = load_results(results_folder, optimizer, beta2)
            averaged = average_metrics(data_by_batch)
            epoch_data = averaged.get(batch_size)

            if not epoch_data:
                print(f"[warn] no data for {task_label}, {optimizer}, beta2={beta2}, BS={batch_size}")
                ax.axis("off")
                continue

            epochs = sorted(epoch_data.keys())
            for metric in metrics:
                means, stds, valid_epochs = [], [], []
                for e in epochs:
                    if metric in epoch_data[e] and epoch_data[e][metric] is not None:
                        means.append(epoch_data[e][metric]["mean"])
                        stds.append(epoch_data[e][metric]["std"])
                        valid_epochs.append(e)
                if means:
                    means, stds = np.array(means), np.array(stds)
                    ax.plot(valid_epochs, means, color=colors[metric], linewidth=1.2)
                    ax.fill_between(valid_epochs, means - stds, means + stds,
                                    alpha=0.15, color=colors[metric])

            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=6)

            if row_idx == 0:
                ax.set_title(FIG1_COL_TITLES[col_idx], fontsize=10, fontweight="bold")

            # Each cell's own row label (task + beta2 + optimizer), since --
            # unlike the other grids -- a single left-column label wouldn't
            # correctly describe every row (column 1's rows differ by beta2,
            # columns 2-3's rows differ by task).
            ax.set_ylabel(f"{task_label}\n{optimizer}, beta2={beta2}\nCosine sim.", fontsize=7)

            if row_idx == n_rows - 1:
                ax.set_xlabel("Epoch", fontsize=8)
            else:
                ax.set_xticklabels([])

    legend_handles = [
        Line2D([0], [0], color=colors[m], lw=2, label=metric_labels[m]) for m in metrics
    ]
    fig.legend(handles=legend_handles, loc="outside upper center",
               ncol=len(metrics), fontsize=9, frameon=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Figure-1-style plot saved to {output_path}")
    plt.close(fig)

def run_one(results_folder, optimizer, beta2, output, plot_type):
    print(f"Loading results from {results_folder}...")
    print(f"Optimizer: {optimizer}, Beta2: {beta2}")

    data_by_batch_size = load_results(results_folder, optimizer, beta2)
    if not data_by_batch_size:
        print(f"No JSON files found matching pattern: {optimizer}_{beta2}_*.json")
        return

    print(f"Found {len(data_by_batch_size)} batch sizes: {sorted(data_by_batch_size.keys())}")
    averaged_data = average_metrics(data_by_batch_size)
    for batch_size in sorted(averaged_data.keys()):
        epochs = sorted(averaged_data[batch_size].keys())
        print(f"  Batch size {batch_size}: {len(epochs)} epochs")

    plot_metrics(averaged_data, optimizer, beta2, output, plot_type)


def run_sweep():
    """Runs plot_type='combined' for every (optimizer, beta2) combo actually
    found in each of the three non-MNIST experiments' results folders."""
    for exp_name, results_folder in SWEEP_EXPERIMENTS:
        combos = discover_combos(results_folder)
        if not combos:
            print(f"[skip] {exp_name}: no JSON files found in {results_folder}")
            continue
        print(f"[{exp_name}] discovered {len(combos)} (optimizer, beta2) combos: {combos}")
        for optimizer, beta2 in combos:
            output = f"{results_folder}/fisher_over_time_{optimizer}_{beta2}.png"
            run_one(results_folder, optimizer, beta2, output, plot_type="combined")


def run_appendix():
    """One big cosine-similarity-only grid per experiment, for the paper
    appendix. Rows = every discovered (optimizer, beta2) combo, columns =
    every discovered batch size, one shared legend per figure."""
    for exp_name, results_folder in SWEEP_EXPERIMENTS:
        output = f"{results_folder}/appendix_cosine_grid_{exp_name}.png"
        plot_cosine_grid(exp_name, results_folder, output)


def main():
    parser = argparse.ArgumentParser(description="Plot Fisher metrics over time from JSON results")
    parser.add_argument("--results_folder", type=str, help="Path to results folder containing JSON files")
    parser.add_argument("--beta2", type=float, help="Beta2 value (e.g., 0.9999)")
    parser.add_argument("--optimizer", type=str, help="Optimizer name (e.g., EFAdam, ReAdam, Adam)")
    parser.add_argument("--output", type=str, default=None, help="Output file path for saving the plot")
    parser.add_argument("--plot_type", type=str, choices=["metrics", "distances", "both", "combined"],
                        default="both", help="Plot type")
    parser.add_argument("--sweep", action="store_true",
                        help="Run the full sweep (linearreg/sinexperiment/easy_classification, "
                             "every optimizer/beta2 combo found, MNIST excluded) instead of a single run.")
    parser.add_argument("--appendix", action="store_true",
                        help="Produce one big cosine-similarity-only grid per experiment "
                             "(rows=combos, columns=batch sizes, one shared legend) for the paper appendix.")
    parser.add_argument("--figure4", action="store_true",
                        help="Reproduce the reference figure's layout: rows=linreg/binary "
                             "classification, columns=BS 50/100/150/300, fixed ReAdam, beta2=0.99.")   
    parser.add_argument("--figure1", action="store_true",
                        help="Reproduce the reference 2x3 figure: col1=sine (beta2 0.9/0.9999, BS=10, "
                             "ReAdam), col2=linreg/binclass (BS=150, beta2=0.9999, EFAdam), "
                             "col3=sine/binclass (BS=300, beta2=0.9999, EFAdam).")
    args = parser.parse_args()

    if args.figure1:
        plot_figure1_style("fig1_style.png")
        return
    
    if args.figure4:
        plot_figure4_style(f"fig4_style_{FIG4_OPTIMIZER}_beta{FIG4_BETA2}.png")
        return
    
    if args.appendix:
        run_appendix()
        return

    if args.sweep:
        run_sweep()
        return

    if not (args.results_folder and args.beta2 is not None and args.optimizer):
        parser.error("--results_folder, --beta2, and --optimizer are required unless --sweep or --appendix is given")

    run_one(args.results_folder, args.optimizer, args.beta2, args.output, args.plot_type)


if __name__ == "__main__":
    main()