"""Plot each individual model's own shard (train) and test accuracy, for
either merging suite.

This is a diagnostic check, separate from the merge-scheme comparison plots:
it asks "did each model actually learn its own subset well" (shard/train
accuracy) and "did it generalize" (test accuracy), independent of anything
about merging or Fisher schemes. A large gap between the two for one model
-- shard accuracy near 100% while test accuracy stays flat or low -- is a
direct sign that model has memorized its shard rather than learned
generalizable structure, which matters for interpreting any merge-scheme
comparison downstream.

Two suites, two different x-axes -- picking the wrong one per suite gives a
degenerate plot (everything collapsing to the same x value), the same
failure mode already hit once with plot_merge_results.py:
    --suite equal_exposure_trend   x-axis = batch-size ratio (m_A / m_B)
                                    (shard sizes are fixed/equal in this suite)
    --suite unbalanced_shards      x-axis = shard-size ratio (n_A / n_B)
                                    (batch size is fixed/equal in this suite)

Requires each result JSON to have a top-level "shard_accuracy" key (added to
fisher_merging_demo.py's output alongside the existing "endpoint_accuracy").
Older result JSONs without this key are skipped with a warning, not silently
plotted as zero.

Usage (from the repo root):
    python merging_experiments/plot_shard_performance.py --suite equal_exposure_trend
    python merging_experiments/plot_shard_performance.py --suite unbalanced_shards

Writes two PNGs into --output-dir, filenames tagged by suite so the two
suites can never silently overwrite each other's output:
    shard_accuracy_vs_batch_ratio__equal_exposure_trend.png
    test_accuracy_vs_batch_ratio__equal_exposure_trend.png
    shard_accuracy_vs_shard_ratio__unbalanced_shards.png
    test_accuracy_vs_shard_ratio__unbalanced_shards.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a cluster login/compute node
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_OUTPUT_DIR = Path("merging_experiments/results")

SUITE_CONFIG = {
    "equal_exposure_trend": {
        "default_results_dir": "merging_experiments/results/equal_exposure_trend_cifar_trimmed",
        "ratio_keys": ("batch_size_a", "batch_size_b"),
        "xlabel": "batch size ratio (m_A / m_B)",
        "file_tag": "batch_ratio",
        "model_a_label": "model A (batch swept)",
        "model_b_label": "model B (batch fixed)",
    },
    "unbalanced_shards": {
        "default_results_dir": "merging_experiments/results/unbalanced_shards_cifar_trimmed",
        "ratio_keys": ("shard_size_a_resolved", "shard_size_b_resolved"),
        "xlabel": "shard size ratio (n_A / n_B)",
        "file_tag": "shard_ratio",
        "model_a_label": "model A (shard swept)",
        "model_b_label": "model B (shard complement)",
    },
}

# Okabe-Ito colorblind-safe, consistent with plot_merge_results.py / plot_batch_ratio.py
MODEL_COLORS = {
    "model_a": "#0072B2",  # blue
    "model_b": "#D55E00",  # vermillion
}


def load_runs(results_dir: Path):
    runs = []
    skipped_no_shard_acc = 0
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as handle:
            data = json.load(handle)
        if "shard_accuracy" not in data:
            skipped_no_shard_acc += 1
            continue
        runs.append(data)
    if skipped_no_shard_acc:
        print(f"[warn] skipped {skipped_no_shard_acc} run(s) with no 'shard_accuracy' key "
              f"(predates the shard_accuracy fix -- rerun to include them)")
    if not runs:
        raise SystemExit(
            f"No usable run JSONs (with 'shard_accuracy') found under {results_dir}."
        )
    return runs


def group_by_ratio_key(runs, ratio_keys):
    key_a, key_b = ratio_keys
    grouped = defaultdict(list)
    for run in runs:
        cfg = run["config"]
        key = (cfg[key_a], cfg[key_b])
        grouped[key].append(run)
    return grouped


def plot_metric(grouped, metric_key, suite_cfg, title, ylabel, out_path: Path):
    """metric_key is 'shard_accuracy' or 'endpoint_accuracy' -- both have the
    same {"model_a": ..., "model_b": ...} shape in each run's JSON."""
    pairs = sorted(grouped, key=lambda p: p[0] / p[1])
    ratios = [a / b for a, b in pairs]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_key, label in [
        ("model_a", suite_cfg["model_a_label"]),
        ("model_b", suite_cfg["model_b_label"]),
    ]:
        means, stds = [], []
        for pair in pairs:
            values = [run[metric_key][model_key] for run in grouped[pair]]
            means.append(np.mean(values))
            stds.append(np.std(values))
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(ratios, means, yerr=stds, marker="o", markersize=5, capsize=3,
                    label=label, color=MODEL_COLORS[model_key], linewidth=1.6)

    ax.set_xscale("log")
    ax.set_xlabel(suite_cfg["xlabel"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("equal_exposure_trend", "unbalanced_shards"),
        required=True,
        help="Which suite's data this is -- determines the x-axis (batch ratio vs shard ratio) "
             "and the default --results-dir. Picking the wrong suite for a given results "
             "directory gives a degenerate plot (all points at the same x value).",
    )
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    suite_cfg = SUITE_CONFIG[args.suite]
    results_dir = Path(args.results_dir or suite_cfg["default_results_dir"])
    output_dir = Path(args.output_dir)

    runs = load_runs(results_dir)
    grouped = group_by_ratio_key(runs, suite_cfg["ratio_keys"])
    print(f"[{args.suite}] {len(runs)} runs across {len(grouped)} distinct ratios")

    tag = suite_cfg["file_tag"]
    plot_metric(
        grouped, "shard_accuracy", suite_cfg,
        title=f"Each model's accuracy on its OWN training shard ({args.suite})",
        ylabel="shard (train) accuracy",
        out_path=output_dir / f"shard_accuracy_vs_{tag}__{args.suite}.png",
    )
    plot_metric(
        grouped, "endpoint_accuracy", suite_cfg,
        title=f"Each model's test accuracy ({args.suite})",
        ylabel="test accuracy",
        out_path=output_dir / f"test_accuracy_vs_{tag}__{args.suite}.png",
    )


if __name__ == "__main__":
    main()