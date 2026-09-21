"""Plot each individual model's own shard (train) and test accuracy against
the batch-size ratio, for the equal_exposure_trend suite.

This is a diagnostic check, separate from the merge-scheme comparison plots:
it asks "did each model actually learn its own subset well" (shard/train
accuracy) and "did it generalize" (test accuracy), independent of anything
about merging or Fisher schemes. A large gap between the two for one model
-- shard accuracy near 100% while test accuracy stays flat or low -- is a
direct sign that model has memorized its shard rather than learned
generalizable structure, which matters for interpreting any merge-scheme
comparison downstream (a merge combining a memorized model with a
well-generalizing one is not a fair comparison of importance-weighting
schemes on their own terms).

Requires each result JSON to have a top-level "shard_accuracy" key
(added to fisher_merging_demo.py's output alongside the existing
"endpoint_accuracy" -- older result JSONs without this key are skipped
with a warning, not silently plotted as zero).

Usage (from the repo root):
    python merging_experiments/plot_shard_performance.py

Writes two PNGs into merging_experiments/results/:
    shard_accuracy_vs_batch_ratio.png   (train/shard accuracy)
    test_accuracy_vs_batch_ratio.png    (test accuracy, i.e. endpoint_accuracy)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a cluster login/compute node
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_DIR = Path("merging_experiments/results/equal_exposure_trend_cifar_trimmed")
DEFAULT_OUTPUT_DIR = Path("merging_experiments/results")

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


def group_by_batch_pair(runs):
    grouped = defaultdict(list)
    for run in runs:
        cfg = run["config"]
        key = (cfg["batch_size_a"], cfg["batch_size_b"])
        grouped[key].append(run)
    return grouped


def plot_metric(grouped, metric_key, title, ylabel, out_path: Path):
    """metric_key is 'shard_accuracy' or 'endpoint_accuracy' -- both have the
    same {"model_a": ..., "model_b": ...} shape in each run's JSON."""
    pairs = sorted(grouped, key=lambda p: p[0] / p[1])
    ratios = [m_a / m_b for m_a, m_b in pairs]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_key, label in [("model_a", "model A (swept batch)"), ("model_b", "model B (fixed batch)")]:
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
    ax.set_xlabel("batch size ratio (m_A / m_B)")
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
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    runs = load_runs(Path(args.results_dir))
    grouped = group_by_batch_pair(runs)
    print(f"{len(runs)} runs across {len(grouped)} batch-size pairs")

    output_dir = Path(args.output_dir)
    plot_metric(
        grouped, "shard_accuracy",
        title="Each model's accuracy on its OWN training shard, vs batch-size ratio",
        ylabel="shard (train) accuracy",
        out_path=output_dir / "shard_accuracy_vs_batch_ratio.png",
    )
    plot_metric(
        grouped, "endpoint_accuracy",
        title="Each model's test accuracy, vs batch-size ratio",
        ylabel="test accuracy",
        out_path=output_dir / "test_accuracy_vs_batch_ratio.png",
    )


if __name__ == "__main__":
    main()