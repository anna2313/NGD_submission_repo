"""Plot performance and importance-weight ratios against the batch-size ratio,
for the equal_exposure_trend merging suite (the only suite where the two
models' batch sizes actually differ -- unbalanced_shards holds batch size
fixed at 128 for both models, so its batch-size ratio is always 1 there).

Reads the raw per-run JSONs directly (not the aggregated CSV), since the
importance-ratio plot needs per-model diagnostics (exp_avg_sq_norm,
correction_factor, probe_fisher_norm, etc.) that aggregate_results.py does
not carry into its output.

Usage (from the repo root):
    python merging_experiments/plot_batch_ratio.py

Writes two PNGs into merging_experiments/results/:
    performance_vs_batch_ratio.png
    importance_ratio_vs_batch_ratio.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a cluster login/compute node
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_DIR = Path("merging_experiments/results/equal_exposure_trend")
DEFAULT_OUTPUT_DIR = Path("merging_experiments/results")

# Schemes whose per-model importance NORM can be reconstructed exactly from
# the diagnostics saved in each run's JSON. This is exact, not approximate:
# every one of these schemes is the raw accumulator or probe Fisher
# multiplied by a single scalar that is uniform across all parameters
# (batch size, a correction factor, a bias-correction term), so
# norm(c * v) = c * norm(v) exactly for any positive scalar c.
# uniform is excluded: both models get an all-ones vector, so its ratio is
# always exactly 1 by construction and carries no information.
SCHEME_COLORS = {
    "probe_fisher": "#0072B2",         # blue
    "probe_theory_raw": "#E69F00",     # orange
    "squisher_raw": "#009E73",         # green
    "squisher_nscaled": "#D55E00",     # vermillion
    "squisher_mscaled": "#CC79A7",     # pink
    "squisher_corrected": "#6A3D9A",   # purple
}


def importance_norm(scheme, diag, batch_size):
    """Reconstructs the L2 norm of one model's importance vector for a given
    scheme, from that model's saved diagnostics."""
    raw_norm = diag["exp_avg_sq_norm"]
    bc2 = diag["second_moment_bias_correction"]
    factor = diag["correction_factor"]
    probe_norm = diag["probe_fisher_norm"]
    if scheme == "squisher_raw":
        return raw_norm
    if scheme == "squisher_nscaled":
        return raw_norm * batch_size
    if scheme == "squisher_mscaled":
        return raw_norm / bc2 * batch_size
    if scheme == "squisher_corrected":
        return raw_norm / bc2 * factor
    if scheme == "probe_fisher":
        return probe_norm
    if scheme == "probe_theory_raw":
        return probe_norm * bc2 / factor
    raise ValueError(f"no norm reconstruction for scheme {scheme!r}")

def load_runs(results_dir: Path):
    runs = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path) as handle:
            runs.append(json.load(handle))
    if not runs:
        raise SystemExit(
            f"No run JSONs found under {results_dir} -- run the equal_exposure_trend "
            "suite first."
        )
    return runs


def group_by_batch_pair(runs):
    """Returns {(batch_size_a, batch_size_b): [run, run, ...]} -- one entry
    per seed sharing that batch-size pair."""
    grouped = defaultdict(list)
    for run in runs:
        cfg = run["config"]
        key = (cfg["batch_size_a"], cfg["batch_size_b"])
        grouped[key].append(run)
    return grouped


def plot_performance(grouped, out_path: Path):
    pairs = sorted(grouped, key=lambda p: p[0] / p[1])
    ratios = [m_a / m_b for m_a, m_b in pairs]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    schemes = sorted({s for runs in grouped.values() for s in runs[0]["merges"]})
    for scheme in schemes:
        means, stds = [], []
        for pair in pairs:
            accs = [run["merges"][scheme]["test_accuracy"] for run in grouped[pair]]
            means.append(np.mean(accs))
            stds.append(np.std(accs))
        color = SCHEME_COLORS.get(scheme, "gray")
        linestyle = "--" if scheme.endswith("_sum") else ("-." if scheme == "uniform" else "-")
        ax.errorbar(ratios, means, yerr=stds, marker="o", markersize=4, capsize=3,
                    label=scheme, color=color, linestyle=linestyle, linewidth=1.3)

    ax.axvline(1.0, color="black", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("batch size ratio (m_A / m_B)")
    ax.set_ylabel("merged model test accuracy")
    ax.set_title("Merged accuracy vs batch-size ratio (equal_exposure_trend)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_importance_ratio(grouped, out_path: Path):
    pairs = sorted(grouped, key=lambda p: p[0] / p[1])
    batch_ratios = [m_a / m_b for m_a, m_b in pairs]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for scheme, color in SCHEME_COLORS.items():
        means, stds = [], []
        for (m_a, m_b) in pairs:
            per_seed_ratios = []
            for run in grouped[(m_a, m_b)]:
                diag_a = run["diagnostics"]["model_a"]
                diag_b = run["diagnostics"]["model_b"]
                norm_a = importance_norm(scheme, diag_a, m_a)
                norm_b = importance_norm(scheme, diag_b, m_b)
                per_seed_ratios.append(norm_a / norm_b)
            means.append(np.mean(per_seed_ratios))
            stds.append(np.std(per_seed_ratios))
        ax.errorbar(batch_ratios, means, yerr=stds, marker="o", markersize=4, capsize=3,
                    label=scheme, color=color)

    ref_x = np.array(sorted(batch_ratios))
    ax.plot(ref_x, ref_x, color="black", linestyle=":", linewidth=1,
            label="importance ratio = batch ratio")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size ratio (m_A / m_B)")
    ax.set_ylabel("importance-weight norm ratio (model A / model B)")
    ax.set_title("Importance ratio vs batch-size ratio (equal_exposure_trend)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
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
    plot_performance(grouped, output_dir / "performance_vs_batch_ratio.png")
    plot_importance_ratio(grouped, output_dir / "importance_ratio_vs_batch_ratio.png")


if __name__ == "__main__":
    main()