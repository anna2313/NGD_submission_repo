"""Aggregate and plot the Squisher validation-tuned grid's two result files:
results.csv (accuracy, from parse_results_valid.py) and fidelity.csv (the
per-context SQUISHER-FIDELITY diagnostics logged during training).

Usage (from squisher_experiments/continual-learning-squisher/):
    python squisher_experiments/analyze_fidelity.py

Reads logs_valid_tuned/{results,fidelity}.csv by default; writes
logs_valid_tuned/fidelity_summary.csv and three PNGs into the same folder.
Either input file may be missing -- whichever plots need it are skipped
with a warning rather than crashing, so this is safe to run after only
the tune stage, only the transfer stage, or both.

fidelity.csv has no batch/lambda/tuned columns of its own -- those are
parsed from its "run" column, using the exact same naming scheme
run_grid_valid.run_key() writes (e.g. "squisher_raw_b128_lam1e+06_s1_vs0.1").
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a cluster login/compute node
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent / "logs_valid_tuned_cifar"

KEY_PATTERN = re.compile(
    r"^(?P<source>none|empirical_nscaled|empirical|squisher_\w+?)_b(?P<batch>\d+)(?:_lam(?P<lam>[0-9e+.-]+))?_s(?P<seed>\d+)"
    r"(?P<tuned>_vs[0-9.]+)?$"
)

# Fixed source order/colors so every plot uses the same mapping and is easy
# to compare across figures.
SOURCE_ORDER = [
    "empirical",
    "empirical_nscaled",
    "squisher_raw",
    "squisher_biascorrected",
    "squisher_nscaled",
    "squisher_mscaled",
    "squisher_corrected",
]
SOURCE_COLORS = dict(zip(SOURCE_ORDER, plt.cm.tab10.colors))


def parse_run_key(run: str):
    """Returns (source, batch, lam, seed, tuned) parsed from a run_key()
    string, or None if it doesn't match (e.g. an unrecognized source)."""
    match = KEY_PATTERN.match(run)
    if match is None:
        return None
    info = match.groupdict()
    return (
        info["source"],
        int(info["batch"]),
        float(info["lam"]) if info["lam"] else None,
        int(info["seed"]),
        bool(info["tuned"]),
    )


def load_results(path: Path):
    if not path.exists():
        print(f"[warn] {path} not found -- skipping anything that needs results.csv")
        return []
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "source": row["source"],
                "batch": int(row["batch"]),
                "lambda": float(row["lambda"]) if row["lambda"] else None,
                "seed": int(row["seed"]),
                "tuned": row["tuned_with_valid_split"] in ("True", "1", "true"),
                "test_accuracy": float(row["test_accuracy"]),
                "validation_accuracy": (
                    float(row["validation_accuracy"]) if row["validation_accuracy"] else None
                ),
            })
    return rows


def load_fidelity(path: Path):
    if not path.exists():
        print(f"[warn] {path} not found -- skipping anything that needs fidelity.csv")
        return []
    rows = []
    skipped = 0
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = parse_run_key(row["run"])
            if parsed is None:
                skipped += 1
                continue
            source, batch, lam, seed, tuned = parsed
            rows.append({
                "source": source,
                "batch": batch,
                "lambda": lam,
                "seed": seed,
                "tuned": tuned,
                "context": int(row["context"]),
                "cosine": float(row["cosine"]),
                "norm_ratio": float(row["norm_ratio"]),
                "norm_est": float(row["norm_est"]),
            })
    if skipped:
        print(f"[warn] {skipped} fidelity.csv rows had an unparseable 'run' value and were skipped")
    return rows


def _finite_mean_std(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    frac_finite = len(finite) / len(values) if len(values) else float("nan")
    if not len(finite):
        return float("nan"), float("nan"), frac_finite
    return float(finite.mean()), float(finite.std(ddof=0)), frac_finite


def _finite_median(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else float("nan")


def aggregate_fidelity(rows):
    """Groups by (source, batch, lambda, tuned), pooling across seeds AND
    contexts. Returns rows for fidelity_summary.csv, plus prints a flag for
    any group where the accumulator blew up to inf/nan on a real fraction
    of its samples -- this is itself a finding, not just plotting input."""
    grouped = defaultdict(lambda: {"cosine": [], "norm_ratio": [], "norm_est": []})
    for row in rows:
        key = (row["source"], row["batch"], row["lambda"], row["tuned"])
        grouped[key]["cosine"].append(row["cosine"])
        grouped[key]["norm_ratio"].append(row["norm_ratio"])
        grouped[key]["norm_est"].append(row["norm_est"])

    summary = []
    blowups = []
    for (source, batch, lam, tuned), values in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
        cosine_mean, cosine_std, cosine_frac_finite = _finite_mean_std(values["cosine"])
        norm_ratio_median = _finite_median(values["norm_ratio"])
        _, _, norm_ratio_frac_finite = _finite_mean_std(values["norm_ratio"])
        norm_est_median = _finite_median(values["norm_est"])
        n = len(values["cosine"])
        summary.append({
            "source": source,
            "batch": batch,
            "lambda": lam if lam is not None else "",
            "tuned": tuned,
            "n_samples": n,
            "cosine_mean": cosine_mean,
            "cosine_std": cosine_std,
            "cosine_frac_finite": cosine_frac_finite,
            "norm_ratio_median": norm_ratio_median,
            "norm_ratio_frac_finite": norm_ratio_frac_finite,
            "norm_est_median": norm_est_median,
        })
        if cosine_frac_finite < 1.0 or norm_ratio_frac_finite < 1.0:
            blowups.append((source, batch, lam, cosine_frac_finite, norm_ratio_frac_finite, n))

    if blowups:
        print(f"\n[flag] {len(blowups)} (source, batch, lambda) groups had non-finite "
              f"(inf/nan) cosine and/or norm_ratio in at least one context/seed:")
        print(f"{'source':24s} {'batch':>6s} {'lambda':>10s} {'cosine ok':>10s} {'norm ok':>10s} {'n':>4s}")
        for source, batch, lam, cf, nf, n in blowups:
            lam_str = f"{lam:.0e}" if lam is not None else "-"
            print(f"{source:24s} {batch:6d} {lam_str:>10s} {cf:10.2f} {nf:10.2f} {n:4d}")

    return summary


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def plot_accuracy_vs_lambda(results_rows, out_path: Path):
    """Test (solid) and validation (dashed) accuracy vs lambda at batch=128,
    one line per source, plus the untuned 'none' baseline as a horizontal
    reference. This is the plot that makes the grid-boundary problem
    (a source still improving at the edge of LAMBDA_GRID) visually obvious."""
    if not results_rows:
        print("[skip] plot_accuracy_vs_lambda: no results.csv data")
        return

    tuned_rows = [r for r in results_rows if r["batch"] == 128 and r["tuned"] and r["source"] != "none"]
    if not tuned_rows:
        print("[skip] plot_accuracy_vs_lambda: no tuned batch=128 rows found")
        return

    none_rows = [r for r in results_rows if r["source"] == "none" and r["batch"] == 128]

    fig, ax = plt.subplots(figsize=(7, 5))
    for source in SOURCE_ORDER:
        source_rows = [r for r in tuned_rows if r["source"] == source]
        if not source_rows:
            continue
        by_lambda = defaultdict(list)
        by_lambda_valid = defaultdict(list)
        for r in source_rows:
            by_lambda[r["lambda"]].append(r["test_accuracy"])
            if r["validation_accuracy"] is not None:
                by_lambda_valid[r["lambda"]].append(r["validation_accuracy"])
        lambdas = sorted(by_lambda)
        means = [np.mean(by_lambda[lam]) for lam in lambdas]
        stds = [np.std(by_lambda[lam]) for lam in lambdas]
        color = SOURCE_COLORS.get(source)
        ax.errorbar(lambdas, means, yerr=stds, marker="o", label=source, color=color, capsize=3)
        if by_lambda_valid:
            valid_means = [np.mean(by_lambda_valid.get(lam, [np.nan])) for lam in lambdas]
            ax.plot(lambdas, valid_means, linestyle="--", color=color, alpha=0.5)

    if none_rows:
        none_mean = np.mean([r["test_accuracy"] for r in none_rows])
        ax.axhline(none_mean, color="gray", linestyle=":", label="none (no EWC)")

    ax.set_xscale("log")
    ax.set_xlabel("lambda (reg-strength)")
    ax.set_ylabel("accuracy (solid=test, dashed=validation)")
    ax.set_title("Accuracy vs lambda at batch=128 (tune stage)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_fidelity_vs_lambda(fidelity_summary_rows, out_path: Path):
    """Cosine similarity to the probe Fisher (top) and median norm_ratio on
    a log scale (bottom) vs lambda, one line per source. Point size shrinks
    for groups where a meaningful fraction of samples were inf/nan, since
    the mean/median there is less trustworthy."""
    if not fidelity_summary_rows:
        print("[skip] plot_fidelity_vs_lambda: no fidelity.csv data")
        return

    tuned = [r for r in fidelity_summary_rows if r["batch"] == 128 and r["tuned"] and r["lambda"] != ""]
    if not tuned:
        print("[skip] plot_fidelity_vs_lambda: no tuned batch=128 fidelity rows found")
        return

    fig, (ax_cos, ax_norm) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for source in SOURCE_ORDER:
        source_rows = sorted((r for r in tuned if r["source"] == source), key=lambda r: r["lambda"])
        if not source_rows:
            continue
        color = SOURCE_COLORS.get(source)
        lambdas = [r["lambda"] for r in source_rows]
        cosines = [r["cosine_mean"] for r in source_rows]
        sizes = [20 + 60 * r["cosine_frac_finite"] for r in source_rows]
        ax_cos.plot(lambdas, cosines, color=color, label=source, alpha=0.6)
        ax_cos.scatter(lambdas, cosines, color=color, s=sizes)

        norm_medians = [r["norm_ratio_median"] for r in source_rows]
        ax_norm.plot(lambdas, norm_medians, color=color, marker="o", label=source)

    ax_cos.set_ylabel("mean cosine(accumulator, probe Fisher)\n(point size ~ fraction finite)")
    ax_cos.set_title("Fidelity vs lambda at batch=128 (tune stage)")
    ax_cos.legend(fontsize=8)

    ax_norm.set_xscale("log")
    ax_norm.set_yscale("log")
    ax_norm.set_xlabel("lambda (reg-strength)")
    ax_norm.set_ylabel("median norm_ratio (log scale)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_batch_transfer(results_rows, out_path: Path):
    """The headline figure: for each source's SELECTED lambda (the one that
    also has a batch=32/512 transfer row), plot test accuracy across
    batch in {32, 25, 512}, plus the untuned 'none' baseline. This is what
    actually answers "does correction survive batch-size transfer"."""
    if not results_rows:
        print("[skip] plot_batch_transfer: no results.csv data")
        return

    transfer_rows = [r for r in results_rows if r["batch"] != 128  and r["source"] != "none"]
    if not transfer_rows:
        print("[skip] plot_batch_transfer: no transfer-stage rows found")
        return
    transfer_batches = sorted({r["batch"] for r in transfer_rows})

    # The transferred lambda per source is whatever lambda actually has a
    # batch=32/512 row -- no need to read best_lambdas.json separately.
    selected_lambda = {}
    for r in transfer_rows:
        selected_lambda.setdefault(r["source"], r["lambda"])

    none_rows = [r for r in results_rows if r["source"] == "none"]
    none_by_batch = defaultdict(list)
    for r in none_rows:
        none_by_batch[r["batch"]].append(r["test_accuracy"])

    fig, ax = plt.subplots(figsize=(7, 5))
    batches = sorted({128, *transfer_batches})
    for source, lam in selected_lambda.items():
        means, stds = [], []
        for batch in batches:
            matching = [r for r in results_rows
                        if r["source"] == source and r["batch"] == batch and r["lambda"] == lam]
            if matching:
                accs = [r["test_accuracy"] for r in matching]
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            else:
                means.append(np.nan)
                stds.append(0)
        color = SOURCE_COLORS.get(source)
        ax.errorbar(batches, means, yerr=stds, marker="o", label=f"{source} (lambda={lam:.0e})",
                    color=color, capsize=3)

    if none_by_batch:
        none_means = [np.mean(none_by_batch.get(b, [np.nan])) for b in batches]
        ax.plot(batches, none_means, linestyle=":", color="gray", marker="x", label="none (no EWC)")

    ax.set_xscale("log")
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("batch size")
    ax.set_ylabel("test accuracy")
    ax.set_title("Batch-size transfer at each source's selected lambda")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity", default=str(DEFAULT_DIR / "fidelity.csv"))
    parser.add_argument("--results", default=str(DEFAULT_DIR / "results.csv"))
    parser.add_argument("--output-dir", default=str(DEFAULT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_rows = load_results(Path(args.results))
    fidelity_rows = load_fidelity(Path(args.fidelity))
    fidelity_summary = aggregate_fidelity(fidelity_rows)
    write_csv(output_dir / "fidelity_summary.csv", fidelity_summary)

    plot_accuracy_vs_lambda(results_rows, output_dir / "accuracy_vs_lambda.png")
    plot_fidelity_vs_lambda(fidelity_summary, output_dir / "fidelity_vs_lambda.png")
    plot_batch_transfer(results_rows, output_dir / "batch_transfer.png")


if __name__ == "__main__":
    main()