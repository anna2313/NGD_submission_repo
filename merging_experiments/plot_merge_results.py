"""Plot merged-model accuracy and merge-weight alignment against the shard ratio.

Reads the aggregated CSV written by aggregate_results.py and produces two
figures, one line per importance scheme:

  merge_accuracy_vs_ratio.png   y = merged test accuracy
  merge_alignment_vs_ratio.png  y = mean coordinatewise merge weight assigned
                                    to model A

The x-axis in both is model A's share of the training data,
shard_size_a / (shard_size_a + shard_size_b) -- so 0.25 means the 15k/45k
split, 0.5 the balanced 30k/30k one, and so on.

Usage (from the repo root):
    python merging_experiments/plot_merge_results.py
    python merging_experiments/plot_merge_results.py --suite equal_exposure_trend

Note on suites: the ratio axis only varies in the `unbalanced_shards` suite.
`equal_exposure_trend` holds both shards at 30000 and varies batch size
instead, so plotting it against the ratio collapses every row onto a single
x value. The script warns and still writes the figures (the vertical spread at
that one x is readable as a per-scheme comparison), but a batch-size x-axis is
the right view for that suite.

On the alignment plot, `probe_fisher` is drawn thicker: it is the independent
empirical-Fisher probe, i.e. the quantity the accumulator-based schemes are
trying to approximate, so it reads as the reference the others should track.
The dotted line at 0.5 marks equal weighting (what `uniform` produces by
construction).

Schemes whose values coincide will overplot each other -- at balanced settings
squisher_raw/nscaled/corrected often agree to within a few thousandths, so a
line appearing "missing" usually means it is hidden under another.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a cluster login/compute node
import matplotlib.pyplot as plt

DEFAULT_RESULTS_DIR = Path("merging_experiments/results")

# Fixed order/colour per scheme so both figures use the same mapping and can be
# read side by side. Mean-convention schemes get solid lines, their _sum
# counterparts dashed in the same colour, so the convention is visible at a
# glance without doubling the number of colours.
BASE_SCHEMES = [
    "probe_fisher",
    "probe_theory_raw",
    "squisher_raw",
    "squisher_nscaled",
    "squisher_corrected",
]
SCHEME_COLORS = dict(zip(BASE_SCHEMES, plt.cm.tab10.colors))
UNIFORM_COLOR = "0.4"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("unbalanced_shards", "equal_exposure_trend"),
        default="unbalanced_shards",
        help="Which aggregated suite to plot (default: unbalanced_shards).",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Aggregated CSV path. Defaults to "
             "review_outputs/merging_experiments/<suite>.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Where to write the PNGs (default: merging_experiments/results).",
    )
    return parser.parse_args()


def load_rows(path: Path):
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run aggregate_results.py for this suite first."
        )
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no data rows.")
    return rows


def discover_schemes(row):
    """Return the scheme names actually present, in BASE_SCHEMES order with
    each _sum variant after its base, plus uniform first if present. Read from
    the CSV header rather than hardcoded so a newly added scheme shows up
    without touching this script."""
    present = []
    if "uniform_share_a_mean" in row:
        present.append("uniform")
    for base in BASE_SCHEMES:
        for name in (base, f"{base}_sum"):
            if f"{name}_share_a_mean" in row:
                present.append(name)
    # Anything in the CSV we didn't anticipate, so it still gets plotted.
    for key in row:
        if key.endswith("_share_a_mean"):
            name = key[: -len("_share_a_mean")]
            if name not in present:
                present.append(name)
    return present


def scheme_style(scheme):
    """Colour by base scheme, dashed for the _sum convention."""
    if scheme == "uniform":
        return {"color": UNIFORM_COLOR, "linestyle": "-.", "linewidth": 1.2}
    base = scheme[: -len("_sum")] if scheme.endswith("_sum") else scheme
    style = {
        "color": SCHEME_COLORS.get(base, "black"),
        "linestyle": "--" if scheme.endswith("_sum") else "-",
        "linewidth": 1.4,
    }
    if base == "probe_fisher":
        # The reference the accumulator schemes are approximating.
        style["linewidth"] = 2.6
    return style


def ratio_of(row):
    shard_a = float(row["shard_size_a"])
    shard_b = float(row["shard_size_b"])
    return shard_a / (shard_a + shard_b)


def make_plot(rows, schemes, metric, ylabel, title, out_path, reference_line=None):
    """metric is 'accuracy' or 'share_a'; both are stored as
    <scheme>_<metric>_mean / _std in the aggregated CSV."""
    ratios = [ratio_of(row) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: ratios[i])
    xs = [ratios[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for scheme in schemes:
        mean_key = f"{scheme}_{metric}_mean"
        std_key = f"{scheme}_{metric}_std"
        if mean_key not in rows[0]:
            continue
        means, stds = [], []
        for i in order:
            raw_mean = rows[i][mean_key]
            raw_std = rows[i].get(std_key, "")
            means.append(float(raw_mean) if raw_mean not in ("", None) else float("nan"))
            stds.append(float(raw_std) if raw_std not in ("", None) else 0.0)
        ax.errorbar(
            xs, means, yerr=stds, marker="o", markersize=4, capsize=3,
            label=scheme, **scheme_style(scheme),
        )

    if reference_line is not None:
        ax.axhline(
            reference_line, color="black", linestyle=":", linewidth=1,
            label=f"equal weighting ({reference_line:g})",
        )

    ax.set_xlabel("model A's share of the training data\n(shard_size_a / total)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(set(xs)) == 1:
        # Single x value (e.g. the equal_exposure suite): widen so the markers
        # aren't stacked on the axis edge.
        ax.set_xlim(xs[0] - 0.1, xs[0] + 0.1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    args = parse_args()
    csv_path = Path(
        args.csv or f"review_outputs/merging_experiments/{args.suite}.csv"
    )
    rows = load_rows(csv_path)
    schemes = discover_schemes(rows[0])
    output_dir = Path(args.output_dir)

    distinct_ratios = {ratio_of(row) for row in rows}
    if len(distinct_ratios) == 1:
        print(
            f"[warn] every row in {csv_path} has the same shard ratio "
            f"({distinct_ratios.pop():.3f}), so the ratio axis does not vary. "
            "The unbalanced_shards suite is the one that sweeps this; "
            "equal_exposure_trend sweeps batch size instead."
        )

    print(f"{len(rows)} settings, {len(schemes)} schemes: {', '.join(schemes)}")

    make_plot(
        rows, schemes, "accuracy",
        ylabel="merged model test accuracy",
        title=f"Merged accuracy vs data split ratio ({args.suite})",
        out_path=output_dir / "merge_accuracy_vs_ratio.png",
    )
    make_plot(
        rows, schemes, "share_a",
        ylabel="mean merge weight share assigned to model A",
        title=f"Merge-weight alignment vs data split ratio ({args.suite})",
        out_path=output_dir / "merge_alignment_vs_ratio.png",
        reference_line=0.5,
    )


if __name__ == "__main__":
    main()