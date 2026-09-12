"""Collect the batch-size grid results (accuracies + fidelity diagnostics) into CSVs.

Reads squisher_experiments/logs/*.log written by run_grid.py and writes:
- results.csv:  source, batch, lambda, seed, average_accuracy
- fidelity.csv: one row per SQUISHER-FIDELITY line (per run, per context)
and prints a mean +- std pivot of accuracy per (source x batch).
"""

import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

LOG_DIR = Path(__file__).resolve().parent / "logs_equal_examples_v2"

ACC_PATTERN = re.compile(r"average accuracy over all \d+ contexts: ([0-9.]+)")
KEY_PATTERN = re.compile(r"^(?P<source>none|empirical|squisher_\w+?)_b(?P<batch>\d+)(?:_lam(?P<lam>[0-9e+.-]+))?_s(?P<seed>\d+)$")
FIDELITY_PATTERN = re.compile(
    r"SQUISHER-FIDELITY context=(\d+) source=(\S+) n=(\d+) m=(\d+) cosine=(\S+) "
    r"norm_est=(\S+) norm_probe=(\S+) norm_ratio=(\S+) raw_v_norm=(\S+) exp_avg_norm=(\S+) "
    r"signal_sq_norm_ratio=(\S+) signal_sq_cosine=(\S+)"
)


def main() -> None:
    results = []
    fidelity = []
    for log_path in sorted(LOG_DIR.glob("*.log")):
        key_match = KEY_PATTERN.match(log_path.stem)
        if key_match is None:
            continue
        text = log_path.read_text()
        acc_match = ACC_PATTERN.search(text)
        info = key_match.groupdict()
        if acc_match:
            results.append({
                "source": info["source"],
                "batch": int(info["batch"]),
                "lambda": info["lam"] or "",
                "seed": int(info["seed"]),
                "average_accuracy": float(acc_match.group(1)),
            })
        for fid in FIDELITY_PATTERN.finditer(text):
            fidelity.append({
                "run": log_path.stem,
                "context": int(fid.group(1)),
                "source": fid.group(2),
                "n": int(fid.group(3)),
                "m": int(fid.group(4)),
                "cosine": float(fid.group(5)),
                "norm_est": float(fid.group(6)),
                "norm_probe": float(fid.group(7)),
                "norm_ratio": float(fid.group(8)),
                "raw_v_norm": float(fid.group(9)),
                "exp_avg_norm": float(fid.group(10)),
                "signal_sq_norm_ratio": float(fid.group(11)),
                "signal_sq_cosine": float(fid.group(12)),
            })

    for name, rows in [("results.csv", results), ("fidelity.csv", fidelity)]:
        if rows:
            with open(LOG_DIR / name, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"wrote {LOG_DIR / name} ({len(rows)} rows)")

    grouped = defaultdict(list)
    for row in results:
        grouped[(row["source"], row["batch"], row["lambda"])].append(row["average_accuracy"])
    print(f"\n{'source':24s} {'batch':>6s} {'lambda':>8s} {'acc mean':>9s} {'std':>7s} {'n':>2s}")
    for (source, batch, lam), accs in sorted(grouped.items()):
        spread = stdev(accs) if len(accs) > 1 else 0.0
        print(f"{source:24s} {batch:6d} {lam:>8s} {mean(accs):9.4f} {spread:7.4f} {len(accs):2d}")


if __name__ == "__main__":
    main()
