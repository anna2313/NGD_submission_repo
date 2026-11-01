"""Collect the CIFAR-100 paper-reproduction results into CSVs and print the
Table-8-style summary.

Reads squisher_experiments/logs_cifar100_paper/*.log written by run_cifar100_paper.py
and writes, next to the logs:
- results.csv:  config, source, optimizer, lambda, seed, stage (tune|final),
                test_accuracy, validation_accuracy (blank for final rows)
- fidelity.csv: one row per SQUISHER-FIDELITY line (per run, per context)
and prints mean +- std TEST accuracy per config for the final stage, alongside the
paper's reported numbers (Fisher 75.96 / Squisher 75.30 / baseline 61.82), plus the
tune-stage validation curve per config for inspection.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from run_cifar100_paper import CONFIGS, LAMBDA_GRID, LOG_DIR, VALID_SIZE

ACC_PATTERN = re.compile(r"average accuracy over all \d+ contexts: ([0-9.]+)")
KEY_PATTERN = re.compile(
    r"^(?P<label>[a-z_]+?)(?:_lam(?P<lam>[0-9e+.-]+))?_s(?P<seed>\d+)(?P<stage>_vs[0-9.]+|_full)$"
)
FIDELITY_PATTERN = re.compile(
    r"SQUISHER-FIDELITY context=(\d+) source=(\S+) n=(\d+) m=(\d+) cosine=(\S+) "
    r"norm_est=(\S+) norm_probe=(\S+) norm_ratio=(\S+) raw_v_norm=(\S+) exp_avg_norm=(\S+) "
    r"signal_sq_norm_ratio=(\S+) signal_sq_cosine=(\S+)"
)
PAPER = {"fisher_adam": 75.96, "squisher_adam": 75.30, "squisher_adamreset": 75.30,
         "none_adam": 61.82, "none_adamreset": 61.82}
CONFIG_INFO = {label: (source or "none", optimizer) for label, source, optimizer in CONFIGS}


def fmt(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{100 * values[0]:.2f} (n=1)"
    return f"{100 * mean(values):.2f} +- {100 * stdev(values):.2f} (n={len(values)})"


def main() -> None:
    results, fidelity = [], []
    for log_path in sorted(LOG_DIR.glob("*.log")):
        key_match = KEY_PATTERN.match(log_path.stem)
        if key_match is None or key_match["label"] not in CONFIG_INFO:
            continue
        text = log_path.read_text()
        accs = ACC_PATTERN.findall(text)
        if not accs:
            continue
        label = key_match["label"]
        source, optimizer = CONFIG_INFO[label]
        stage = "tune" if key_match["stage"].startswith("_vs") else "final"
        lam = key_match["lam"] or ""
        seed = int(key_match["seed"])
        test_acc = float(accs[0])
        valid_acc = float(accs[1]) if (stage == "tune" and len(accs) >= 2) else None
        results.append(dict(config=label, source=source, optimizer=optimizer, **{"lambda": lam}, seed=seed,
                            stage=stage, test_accuracy=test_acc,
                            validation_accuracy="" if valid_acc is None else valid_acc))
        for m in FIDELITY_PATTERN.finditer(text):
            fidelity.append(dict(config=label, **{"lambda": lam}, seed=seed, stage=stage, context=int(m.group(1)),
                                 source=m.group(2), n=int(m.group(3)), m=int(m.group(4)), cosine=float(m.group(5)),
                                 norm_est=float(m.group(6)), norm_probe=float(m.group(7)), norm_ratio=float(m.group(8)),
                                 raw_v_norm=float(m.group(9)), exp_avg_norm=float(m.group(10)),
                                 signal_sq_norm_ratio=float(m.group(11)), signal_sq_cosine=float(m.group(12))))

    with open(LOG_DIR / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "source", "optimizer", "lambda", "seed", "stage",
                                                    "test_accuracy", "validation_accuracy"])
        writer.writeheader()
        writer.writerows(results)
    if fidelity:
        with open(LOG_DIR / "fidelity.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fidelity[0].keys()))
            writer.writeheader()
            writer.writerows(fidelity)

    print(f"\n== FINAL stage: test accuracy, full training data (paper Table 8, split CIFAR-100, task-IL) ==")
    final = defaultdict(list)
    final_lam = {}
    for r in results:
        if r["stage"] == "final":
            final[r["config"]].append(r["test_accuracy"])
            final_lam[r["config"]] = r["lambda"]
    for label, _, _ in CONFIGS:
        lam = final_lam.get(label, "")
        lam_str = f" lambda={lam}" if lam else ""
        print(f"  {label:20s}{lam_str:14s} {fmt(final[label]):28s} paper: {PAPER.get(label, float('nan')):.2f}")

    print(f"\n== TUNE stage: mean validation accuracy per lambda ({VALID_SIZE:.0%} hold-out) ==")
    tune = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["stage"] == "tune" and r["validation_accuracy"] != "":
            tune[r["config"]][float(r["lambda"])].append(r["validation_accuracy"])
    for label, source, _ in CONFIGS:
        if source is None:
            continue
        row = "  ".join(f"{lam:.0e}:{100 * mean(tune[label][lam]):.1f}" if tune[label][lam] else f"{lam:.0e}:--"
                        for lam in LAMBDA_GRID)
        print(f"  {label:20s} {row}")
    print(f"\nwrote {LOG_DIR / 'results.csv'} ({len(results)} runs)"
          + (f" and fidelity.csv ({len(fidelity)} rows)" if fidelity else ""))


if __name__ == "__main__":
    main()
