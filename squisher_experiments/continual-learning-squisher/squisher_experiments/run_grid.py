"""Batch-size sensitivity grid for accumulator-recycled EWC ("Squisher") on Split MNIST.

Protocol (spec: NGD/WS3_SQUISHER_NOTES.md, Stage B):
1. "none":     no-regularization reference at every batch size.
2. "tune":     tune the EWC regularization strength once per importance source at
               batch size 128, over a wide shared log-grid.
3. "transfer": re-use each source's selected strength, unchanged, at batch sizes
               32 and 512.

Prediction: the raw and N-scaled accumulator sources degrade away from the tuning
batch size (their scale carries the 1/m factor of Theorem 1), while the
theorem-corrected source and the empirical Fisher stay batch-size stable.

Fidelity diagnostics (Stage A) are printed by the patched estimate_fisher_from_accumulator
as SQUISHER-FIDELITY lines and land in the same logs.

Resumable: a run whose log already contains the final-accuracy line is skipped.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / "logs_equal_examples_v2"

SOURCES = [
    "empirical",
    "squisher_raw",
    "squisher_biascorrected",
    "squisher_nscaled",
    "squisher_corrected",
]
SEEDS = [1, 2, 3]
TUNE_BATCH = 128
TRANSFER_BATCHES = [32, 512]
LAMBDA_GRID = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
EXAMPLES_PER_CONTEXT = 256_000
LR = 0.001

ACC_PATTERN = re.compile(r"average accuracy over all \d+ contexts: ([0-9.]+)")


def run_key(source: str | None, batch: int, lam: float | None, seed: int) -> str:
    if source is None:
        return f"none_b{batch}_s{seed}"
    return f"{source}_b{batch}_lam{lam:.0e}_s{seed}"


def build_command(source: str | None, batch: int, lam: float | None, seed: int) -> list[str]:
    iters = max(1, round(EXAMPLES_PER_CONTEXT / batch))
    command = [
        sys.executable, "main.py",
        "--experiment", "splitMNIST",
        "--scenario", "task",
        "--iters", str(iters),
        "--batch", str(batch),
        "--lr", str(LR),
        "--optimizer", "adam_reset",
        "--seed", str(seed),
        "--no-save",
    ]
    if source is not None:
        command += [
            "--ewc",
            "--reg-strength", str(lam),
            "--fisher-labels", "true",
            "--fisher-source", source,
            "--squisher-probe-n", "512",
        ]
    return command


def read_accuracy(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    match = ACC_PATTERN.search(log_path.read_text())
    return float(match.group(1)) if match else None


def run_one(source: str | None, batch: int, lam: float | None, seed: int) -> float | None:
    key = run_key(source, batch, lam, seed)
    log_path = LOG_DIR / f"{key}.log"
    accuracy = read_accuracy(log_path)
    if accuracy is not None:
        print(f"[skip] {key}: {accuracy:.4f}")
        return accuracy
    command = build_command(source, batch, lam, seed)
    with open(log_path, "w") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    accuracy = read_accuracy(log_path)
    status = f"{accuracy:.4f}" if accuracy is not None else f"FAILED (exit {result.returncode})"
    print(f"[done] {key}: {status}")
    return accuracy


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)

    print("== phase: none (no-regularization reference) ==")
    for batch in [TUNE_BATCH, *TRANSFER_BATCHES]:
        for seed in SEEDS:
            run_one(None, batch, None, seed)

    print(f"== phase: tune (batch {TUNE_BATCH}) ==")
    tuning: dict[str, dict[float, list[float]]] = {source: {} for source in SOURCES}
    for source in SOURCES:
        for lam in LAMBDA_GRID:
            accs = [run_one(source, TUNE_BATCH, lam, seed) for seed in SEEDS]
            accs = [acc for acc in accs if acc is not None]
            if accs:
                tuning[source][lam] = accs

    best_lambdas = {}
    for source in SOURCES:
        if not tuning[source]:
            print(f"[warn] no completed tuning runs for {source}; skipping transfer")
            continue
        best_lam = max(tuning[source], key=lambda lam: sum(tuning[source][lam]) / len(tuning[source][lam]))
        mean_acc = sum(tuning[source][best_lam]) / len(tuning[source][best_lam])
        best_lambdas[source] = best_lam
        print(f"[select] {source}: lambda={best_lam:.0e} (mean acc {mean_acc:.4f})")
    with open(LOG_DIR / "best_lambdas.json", "w") as handle:
        json.dump(best_lambdas, handle, indent=2)

    print("== phase: transfer (fixed lambda across batch sizes) ==")
    for source, lam in best_lambdas.items():
        for batch in TRANSFER_BATCHES:
            for seed in SEEDS:
                run_one(source, batch, lam, seed)

    print("all phases complete; parse with squisher_experiments/parse_results.py")


if __name__ == "__main__":
    main()
