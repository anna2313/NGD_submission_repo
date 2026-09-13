"""Batch-size sensitivity grid for accumulator-recycled EWC ("Squisher") on Split MNIST.

Same protocol as run_grid.py (same sources, seeds, batch sizes, lambda grid, and
per-context example budget). The only thing that changes is how lambda gets picked:

  - OLD (run_grid.py): the "tune" phase picked lambda using the exact same metric
    (final test-set accuracy) that later gets reported as the result. Oracle-style /
    cherry-picking, as flagged in squisher_experiments/README.md.
  - NEW (this script): "tune"-phase runs pass --valid-size VALID_SIZE, so main.py
    holds out part of each class's TRAINING data (the same physical examples every
    time, regardless of --seed -- see data/load.py:split_off_validation) and reports
    two numbers per run: test accuracy (1st "average accuracy..." line in the log /
    acc-*.txt) and validation accuracy (2nd line / acc-valid-*.txt). Lambda is now
    selected by validation accuracy. Test accuracy is never touched by selection.

"none" (no-EWC baseline) and "transfer" (frozen lambda at batch 32/512) phases do
NOT hold out a validation set -- no further tuning happens there, so there's no
reason to shrink the training set -- and are scored, as before, on the untouched
test set. That test-set number is now a genuinely unbiased result, since it played
no role in choosing lambda.

Resumable: a run whose log already contains the accuracy line(s) it needs is
skipped. Uses a NEW log directory (logs_valid_tuned/, not the original grid's
logs_equal_examples_v2/) on purpose: old tune-phase logs never used --valid-size
and only ever contain ONE accuracy line, so they must not be mistaken for
completed validation-based tuning runs.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / "logs_valid_tuned"

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
VALID_SIZE = 0.1  # fraction of each class's training data held out -- tune phase only

ACC_PATTERN = re.compile(r"average accuracy over all \d+ contexts: ([0-9.]+)")


def run_key(source: str | None, batch: int, lam: float | None, seed: int, tuning: bool) -> str:
    tag = "_vs{:g}".format(VALID_SIZE) if tuning else ""
    if source is None:
        return f"none_b{batch}_s{seed}"
    return f"{source}_b{batch}_lam{lam:.0e}_s{seed}{tag}"


def build_command(source: str | None, batch: int, lam: float | None, seed: int, tuning: bool) -> list[str]:
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
    if tuning:
        # -only hold out a validation-set while lambda is actually being selected
        command += ["--valid-size", str(VALID_SIZE)]
    if source is not None:
        command += [
            "--ewc",
            "--reg-strength", str(lam),
            "--fisher-labels", "true",
            "--fisher-source", source,
            "--squisher-probe-n", "512",
        ]
    return command


def read_accuracies(log_path: Path) -> tuple[float | None, float | None]:
    """Returns (test_accuracy, validation_accuracy).

    main.py always prints the test-set block before the (optional) validation-set
    block, so the 1st "average accuracy..." match is always test and the 2nd
    (only present when --valid-size was passed) is always validation.
    """
    if not log_path.exists():
        return None, None
    matches = ACC_PATTERN.findall(log_path.read_text())
    test_acc = float(matches[0]) if len(matches) >= 1 else None
    valid_acc = float(matches[1]) if len(matches) >= 2 else None
    return test_acc, valid_acc


def run_one(source: str | None, batch: int, lam: float | None, seed: int,
           tuning: bool = False) -> tuple[float | None, float | None]:
    key = run_key(source, batch, lam, seed, tuning)
    log_path = LOG_DIR / f"{key}.log"
    test_acc, valid_acc = read_accuracies(log_path)
    # -a tune-phase run needs BOTH numbers to count as done; none/transfer runs
    #  never produce a validation number, so the test number alone is enough there.
    already_done = (test_acc is not None and valid_acc is not None) if tuning else (test_acc is not None)
    if already_done:
        suffix = f" valid={valid_acc:.4f}" if tuning else ""
        print(f"[skip] {key}: test={test_acc:.4f}{suffix}")
        return test_acc, valid_acc
    command = build_command(source, batch, lam, seed, tuning)
    with open(log_path, "w") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    test_acc, valid_acc = read_accuracies(log_path)
    if test_acc is None:
        print(f"[done] {key}: FAILED (exit {result.returncode})")
    else:
        suffix = f" valid={valid_acc:.4f}" if tuning else ""
        print(f"[done] {key}: test={test_acc:.4f}{suffix}")
    return test_acc, valid_acc


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)

    print("== phase: none (no-regularization reference; full training data, no validation split) ==")
    for batch in [TUNE_BATCH, *TRANSFER_BATCHES]:
        for seed in SEEDS:
            run_one(None, batch, None, seed, tuning=False)

    print(f"== phase: tune (batch {TUNE_BATCH}, selecting lambda by VALIDATION accuracy) ==")
    tuning_scores: dict[str, dict[float, list[float]]] = {source: {} for source in SOURCES}
    for source in SOURCES:
        for lam in LAMBDA_GRID:
            valid_accs = []
            for seed in SEEDS:
                _test_acc, valid_acc = run_one(source, TUNE_BATCH, lam, seed, tuning=True)
                if valid_acc is not None:
                    valid_accs.append(valid_acc)
            if valid_accs:
                tuning_scores[source][lam] = valid_accs

    best_lambdas = {}
    for source in SOURCES:
        if not tuning_scores[source]:
            print(f"[warn] no completed tuning runs for {source}; skipping transfer")
            continue
        best_lam = max(
            tuning_scores[source],
            key=lambda lam: sum(tuning_scores[source][lam]) / len(tuning_scores[source][lam]),
        )
        mean_valid_acc = sum(tuning_scores[source][best_lam]) / len(tuning_scores[source][best_lam])
        best_lambdas[source] = best_lam
        print(f"[select] {source}: lambda={best_lam:.0e} (mean VALIDATION acc {mean_valid_acc:.4f})")
    with open(LOG_DIR / "best_lambdas.json", "w") as handle:
        json.dump(best_lambdas, handle, indent=2)

    print("== phase: transfer (fixed lambda across batch sizes; scored on held-out test set) ==")
    for source, lam in best_lambdas.items():
        for batch in TRANSFER_BATCHES:
            for seed in SEEDS:
                run_one(source, batch, lam, seed, tuning=False)

    print("all phases complete; parse with squisher_experiments/parse_results_valid.py")


if __name__ == "__main__":
    main()
