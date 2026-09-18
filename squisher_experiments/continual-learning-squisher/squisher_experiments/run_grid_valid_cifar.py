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

Two ways to run this:

1. Sequentially, on one machine (e.g. for the one-cell smoke test, or a full local
   run): `python run_grid_valid.py`. Resumable -- a run whose log already contains
   the accuracy line(s) it needs is skipped.

2. As a cluster array job, two stages, one combo per array-task-slot:
   - Stage "tune":     `python run_grid_valid.py --stage tune --combo-index N`
                        (N in 0..167, the "none" + "tune" combos -- see
                        build_none_tune_combos() for the exact count, which
                        depends on TRANSFER_BATCHES/TUNE_SEEDS/OTHER_SEEDS/
                        LAMBDA_GRID above)
   - (in between)       `python select_lambdas.py` -- reads the completed tune-phase
                        logs and writes best_lambdas.json (needs ALL of stage
                        "tune" finished first)
   - Stage "transfer": `python run_grid_valid.py --stage transfer --combo-index N`
                        (N in 0..149, needs best_lambdas.json to exist)
   See run_grid_valid_tune.sbatch / select_lambdas.sbatch / run_grid_valid_transfer.sbatch.

Uses a dedicated log directory (logs_valid_tuned/, not the original grid's
logs_equal_examples_v2/) on purpose: old tune-phase logs never used --valid-size
and only ever contain ONE accuracy line, so they must not be mistaken for
completed validation-based tuning runs.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / "logs_valid_tuned_cifar"
BEST_LAMBDAS_PATH = LOG_DIR / "best_lambdas.json"

SOURCES = [
    "empirical",
    "squisher_raw",
    "squisher_biascorrected",
    "squisher_nscaled",
    "squisher_corrected",
]
TUNE_BATCH = 256
EXPERIMENT = "CIFAR10"
CONTEXTS = 5            # CIFAR10 default; 50,000 train / 5 = 10,000 per context
TRANSFER_BATCHES = [64, 512, 1024]
MIN_ITERS = 2000  # floor on iterations/context (2026-09-14) -- with a fixed EXAMPLES_PER_CONTEXT budget,
                   # large batches get too few iterations for the exp_avg_sq accumulator to burn in
                   # against beta2=0.999's ~1,000-step horizon (e.g. batch=4096 would only get ~63
                   # iterations otherwise). Matches batch=128's own iteration count, which is already
                   # known to burn in adequately. Only raises iters (and therefore compute cost) for
                   # batches where EXAMPLES_PER_CONTEXT/batch would otherwise fall short of this floor.
TUNE_SEEDS = [1, 2, 3]            # unchanged -- keeps tune-phase (batch=128, all lambdas) cost stable
OTHER_SEEDS = [1, 2, 3, 4, 5]     # increased from 3 -- the "none" baseline at non-128 batches, and
                                   # every "transfer" run, since these feed the final
                                   # batch-size-transfer comparison and benefit most from lower variance
LAMBDA_GRID = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]  # extended one order of magnitude each
                                   # way (2026-09-14): the original [1e1..1e7] grid had two sources
                                   # (empirical, nscaled) select a boundary value, meaning the true
                                   # optimum likely lay outside the tested range
EXAMPLES_PER_CONTEXT = 256_000
LR = 0.0001
VALID_SIZE = 0.25  # fraction of each class's training data held out -- tune phase only (was 0.1)

ACC_PATTERN = re.compile(r"average accuracy over all \d+ contexts: ([0-9.]+)")


def run_key(source: str | None, batch: int, lam: float | None, seed: int, tuning: bool) -> str:
    tag = "_vs{:g}".format(VALID_SIZE) if tuning else ""
    if source is None:
        return f"none_b{batch}_s{seed}"
    return f"{source}_b{batch}_lam{lam:.0e}_s{seed}{tag}"


def build_command(source: str | None, batch: int, lam: float | None, seed: int, tuning: bool) -> list[str]:
    iters = max(round(EXAMPLES_PER_CONTEXT / batch), MIN_ITERS)
    command = [
        sys.executable, "main.py",
        "--experiment", EXPERIMENT,
        "--scenario", "task",
        "--contexts", str(CONTEXTS),
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


# --------------------------------------------------------------------------------- #
# Combo lists -- used both by the sequential main() below and by the cluster-array
# CLI mode (--stage / --combo-index), so the two never drift out of sync.
# --------------------------------------------------------------------------------- #

def build_none_tune_combos() -> list[dict]:
    """The 'none' runs (batch=128 at TUNE_SEEDS + each TRANSFER_BATCHES at
    OTHER_SEEDS) + 'tune' runs (5 sources x LAMBDA_GRID x TUNE_SEEDS) -- all
    independent of each other, none of them need best_lambdas.json, so these
    are safe to fan out across an array job in any order."""
    combos = []
    for seed in TUNE_SEEDS:
        combos.append(dict(source=None, batch=TUNE_BATCH, lam=None, seed=seed, tuning=False))
    for batch in TRANSFER_BATCHES:
        for seed in OTHER_SEEDS:
            combos.append(dict(source=None, batch=batch, lam=None, seed=seed, tuning=False))
    for source in SOURCES:
        for lam in LAMBDA_GRID:
            for seed in TUNE_SEEDS:
                combos.append(dict(source=source, batch=TUNE_BATCH, lam=lam, seed=seed, tuning=True))
    return combos


def build_transfer_combos(best_lambdas: dict) -> list[dict]:
    """The 'transfer' runs: each selected source/lambda re-run at every
    TRANSFER_BATCHES value, OTHER_SEEDS seeds each. Requires
    best_lambdas.json to already exist (i.e. the 'tune' stage above must be
    fully complete first)."""
    combos = []
    for source, lam in best_lambdas.items():
        for batch in TRANSFER_BATCHES:
            for seed in OTHER_SEEDS:
                combos.append(dict(source=source, batch=batch, lam=lam, seed=seed, tuning=False))
    return combos


def select_lambdas(verbose: bool = True) -> dict:
    """Reads the completed 'tune'-phase logs and picks, per source, the lambda
    with the highest mean VALIDATION accuracy across seeds. Writes and returns
    best_lambdas.json. Safe to re-run; only reads logs, runs nothing."""
    tuning_scores: dict[str, dict[float, list[float]]] = {source: {} for source in SOURCES}
    for source in SOURCES:
        for lam in LAMBDA_GRID:
            valid_accs = []
            for seed in TUNE_SEEDS:
                _test_acc, valid_acc = read_accuracies(LOG_DIR / f"{run_key(source, TUNE_BATCH, lam, seed, True)}.log")
                if valid_acc is not None:
                    valid_accs.append(valid_acc)
            if valid_accs:
                tuning_scores[source][lam] = valid_accs

    best_lambdas = {}
    for source in SOURCES:
        if not tuning_scores[source]:
            if verbose:
                print(f"[warn] no completed tuning runs for {source}; skipping transfer")
            continue
        best_lam = max(
            tuning_scores[source],
            key=lambda lam: sum(tuning_scores[source][lam]) / len(tuning_scores[source][lam]),
        )
        mean_valid_acc = sum(tuning_scores[source][best_lam]) / len(tuning_scores[source][best_lam])
        best_lambdas[source] = best_lam
        if verbose:
            print(f"[select] {source}: lambda={best_lam:.0e} (mean VALIDATION acc {mean_valid_acc:.4f})")
    LOG_DIR.mkdir(exist_ok=True)
    with open(BEST_LAMBDAS_PATH, "w") as handle:
        json.dump(best_lambdas, handle, indent=2)
    return best_lambdas


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)

    print("== phase: none (no-regularization reference; full training data, no validation split) ==")
    print(f"== phase: tune (batch {TUNE_BATCH}, selecting lambda by VALIDATION accuracy) ==")
    for combo in build_none_tune_combos():
        run_one(combo["source"], combo["batch"], combo["lam"], combo["seed"], tuning=combo["tuning"])

    best_lambdas = select_lambdas()

    print("== phase: transfer (fixed lambda across batch sizes; scored on held-out test set) ==")
    for combo in build_transfer_combos(best_lambdas):
        run_one(combo["source"], combo["batch"], combo["lam"], combo["seed"], tuning=combo["tuning"])

    print("all phases complete; parse with squisher_experiments/parse_results_valid.py")


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["tune", "transfer"],
                       help="run exactly one combo from the given stage (for cluster array jobs)")
    parser.add_argument("--combo-index", type=int,
                       help="which combo to run (0-based) within --stage's list")
    args = parser.parse_args()

    if args.stage is None:
        main()
        return

    if args.combo_index is None:
        parser.error("--combo-index is required when --stage is given")

    if args.stage == "tune":
        combos = build_none_tune_combos()
    else:
        if not BEST_LAMBDAS_PATH.exists():
            parser.error(f"{BEST_LAMBDAS_PATH} not found -- run the 'tune' stage and select_lambdas.py first")
        with open(BEST_LAMBDAS_PATH) as handle:
            best_lambdas = json.load(handle)
        combos = build_transfer_combos(best_lambdas)

    if not (0 <= args.combo_index < len(combos)):
        parser.error(f"--combo-index must be in [0, {len(combos) - 1}] for --stage {args.stage} "
                     f"({len(combos)} combos)")

    combo = combos[args.combo_index]
    LOG_DIR.mkdir(exist_ok=True)
    run_one(combo["source"], combo["batch"], combo["lam"], combo["seed"], tuning=combo["tuning"])


if __name__ == "__main__":
    _cli_main()