"""Reproduction of the split CIFAR-100 EWC experiment of Li et al. (ICML 2025),
"Fishers for Free?" (arXiv:2507.18807), Section 3.6 / Table 8, task-incremental row:

    Fisher 75.96   Squisher 75.30   no-regularization baseline 61.82   (5 trials)

Protocol, as pinned by the paper plus the upstream code it inherits from
(GMvandeVen/continual-learning, all_results.sh):

  - split CIFAR-100, 10 contexts x 10 classes, task-incremental scenario
  - 5-layer conv encoder C3-5x16-bn (393,088 params -- the number the paper quotes),
    PRETRAINED on CIFAR-10 for 100 epochs with augmentation and FROZEN during the
    continual-learning runs (--pre-convE --freeze-convE), one encoder per seed
    (--seed-to-ltag -> loads store/models/C3-5x16-bn-e100-s<seed>; see
    pretrain_cifar100.sbatch). Only the FC head 1024-2000-2000-100 is trained.
  - Adam, lr 1e-4, batch 256, 5,000 iterations per context (upstream defaults)
  - EWC with the EMPIRICAL Fisher (true labels, full pass over the context's
    training data at batch 1 -- consistent with the paper's Table 18 runtime of
    ~138 s per context) vs. the Squisher, which the paper defines as N * v_T with
    N the per-context training-set size -> the repo's `squisher_nscaled` source
    (raw accumulator, no bias correction, times n).
  - lambda: the paper grid-searches around lambda_Squisher = N * lambda_Fisher.
    Here every regularized config is tuned over LAMBDA_GRID on a 25% validation
    hold-out (3 seeds), then the selected lambda is re-run for FINAL_SEEDS on the
    FULL per-context training data (n = 5,000) with no hold-out, so the reported
    numbers are produced under the paper's exact data conditions and the test
    set never participates in selection.
  - Adam state across context boundaries is not stated in the paper. Upstream's
    default `adam` keeps it (the harvested accumulator then carries history from
    earlier contexts); the repo's batch-transfer grids use `adam_reset`. Both are
    run for the Squisher (and for the baseline, which is cheap) so the one that
    matches the paper can be reported.

Stages (mirrors run_grid_valid_cifar.py so the sbatch pattern is identical):

  pretrain : pretrain_cifar100.sbatch (array over seeds, main_pretrain.py)
  tune     : python run_cifar100_paper.py --stage tune  --combo-index N
  select   : python select_lambdas_cifar100.py           (reads tune logs)
  final    : python run_cifar100_paper.py --stage final --combo-index N
  parse    : python parse_results_cifar100.py

`python run_cifar100_paper.py` with no arguments runs everything sequentially
(resumable: a log that already holds the accuracy line(s) it needs is skipped).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / "logs_cifar100_paper"
BEST_LAMBDAS_PATH = LOG_DIR / "best_lambdas.json"

EXPERIMENT = "CIFAR100"
CONTEXTS = 10
BATCH = 256
ITERS = 5000
LR = 0.0001
VALID_SIZE = 0.25       # tune stage only; final stage uses the full training data
SQUISHER_PROBE_N = 512  # fidelity diagnostic only; does not affect training

# (label, fisher_source, optimizer). fisher_source=None -> no regularization.
CONFIGS = [
    ("none_adam",          None,               "adam"),
    ("none_adamreset",     None,               "adam_reset"),
    ("fisher_adam",        "empirical",        "adam"),
    ("squisher_adam",      "squisher_nscaled", "adam"),
    ("squisher_adamreset", "squisher_nscaled", "adam_reset"),
]
LAMBDA_GRID = [                                     # upstream default for CIFAR-100 task-IL is 1e3
    1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10,
]
# History (2026-10-19): the first pass used decades 1e1..1e7. fisher_adam selected the boundary value
# 1e7 with validation accuracy still rising (62.8 -> 73.5 -> 75.0 over 1e5..1e7), so the grid was
# extended upward to 1e10. squisher_adam went 74.7 at 1e3 -> 10.0 (chance; training diverged) at 1e4,
# so half-decade points were added everywhere to give the Fisher and Squisher the same selection
# resolution. Completed combos are skipped on re-run; only the new (config, lambda, seed) cells cost.
TUNE_SEEDS = [1, 2, 3]
FINAL_SEEDS = [1, 2, 3, 4, 5]                       # "5 trials" in the paper

ACC_PATTERN = re.compile(r"average accuracy over all \d+ contexts: ([0-9.]+)")


def config_by_label(label: str) -> tuple[str, str | None, str]:
    for cfg in CONFIGS:
        if cfg[0] == label:
            return cfg
    raise KeyError(label)


def run_key(label: str, lam: float | None, seed: int, tuning: bool) -> str:
    lam_tag = "" if lam is None else f"_lam{lam:.0e}"
    tag = f"_vs{VALID_SIZE:g}" if tuning else "_full"
    return f"{label}{lam_tag}_s{seed}{tag}"


def build_command(label: str, lam: float | None, seed: int, tuning: bool) -> list[str]:
    _, source, optimizer = config_by_label(label)
    command = [
        sys.executable, "main.py",
        "--experiment", EXPERIMENT,
        "--scenario", "task",
        "--contexts", str(CONTEXTS),
        "--iters", str(ITERS),
        "--batch", str(BATCH),
        "--lr", str(LR),
        "--optimizer", optimizer,
        "--seed", str(seed),
        "--no-save",
        # paper / upstream recipe: CIFAR-10-pretrained, frozen conv encoder, one per seed
        "--pre-convE", "--freeze-convE", "--seed-to-ltag",
    ]
    if tuning:
        command += ["--valid-size", str(VALID_SIZE)]
    if source is not None:
        command += [
            "--ewc",
            "--reg-strength", str(lam),
            "--fisher-labels", "true",          # empirical Fisher (paper: "we focus on the empirical FIM")
            "--fisher-source", source,
            "--squisher-probe-n", str(SQUISHER_PROBE_N),
        ]
    return command


def read_accuracies(log_path: Path) -> tuple[float | None, float | None]:
    """(test_accuracy, validation_accuracy). main.py prints the test block first, so the
    1st match is always test; the 2nd exists only when --valid-size was passed."""
    if not log_path.exists():
        return None, None
    matches = ACC_PATTERN.findall(log_path.read_text())
    test_acc = float(matches[0]) if len(matches) >= 1 else None
    valid_acc = float(matches[1]) if len(matches) >= 2 else None
    return test_acc, valid_acc


def run_one(label: str, lam: float | None, seed: int, tuning: bool) -> tuple[float | None, float | None]:
    key = run_key(label, lam, seed, tuning)
    log_path = LOG_DIR / f"{key}.log"
    test_acc, valid_acc = read_accuracies(log_path)
    already_done = (test_acc is not None) and (valid_acc is not None or not tuning)
    if already_done:
        suffix = f" valid={valid_acc:.4f}" if tuning else ""
        print(f"[skip] {key}: test={test_acc:.4f}{suffix}")
        return test_acc, valid_acc
    command = build_command(label, lam, seed, tuning)
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


# ----------------------------------------------------------------------------- #
# Combo lists (shared by the sequential main() and the --stage/--combo-index CLI)
# ----------------------------------------------------------------------------- #

def build_tune_combos() -> list[dict]:
    """Regularized configs x LAMBDA_GRID x TUNE_SEEDS, with the validation hold-out.
    The 'none' configs have nothing to tune and only appear in the final stage."""
    combos = []
    for label, source, _ in CONFIGS:
        if source is None:
            continue
        for lam in LAMBDA_GRID:
            for seed in TUNE_SEEDS:
                combos.append(dict(label=label, lam=lam, seed=seed, tuning=True))
    return combos


def build_final_combos(best_lambdas: dict) -> list[dict]:
    """Every config x FINAL_SEEDS on the full training data: 'none' configs with no
    lambda, regularized configs at their selected lambda."""
    combos = []
    for label, source, _ in CONFIGS:
        if source is None:
            lam = None
        elif label in best_lambdas:
            lam = best_lambdas[label]
        else:
            print(f"[warn] no selected lambda for {label}; skipping its final runs")
            continue
        for seed in FINAL_SEEDS:
            combos.append(dict(label=label, lam=lam, seed=seed, tuning=False))
    return combos


def select_lambdas(verbose: bool = True) -> dict:
    """Per regularized config, pick the lambda with the best mean VALIDATION accuracy
    over TUNE_SEEDS. Writes best_lambdas.json. Only reads logs."""
    best_lambdas = {}
    for label, source, _ in CONFIGS:
        if source is None:
            continue
        scores: dict[float, list[float]] = {}
        for lam in LAMBDA_GRID:
            valid_accs = []
            for seed in TUNE_SEEDS:
                _test, valid = read_accuracies(LOG_DIR / f"{run_key(label, lam, seed, True)}.log")
                if valid is not None:
                    valid_accs.append(valid)
            if valid_accs:
                scores[lam] = valid_accs
        if not scores:
            if verbose:
                print(f"[warn] no completed tuning runs for {label}; skipping")
            continue
        best_lam = max(scores, key=lambda lam: sum(scores[lam]) / len(scores[lam]))
        best_lambdas[label] = best_lam
        if verbose:
            mean_valid = sum(scores[best_lam]) / len(scores[best_lam])
            at_edge = best_lam in (LAMBDA_GRID[0], LAMBDA_GRID[-1])
            print(f"[select] {label}: lambda={best_lam:.0e} (mean VALIDATION acc {mean_valid:.4f})"
                  + ("  <-- grid boundary, consider extending LAMBDA_GRID" if at_edge else ""))
    LOG_DIR.mkdir(exist_ok=True)
    with open(BEST_LAMBDAS_PATH, "w") as handle:
        json.dump(best_lambdas, handle, indent=2)
    return best_lambdas


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    print(f"== stage: tune (batch {BATCH}, {VALID_SIZE:.0%} validation hold-out, selecting lambda by validation acc) ==")
    for combo in build_tune_combos():
        run_one(**combo)
    best_lambdas = select_lambdas()
    print("== stage: final (selected lambda, full training data, 5 seeds; scored on the test set) ==")
    for combo in build_final_combos(best_lambdas):
        run_one(**combo)
    print("all stages complete; parse with squisher_experiments/parse_results_cifar100.py")


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["tune", "final"],
                        help="run exactly one combo from the given stage (for cluster array jobs)")
    parser.add_argument("--combo-index", type=int, help="which combo to run (0-based) within --stage's list")
    parser.add_argument("--count", action="store_true", help="print the number of combos in --stage and exit")
    args = parser.parse_args()

    if args.stage is None:
        main()
        return

    if args.stage == "tune":
        combos = build_tune_combos()
    else:
        if not BEST_LAMBDAS_PATH.exists():
            parser.error(f"{BEST_LAMBDAS_PATH} not found -- run the 'tune' stage and select_lambdas_cifar100.py first")
        with open(BEST_LAMBDAS_PATH) as handle:
            combos = build_final_combos(json.load(handle))

    if args.count:
        print(len(combos))
        return
    if args.combo_index is None:
        parser.error("--combo-index is required when --stage is given")
    if not (0 <= args.combo_index < len(combos)):
        parser.error(f"--combo-index must be in [0, {len(combos) - 1}] for --stage {args.stage} ({len(combos)} combos)")

    LOG_DIR.mkdir(exist_ok=True)
    run_one(**combos[args.combo_index])


if __name__ == "__main__":
    _cli_main()
