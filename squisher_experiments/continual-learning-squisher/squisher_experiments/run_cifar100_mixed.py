"""Mixed-batch-size EWC experiment on split CIFAR-100 (follow-up to the Li et al. 2025 reproduction).

Question: when every task (context) is trained with its own batch size, does recycling Adam's
exp_avg_sq accumulator as the EWC importance ("Squisher") still work? In online EWC the per-task
estimates are summed and one global lambda scales the sum. With equal batch sizes any scaling error is
a global constant that lambda absorbs; with mixed batch sizes the error is RELATIVE between tasks and
no lambda can absorb it. The theorem's per-task correction m_t(n-1)/(n-m_t) should remove it.

Three stages (plus selection steps in select_cifar100_mixed.py, summary in parse_results_cifar100_mixed.py):

  lrcal   batch {32,128,512,2048} x lr {3e-5,1e-4,3e-4,1e-3} x 3 seeds, no EWC, every context at the
          same batch/lr. Score = "just-learned" validation accuracy (accuracy on each context right
          after training it, before any forgetting), averaged over the 10 contexts.
          -> select_cifar100_mixed.py --stage lr  writes best_lr.json (one lr per batch size)
  lambda  each (batch, best lr) pair x 4 sources x 19-point lambda grid x 3 seeds, every context at the
          same batch/lr (homogeneous). Score = final average validation accuracy.
          -> select_cifar100_mixed.py --stage lambda  writes best_lambdas.json (one "middle-ground"
             lambda per source: best mean validation accuracy across the four batch sizes)
  test    5 random batch sequences (each context draws its batch size uniformly from the four; every
          sequence contains all four) x 3 seeds x 5 arms (none + 4 sources). Each context uses the lr
          chosen for its batch size; each source uses its middle-ground lambda. All arms see the same
          sequences (paired comparison).

Fixed throughout: CIFAR-10-pretrained frozen encoder (one per seed), 25% validation hold-out in EVERY
stage (n = 3,750 per context, so the N-scaled source's scale never changes between tuning and testing),
1.28M training examples per context (= paper's 5,000 x 256) with a 1,000-step floor, Adam reset per
context (required: per-context batch/lr and per-task correction only make sense with a fresh
accumulator per context).

Requires main.py's --batch-schedule / --lr-schedule / --iters-schedule options and the JUST-LEARNED
callback (commits 05ee436, 95ed7b7).

Usage (from the checkout root, i.e. the directory containing main.py):
  python squisher_experiments/run_cifar100_mixed.py --stage lrcal --count
  python squisher_experiments/run_cifar100_mixed.py --stage lrcal --combo-index 0
  python squisher_experiments/run_cifar100_mixed.py --stage lrcal --combo-index 0 --show   # print command only
  python squisher_experiments/run_cifar100_mixed.py --stage lrcal                          # run all, sequentially
Runs are resumable: a log that already contains both accuracy lines is skipped.

NOTE: lambda-stage keys contain the batch's lr and test-stage keys contain all four lrs, so changing
best_lr.json never silently reuses runs made with a different lr. Test-stage keys also contain the
lambda. The random sequences are fixed by SEQUENCE_SEED.
"""

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / "logs_cifar100_mixed"
BEST_LR_PATH = LOG_DIR / "best_lr.json"
BEST_LAMBDAS_PATH = LOG_DIR / "best_lambdas.json"

EXPERIMENT = "CIFAR100"
CONTEXTS = 10
VALID_SIZE = 0.25
EXAMPLES_PER_CONTEXT = 1_280_000          # = 5,000 steps x batch 256, the paper's budget
MIN_ITERS = 1000                          # floor so large batches still fill beta2=0.999's ~1,000-step window
PREPROCESS_BATCH = 256                    # only used by main.py to cache encoder features; training uses the schedule
BATCHES = [32, 128, 512, 2048]
LR_GRID = [3e-5, 1e-4, 3e-4, 1e-3]
TUNE_SEEDS = [1, 2, 3]

# arm name -> main.py --fisher-source
SOURCES = {
    "fisher": "empirical",                # control: computed from data at batch 1, batch-size independent
    "nscaled": "squisher_nscaled",        # the paper's Squisher: raw accumulator x n
    "mscaled": "squisher_mscaled",        # bias-corrected accumulator x m_t
    "corrected": "squisher_corrected",    # bias-corrected accumulator x m_t(n-1)/(n-m_t)  (Theorem 1)
}
LAMBDA_GRID = [
    1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5,
    1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10,
]

N_SEQUENCES = 5
SEQUENCE_SEED = 20261102
TEST_SEEDS = [1, 2, 3]
TEST_ARMS = ["none", "fisher", "nscaled", "mscaled", "corrected"]
SQUISHER_PROBE_N = 512                    # fidelity diagnostic only; does not affect training

ACC_PATTERN = re.compile(r"average accuracy over all \d+ contexts: ([0-9.]+)")
CONTEXT_ACC_PATTERN = re.compile(r"^\s*- Context (\d+): ([0-9.]+)")
SCHEDULE_PATTERN = re.compile(r"^SCHEDULE context=(\d+) batch=(\d+) lr=(\S+) iters=(\d+)")
JUST_LEARNED_PATTERN = re.compile(r"^JUST-LEARNED context=(\d+) valid_acc=(\S+) test_acc=(\S+)")
FIDELITY_PATTERN = re.compile(
    r"SQUISHER-FIDELITY context=(\d+) source=(\S+) n=(\d+) m=(\d+) cosine=(\S+) "
    r"norm_est=(\S+) norm_probe=(\S+) norm_ratio=(\S+) raw_v_norm=(\S+) exp_avg_norm=(\S+) "
    r"signal_sq_norm_ratio=(\S+) signal_sq_cosine=(\S+)"
)


# ----------------------------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------------------------- #

def iters_for(batch: int) -> int:
    return max(round(EXAMPLES_PER_CONTEXT / batch), MIN_ITERS)


def lr_tag(best_lr: dict) -> str:
    return "-".join(f"{best_lr[b]:.0e}" for b in BATCHES)


def build_sequences() -> list[list[int]]:
    """N_SEQUENCES random batch sequences, one batch size per context drawn uniformly from BATCHES.
    Sequences missing any of the four batch sizes are redrawn, so every run contains all of them.
    Deterministic given SEQUENCE_SEED."""
    rng = random.Random(SEQUENCE_SEED)
    sequences = []
    while len(sequences) < N_SEQUENCES:
        seq = [rng.choice(BATCHES) for _ in range(CONTEXTS)]
        if set(seq) == set(BATCHES):
            sequences.append(seq)
    return sequences


def load_best_lr() -> dict:
    if not BEST_LR_PATH.exists():
        raise SystemExit(f"{BEST_LR_PATH} not found -- run the 'lrcal' stage and "
                         f"'select_cifar100_mixed.py --stage lr' first")
    with open(BEST_LR_PATH) as handle:
        return {int(b): float(lr) for b, lr in json.load(handle).items()}


def load_best_lambdas() -> dict:
    if not BEST_LAMBDAS_PATH.exists():
        raise SystemExit(f"{BEST_LAMBDAS_PATH} not found -- run the 'lambda' stage and "
                         f"'select_cifar100_mixed.py --stage lambda' first")
    with open(BEST_LAMBDAS_PATH) as handle:
        return {arm: float(lam) for arm, lam in json.load(handle).items()}


def parse_log(log_path: Path) -> dict | None:
    """Everything the selection and summary scripts need from one run's log, or None if missing."""
    if not log_path.exists():
        return None
    out = dict(schedule={}, just_learned={}, final_test={}, final_valid={}, fidelity=[],
               test_avg=None, valid_avg=None)
    averages = []
    mode = None
    for line in log_path.read_text().splitlines():
        if "Accuracy of final model on test-set" in line:
            mode = "final_test"
            continue
        if "Accuracy of final model on validation-set" in line:
            mode = "final_valid"
            continue
        m = ACC_PATTERN.search(line)
        if m:
            averages.append(float(m.group(1)))
            mode = None
            continue
        m = CONTEXT_ACC_PATTERN.match(line)
        if m and mode is not None:
            out[mode][int(m.group(1))] = float(m.group(2))
            continue
        m = SCHEDULE_PATTERN.match(line)
        if m:
            out["schedule"][int(m.group(1))] = (int(m.group(2)), float(m.group(3)), int(m.group(4)))
            continue
        m = JUST_LEARNED_PATTERN.match(line)
        if m:
            out["just_learned"][int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
            continue
        m = FIDELITY_PATTERN.search(line)
        if m:
            g = m.groups()
            out["fidelity"].append(dict(context=int(g[0]), source=g[1], n=int(g[2]), m=int(g[3]),
                                        cosine=float(g[4]), norm_ratio=float(g[7]),
                                        signal_sq_norm_ratio=float(g[10])))
    # main.py prints the test-set average first, then the validation-set average
    out["test_avg"] = averages[0] if len(averages) >= 1 else None
    out["valid_avg"] = averages[1] if len(averages) >= 2 else None
    return out


# ----------------------------------------------------------------------------------------------- #
# Combo lists -- shared by the CLI, select_cifar100_mixed.py and parse_results_cifar100_mixed.py
# ----------------------------------------------------------------------------------------------- #

def build_lrcal_combos() -> list[dict]:
    combos = []
    for batch in BATCHES:
        for lr in LR_GRID:
            for seed in TUNE_SEEDS:
                combos.append(dict(stage="lrcal", arm="none", lam=None, seed=seed, seq=None,
                                   batches=[batch] * CONTEXTS, lrs=[lr] * CONTEXTS,
                                   key=f"lrcal_b{batch}_lr{lr:.0e}_s{seed}"))
    return combos


def build_lambda_combos(best_lr: dict) -> list[dict]:
    combos = []
    for arm in SOURCES:
        for batch in BATCHES:
            lr = best_lr[batch]
            for lam in LAMBDA_GRID:
                for seed in TUNE_SEEDS:
                    combos.append(dict(stage="lambda", arm=arm, lam=lam, seed=seed, seq=None,
                                       batches=[batch] * CONTEXTS, lrs=[lr] * CONTEXTS,
                                       key=f"lambda_{arm}_b{batch}_lr{lr:.0e}_lam{lam:.0e}_s{seed}"))
    return combos


def build_test_combos(best_lr: dict, best_lambdas: dict) -> list[dict]:
    combos = []
    sequences = build_sequences()
    tag = lr_tag(best_lr)
    for arm in TEST_ARMS:
        if arm != "none" and arm not in best_lambdas:
            print(f"[warn] no selected lambda for '{arm}'; skipping its test runs", file=sys.stderr)
            continue
        lam = None if arm == "none" else best_lambdas[arm]
        lam_tag = "" if lam is None else f"_lam{lam:.0e}"
        for q, seq in enumerate(sequences, 1):
            for seed in TEST_SEEDS:
                combos.append(dict(stage="test", arm=arm, lam=lam, seed=seed, seq=q,
                                   batches=list(seq), lrs=[best_lr[b] for b in seq],
                                   key=f"test_{arm}{lam_tag}_lrs{tag}_q{q}_s{seed}"))
    return combos


def combos_for(stage: str) -> list[dict]:
    if stage == "lrcal":
        return build_lrcal_combos()
    best_lr = load_best_lr()
    if stage == "lambda":
        return build_lambda_combos(best_lr)
    return build_test_combos(best_lr, load_best_lambdas())


# ----------------------------------------------------------------------------------------------- #
# Running
# ----------------------------------------------------------------------------------------------- #

def build_command(combo: dict) -> list[str]:
    batches, lrs = combo["batches"], combo["lrs"]
    command = [
        sys.executable, "main.py",
        "--experiment", EXPERIMENT,
        "--scenario", "task",
        "--contexts", str(CONTEXTS),
        "--pre-convE", "--freeze-convE", "--seed-to-ltag",
        "--seed", str(combo["seed"]),
        "--no-save",
        "--valid-size", str(VALID_SIZE),
        "--optimizer", "adam_reset",
        "--batch", str(PREPROCESS_BATCH),
        "--batch-schedule", ",".join(str(b) for b in batches),
        "--lr-schedule", ",".join(f"{lr:g}" for lr in lrs),
        "--iters-schedule", ",".join(str(iters_for(b)) for b in batches),
    ]
    if combo["arm"] != "none":
        command += [
            "--ewc",
            "--reg-strength", str(combo["lam"]),
            "--fisher-labels", "true",
            "--fisher-source", SOURCES[combo["arm"]],
            "--squisher-probe-n", str(SQUISHER_PROBE_N),
        ]
    return command


def is_done(log_path: Path) -> bool:
    parsed = parse_log(log_path)
    return parsed is not None and parsed["test_avg"] is not None and parsed["valid_avg"] is not None


def run_one(combo: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{combo['key']}.log"
    if is_done(log_path):
        parsed = parse_log(log_path)
        print(f"[skip] {combo['key']}: test={parsed['test_avg']:.4f} valid={parsed['valid_avg']:.4f}")
        return
    command = build_command(combo)
    with open(log_path, "w") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    parsed = parse_log(log_path)
    if parsed is None or parsed["test_avg"] is None or parsed["valid_avg"] is None:
        print(f"[done] {combo['key']}: FAILED (exit {result.returncode}) -- see {log_path}")
    else:
        print(f"[done] {combo['key']}: test={parsed['test_avg']:.4f} valid={parsed['valid_avg']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["lrcal", "lambda", "test"])
    parser.add_argument("--combo-index", type=int, help="run only this combo (0-based); omit to run all sequentially")
    parser.add_argument("--count", action="store_true", help="print the number of combos in --stage and exit")
    parser.add_argument("--show", action="store_true", help="print the command(s) instead of running them")
    args = parser.parse_args()

    combos = combos_for(args.stage)
    if args.count:
        print(len(combos))
        return
    if args.combo_index is not None:
        if not (0 <= args.combo_index < len(combos)):
            parser.error(f"--combo-index must be in [0, {len(combos) - 1}] for --stage {args.stage}")
        combos = [combos[args.combo_index]]
    for combo in combos:
        if args.show:
            print(f"# {combo['key']}\n" + " ".join(build_command(combo)))
        else:
            run_one(combo)


if __name__ == "__main__":
    main()
