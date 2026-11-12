"""Pilot: training budget and lr decay on context 1 of the mixed-batch CIFAR-100 setup (2026-11-12).

Question: at the end of a context, does Adam's exp_avg_sq reflect minibatch noise at a settled point
(Theorem 1), and how does that depend on the number of steps and on lr decay? Only context 1 is
trained for real (no EWC penalty is active there, so lambda is irrelevant); contexts 2-10 get 1 step
each so that main.py finishes normally. squisher_corrected is used only to get the SQUISHER-FIDELITY
line for context 1.

Grid (all x seeds {1, 2, 3}; new combos are always appended, so earlier indices and run names never
change and finished runs are skipped):
  0-47   round 1: (b32, lr 1e-4) x steps {2k, 5k, 10k, 40k} and (b2048, lr 1e-3) x steps {1k, 2k, 5k, 10k}
         x lr decay {none, cosine}. lrs = best_lr.json of phase 1.
  48-56  extension, cosine: (b2048, lr 1e-3, 40k) and (b2048, lr 1e-4, {10k, 40k}) -- at b2048, is it
         the length of the low-lr tail, or the lr?
  57-89  'cosine-tail' (cosine to 1% over the first 75% of steps, then 1% held) on all 8 round-1
         settings and the 3 extension settings.

Usage: --count | --combo-index I | --summary     (logs in logs_cifar100_pilot/)
"""

import argparse
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

from run_cifar100_mixed import (
    CONTEXTS, EXPERIMENT, FIDELITY_PATTERN, JUST_LEARNED_PATTERN, PREPROCESS_BATCH, REPO_ROOT,
    SQUISHER_PROBE_N, VALID_SIZE,
)

LOG_DIR = Path(__file__).resolve().parent / "logs_cifar100_pilot"
SETTINGS = [(32, 1e-4, [2000, 5000, 10000, 40000]), (2048, 1e-3, [1000, 2000, 5000, 10000])]
DECAYS = ["none", "cosine"]
# appended after the original 48 so their combo indices stay the same: (batch, lr, steps), cosine only
EXTENSION = [(2048, 1e-3, 40000), (2048, 1e-4, 10000), (2048, 1e-4, 40000)]
TAIL_DECAY = "cosine-tail"  # appended after the extension (indices 57-89)
SEEDS = [1, 2, 3]
LAMBDA = 1e1  # irrelevant for context 1


def combos():
    out = []
    for batch, lr, budgets in SETTINGS:
        for steps in budgets:
            for decay in DECAYS:
                for seed in SEEDS:
                    out.append(dict(batch=batch, lr=lr, steps=steps, decay=decay, seed=seed,
                                    key=f"pilot_b{batch}_lr{lr:.0e}_it{steps}_{decay}_s{seed}"))
    for batch, lr, steps in EXTENSION:
        for seed in SEEDS:
            out.append(dict(batch=batch, lr=lr, steps=steps, decay="cosine", seed=seed,
                            key=f"pilot_b{batch}_lr{lr:.0e}_it{steps}_cosine_s{seed}"))
    tail_settings = [(b, lr, st_) for b, lr, budgets in SETTINGS for st_ in budgets] + EXTENSION
    for batch, lr, steps in tail_settings:
        for seed in SEEDS:
            out.append(dict(batch=batch, lr=lr, steps=steps, decay=TAIL_DECAY, seed=seed,
                            key=f"pilot_b{batch}_lr{lr:.0e}_it{steps}_{TAIL_DECAY}_s{seed}"))
    return out


def command(c):
    rest = CONTEXTS - 1
    return [
        sys.executable, "main.py", "--experiment", EXPERIMENT, "--scenario", "task",
        "--contexts", str(CONTEXTS), "--pre-convE", "--freeze-convE", "--seed-to-ltag",
        "--seed", str(c["seed"]), "--no-save", "--valid-size", str(VALID_SIZE),
        "--optimizer", "adam_reset", "--batch", str(PREPROCESS_BATCH),
        "--batch-schedule", ",".join([str(c["batch"])] + ["32"] * rest),
        "--lr-schedule", ",".join([f"{c['lr']:g}"] + ["1e-4"] * rest),
        "--iters-schedule", ",".join([str(c["steps"])] + ["1"] * rest),
        "--lr-decay", c["decay"],
        "--ewc", "--reg-strength", str(LAMBDA), "--fisher-labels", "true",
        "--fisher-source", "squisher_corrected", "--squisher-probe-n", str(SQUISHER_PROBE_N),
    ]


LOSS_PATTERN = re.compile(r"Context: 1/\d+ \| training loss: ([^ ]+) .*?\|\s*(\d+)/(\d+)")


def parse(path):
    """Context-1 numbers from one log, or None if the run did not finish."""
    if not path.exists():
        return None
    text = path.read_text()
    jl = fid = None
    for line in text.splitlines():
        m = JUST_LEARNED_PATTERN.match(line)
        if m and m.group(1) == "1":
            jl = float(m.group(2))
        m = FIDELITY_PATTERN.search(line)
        if m and m.group(1) == "1":
            fid = dict(norm_ratio=float(m.group(8)), raw_v=float(m.group(9)),
                       exp_avg=float(m.group(10)), cosine=float(m.group(5)))
    if jl is None or fid is None:
        return None
    losses = [(int(m.group(2)), float(m.group(1)))
              for seg in text.replace("\r", "\n").split("\n") for m in [LOSS_PATTERN.search(seg)] if m]
    T = max((s for s, _ in losses), default=0)
    second_half = [l for s, l in losses if s > T / 2]
    last = [l for s, l in losses if s > 0.95 * T]
    return dict(valid=jl, **fid,
                final_loss=st.median(last) if last else float("nan"),
                spike=(max(second_half) / st.median(second_half)) if second_half else float("nan"))


def run(c):
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{c['key']}.log"
    if parse(path) is not None:
        print(f"[skip] {c['key']}")
        return
    cmd = command(c)
    with open(path, "w") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if parse(path) is None:
        print(f"[done] {c['key']}: FAILED (exit {result.returncode}) -- see {path}")
        sys.exit(result.returncode or 1)
    print(f"[done] {c['key']}")


def summary():
    print("Context 1 only; medians over seeds (n = number of finished seeds).")
    print("valid = just-learned validation acc; norm_ratio = corrected estimate / probe Fisher (1 = Theorem 1);")
    print("final_loss = median training loss over the last 5% of steps; spike = max/median loss in the 2nd half.\n")
    print(f"{'batch':>5} {'lr':>6} {'steps':>6} {'decay':>11} {'n':>2} {'valid':>7} {'norm_ratio':>11} {'raw_v':>10} "
          f"{'final_loss':>11} {'spike':>9}  norm_ratio per seed")
    rows = {}
    for c in combos():
        rows.setdefault((c["batch"], c["lr"], c["steps"], c["decay"]), []).append(
            parse(LOG_DIR / f"{c['key']}.log"))
    order = {"none": 0, "cosine": 1, "cosine-tail": 2}
    for (b, lr, s, d) in sorted(rows, key=lambda k: (k[0], -k[1], k[2], order[k[3]])):
        rs = [r for r in rows[(b, lr, s, d)] if r is not None]
        if not rs:
            print(f"{b:>5} {lr:>6.0e} {s:>6} {d:>11} {0:>2}  (no finished runs)")
            continue
        med = lambda k: st.median(r[k] for r in rs)
        print(f"{b:>5} {lr:>6.0e} {s:>6} {d:>11} {len(rs):>2} {100 * med('valid'):>7.2f} {med('norm_ratio'):>11.3g} "
              f"{med('raw_v'):>10.3g} {med('final_loss'):>11.3g} {med('spike'):>9.3g}  "
              + " ".join(f"{r['norm_ratio']:.3g}" for r in rs))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--combo-index", type=int)
    g.add_argument("--summary", action="store_true")
    a = p.parse_args()
    cs = combos()
    if a.count:
        print(len(cs))
    elif a.summary:
        summary()
    else:
        run(cs[a.combo_index])


if __name__ == "__main__":
    main()
