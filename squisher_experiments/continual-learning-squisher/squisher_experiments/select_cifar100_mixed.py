"""Selection steps of the mixed-batch CIFAR-100 experiment (see run_cifar100_mixed.py).

  --stage lr      after the 'lrcal' stage: for each batch size, the lr with the best mean
                  just-learned VALIDATION accuracy (averaged over the 10 contexts, then over seeds).
                  Writes logs_cifar100_mixed/best_lr.json.
  --stage lambda  after the 'lambda' stage: for each source, the per-batch best lambda (reported) and the
                  middle-ground lambda = best mean final VALIDATION accuracy across the four batch sizes.
                  Writes best_lambdas.json (used by the test stage), lambda_selection.json (full detail)
                  and sequences.json (the test-stage batch sequences, for the record).

Only reads logs; launches no training. Safe to re-run. Test-set numbers are never used for selection.
"""

import argparse
import json
from statistics import mean

from run_cifar100_mixed import (
    BATCHES, BEST_LAMBDAS_PATH, BEST_LR_PATH, CONTEXTS, LAMBDA_GRID, LOG_DIR, LR_GRID, SOURCES,
    build_lambda_combos, build_lrcal_combos, build_sequences, iters_for, load_best_lr, parse_log,
)


def select_lr() -> None:
    scores = {b: {lr: [] for lr in LR_GRID} for b in BATCHES}
    missing = 0
    for combo in build_lrcal_combos():
        parsed = parse_log(LOG_DIR / f"{combo['key']}.log")
        if parsed is None or len(parsed["just_learned"]) < CONTEXTS or parsed["valid_avg"] is None:
            missing += 1
            continue
        just_learned_valid = mean(v for v, _t in parsed["just_learned"].values())
        scores[combo["batches"][0]][combo["lrs"][0]].append(just_learned_valid)

    print("== lr calibration: mean just-learned VALIDATION accuracy (%), mean over seeds ==")
    print("  batch  " + "  ".join(f"{lr:>8.0e}" for lr in LR_GRID) + "   -> chosen")
    best_lr = {}
    for b in BATCHES:
        cells = {lr: mean(v) for lr, v in scores[b].items() if v}
        row = "  ".join(f"{100 * cells[lr]:8.2f}" if lr in cells else f"{'--':>8}" for lr in LR_GRID)
        if not cells:
            print(f"  {b:>5}  {row}   -> none (no completed runs)")
            continue
        chosen = max(cells, key=cells.get)
        best_lr[b] = chosen
        edge = "  <-- grid edge, consider extending LR_GRID" if chosen in (LR_GRID[0], LR_GRID[-1]) else ""
        print(f"  {b:>5}  {row}   -> {chosen:g}{edge}")
    if missing:
        print(f"\n[warn] {missing} lrcal runs missing or incomplete")
    if len(best_lr) < len(BATCHES):
        raise SystemExit("[error] not every batch size has an lr; not writing best_lr.json")

    spread = max(mean(scores[b][best_lr[b]]) for b in BATCHES) - min(mean(scores[b][best_lr[b]]) for b in BATCHES)
    print(f"\nspread of just-learned accuracy across batch sizes at the chosen lrs: {100 * spread:.2f} points")
    LOG_DIR.mkdir(exist_ok=True)
    with open(BEST_LR_PATH, "w") as handle:
        json.dump({str(b): best_lr[b] for b in BATCHES}, handle, indent=2)
    print(f"wrote {BEST_LR_PATH}")


def select_lambda() -> None:
    best_lr = load_best_lr()
    # scores[arm][batch][lam] -> list of final validation accuracies over seeds
    scores = {arm: {b: {lam: [] for lam in LAMBDA_GRID} for b in BATCHES} for arm in SOURCES}
    missing = 0
    for combo in build_lambda_combos(best_lr):
        parsed = parse_log(LOG_DIR / f"{combo['key']}.log")
        if parsed is None or parsed["valid_avg"] is None:
            missing += 1
            continue
        scores[combo["arm"]][combo["batches"][0]][combo["lam"]].append(parsed["valid_avg"])

    best_lambdas, detail = {}, {}
    for arm in SOURCES:
        print(f"\n== {arm} ({SOURCES[arm]}): mean final VALIDATION accuracy (%) per batch size ==")
        means = {b: {lam: mean(v) for lam, v in scores[arm][b].items() if v} for b in BATCHES}
        per_batch_best = {}
        for b in BATCHES:
            if means[b]:
                per_batch_best[b] = max(means[b], key=means[b].get)
        for lam in LAMBDA_GRID:
            row = "  ".join(f"{100 * means[b][lam]:6.1f}" if lam in means[b] else f"{'--':>6}" for b in BATCHES)
            print(f"  lambda {lam:7.0e}  {row}")
        print("  per-batch best: " + ", ".join(f"b{b}: {per_batch_best[b]:.0e}" for b in per_batch_best))

        # middle-ground lambda: only lambdas with results at every batch size are eligible
        eligible = [lam for lam in LAMBDA_GRID if all(lam in means[b] for b in BATCHES)]
        if not eligible:
            print(f"  [warn] no lambda has results at all four batch sizes; skipping {arm}")
            continue
        avg = {lam: mean(means[b][lam] for b in BATCHES) for lam in eligible}
        worst = {lam: min(means[b][lam] for b in BATCHES) for lam in eligible}
        chosen = max(avg, key=avg.get)
        maximin = max(worst, key=worst.get)
        best_lambdas[arm] = chosen
        edge = "  <-- grid edge, consider extending LAMBDA_GRID" if chosen in (LAMBDA_GRID[0], LAMBDA_GRID[-1]) else ""
        print(f"  middle-ground lambda (best mean over batch sizes): {chosen:.0e}  "
              f"(mean {100 * avg[chosen]:.2f}, worst batch {100 * worst[chosen]:.2f}){edge}")
        print(f"  for reference, maximin lambda: {maximin:.0e}  (worst batch {100 * worst[maximin]:.2f})")
        regret = {b: 100 * (means[b][per_batch_best[b]] - means[b][chosen]) for b in BATCHES}
        print("  cost of one lambda vs per-batch best (points): "
              + ", ".join(f"b{b}: {regret[b]:.2f}" for b in BATCHES))
        detail[arm] = dict(
            middle_ground=chosen, maximin=maximin,
            per_batch_best={str(b): per_batch_best[b] for b in BATCHES},
            regret_points={str(b): regret[b] for b in BATCHES},
            mean_valid={str(b): {f"{lam:.0e}": means[b][lam] for lam in means[b]} for b in BATCHES},
        )

    if missing:
        print(f"\n[warn] {missing} lambda runs missing or incomplete")
    LOG_DIR.mkdir(exist_ok=True)
    with open(BEST_LAMBDAS_PATH, "w") as handle:
        json.dump(best_lambdas, handle, indent=2)
    with open(LOG_DIR / "lambda_selection.json", "w") as handle:
        json.dump(detail, handle, indent=2)
    sequences = build_sequences()
    with open(LOG_DIR / "sequences.json", "w") as handle:
        json.dump(dict(batches=sequences,
                       lrs=[[best_lr[b] for b in seq] for seq in sequences],
                       iters=[[iters_for(b) for b in seq] for seq in sequences]), handle, indent=2)
    print(f"\nwrote {BEST_LAMBDAS_PATH} for {len(best_lambdas)}/{len(SOURCES)} sources, "
          f"lambda_selection.json and sequences.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["lr", "lambda"])
    args = parser.parse_args()
    select_lr() if args.stage == "lr" else select_lambda()
