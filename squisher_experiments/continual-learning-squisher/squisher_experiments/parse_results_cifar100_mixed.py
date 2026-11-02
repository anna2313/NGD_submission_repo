"""Summary of the mixed-batch CIFAR-100 experiment (see run_cifar100_mixed.py).

Reads logs_cifar100_mixed/*.log and writes, next to the logs:
  runs.csv         one row per completed run (all stages): stage, arm, lambda, seed, sequence, batches, lrs,
                   final test/valid average, mean just-learned test/valid accuracy
  per_context.csv  one row per (test-stage run, context): batch, lr, just-learned test accuracy, final test
                   accuracy, forgetting (= just-learned - final)
  fidelity.csv     one row per SQUISHER-FIDELITY line (every stage)
and prints, for whatever stages have results:
  - lrcal: just-learned validation accuracy per batch x lr
  - lambda: per source, the best lambda per batch size (selection details: select_cifar100_mixed.py)
  - test: mean +- std final test accuracy per arm; paired differences vs the Fisher and vs the corrected
    Squisher; per-arm slope of per-context final test accuracy (points) against log2(batch size);
    final accuracy and forgetting per batch size; Squisher fidelity (norm_ratio) per batch size.

Prediction: the N-scaled Squisher over-weights small-batch tasks (negative slope: small-batch tasks
retained, large-batch tasks forgotten); the corrected Squisher and the Fisher show ~zero slope.
"""

import csv
import math
from collections import defaultdict
from statistics import mean, stdev

from run_cifar100_mixed import (
    BATCHES, BEST_LAMBDAS_PATH, BEST_LR_PATH, LAMBDA_GRID, LOG_DIR, LR_GRID, SOURCES, TEST_ARMS,
    build_lambda_combos, build_lrcal_combos, build_test_combos, load_best_lambdas, load_best_lr, parse_log,
)


def fmt(values: list[float], scale: float = 100.0) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{scale * values[0]:.2f} (n=1)"
    return f"{scale * mean(values):.2f} +- {scale * stdev(values):.2f} (n={len(values)})"


def slope(xs: list[float], ys: list[float]) -> float | None:
    if len(set(xs)) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def collect(combos: list[dict]) -> list[tuple[dict, dict]]:
    out = []
    for combo in combos:
        parsed = parse_log(LOG_DIR / f"{combo['key']}.log")
        if parsed is not None and parsed["test_avg"] is not None and parsed["valid_avg"] is not None:
            out.append((combo, parsed))
    return out


def main() -> None:
    lrcal = collect(build_lrcal_combos())
    lam_runs, test_runs = [], []
    if BEST_LR_PATH.exists():
        best_lr = load_best_lr()
        lam_runs = collect(build_lambda_combos(best_lr))
        if BEST_LAMBDAS_PATH.exists():
            test_runs = collect(build_test_combos(best_lr, load_best_lambdas()))

    # ---------------------------------------------------------------- CSVs
    run_rows, ctx_rows, fid_rows = [], [], []
    for combo, parsed in lrcal + lam_runs + test_runs:
        jl = parsed["just_learned"]
        run_rows.append(dict(
            stage=combo["stage"], arm=combo["arm"], **{"lambda": "" if combo["lam"] is None else combo["lam"]},
            seed=combo["seed"], sequence="" if combo["seq"] is None else combo["seq"],
            batches=" ".join(map(str, combo["batches"])), lrs=" ".join(f"{x:g}" for x in combo["lrs"]),
            test_avg=parsed["test_avg"], valid_avg=parsed["valid_avg"],
            just_learned_test=mean(t for _v, t in jl.values()) if jl else "",
            just_learned_valid=mean(v for v, _t in jl.values()) if jl else "",
        ))
        for f in parsed["fidelity"]:
            fid_rows.append(dict(stage=combo["stage"], arm=combo["arm"], key=combo["key"], **f))
    for combo, parsed in test_runs:
        for c in range(1, len(combo["batches"]) + 1):
            jl_test = parsed["just_learned"].get(c, (None, None))[1]
            final = parsed["final_test"].get(c)
            ctx_rows.append(dict(
                arm=combo["arm"], **{"lambda": "" if combo["lam"] is None else combo["lam"]},
                sequence=combo["seq"], seed=combo["seed"], context=c,
                batch=combo["batches"][c - 1], lr=combo["lrs"][c - 1],
                just_learned_test=jl_test, final_test=final,
                forgetting=(jl_test - final) if (jl_test is not None and final is not None) else "",
            ))
    for name, rows in (("runs.csv", run_rows), ("per_context.csv", ctx_rows), ("fidelity.csv", fid_rows)):
        if rows:
            with open(LOG_DIR / name, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    # ---------------------------------------------------------------- lrcal
    if lrcal:
        print("== LRCAL: mean just-learned VALIDATION accuracy (%) per batch x lr, mean over seeds ==")
        cells = defaultdict(list)
        for combo, parsed in lrcal:
            cells[(combo["batches"][0], combo["lrs"][0])].append(mean(v for v, _t in parsed["just_learned"].values()))
        print("  batch  " + "  ".join(f"{lr:>8.0e}" for lr in LR_GRID))
        for b in BATCHES:
            print(f"  {b:>5}  " + "  ".join(f"{100 * mean(cells[(b, lr)]):8.2f}" if cells[(b, lr)] else f"{'--':>8}"
                                            for lr in LR_GRID))
        print(f"  ({len(lrcal)} runs)\n")

    # ---------------------------------------------------------------- lambda
    if lam_runs:
        print("== LAMBDA: best lambda per batch size (mean final VALIDATION accuracy %) ==")
        cells = defaultdict(list)
        for combo, parsed in lam_runs:
            cells[(combo["arm"], combo["batches"][0], combo["lam"])].append(parsed["valid_avg"])
        for arm in SOURCES:
            parts = []
            for b in BATCHES:
                avail = {lam: mean(cells[(arm, b, lam)]) for lam in LAMBDA_GRID if cells[(arm, b, lam)]}
                if avail:
                    best = max(avail, key=avail.get)
                    parts.append(f"b{b}: {best:.0e} ({100 * avail[best]:.1f})")
            print(f"  {arm:<10} " + "  ".join(parts))
        print(f"  ({len(lam_runs)} runs; full tables and the middle-ground choice: select_cifar100_mixed.py --stage lambda)\n")

    # ---------------------------------------------------------------- test
    if not test_runs:
        print("no completed test-stage runs yet")
        print(f"\nwrote CSVs to {LOG_DIR}")
        return

    print("== TEST: mixed batch sizes, final TEST accuracy (%), mean +- std over sequences x seeds ==")
    by_arm = defaultdict(dict)                       # arm -> (seq, seed) -> (combo, parsed)
    for combo, parsed in test_runs:
        by_arm[combo["arm"]][(combo["seq"], combo["seed"])] = (combo, parsed)
    for arm in TEST_ARMS:
        if arm in by_arm:
            lam = next(iter(by_arm[arm].values()))[0]["lam"]
            lam_str = "" if lam is None else f"lambda={lam:.0e}"
            print(f"  {arm:<10} {lam_str:<14} {fmt([p['test_avg'] for _c, p in by_arm[arm].values()])}")

    for ref in ("fisher", "corrected"):
        if ref not in by_arm:
            continue
        print(f"\n  paired difference vs {ref} (points; same sequence and seed):")
        for arm in TEST_ARMS:
            if arm == ref or arm not in by_arm:
                continue
            shared = sorted(set(by_arm[arm]) & set(by_arm[ref]))
            diffs = [by_arm[arm][k][1]["test_avg"] - by_arm[ref][k][1]["test_avg"] for k in shared]
            if diffs:
                wins = sum(d > 0 for d in diffs)
                print(f"    {arm:<10} - {ref:<10} {fmt(diffs)}   higher in {wins}/{len(diffs)}")

    print("\n  slope of per-context final test accuracy (points) vs log2(batch size), per run then mean +- std:")
    print("  (negative = small-batch tasks retained better than large-batch tasks)")
    for arm in TEST_ARMS:
        if arm not in by_arm:
            continue
        slopes = []
        for combo, parsed in by_arm[arm].values():
            xs = [math.log2(combo["batches"][c - 1]) for c in parsed["final_test"]]
            ys = [100 * parsed["final_test"][c] for c in parsed["final_test"]]
            s = slope(xs, ys)
            if s is not None:
                slopes.append(s)
        print(f"    {arm:<10} {fmt(slopes, scale=1.0)}")

    print("\n  per batch size, mean over all contexts trained at that batch (%):")
    header = "  ".join(f"{'b' + str(b):>8}" for b in BATCHES)
    for title, field in (("final test accuracy", "final_test"), ("forgetting (just-learned - final)", "forgetting")):
        print(f"    {title}")
        print(f"    {'':<10} {header}")
        for arm in TEST_ARMS:
            vals = defaultdict(list)
            for row in ctx_rows:
                if row["arm"] == arm and row[field] not in ("", None):
                    vals[row["batch"]].append(row[field])
            if vals:
                print(f"    {arm:<10} " + "  ".join(f"{100 * mean(vals[b]):8.2f}" if vals[b] else f"{'--':>8}"
                                                     for b in BATCHES))

    test_fid = [r for r in fid_rows if r["stage"] == "test"]
    if test_fid:
        print("\n  Squisher fidelity in the test stage: mean norm_ratio (estimate / probe Fisher; 1 = right scale)")
        print(f"    {'':<10} {header}")
        for arm in ("nscaled", "mscaled", "corrected"):
            vals = defaultdict(list)
            for r in test_fid:
                if r["arm"] == arm:
                    vals[r["m"]].append(r["norm_ratio"])
            if vals:
                print(f"    {arm:<10} " + "  ".join(f"{mean(vals[b]):8.3g}" if vals[b] else f"{'--':>8}"
                                                     for b in BATCHES))

    print(f"\n({len(test_runs)} test runs)  wrote runs.csv, per_context.csv, fidelity.csv to {LOG_DIR}")


if __name__ == "__main__":
    main()
