"""Standalone lambda-selection step, meant to run BETWEEN the two cluster array
jobs (run_grid_valid_tune.sbatch -> select_lambdas.sbatch -> run_grid_valid_transfer.sbatch).

Requires every combo in run_grid_valid.build_none_tune_combos() (9 "none" + 105
"tune" runs) to have already completed -- i.e. the full "tune" array job must be
finished first. Only reads existing logs; launches no training. Writes
logs_valid_tuned/best_lambdas.json, which run_grid_valid.py --stage transfer reads.
"""

from run_grid_valid_cifar import select_lambdas

if __name__ == "__main__":
    best_lambdas = select_lambdas()
    print(f"\nwrote best_lambdas.json for {len(best_lambdas)}/5 sources")
