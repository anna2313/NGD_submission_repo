"""Lambda-selection step for the CIFAR-100 paper reproduction, run BETWEEN the two
cluster array jobs (run_cifar100_tune.sbatch -> select_lambdas_cifar100.sbatch ->
run_cifar100_final.sbatch).

Requires every combo in run_cifar100_paper.build_tune_combos() to have completed.
Only reads logs; launches no training. Writes logs_cifar100_paper/best_lambdas.json,
which run_cifar100_paper.py --stage final reads.
"""

from run_cifar100_paper import CONFIGS, select_lambdas

if __name__ == "__main__":
    best_lambdas = select_lambdas()
    n_regularized = sum(1 for _, source, _ in CONFIGS if source is not None)
    print(f"\nwrote best_lambdas.json for {len(best_lambdas)}/{n_regularized} regularized configs")
