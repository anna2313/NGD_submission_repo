#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-cifar100-select
#SBATCH --partition=compute       # CPU-only: reads existing logs only, launches no training
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=squisher_experiments/logs/squisher-cifar100-select-%j.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Stage 2 of 4: reads every completed tune-stage log and picks, per regularized
# config, the lambda with the best mean VALIDATION accuracy. Requires ALL 63 tasks
# of run_cifar100_tune.sbatch to have finished first -- this doesn't check that
# itself, so submit it with `sbatch --dependency=afterok:<tune_job_id> ...`.
#
# Prints a "grid boundary" warning if a selected lambda sits at either end of
# LAMBDA_GRID -- extend the grid and re-tune in that case rather than proceeding.
#
# Writes logs_cifar100_paper/best_lambdas.json, which the final stage reads.
# Safe to re-run.

python squisher_experiments/select_lambdas_cifar100.py
