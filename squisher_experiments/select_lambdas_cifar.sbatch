#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-cifar-select
#SBATCH --partition=compute       # CPU-only: reads existing logs only, launches no training
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=squisher_experiments/logs/squisher-cifar-select-%j.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"

# Uses the same conda env as every other experiment's scripts here (NOT a
# separate uv venv, contrary to the outer README's "Prepare And Smoke"
# section) -- torch/torchvision are already installed and GPU-verified for
# this cluster's V100s in ngd (see cluster_run_log.md's PyTorch/V100 fix
# section), and this cluster's proxy blocks pypi.org/download.pytorch.org
# entirely, so a fresh uv venv install would very likely hit that same wall
# trying to fetch torch from scratch.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Stage 2 of 3: reads every completed tune-stage log and picks, per source, the
# lambda with the best mean VALIDATION accuracy (see run_grid_valid.select_lambdas).
# Requires ALL 57 tasks of run_grid_valid_tune.sbatch to have finished first --
# this doesn't check that itself, so submit it with
# `sbatch --dependency=afterok:<tune_job_id> select_lambdas.sbatch`, or confirm
# via `squeue`/`sacct` that the tune array is fully COMPLETED before submitting.
#
# Writes logs_valid_tuned/best_lambdas.json, which the transfer stage reads next.
# Safe to re-run.

python squisher_experiments/select_lambdas_cifar.py
