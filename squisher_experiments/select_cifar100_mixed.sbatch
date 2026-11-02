#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-mixed-select
#SBATCH --partition=compute       # CPU-only: reads existing logs only, launches no training
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --propagate=NONE
#SBATCH --output=squisher_experiments/logs/squisher-mixed-select-%j.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"
STAGE="${1:?usage: sbatch select_cifar100_mixed.sbatch lr|lambda}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd

cd "${CHECKOUT_DIR}"

# lr     : after run_cifar100_mixed_lrcal.sbatch  -> writes logs_cifar100_mixed/best_lr.json
# lambda : after run_cifar100_mixed_lambda.sbatch -> writes best_lambdas.json, lambda_selection.json,
#          sequences.json
# Submit with --dependency=afterok:<array_job_id> or after squeue shows the array finished.
# Check the output for "grid edge" warnings before starting the next stage.
cd squisher_experiments
python select_cifar100_mixed.py --stage "${STAGE}"
