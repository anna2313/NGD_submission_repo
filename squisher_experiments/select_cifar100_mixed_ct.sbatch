#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-ct-select
#SBATCH --partition=compute       # CPU-only: reads existing logs only, launches no training
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00           # the base-recipe selection timed out at 10 min on 912 logs (2026-11-12)
#SBATCH --propagate=NONE
#SBATCH --output=squisher_experiments/logs/squisher-ct-select-%j.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"
export MIXED_RECIPE=ct

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd

cd "${CHECKOUT_DIR}/squisher_experiments"

# Cosine-tail recipe: lambda selection after run_cifar100_mixed_ct_lambda.sbatch.
# Writes logs_cifar100_mixed_ct/{best_lambdas.json, lambda_selection.json, sequences.json}.
# Check the output for "missing" warnings and "grid edge" flags before the test stage.
python select_cifar100_mixed.py --stage lambda
