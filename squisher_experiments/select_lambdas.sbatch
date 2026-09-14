#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-select-lambdas
#SBATCH --partition=compute       # CPU-only: reads existing logs only, launches no training
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=squisher_experiments/logs/squisher-select-%j.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"
PYTHON="${PYTHON:-${CHECKOUT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python executable not found: %s\n' "${PYTHON}" >&2
  exit 2
fi

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

"${PYTHON}" squisher_experiments/select_lambdas.py
