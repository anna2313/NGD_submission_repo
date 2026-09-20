#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=merge-equal-cifar-trimmed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G                 # MEASURED (2026-09-18): smoke test at batch=8000 (the worst case
                                  # in this grid) used 1.15GB host RAM against an 8G request. Note
                                  # this is HOST RAM only -- separate from the GPU memory that
                                  # actually OOM'd before microbatching was added; --mem never
                                  # governs GPU memory, so this number says nothing about GPU safety.
#SBATCH --time=03:00:00          # ESTIMATE, unmeasured -- bumped over the full grid's 2h given the
                                  # MIN_STEPS floor's extra compute at batch=2048/8000, and
                                  # probe_fisher()'s per-example sequential passes on ResNet18.
#SBATCH --array=0-11
#SBATCH --output=merging_experiments/logs/merge-equal-cifar-trimmed-%A_%a.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this from the repo root}"

PLAN=merging_experiments/plans/equal_exposure_trend_cifar_trimmed.json

if [[ ! -f "${PLAN}" ]]; then
  printf 'Plan not found: %s\nRun: python merging_experiments/write_sweep_plan_cifar_trimmed.py --suite equal_exposure_trend\n' "${PLAN}" >&2
  exit 2
fi

COMMAND=$(python -c "
import json, sys
with open('${PLAN}') as f:
    commands = json.load(f)['commands']
i = ${SLURM_ARRAY_TASK_ID}
if not 0 <= i < len(commands):
    raise SystemExit(f'task {i} out of range for {len(commands)} commands')
print(' '.join(commands[i]))
")

echo "Task ${SLURM_ARRAY_TASK_ID}: ${COMMAND}"
${COMMAND} --device cuda

echo "Task ${SLURM_ARRAY_TASK_ID} finished."