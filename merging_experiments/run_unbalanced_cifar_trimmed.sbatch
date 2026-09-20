#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=merge-unbalanced-cifar-trimmed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                 # ESTIMATE, unmeasured. ResNet18 on CIFAR; the batch=2048/8000
                                  # cells also process far more examples than 256,000 due to the
                                  # MIN_STEPS=1000 floor (up to 8,000,000 examples at batch=8000).
                                  # Check seff after the first tasks land.
#SBATCH --time=03:00:00          # ESTIMATE, unmeasured -- bumped over the full grid's 2h given the
                                  # MIN_STEPS floor's extra compute at batch=2048/8000, and
                                  # probe_fisher()'s per-example sequential passes on ResNet18.
#SBATCH --array=0-11
#SBATCH --output=merging_experiments/logs/merge-unbalanced-cifar-trimmed-%A_%a.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this from the repo root}"

PLAN=merging_experiments/plans/unbalanced_shards_cifar_trimmed.json

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