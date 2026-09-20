#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=merge-unbalanced-cifar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                 # ESTIMATE, unmeasured. ResNet18 on CIFAR is a much larger model
                                  # than LeNet on MNIST -- do not assume MNIST's measured ~275MB holds
                                  # here. Check seff after the first tasks land.
#SBATCH --time=02:00:00          # ESTIMATE, unmeasured -- also uncertain because probe_fisher() does
                                  # one full forward+backward pass PER SINGLE EXAMPLE, sequentially, and
                                  # ResNet18 is far more expensive per pass than LeNet.
#SBATCH --array=0-8
#SBATCH --output=merging_experiments/logs/merge-unbalanced-cifar-%A_%a.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this from the repo root}"

PLAN=merging_experiments/plans/unbalanced_shards_cifar.json

if [[ ! -f "${PLAN}" ]]; then
  printf 'Plan not found: %s\nRun: python merging_experiments/write_sweep_plan_cifar.py --suite unbalanced_shards\n' "${PLAN}" >&2
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