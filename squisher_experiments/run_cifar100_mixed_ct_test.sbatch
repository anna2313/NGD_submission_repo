#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-ct-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-[01,03,07,08]  # GPUs the ngd torch build (2.9.0+cu126, sm_50..sm_90) cannot run
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=05:00:00          # 10 contexts x 40,000 steps at mixed batch sizes; worst case (several b2048
                                  # contexts) ~1.5-2 h
#SBATCH --propagate=NONE
#SBATCH --array=0-74
#SBATCH --output=squisher_experiments/logs/squisher-ct-test-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"
export MIXED_RECIPE=ct

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Cosine-tail recipe, stage 2 of 2: mixed-batch test -- 5 random batch sequences x 3 seeds x 5 arms
# (none, fisher, nscaled, mscaled, corrected) = 75 combos, one per task. Every context: 40k steps, lr 1e-4,
# cosine-tail; each source uses its middle-ground lambda from logs_cifar100_mixed_ct/best_lambdas.json.
# Afterwards (login node is fine, no torch):
#   MIXED_RECIPE=ct python squisher_experiments/parse_results_cifar100_mixed.py

echo "Task ${SLURM_ARRAY_TASK_ID}: --stage test --combo-index ${SLURM_ARRAY_TASK_ID}"
python squisher_experiments/run_cifar100_mixed.py --stage test --combo-index "${SLURM_ARRAY_TASK_ID}"
