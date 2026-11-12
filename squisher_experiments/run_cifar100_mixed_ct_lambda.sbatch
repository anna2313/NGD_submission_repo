#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-ct-lambda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-[01,03,07,08]  # GPUs the ngd torch build (2.9.0+cu126, sm_50..sm_90) cannot run
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=04:00:00          # default = enough for b2048 (~2 h); override per batch size at submission
#SBATCH --propagate=NONE
#SBATCH --array=0-119
#SBATCH --output=squisher_experiments/logs/squisher-ct-lambda-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"
BATCH="${1:?usage: sbatch [--time=...] run_cifar100_mixed_ct_lambda.sbatch 32|128|512|2048}"
export MIXED_RECIPE=ct           # cosine-tail recipe: 40k steps/context, lr 1e-4, logs_cifar100_mixed_ct/

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Cosine-tail recipe, stage 1 of 2: lambda tuning on homogeneous runs -- 4 sources (fisher, nscaled,
# mscaled, corrected) x 4 batch sizes x 10 lambdas (1e1..1e10) x 3 seeds = 480 combos.
# Submitted as FOUR arrays, one per batch size (argument), each 120 tasks x 1 run, so that every run is
# its own task and each batch size gets a matching time limit:
#   sbatch --time=01:00:00 ... run_cifar100_mixed_ct_lambda.sbatch 32
#   sbatch --time=01:30:00 ... run_cifar100_mixed_ct_lambda.sbatch 128
#   sbatch --time=02:30:00 ... run_cifar100_mixed_ct_lambda.sbatch 512
#   sbatch --time=04:00:00 ... run_cifar100_mixed_ct_lambda.sbatch 2048
# The combo list is ordered source -> batch -> lambda -> seed (16 blocks of 30). Task t of the array for
# batch index b runs combo (source * 4 + b) * 30 + j with source = t / 30, j = t % 30.
# Resubmitting is safe: finished runs are skipped.
# Afterwards: select_cifar100_mixed_ct.sbatch with --dependency=afterok on all four array jobs.

case "${BATCH}" in
  32) B=0 ;; 128) B=1 ;; 512) B=2 ;; 2048) B=3 ;;
  *) echo "batch must be 32, 128, 512 or 2048 (got ${BATCH})"; exit 1 ;;
esac
N_BATCHES=4
TOTAL_COMBOS=$(python squisher_experiments/run_cifar100_mixed.py --stage lambda --count)
BLOCK=$(( TOTAL_COMBOS / (4 * N_BATCHES) ))
if [[ "${TOTAL_COMBOS}" -ne 480 || "${BLOCK}" -ne 30 ]]; then
  echo "Unexpected combo count ${TOTAL_COMBOS} (block ${BLOCK}); expected 480 (30). Is MIXED_RECIPE=ct?"
  exit 1
fi
SOURCE=$(( SLURM_ARRAY_TASK_ID / BLOCK ))
J=$(( SLURM_ARRAY_TASK_ID % BLOCK ))
i=$(( (SOURCE * N_BATCHES + B) * BLOCK + J ))

echo "Batch ${BATCH}, task ${SLURM_ARRAY_TASK_ID}: --stage lambda --combo-index ${i}"
python squisher_experiments/run_cifar100_mixed.py --stage lambda --combo-index "${i}"
