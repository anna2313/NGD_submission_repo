#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-mixed-lambda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-[01,03,07,08]  # GPUs the ngd torch build (2.9.0+cu126, sm_50..sm_90) cannot run:
                                              # 01 = V100 (2026-10-19); 03/07/08 = 'no kernel image' (2026-11-02, job 9957294)
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G                 # measured ~1 GB for pretraining on 2026-10-19; continual-learning runs are similar
#SBATCH --time=10:00:00          # 16 runs per task, back to back: 4 each at batch 32/128/512/2048.
                                  # Batch-32 runs took 18-30 min in lrcal, larger batches less -> ~3-4 h per task expected.
#SBATCH --propagate=NONE         # do not inherit the login node's per-process memory cap
#SBATCH --array=0-56
#SBATCH --output=squisher_experiments/logs/squisher-mixed-lambda-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Stage 2 of 3: lambda tuning on homogeneous runs -- 4 sources (fisher, nscaled, mscaled, corrected)
# x 4 batch sizes (each at its lr from best_lr.json) x 19 lambdas (1e1..1e10) x 3 seeds = 912 combos,
# packed into N_TASKS = 57 array tasks of 16 runs each.
#
# Packing is STRIDED: task t runs combos t, t+57, t+114, ... The combo list is ordered source -> batch
# -> lambda -> seed, i.e. 16 blocks of 57 (one block per source x batch), so every task gets exactly one
# run from each block: 4 runs at each batch size, which balances the run time across tasks.
#
# A failed run does not stop the task's remaining runs; the task exits non-zero at the end if any run
# failed, so Slurm marks it FAILED and the afterok-dependent selection does not start.
# Keep --array=0-56 together with N_TASKS=57 (the mapping depends on it). Resubmitting the array is
# safe: finished runs are skipped.
#
# Requires logs_cifar100_mixed/best_lr.json (select_cifar100_mixed.sbatch lr).
# Afterwards: sbatch --export=ALL,CHECKOUT_DIR select_cifar100_mixed.sbatch lambda

N_TASKS=57
TOTAL_COMBOS=$(python squisher_experiments/run_cifar100_mixed.py --stage lambda --count)

echo "Task ${SLURM_ARRAY_TASK_ID} of ${N_TASKS}: combos ${SLURM_ARRAY_TASK_ID}, +${N_TASKS}, ... (< ${TOTAL_COMBOS})"
FAILED=0
for (( i = SLURM_ARRAY_TASK_ID; i < TOTAL_COMBOS; i += N_TASKS )); do
  echo "-> --stage lambda --combo-index ${i}"
  python squisher_experiments/run_cifar100_mixed.py --stage lambda --combo-index "${i}" || FAILED=1
done

if [[ "${FAILED}" -ne 0 ]]; then
  echo "Task ${SLURM_ARRAY_TASK_ID} finished WITH FAILURES (see [done] ... FAILED lines above)."
  exit 1
fi
echo "Task ${SLURM_ARRAY_TASK_ID} finished."
