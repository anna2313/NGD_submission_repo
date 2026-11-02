#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-mixed-lambda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-[01,03,07,08]  # GPUs the ngd torch build (2.9.0+cu126, sm_50..sm_90) cannot run:
                                              # 01 = V100 (2026-10-19); 03/07/08 = 'no kernel image' (2026-11-02, job 9957294)
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G                 # measured ~1 GB for pretraining on 2026-10-19; continual-learning runs are similar
#SBATCH --time=03:00:00          # batch-32 contexts take 40,000 steps (1.28M examples); ~30-45 min per run expected
#SBATCH --propagate=NONE         # do not inherit the login node's per-process memory cap
#SBATCH --array=0-911
#SBATCH --output=squisher_experiments/logs/squisher-mixed-lambda-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Stage 2 of 3: lambda tuning on homogeneous runs -- 4 sources (fisher, nscaled, mscaled, corrected)
# x 4 batch sizes (each at its lr from best_lr.json) x 19 lambdas (1e1..1e10) x 3 seeds = 912 combos.
# Requires logs_cifar100_mixed/best_lr.json (select_cifar100_mixed.sbatch lr).
# If the cluster limits array size or queued jobs below 912, submit in chunks, e.g. --array=0-455 then
# --array=456-911 (the command-line --array overrides the one above).
# Afterwards: sbatch --export=ALL,CHECKOUT_DIR select_cifar100_mixed.sbatch lambda

RUNS_PER_TASK=1
TOTAL_COMBOS=$(python squisher_experiments/run_cifar100_mixed.py --stage lambda --count)

START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_TASK ))
END=$(( START + RUNS_PER_TASK - 1 ))
if [[ "${START}" -ge "${TOTAL_COMBOS}" ]]; then
  echo "Task ${SLURM_ARRAY_TASK_ID}: nothing to do (only ${TOTAL_COMBOS} combos)"
  exit 0
fi
if [[ "${END}" -ge "${TOTAL_COMBOS}" ]]; then
  END=$(( TOTAL_COMBOS - 1 ))
fi

echo "Task ${SLURM_ARRAY_TASK_ID} handling combo indices ${START}-${END} of ${TOTAL_COMBOS}"
for i in $(seq "${START}" "${END}"); do
  echo "-> --stage lambda --combo-index ${i}"
  python squisher_experiments/run_cifar100_mixed.py --stage lambda --combo-index "${i}"
done
echo "Task ${SLURM_ARRAY_TASK_ID} finished."
