#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-cifar100-tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-01  # V100 node: the ngd torch build's cuDNN cannot run convolutions there
                                  # (see pretrain_cifar100.sbatch). Extend this list if other V100 nodes exist.
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G                 # ESTIMATE -- CIFAR-100 in memory, frozen 393k-param encoder, 6.2M-param FC head.
                                  # Check seff after the first few tasks land and correct before resubmitting.
#SBATCH --time=01:00:00          # MEASURED 2026-10-19 (job 9774069): 5-7 min per squisher/none combo, 7-12 min per
                                  # fisher combo (features are cached once the encoder is frozen, so the full-pass
                                  # Fisher is cheap). 1 h leaves ample margin for slow nodes.
#SBATCH --array=0-170            # = len(build_tune_combos()) = 3 configs x 19 lambdas x 3 seeds; completed combos
                                  # are skipped in seconds, so re-running the whole range after a grid change is fine.
#SBATCH --output=squisher_experiments/logs/squisher-cifar100-tune-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Stage 1 of 4: lambda tuning with a 25% validation hold-out -- 3 regularized
# configs (fisher_adam, squisher_adam, squisher_adamreset) x 19-value LAMBDA_GRID
# x 3 seeds = 171 combos, as defined once in run_cifar100_paper.build_tune_combos().
# The "none" baselines have nothing to tune and only run in the final stage.
#
# Requires store/models/C3-5x16-bn-e100-s{1,2,3} from pretrain_cifar100.sbatch.
#
# Once ALL tasks finish: submit select_lambdas_cifar100.sbatch, then (once THAT's
# done) run_cifar100_final.sbatch. Chain with
# `sbatch --dependency=afterok:<this_job_id> ...`, or wait for `squeue` to show
# this array fully finished before submitting.

RUNS_PER_TASK=1
TOTAL_COMBOS=$(python squisher_experiments/run_cifar100_paper.py --stage tune --count)

START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_TASK ))
END=$(( START + RUNS_PER_TASK - 1 ))
if [[ "${END}" -ge "${TOTAL_COMBOS}" ]]; then
  END=$(( TOTAL_COMBOS - 1 ))
fi
if [[ "${START}" -ge "${TOTAL_COMBOS}" ]]; then
  echo "Task ${SLURM_ARRAY_TASK_ID}: nothing to do (only ${TOTAL_COMBOS} combos)"
  exit 0
fi

echo "Task ${SLURM_ARRAY_TASK_ID} handling combo indices ${START}-${END} of ${TOTAL_COMBOS}"

for i in $(seq "${START}" "${END}"); do
  echo "-> --stage tune --combo-index ${i}"
  python squisher_experiments/run_cifar100_paper.py --stage tune --combo-index "${i}"
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished."
