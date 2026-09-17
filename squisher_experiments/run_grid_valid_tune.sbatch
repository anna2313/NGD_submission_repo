#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-tune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                 # ESTIMATE, not yet measured on this cluster -- tiny MLP (478K params),
                                  # similar footprint to the MNIST experiments elsewhere in this project.
                                  # Check seff/sacct after the first handful of tasks land and correct
                                  # before resubmitting anything that fails.
#SBATCH --time=06:00:00          # ESTIMATE, bumped 2026-09-14 -- batch=4096 "none" runs (under the new
                                  # MIN_ITERS floor) TIMED OUT at the previous 01:00:00 limit (tasks 14/15).
                                  # A single batch=4096 combo measured at ~63 min on one node but only
                                  # ~16 min (half of a 31:48 two-combo task) on another -- real node-speed
                                  # variance on this cluster, not just batch-size cost. Sized with margin
                                  # for two slow-node batch=4096 combos back to back in one task. No
                                  # memory issue observed (still well under 4G).
#SBATCH --array=0-83
#SBATCH --output=squisher_experiments/logs/squisher-tune-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Stage 1 of 3: the "none"-baseline runs (batch=128 at 3 seeds, plus each of
# TRANSFER_BATCHES=[16,32,512,1024] at 5 seeds) + "tune" runs (5 sources x
# 9-value LAMBDA_GRID x 3 seeds) -- 158 combos total, as defined once in
# run_grid_valid.build_none_tune_combos() -- this script never needs to know
# the actual sources/lambdas/seeds itself, only which combo INDICES belong to
# this task. 2 combos per task -> 79 tasks (0-78); 158 is evenly divisible by
# 2 so no task ever runs fewer than 2.
#
# Neither "none" nor "tune" needs best_lambdas.json, so this whole array is
# independent and can run in any order.
#
# Once ALL 79 tasks finish: submit select_lambdas.sbatch, then (once THAT's
# done) run_grid_valid_transfer.sbatch. Chain with
# `sbatch --dependency=afterok:<this_job_id> select_lambdas.sbatch`, or just
# wait for `squeue` to show this array fully finished before submitting.

RUNS_PER_TASK=2
TOTAL_COMBOS=168   # must match len(run_grid_valid.build_none_tune_combos())

START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_TASK ))
END=$(( START + RUNS_PER_TASK - 1 ))
if [[ "${END}" -ge "${TOTAL_COMBOS}" ]]; then
  END=$(( TOTAL_COMBOS - 1 ))
fi

echo "Task ${SLURM_ARRAY_TASK_ID} handling combo indices ${START}-${END} of ${TOTAL_COMBOS}"

for i in $(seq "${START}" "${END}"); do
  echo "-> --stage tune --combo-index ${i}"
  python squisher_experiments/run_grid_valid.py --stage tune --combo-index "${i}"
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished."