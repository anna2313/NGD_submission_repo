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
#SBATCH --time=00:30:00          # ESTIMATE -- these are all batch=128 runs (2,000 iters/context x 5
                                  # contexts = 10,000 iterations), 2 runs per task. Correct after the
                                  # first batch completes, same practice as this project's other array jobs.
#SBATCH --array=0-56
#SBATCH --output=logs/squisher-tune-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"
PYTHON="${PYTHON:-${CHECKOUT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python executable not found: %s\n' "${PYTHON}" >&2
  exit 2
fi

cd "${CHECKOUT_DIR}"

# Stage 1 of 3: the 9 "none"-baseline runs + 105 "tune" runs (114 combos total),
# as defined once in run_grid_valid.build_none_tune_combos() -- this script never
# needs to know the actual sources/lambdas/seeds itself, only which combo
# INDICES belong to this task. 2 combos per task -> 57 tasks (0-56); 114 is
# evenly divisible by 2 so no task ever runs fewer than 2.
#
# Neither "none" nor "tune" needs best_lambdas.json, so this whole array is
# independent and can run in any order.
#
# Once ALL 57 tasks finish: submit select_lambdas.sbatch, then (once THAT's
# done) run_grid_valid_transfer.sbatch. Chain with
# `sbatch --dependency=afterok:<this_job_id> select_lambdas.sbatch`, or just
# wait for `squeue` to show this array fully finished before submitting.

RUNS_PER_TASK=2
TOTAL_COMBOS=114   # must match len(run_grid_valid.build_none_tune_combos())

START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_TASK ))
END=$(( START + RUNS_PER_TASK - 1 ))
if [[ "${END}" -ge "${TOTAL_COMBOS}" ]]; then
  END=$(( TOTAL_COMBOS - 1 ))
fi

echo "Task ${SLURM_ARRAY_TASK_ID} handling combo indices ${START}-${END} of ${TOTAL_COMBOS}"

for i in $(seq "${START}" "${END}"); do
  echo "-> --stage tune --combo-index ${i}"
  "${PYTHON}" squisher_experiments/run_grid_valid.py --stage tune --combo-index "${i}"
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished."
