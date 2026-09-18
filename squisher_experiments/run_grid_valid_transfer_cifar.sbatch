#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-cifar-transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                 # ESTIMATE, not yet measured -- same tiny MLP as the tune stage.
                                  # Correct after the first batch of this array completes.
#SBATCH --time=06:00:00          # ESTIMATE, bumped 2026-09-14 -- batch=4096 transfer runs will hit the
                                  # same node-speed variance seen in the tune stage (a single batch=4096
                                  # combo measured up to ~63 min on a slow node, another completed in
                                  # ~16 min on a different node). Matches the tune stage's bump for the
                                  # same reason.
#SBATCH --array=0-199
#SBATCH --output=squisher_experiments/logs/squisher-cifar-transfer-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Stage 3 of 3: the "transfer" runs -- each source's lambda, as selected in
# stage 2, re-run at batch 16, 32, 512, and 1024 (5 seeds each). Requires
# logs_valid_tuned/best_lambdas.json to already exist; run_grid_valid.py
# itself errors out clearly (not silently) if it's missing.
#
# TOTAL_COMBOS below assumes all 5 sources got a selected lambda in stage 2
# (5 sources x 4 batches x 5 seeds = 100). If select_lambdas.py's summary
# printed a "[warn] no completed tuning runs for <source>; skipping transfer"
# for ANY source, the real combo list is shorter than 100 -- check
# logs_valid_tuned/best_lambdas.json (should have exactly 5 keys) and lower
# TOTAL_COMBOS to match len(...) before submitting, or some of the higher
# array indices below will fail with an out-of-range error from
# run_grid_valid.py's own argument check.
#
# 1 combo per task -> 200 tasks (0-199), 200/1 exactly, no remainder.

RUNS_PER_TASK=1
TOTAL_COMBOS=200  # must match len(run_grid_valid.build_transfer_combos(best_lambdas)) -- see note above

START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_TASK ))
END=$(( START + RUNS_PER_TASK - 1 ))
if [[ "${END}" -ge "${TOTAL_COMBOS}" ]]; then
  END=$(( TOTAL_COMBOS - 1 ))
fi

echo "Task ${SLURM_ARRAY_TASK_ID} handling combo indices ${START}-${END} of ${TOTAL_COMBOS}"

for i in $(seq "${START}" "${END}"); do
  echo "-> --stage transfer --combo-index ${i}"
  python squisher_experiments/run_grid_valid_cifar.py --stage transfer --combo-index "${i}"
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished."