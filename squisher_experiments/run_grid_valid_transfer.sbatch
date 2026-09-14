#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G                 # ESTIMATE, not yet measured -- same tiny MLP as the tune stage.
                                  # Correct after the first batch of this array completes.
#SBATCH --time=01:00:00          # ESTIMATE -- batch=32 runs here are the heaviest single combo in the
                                  # whole experiment (8,000 iters/context x 5 contexts = 40,000
                                  # iterations), noticeably more than the tune stage's batch=128 runs
                                  # (10,000 iterations), so this stage gets a more generous budget even
                                  # though it has fewer total combos.
#SBATCH --array=0-14
#SBATCH --output=squisher_experiments/logs/squisher-transfer-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"
PYTHON="${PYTHON:-${CHECKOUT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python executable not found: %s\n' "${PYTHON}" >&2
  exit 2
fi

cd "${CHECKOUT_DIR}"

# Stage 3 of 3: the "transfer" runs -- each source's lambda, as selected in
# stage 2, re-run at batch 32 and batch 512 (3 seeds each). Requires
# logs_valid_tuned/best_lambdas.json to already exist; run_grid_valid.py
# itself errors out clearly (not silently) if it's missing.
#
# TOTAL_COMBOS below assumes all 5 sources got a selected lambda in stage 2
# (5 sources x 2 batches x 3 seeds = 30). If select_lambdas.py's summary
# printed a "[warn] no completed tuning runs for <source>; skipping transfer"
# for ANY source, the real combo list is shorter than 30 -- check
# logs_valid_tuned/best_lambdas.json (should have exactly 5 keys) and lower
# TOTAL_COMBOS to match len(...) before submitting, or some of the higher
# array indices below will fail with an out-of-range error from
# run_grid_valid.py's own argument check.
#
# 2 combos per task -> 15 tasks (0-14), 30/2 exactly, no remainder.

RUNS_PER_TASK=2
TOTAL_COMBOS=30   # must match len(run_grid_valid.build_transfer_combos(best_lambdas)) -- see note above

START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_TASK ))
END=$(( START + RUNS_PER_TASK - 1 ))
if [[ "${END}" -ge "${TOTAL_COMBOS}" ]]; then
  END=$(( TOTAL_COMBOS - 1 ))
fi

echo "Task ${SLURM_ARRAY_TASK_ID} handling combo indices ${START}-${END} of ${TOTAL_COMBOS}"

for i in $(seq "${START}" "${END}"); do
  echo "-> --stage transfer --combo-index ${i}"
  "${PYTHON}" squisher_experiments/run_grid_valid.py --stage transfer --combo-index "${i}"
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished."
