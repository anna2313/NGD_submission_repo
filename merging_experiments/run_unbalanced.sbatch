#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=merge-unbalanced
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                 # ESTIMATE, unmeasured. Higher than Experiment 2's 4G because this
                                  # tensorizes an entire 30k-example MNIST shard onto the device up
                                  # front (tensorize() in fisher_merging_demo.py), plus holds 11
                                  # importance dicts in memory. Check seff after the first tasks land.
#SBATCH --time=00:30:00          # MEASURED (2026-09-14): completed tasks took ~3 min each, including
                                  # the heaviest equal-exposure cell (batch_size_a=32 -> 8,000 steps for
                                  # model A plus 2,000 for model B, plus two 512-sample sequential probe
                                  # loops). 10x headroom. Was 02:00:00, a guess made without a smoke
                                  # test. Note this cluster has real node-speed variance (established
                                  # during Experiment 2's batch=4096 work), so keep margin rather than
                                  # trimming to ~5 min.
#SBATCH --array=0-8
#SBATCH --output=merging_experiments/logs/merge-unbalanced-%A_%a.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# Run from the repo root: fisher_merging_demo.py's --data_dir and --output_dir
# defaults are both relative paths resolved from there.
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this from the repo root}"

PLAN=merging_experiments/plans/unbalanced_shards.json

if [[ ! -f "${PLAN}" ]]; then
  printf 'Plan not found: %s\nRun: python merging_experiments/write_sweep_plan.py --suite equal_exposure_trend\n' "${PLAN}" >&2
  exit 2
fi

# 18 commands = 6 batch pairs x 3 seeds (0,1,2). One command per array task.
# Each command is a full (train A, train B, probe, merge 11 schemes, evaluate) run
# writing its own results JSON, so tasks are fully independent and resumable:
# rerunning this array re-does only the jobs whose output JSON is missing.
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
# --device cuda overrides the script's cpu default; the plan does not set it.
${COMMAND} --device cuda

echo "Task ${SLURM_ARRAY_TASK_ID} finished."