#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=merge-equal-exposure
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G                 # ESTIMATE, unmeasured. Higher than Experiment 2's 4G because this
                                  # tensorizes an entire 30k-example MNIST shard onto the device up
                                  # front (tensorize() in fisher_merging_demo.py), plus holds 11
                                  # importance dicts in memory. Check seff after the first tasks land.
#SBATCH --time=02:00:00          # ESTIMATE, unmeasured. The script's own docstring claims "minutes"
                                  # on CPU/MPS, but that predates the 256,000-example budget; the
                                  # per-sample probe_fisher loop (512 sequential single-example
                                  # backward passes per model) is the likely bottleneck. Correct after
                                  # the first tasks complete.
#SBATCH --array=0-17
#SBATCH --output=merging_experiments/logs/merge-equal-%A_%a.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

# Run from the repo root: fisher_merging_demo.py's --data_dir and --output_dir
# defaults are both relative paths resolved from there.
cd "$(dirname "$(dirname "$(realpath "$0")")")"

PLAN=merging_experiments/plans/equal_exposure_trend.json

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