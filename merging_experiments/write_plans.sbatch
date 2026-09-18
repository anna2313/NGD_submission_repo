#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=merge-write-plans
#SBATCH --partition=compute       # CPU-only: writes two small JSON files, no training
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=merging_experiments/logs/merge-write-plans-%j.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "$(dirname "$(dirname "$(realpath "$0")")")"

mkdir -p merging_experiments/logs

python merging_experiments/write_sweep_plan.py --suite equal_exposure_trend
python merging_experiments/write_sweep_plan.py --suite unbalanced_shards

# Fail loudly here rather than letting the run arrays silently pick the wrong
# plan: run_unbalanced.sbatch shipped once with the equal-exposure path, which
# would have written 9 duplicate equal-exposure results while leaving
# results/unbalanced_shards/ empty -- invisible in squeue, only surfacing later
# as missing data.
grep -q "plans/equal_exposure_trend.json" merging_experiments/run_equal.sbatch \
  || { echo "run_equal.sbatch does not point at equal_exposure_trend.json" >&2; exit 2; }
grep -q "plans/unbalanced_shards.json" merging_experiments/run_unbalanced.sbatch \
  || { echo "run_unbalanced.sbatch does not point at unbalanced_shards.json" >&2; exit 2; }

# Confirm the array sizes in the .sbatch files match the plans just written.
python - <<'EOF'
import json, re, sys
for plan, script, expected_last in [
    ("merging_experiments/plans/equal_exposure_trend.json", "merging_experiments/run_equal.sbatch", None),
    ("merging_experiments/plans/unbalanced_shards.json", "merging_experiments/run_unbalanced.sbatch", None),
]:
    n = len(json.load(open(plan))["commands"])
    text = open(script).read()
    m = re.search(r"#SBATCH --array=0-(\d+)", text)
    if m is None:
        sys.exit(f"{script}: no --array=0-N line found")
    last = int(m.group(1))
    if last != n - 1:
        sys.exit(f"{script}: --array=0-{last} but {plan} has {n} commands (expected 0-{n-1})")
    print(f"ok: {script} --array=0-{last} matches {n} commands in {plan}")
EOF

echo "Plans written and verified."