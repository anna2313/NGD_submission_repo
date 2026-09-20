#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=merge-write-plans-cifar-trimmed
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=merging_experiments/logs/merge-write-plans-cifar-trimmed-%j.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this from the repo root}"

mkdir -p merging_experiments/logs

python merging_experiments/write_sweep_plan_cifar_trimmed.py --suite equal_exposure_trend
python merging_experiments/write_sweep_plan_cifar_trimmed.py --suite unbalanced_shards

grep -q "plans/equal_exposure_trend_cifar_trimmed.json" merging_experiments/run_equal_cifar_trimmed.sbatch \
  || { echo "run_equal_cifar_trimmed.sbatch does not point at equal_exposure_trend_cifar_trimmed.json" >&2; exit 2; }
grep -q "plans/unbalanced_shards_cifar_trimmed.json" merging_experiments/run_unbalanced_cifar_trimmed.sbatch \
  || { echo "run_unbalanced_cifar_trimmed.sbatch does not point at unbalanced_shards_cifar_trimmed.json" >&2; exit 2; }

python - <<'EOF'
import json, re, sys
for plan, script in [
    ("merging_experiments/plans/equal_exposure_trend_cifar_trimmed.json", "merging_experiments/run_equal_cifar_trimmed.sbatch"),
    ("merging_experiments/plans/unbalanced_shards_cifar_trimmed.json", "merging_experiments/run_unbalanced_cifar_trimmed.sbatch"),
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

echo "Trimmed CIFAR plans written and verified."