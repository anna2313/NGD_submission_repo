#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=merge-aggregate
#SBATCH --partition=compute       # CPU-only: reads result JSONs, writes CSV/markdown
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=merging_experiments/logs/merge-aggregate-%j.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "$(dirname "$(dirname "$(realpath "$0")")")"

# --require-complete makes this fail if any expected result cell is missing,
# which is what we want in a dependency chain: a partial aggregate is worse
# than a clear failure. Note aggregate_results.py derives its expected paths
# from write_sweep_plan.build_commands(), so the plans must match the ones the
# run arrays actually used.
python merging_experiments/aggregate_results.py --suite equal_exposure_trend --require-complete
python merging_experiments/aggregate_results.py --suite unbalanced_shards --require-complete

echo "Aggregation complete. Outputs under review_outputs/merging_experiments/."