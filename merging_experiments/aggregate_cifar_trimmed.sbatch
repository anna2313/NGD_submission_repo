#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=merge-aggregate-cifar-trimmed
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=merging_experiments/logs/merge-aggregate-cifar-trimmed-%j.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this from the repo root}"

python merging_experiments/aggregate_results_cifar_trimmed.py --suite equal_exposure_trend --require-complete
python merging_experiments/aggregate_results_cifar_trimmed.py --suite unbalanced_shards --require-complete

echo "Trimmed CIFAR aggregation complete."