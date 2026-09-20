#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=merge-smoketest-cifar-trimmed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-01
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=merging_experiments/logs/merge-smoketest-cifar-trimmed-%j.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set -- submit this from the repo root}"

time python merging_experiments/fisher_merging_demo.py --dataset cifar10 \
  --batch_size_a 8000 --batch_size_b 256 --examples_per_model 256000 \
  --learning_rate 0.001 --beta2 0.999 --probe_size 512 --seed 0 --device cuda \
  --output_dir merging_experiments/results/smoketest_trimmed --microbatch_size 1024

echo "Smoke test finished."