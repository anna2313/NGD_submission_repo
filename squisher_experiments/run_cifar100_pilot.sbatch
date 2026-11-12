#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-[01,03,07,08]  # GPUs the ngd torch build (2.9.0+cu126, sm_50..sm_90) cannot run
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00          # longest runs: context 1 with 40,000 steps at b2048 (indices 48-89)
#SBATCH --propagate=NONE
#SBATCH --array=0-89            # 0-47 round 1 (done), 48-56 cosine extension, 57-89 cosine-tail
#SBATCH --output=squisher_experiments/logs/squisher-pilot-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Pilot (2026-11-12): training budget x lr decay on context 1, see run_cifar100_pilot.py.
# Afterwards (login node is fine): python squisher_experiments/run_cifar100_pilot.py --summary
python squisher_experiments/run_cifar100_pilot.py --combo-index "${SLURM_ARRAY_TASK_ID}"
