#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-cifar100-pretrain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-01  # V100 node: the ngd torch build's cuDNN cannot run convolutions there
                                  # ("GET was unable to find an engine", 2026-10-19, job 9774051). Titan RTX
                                  # nodes (e.g. gpu-biomed-15) work. Extend this list if other V100 nodes exist.
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G                 # ESTIMATE -- CIFAR-10 in memory plus a small CNN; check seff after the first task.
#SBATCH --time=03:00:00          # ESTIMATE -- 100 epochs of CIFAR-10 (50k images, batch 256) on the C3-5x16-bn
                                  # encoder + FC head; expect well under an hour on a V100, with margin for slow nodes.
#SBATCH --array=1-5
#SBATCH --output=squisher_experiments/logs/squisher-cifar100-pretrain-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Stage 0 of 4: pretrain the conv encoder on CIFAR-10, one per seed, exactly as
# upstream's all_results.sh does for its CIFAR-100 experiments:
#
#   ./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=S
#
# Writes store/models/C3-5x16-bn-e100-s<seed> (393,088 parameters -- the count the
# paper quotes for its "5-layer CNN"), which the continual-learning runs then load
# frozen via --pre-convE --freeze-convE --seed-to-ltag. The array index IS the seed
# (1..5 = FINAL_SEEDS in run_cifar100_paper.py; TUNE_SEEDS 1..3 are a subset).
#
# Idempotent: skips if the checkpoint already exists. NOTE: this does not touch the
# pre-existing unsuffixed store/models/C3-5x16-bn-e100, which nothing here uses.
#
# Once ALL 5 tasks finish: submit run_cifar100_tune.sbatch
# (`sbatch --dependency=afterok:<this_job_id> --export=ALL,CHECKOUT_DIR=... run_cifar100_tune.sbatch`).

SEED=${SLURM_ARRAY_TASK_ID}
CKPT="store/models/C3-5x16-bn-e100-s${SEED}"

if [[ -f "${CKPT}" ]]; then
  echo "[skip] ${CKPT} already exists"
  exit 0
fi

echo "-> pretraining encoder for seed ${SEED}"
python main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment \
  --convE-stag=e100 --seed-to-stag --seed="${SEED}"

test -f "${CKPT}" || { echo "[error] ${CKPT} was not written"; exit 1; }
echo "Task ${SLURM_ARRAY_TASK_ID} finished: ${CKPT}"
