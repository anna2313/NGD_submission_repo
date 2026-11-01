#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=squisher-cifar100-final
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu-biomed-01  # V100 node: the ngd torch build's cuDNN cannot run convolutions there
                                  # (see pretrain_cifar100.sbatch). Extend this list if other V100 nodes exist.
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G                 # ESTIMATE -- same model as the tune stage; correct from seff once measured.
#SBATCH --time=01:00:00          # MEASURED on the tune stage (job 9774069): 5-12 min per run; the full-data Fisher
                                  # pass here is over 5,000 instead of 3,750 examples, still minutes. Ample margin.
#SBATCH --array=0-24
#SBATCH --output=squisher_experiments/logs/squisher-cifar100-final-%A_%a.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ngd
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}

cd "${CHECKOUT_DIR}"

# Stage 3 of 4: the reported runs -- all 5 configs (none_adam, none_adamreset,
# fisher_adam, squisher_adam, squisher_adamreset) x 5 seeds = 25 combos, on the
# FULL per-context training data (n = 5,000, no hold-out) at the lambda selected
# in stage 2, scored on the untouched test set. These are the numbers to put
# against the paper's Table 8 (Fisher 75.96 / Squisher 75.30 / baseline 61.82).
#
# Requires store/models/C3-5x16-bn-e100-s{1..5} and logs_cifar100_paper/
# best_lambdas.json; run_cifar100_paper.py errors out clearly if either is missing.
#
# TOTAL_COMBOS is read from the script so that it shrinks automatically if
# select_lambdas printed "[warn] no completed tuning runs for <config>" -- in that
# case some of the higher array indices below simply do nothing.
#
# Afterwards: python squisher_experiments/parse_results_cifar100.py

RUNS_PER_TASK=1
TOTAL_COMBOS=$(python squisher_experiments/run_cifar100_paper.py --stage final --count)

START=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_TASK ))
END=$(( START + RUNS_PER_TASK - 1 ))
if [[ "${END}" -ge "${TOTAL_COMBOS}" ]]; then
  END=$(( TOTAL_COMBOS - 1 ))
fi
if [[ "${START}" -ge "${TOTAL_COMBOS}" ]]; then
  echo "Task ${SLURM_ARRAY_TASK_ID}: nothing to do (only ${TOTAL_COMBOS} combos)"
  exit 0
fi

echo "Task ${SLURM_ARRAY_TASK_ID} handling combo indices ${START}-${END} of ${TOTAL_COMBOS}"

for i in $(seq "${START}" "${END}"); do
  echo "-> --stage final --combo-index ${i}"
  python squisher_experiments/run_cifar100_paper.py --stage final --combo-index "${i}"
done

echo "Task ${SLURM_ARRAY_TASK_ID} finished."
