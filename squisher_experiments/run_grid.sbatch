#!/usr/bin/env bash

# Cluster resource names and limits are site-specific; inspect before use.
#SBATCH --job-name=ngd-squisher
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=logs/squisher-%j.out

set -euo pipefail

: "${CHECKOUT_DIR:?Set CHECKOUT_DIR to the patched external checkout}"
PYTHON="${PYTHON:-${CHECKOUT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python executable not found: %s\n' "${PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${CHECKOUT_DIR}/squisher_experiments/run_grid.py" ]]; then
  printf 'Patched Squisher driver not found under %s\n' "${CHECKOUT_DIR}" >&2
  exit 2
fi

cd "${CHECKOUT_DIR}"
exec "${PYTHON}" squisher_experiments/run_grid.py
