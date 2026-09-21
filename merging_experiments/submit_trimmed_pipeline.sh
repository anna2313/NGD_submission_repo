#!/usr/bin/env bash
# merging_experiments/submit_trimmed_pipeline.sh
# Full CIFAR trimmed-grid pipeline: plans -> train (both suites) -> aggregate -> all plots.
# Run from the repo root: bash merging_experiments/submit_trimmed_pipeline.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

mkdir -p merging_experiments/logs
mkdir -p merging_experiments/results/cifar_trimmed/unbalanced_shards
mkdir -p merging_experiments/results/cifar_trimmed/equal_exposure_trend

# --- pre-flight: confirm the shard_accuracy fix actually made it to the cluster ---
if ! grep -q "shard_accuracy" merging_experiments/fisher_merging_demo.py; then
  echo "ERROR: fisher_merging_demo.py has no 'shard_accuracy' key -- git pull first." >&2
  exit 1
fi
if ! grep -q "def main" merging_experiments/plot_shard_performance.py 2>/dev/null; then
  echo "ERROR: merging_experiments/plot_shard_performance.py not found -- copy it in first." >&2
  exit 1
fi

# --- training + aggregate chain (sbatch, GPU work) ---
plans_id=$(sbatch --parsable merging_experiments/write_plans_cifar_trimmed.sbatch)
echo "plans_id=$plans_id"

equal_id=$(sbatch --parsable --dependency=afterok:${plans_id} --exclude=gpu-biomed-01 \
  merging_experiments/run_equal_cifar_trimmed.sbatch)
unbal_id=$(sbatch --parsable --dependency=afterok:${plans_id} --exclude=gpu-biomed-01 \
  merging_experiments/run_unbalanced_cifar_trimmed.sbatch)
echo "equal_id=$equal_id unbal_id=$unbal_id"

agg_id=$(sbatch --parsable --dependency=afterok:${equal_id}:${unbal_id} \
  merging_experiments/aggregate_cifar_trimmed.sbatch)
echo "agg_id=$agg_id"

# --- plotting (srun, CPU-only, each waits in queue for aggregate via --dependency) ---
srun --partition=compute --mem=1G --time=00:05:00 --dependency=afterok:${agg_id} \
  --job-name=merge-plot-results-cifar-trimmed \
  --output=merging_experiments/logs/merge-plot-results-cifar-trimmed-%j.out bash -c "
    source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ngd &&
    python merging_experiments/plot_merge_results.py \
      --suite unbalanced_shards \
      --csv review_outputs/merging_experiments/unbalanced_shards_cifar_trimmed.csv \
      --output-dir merging_experiments/results/cifar_trimmed/unbalanced_shards &&
    python merging_experiments/plot_merge_results.py \
      --suite equal_exposure_trend \
      --csv review_outputs/merging_experiments/equal_exposure_trend_cifar_trimmed.csv \
      --output-dir merging_experiments/results/cifar_trimmed/equal_exposure_trend
  " &

srun --partition=compute --mem=1G --time=00:05:00 --dependency=afterok:${agg_id} \
  --job-name=merge-plot-batch-ratio-cifar-trimmed \
  --output=merging_experiments/logs/merge-plot-batch-ratio-cifar-trimmed-%j.out bash -c "
    source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ngd &&
    python merging_experiments/plot_batch_ratio.py \
      --results-dir merging_experiments/results/equal_exposure_trend_cifar_trimmed \
      --output-dir merging_experiments/results/cifar_trimmed
  " &

srun --partition=compute --mem=1G --time=00:05:00 --dependency=afterok:${agg_id} \
  --job-name=merge-plot-shard-perf-cifar-trimmed \
  --output=merging_experiments/logs/merge-plot-shard-perf-cifar-trimmed-%j.out bash -c "
    source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ngd &&
    python merging_experiments/plot_shard_performance.py --suite equal_exposure_trend \
      --output-dir merging_experiments/results/cifar_trimmed &&
    python merging_experiments/plot_shard_performance.py --suite unbalanced_shards \
      --output-dir merging_experiments/results/cifar_trimmed
  " &

echo ""
echo "All jobs submitted. Plotting jobs are queued behind aggregate (job $agg_id) and will"
echo "start automatically once it completes. Check status with: squeue -u \$USER"