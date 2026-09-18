# Controlled Fisher-Merging Experiments

This directory contains runnable protocol and aggregation code, not verified
paper results. Every manuscript-facing value must come from a clean rerun of
the reviewed branch. Do not restore the earlier local numbers as evidence.

## Equal-Exposure Trend

The required 18-cell suite trains two shared-initialization LeNets on disjoint
random 30k-example MNIST halves with AdamW, learning rate `1e-3`, and
`beta2=0.999`. Both models process exactly 256,000 examples. Model B uses batch
128; model A uses batch 32, 64, 128, 256, or 512. A 32-versus-512 pair retains
the most extreme comparison. Every pair has three seeds.

The experiment compares uniform merging, independent empirical-Fisher probes,
the theorem-predicted raw scaling of those probes, actual raw Adam
accumulators, and theorem-corrected accumulators. It records both test accuracy
and the mean coordinatewise merge share assigned to model A.

```bash
uv run python merging_experiments/write_sweep_plan.py \
  --suite equal_exposure_trend
uv run ngd-run-sweep \
  --plan merging_experiments/plans/equal_exposure_trend.json \
  --workers 1 --device cpu --dry-run
```

After the Slurm run:

```bash
uv run python merging_experiments/aggregate_results.py \
  --suite equal_exposure_trend --require-complete
```

The primary question is whether corrected cross-model importance ratios track
the independent probe-Fisher ratios better than raw accumulators as training
batch size changes. Accuracy is a secondary check: this controlled random-half
task may favor nearly uniform averaging, so a calibration improvement need not
produce an accuracy improvement.

## Unbalanced Shards

The implemented optional nine-cell suite holds both training batches at 128,
gives model A 15k, 30k, or 45k of the 60k MNIST examples, and retains equal
processed-example budgets. Each result reports both mean-Fisher schemes and
parallel `*_sum` schemes that multiply each importance by source shard size.

```bash
uv run python merging_experiments/write_sweep_plan.py --suite unbalanced_shards
uv run ngd-run-sweep \
  --plan merging_experiments/plans/unbalanced_shards.json \
  --workers 1 --device cpu --dry-run
uv run python merging_experiments/aggregate_results.py \
  --suite unbalanced_shards --require-complete
```

This tests a different normalization choice from the batch correction. The
mean convention treats source tasks equally; the sum convention represents
accumulated likelihood precision. Neither is universally correct, so the paper
must state which merging objective motivates the chosen convention.

## Evidence Gates

- Every expected cell is present, successful, finite, and provenance-bearing.
- Three seeds contribute to every aggregate; tables use population SD
  (`ddof=0`) consistently with the maintained aggregators.
- Endpoint quality is comparable enough that merge-share changes are not
  explained only by failed source training.
- Corrected-to-probe cosine and norm diagnostics are inspected, not only the
  scalar mean merge share.
- Claims distinguish relative calibration from downstream accuracy.

If the unbalanced result is clear, vary shard size and training batch size on
separate axes before considering an interaction grid. A semantic-task or
large-model merging benchmark is a later external-validity decision, not part
of the minimum reconstruction.
