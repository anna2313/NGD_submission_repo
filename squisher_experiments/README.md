# Squisher Boundary Experiment

This directory reconstructs an external continual-learning experiment that
tests whether reusing Adam's second-moment accumulator as an EWC importance
estimate is sensitive to training batch size. It contains no paper-facing
result numbers. A fresh patched checkout and complete rerun are required before
the experiment can support a manuscript claim.

## Exact Experiment

- Upstream: `https://github.com/GMvandeVen/continual-learning.git`
- Base commit: `e6d795a`
- Task: Split MNIST, task-incremental, five contexts.
- Model: MLP 784-400-400-10 (478,410 parameters).
- Training: Adam with state reset at each context, learning rate `1e-3`, and
  256,000 processed examples per context.
- Batch sizes: 128 for the exploratory lambda grid; 32 and 512 for transfer.
- Importance sources: empirical Fisher, raw accumulator, bias-corrected
  accumulator, dataset-size-scaled accumulator, and theorem-corrected
  accumulator, plus no-EWC controls.
- Seeds: 1, 2, and 3; 144 cells in the full driver.

The combined patch is
`source/squisher_from_upstream_e6d795a.patch.gz`. Unlike the superseded patch,
it includes both the local Squisher implementation commit and the later working
tree changes. Its SHA-256 values are:

```text
compressed:   4d9cc8461bd8c7dc170cabd0d1412f60df4160b75f41091070dca458d1e3dd7f
uncompressed: ace907cb631eaec54ea855dc00065a2ce2af8439c81072a242240625fe223ada
```

## Prepare And Smoke

From the NGD repository root:

```bash
bash squisher_experiments/prepare_checkout.sh ../continual-learning-squisher
cd ../continual-learning-squisher
uv venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
```

The upstream requirements are not locked. Before the full submission run,
record `uv pip freeze`, the Python version, CUDA/driver versions, GPU model, and
the NGD and external Git revisions with the logs. Run one cell first:

```bash
.venv/bin/python main.py \
  --experiment splitMNIST --scenario task --iters 2000 --batch 128 \
  --lr 0.001 --optimizer adam_reset --seed 1 --no-save \
  --ewc --reg-strength 100 --fisher-labels true \
  --fisher-source squisher_corrected --squisher-probe-n 512
```

Confirm that training completes, the final average-accuracy line is present,
and every `SQUISHER-FIDELITY` value is finite before scheduling the grid.

## Slurm Rerun And Aggregation

After adapting the resource directives to the target cluster:

```bash
mkdir -p logs
sbatch --export=ALL,CHECKOUT_DIR="$PWD/../continual-learning-squisher" \
  squisher_experiments/run_grid.sbatch
```

The external driver is resumable because it skips logs containing a final
accuracy. After completion:

```bash
cd ../continual-learning-squisher
.venv/bin/python squisher_experiments/parse_results.py
```

Archive the raw logs, `results.csv`, `fidelity.csv`, environment freeze, and
Slurm log together. Check the expected 144 completed cells manually; this
external experiment is deliberately not counted by NGD's internal completion
checker.

## Interpretation And Required Review

The current driver chooses lambda from final benchmark accuracy at batch 128
and transfers it to batches 32 and 512. That is an exploratory, oracle-style
selection procedure, not a validation-selected estimate. A submission should
either report the complete lambda curves, preregister a fixed lambda before the
rerun, or add a genuine validation protocol. Do not headline a selected row as
an unbiased downstream comparison.

The experiment is supporting boundary evidence. Its purpose is to test batch
transfer and accumulator fidelity, not to establish a state-of-the-art
continual-learning method. The following extensions are scientifically useful
only if the clean rerun is stable:

- vary task size and training batch size on separate axes;
- separate finite-time EMA bias correction from the theorem's batch correction;
- compare fixed-lambda and full-curve conclusions;
- move to a larger continual-learning benchmark only after the Split-MNIST
  mechanism result is unambiguous.

A CIFAR-100/ResNet18 Squisher study would be a new experiment. It is not what
this reconstruction runs.
