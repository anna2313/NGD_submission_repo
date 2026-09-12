# Squisher batch-size grid (NGD paper, WS3)

Tests whether recycling Adam's `exp_avg_sq` as an EWC importance estimate
("Squisher"; Li et al., ICML 2025) is batch-size sensitive, as predicted by
Theorem 1 of the NGD paper, and whether the theorem's `m(n-1)/(n-m)` correction
removes that sensitivity. Full spec: `NGD/WS3_SQUISHER_NOTES.md`.

Upstream: `GMvandeVen/continual-learning` pinned at commit `e6d795a`.
Modifications (branch `squisher-batchsize`):
- `--fisher-source {empirical, squisher_raw, squisher_biascorrected, squisher_nscaled, squisher_corrected}`
  (options.py, main.py, param_stamp.py)
- `estimate_fisher_from_accumulator` + empirical-Fisher fidelity probe
  (models/cl/continual_learner.py); prints `SQUISHER-FIDELITY` diagnostics.

## Protocol pins

- Split MNIST, task-incremental, 5 contexts, MLP 784-400-400-10 (478,410 params).
- Adam (`adam_reset`: state reset at each context start, so the harvested
  accumulator reflects only the just-finished context), lr 1e-3, and an equal
  256,000-example budget per context (iterations scale as budget / batch size).
- Empirical-Fisher importance convention: ground-truth labels
  (`--fisher-labels true`), full pass, batch 1, per-sample mean. Accumulator
  fidelity uses a separate 512-example probe.
- Lambda tuned once per source at batch 128 over {1e1 ... 1e7}, then reused
  frozen at batch 32 and 512; 3 seeds everywhere. 144 runs total. The
  bias-corrected-only source isolates finite-time EMA initialization bias from
  the theorem's batch-size scale correction.

## Run

```bash
python squisher_experiments/run_grid.py    # resumable; skips completed logs
python squisher_experiments/parse_results.py
```

On the AWS box: rsync this directory, `cd continual-learning-squisher`, run
under nohup with the NGD repo's venv python. MNIST downloads to ./store/datasets.
Fidelity probes report the dimensionless `||m_hat^2|| / ||F_probe||`
stationarity diagnostic.
