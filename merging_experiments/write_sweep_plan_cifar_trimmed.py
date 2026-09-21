"""Write the trimmed CIFAR-10 Fisher-merging experiment plans.

Trimmed counterpart of write_sweep_plan_cifar.py -- same fisher_merging_demo.py,
same --dataset cifar10 flag, but a smaller, more targeted grid than the full
CIFAR sweep, and a different reference batch.

Design (2026-09-18): batch=256 is the fixed reference throughout -- the same
batch CIFAR Squisher tunes at (TUNE_BATCH=256) -- rather than the full grid's
128, so results here are directly comparable in spirit to the CIFAR EWC grid.

equal_exposure_trend: model A in {32, 256, 2048, 8000}, model B fixed at 256,
equal ~25000/25000 shards. batch=512 and 64/128 deliberately dropped from the
full grid. 8000 chosen over the initially-proposed 8192 specifically for
divisibility: 256_000 = 2**11 * 5**3 and 8000 = 2**6 * 5**3, so
256_000 / 8000 = 32 exactly -- the full grid's 256,000-example budget works
unchanged.

unbalanced_shards: batch fixed at 256 for both models (not the full grid's
128). Model A's shard in {1000, 5000, 15000, 25000} out of CIFAR-10's 50,000
training examples (model B gets the complement: 49000, 45000, 35000, 25000).
The 1000-example shard pushes m/n to 25.6%, deliberately no longer negligible
-- the full grid's smallest split (15,000) never got below ~1%.

MIN_STEPS = 1000 in fisher_merging_demo.py floors optimizer steps so
batch=2048 and batch=8000 -- which would otherwise get only 125 and 32 steps
under the 256,000-example budget, far short of beta2=0.999's ~1,000-step EMA
window -- still get an adequately burned-in exp_avg_sq. batch=32 and 256 are
unaffected (8,000 and exactly 1,000 steps respectively, already >= the floor).
This floor applies to every plan that calls fisher_merging_demo.py, including
the full grid above -- it was added for this trimmed grid's sake but is not
specific to it.
"""

import argparse
import json
from pathlib import Path

EXAMPLES_PER_MODEL = 256_000  # divisible by every batch below: 32, 256, 2048, 8000
BATCH_PAIRS = [(32, 256), (256, 256), (2048, 256), (8000, 256)]
UNBALANCED_SHARD_SIZES_A = (1_000, 5_000, 15_000, 25_000)
UNBALANCED_BATCH = 256
MICROBATCH_CAP = 1024  # confirmed necessary via smoke test (2026-09-18): batch_size_a=8000 through
                        # ResNet18 in one forward pass OOM'd a 24GB GPU (~22.7GB used, crashed in
                        # layer2). Chunking to 1024 fixed it (measured: 29m25s, 1.15GB host RAM, no
                        # GPU OOM). Mathematically identical to one large-batch step -- see
                        # fisher_merging_demo.py's train_model() for why.


def build_commands(suite="equal_exposure_trend", examples_per_model=EXAMPLES_PER_MODEL, seeds=(0, 1, 2)):
    if suite == "equal_exposure_trend":
        settings = [(batch_a, batch_b, None) for batch_a, batch_b in BATCH_PAIRS]
    elif suite == "unbalanced_shards":
        settings = [(UNBALANCED_BATCH, UNBALANCED_BATCH, shard_size_a) for shard_size_a in UNBALANCED_SHARD_SIZES_A]
    else:
        raise ValueError(f"Unknown merging suite: {suite}")

    commands = []
    output_dir = f"merging_experiments/results/{suite}_cifar_trimmed"
    for batch_size_a, batch_size_b, shard_size_a in settings:
        for seed in seeds:
            command = [
                    "python",
                    "merging_experiments/fisher_merging_demo.py",
                    "--dataset", "cifar10",
                    "--batch_size_a", str(batch_size_a),
                    "--batch_size_b", str(batch_size_b),
                    "--examples_per_model", str(examples_per_model),
                    "--min_steps", "300",
                    "--steps", "2000",
                    "--learning_rate", "0.001",
                    "--beta2", "0.999",
                    "--probe_size", "512",
                    "--seed", str(seed),
                    "--output_dir", output_dir,
                ]
            if shard_size_a is not None:
                command.extend(["--shard_size_a", str(shard_size_a)])
            if max(batch_size_a, batch_size_b) > MICROBATCH_CAP:
                command.extend(["--microbatch_size", str(MICROBATCH_CAP)])
            commands.append(command)
    return commands


def suite_description(suite):
    if suite == "equal_exposure_trend":
        return (
            "Trimmed CIFAR-10 equal-exposure trend: model B fixed at batch 256 "
            "(matches CIFAR Squisher's TUNE_BATCH), model A in {32, 256, 2048, "
            "8000}. ResNet18, ~equal 25000/25000 shards. MIN_STEPS=1000 floor "
            "applied for batch=2048/8000."
        )
    return (
        "Trimmed CIFAR-10 unbalanced-shard diagnostic: both models fixed at "
        "batch 256. Model A's shard in {1000, 5000, 15000, 25000} out of "
        "CIFAR-10's 50000 training examples (model B gets the complement: "
        "49000, 45000, 35000, 25000). The 1000-example shard pushes m/n to "
        "25.6%, deliberately no longer negligible."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("equal_exposure_trend", "unbalanced_shards"),
        default="equal_exposure_trend",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    commands = build_commands(args.suite)
    output = Path(args.output or f"merging_experiments/plans/{args.suite}_cifar_trimmed.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "suite": {
                    "name": f"{args.suite}_cifar_trimmed",
                    "description": suite_description(args.suite),
                },
                "n_commands": len(commands),
                "commands": commands,
            },
            indent=2,
        )
    )
    print(f"Wrote {len(commands)} commands to {output}")


if __name__ == "__main__":
    main()