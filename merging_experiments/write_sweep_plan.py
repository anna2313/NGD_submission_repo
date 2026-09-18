"""Write maintained controlled Fisher-merging experiment plans."""

import argparse
import json
from pathlib import Path


BATCH_PAIRS = [(32, 128), (64, 128), (128, 128), (256, 128), (512, 128), (32, 512)]
UNBALANCED_SHARD_SIZES_A = (15_000, 30_000, 45_000)


def build_commands(suite="equal_exposure_trend", examples_per_model=256_000, seeds=(0, 1, 2)):
    if suite == "equal_exposure_trend":
        settings = [(batch_a, batch_b, None) for batch_a, batch_b in BATCH_PAIRS]
    elif suite == "unbalanced_shards":
        settings = [(128, 128, shard_size_a) for shard_size_a in UNBALANCED_SHARD_SIZES_A]
    else:
        raise ValueError(f"Unknown merging suite: {suite}")

    commands = []
    output_dir = f"merging_experiments/results/{suite}"
    for batch_size_a, batch_size_b, shard_size_a in settings:
        for seed in seeds:
            command = [
                "python",
                "merging_experiments/fisher_merging_demo.py",
                "--batch_size_a",
                str(batch_size_a),
                "--batch_size_b",
                str(batch_size_b),
                "--examples_per_model",
                str(examples_per_model),
                "--steps",
                "2000",
                "--learning_rate",
                "0.001",
                "--beta2",
                "0.999",
                "--probe_size",
                "512",
                "--seed",
                str(seed),
                "--output_dir",
                output_dir,
            ]
            if shard_size_a is not None:
                command.extend(["--shard_size_a", str(shard_size_a)])
            commands.append(command)
    return commands


def suite_description(suite):
    if suite == "equal_exposure_trend":
        return (
            "Three-seed Fisher-merging trend with 256000 examples per model. "
            "Model B is fixed at batch 128 while model A spans 32--512; "
            "32-vs-512 preserves the original extreme as an equal-exposure anchor."
        )
    return (
        "Three-seed unbalanced-shard Fisher-merging diagnostic. Both models use "
        "batch 128 and equal training exposure; model A receives 15000, 30000, "
        "or 45000 of the 60000 MNIST examples. Every run reports both mean- and "
        "dataset-size-weighted Fisher conventions."
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
    output = Path(args.output or f"merging_experiments/plans/{args.suite}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "suite": {
                    "name": args.suite,
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
