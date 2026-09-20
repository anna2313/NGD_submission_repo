"""Write maintained CIFAR-10 Fisher-merging experiment plans.

CIFAR counterpart of write_sweep_plan.py. Same batch pairs and shard-size
sweep, added --dataset cifar10 to every generated command so
fisher_merging_demo.py loads CIFAR10 and builds a ResNet18 instead of LeNet.

One real difference worth knowing: CIFAR10's training set is 50,000 examples,
not MNIST's 60,000. equal_exposure_trend's default shard (n_train // 2) is
therefore 25,000 per model here, not 30,000. unbalanced_shards keeps the same
hardcoded (15_000, 30_000, 45_000) split for model A -- meaning model B gets
35,000 / 20,000 / 5,000 here, a more lopsided complement than MNIST's
45,000 / 30,000 / 15,000. Not adjusted to compensate, so the two datasets'
unbalanced_shards suites are not directly comparable cell-for-cell; flagged
here rather than silently rescaled.
"""

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
    output_dir = f"merging_experiments/results/{suite}_cifar"
    for batch_size_a, batch_size_b, shard_size_a in settings:
        for seed in seeds:
            command = [
                "python",
                "merging_experiments/fisher_merging_demo.py",
                "--dataset", "cifar10",
                "--batch_size_a", str(batch_size_a),
                "--batch_size_b", str(batch_size_b),
                "--examples_per_model", str(examples_per_model),
                "--steps", "2000",
                "--learning_rate", "0.001",
                "--beta2", "0.999",
                "--probe_size", "512",
                "--seed", str(seed),
                "--output_dir", output_dir,
            ]
            if shard_size_a is not None:
                command.extend(["--shard_size_a", str(shard_size_a)])
            commands.append(command)
    return commands


def suite_description(suite):
    if suite == "equal_exposure_trend":
        return (
            "CIFAR-10 counterpart of equal_exposure_trend: three-seed Fisher-merging "
            "trend with 256000 examples per model, ResNet18 instead of LeNet. "
            "Model B fixed at batch 128, model A spans 32--512."
        )
    return (
        "CIFAR-10 counterpart of unbalanced_shards: three-seed unbalanced-shard "
        "diagnostic, both models at batch 128, ResNet18. Model A receives "
        "15000/30000/45000 of CIFAR-10's 50000 training examples -- note this is "
        "NOT the same proportional split as MNIST's version; see module docstring."
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
    output = Path(args.output or f"merging_experiments/plans/{args.suite}_cifar.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "suite": {
                    "name": f"{args.suite}_cifar",
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