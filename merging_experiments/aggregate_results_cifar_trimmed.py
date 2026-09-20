"""Validate and aggregate the trimmed CIFAR-10 Fisher-merging suites.

Trimmed counterpart of aggregate_results_cifar.py -- identical logic, imports
from write_sweep_plan_cifar_trimmed instead of write_sweep_plan_cifar, and
defaults to the _cifar_trimmed-suffixed results/output paths.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from merging_experiments.write_sweep_plan_cifar_trimmed import build_commands
from merging_experiments.run_sweep import output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("equal_exposure_trend", "unbalanced_shards"),
        default="equal_exposure_trend",
    )
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Defaults to review_outputs/merging_experiments/<suite>_cifar_trimmed.",
    )
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def expected_paths(suite, results_dir):
    paths = []
    for command in build_commands(suite):
        path = output_path(command)
        if path is None:
            raise RuntimeError(f"Could not derive output path for: {command}")
        paths.append(results_dir / path.name)
    return paths


def load_results(suite, results_dir, require_complete):
    expected = expected_paths(suite, results_dir)
    missing = [path for path in expected if not path.exists()]
    if missing and require_complete:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        raise ValueError(f"Missing {len(missing)} expected results:\n{preview}")
    results = []
    failures = []
    for path in expected:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            failures.append(f"{path}: {error}")
            continue
        if payload.get("status") != "ok":
            failures.append(f"{path}: status={payload.get('status')}")
            continue
        payload["_path"] = str(path)
        results.append(payload)
    if failures:
        raise ValueError("Invalid or failed results:\n" + "\n".join(failures[:10]))
    if not results:
        raise ValueError(f"No valid results found in {results_dir}")
    return results, len(expected), len(missing)


def _setting_key(payload):
    config = payload["config"]
    return (
        int(config["batch_size_a"]),
        int(config["batch_size_b"]),
        int(config["shard_size_a_resolved"]),
        int(config["shard_size_b_resolved"]),
    )


def _mean_std(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    return float(values.mean()), float(values.std(ddof=0))


def aggregate(results):
    grouped = defaultdict(list)
    for payload in results:
        grouped[_setting_key(payload)].append(payload)
    rows = []
    for (batch_a, batch_b, shard_a, shard_b), group in sorted(grouped.items()):
        row = {
            "batch_size_a": batch_a,
            "batch_size_b": batch_b,
            "shard_size_a": shard_a,
            "shard_size_b": shard_b,
            "n_seeds": len(group),
        }
        endpoint_names = sorted(group[0]["endpoint_accuracy"])
        for name in endpoint_names:
            mean, std = _mean_std(payload["endpoint_accuracy"][name] for payload in group)
            row[f"{name}_accuracy_mean"] = mean
            row[f"{name}_accuracy_std"] = std
        scheme_names = sorted(group[0]["merges"])
        for scheme in scheme_names:
            accuracy_mean, accuracy_std = _mean_std(
                payload["merges"][scheme]["test_accuracy"] for payload in group
            )
            share_mean, share_std = _mean_std(
                payload["merges"][scheme]["mean_weight_share_model_a"] for payload in group
            )
            row[f"{scheme}_accuracy_mean"] = accuracy_mean
            row[f"{scheme}_accuracy_std"] = accuracy_std
            row[f"{scheme}_share_a_mean"] = share_mean
            row[f"{scheme}_share_a_std"] = share_std
        rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, suite, rows, expected, missing):
    # Require BOTH keys, not just _accuracy_mean -- the endpoint accuracies
    # (model_a, model_b) also end in "_accuracy_mean" but never get a
    # _share_a_mean (a single trained model has no merge-weight share), so
    # checking _accuracy_mean alone swept them up as if they were schemes.
    schemes = sorted({
        key[: -len("_accuracy_mean")]
        for row in rows for key in row
        if key.endswith("_accuracy_mean")
        and key[: -len("_accuracy_mean")] + "_share_a_mean" in row
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {suite}_cifar_trimmed\n\n")
        handle.write(
            f"Loaded {sum(row['n_seeds'] for row in rows)}/{expected} cells; "
            f"{missing} missing. Values are mean +/- population SD across seeds.\n\n"
        )
        handle.write("| Batch A/B | Shards A/B | Scheme | Accuracy | Share assigned to A |\n")
        handle.write("| --- | --- | --- | ---: | ---: |\n")
        for row in rows:
            for scheme in schemes:
                if f"{scheme}_accuracy_mean" not in row:
                    continue
                handle.write(
                    f"| {row['batch_size_a']}/{row['batch_size_b']} | "
                    f"{row['shard_size_a']}/{row['shard_size_b']} | `{scheme}` | "
                    f"{row[f'{scheme}_accuracy_mean']:.4f} +/- "
                    f"{row[f'{scheme}_accuracy_std']:.4f} | "
                    f"{row[f'{scheme}_share_a_mean']:.4f} +/- "
                    f"{row[f'{scheme}_share_a_std']:.4f} |\n"
                )


def main():
    args = parse_args()
    results_dir = Path(args.results_dir or f"merging_experiments/results/{args.suite}_cifar_trimmed")
    prefix = Path(args.output_prefix or f"review_outputs/merging_experiments/{args.suite}_cifar_trimmed")
    results, expected, missing = load_results(args.suite, results_dir, args.require_complete)
    rows = aggregate(results)
    write_csv(prefix.with_suffix(".csv"), rows)
    write_markdown(prefix.with_suffix(".md"), args.suite, rows, expected, missing)
    print(f"Loaded {len(results)}/{expected} result cells ({missing} missing)")
    print(f"Wrote {prefix.with_suffix('.csv')}")
    print(f"Wrote {prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()