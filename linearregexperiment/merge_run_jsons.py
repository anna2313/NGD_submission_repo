import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Expected run file name format, e.g. EFAdam_0.9_10_0.json
RUN_FILE_PATTERN = re.compile(
    r"^(?P<optimizer>[^_]+)_(?P<beta2>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_(?P<batch>\d+)_(?P<run>\d+)\.json$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-run trainer JSON files into a single summary JSON "
            "that plot_results.py can consume."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="linearregexperiment/results",
        help="Directory containing per-run JSON files",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help="Only merge files for this optimizer (default: auto-detect single optimizer)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output summary JSON path (default: linearregexperiment/results/results_<optimizer>.json)",
    )
    parser.add_argument(
        "--drop_bad_runs",
        action="store_true",
        help="Exclude runs with final_loss > bad_run_loss_threshold from merged similarity statistics",
    )
    parser.add_argument(
        "--bad_run_loss_threshold",
        type=float,
        default=0.01,
        help="Loss threshold used with --drop_bad_runs (default: 0.01)",
    )
    return parser.parse_args()


def safe_float(value):
    if value is None:
        return np.nan
    if isinstance(value, (float, int)):
        return float(value)
    return float(value)


def load_run_record(json_path):
    match = RUN_FILE_PATTERN.match(json_path.name)
    if match is None:
        return None

    optimizer = match.group("optimizer")
    beta2 = float(match.group("beta2"))
    batch_size = int(match.group("batch"))
    run = int(match.group("run"))

    with open(json_path, "r") as handle:
        payload = json.load(handle)

    fisher_approximations = payload.get("fisher_approximations", {})
    similarities = {}
    for fisher_type, fisher_payload in fisher_approximations.items():
        similarities[fisher_type] = safe_float(
            fisher_payload.get("cosine_similarity_with_exp_avg_sq")
        )

    return {
        "optimizer": optimizer,
        "beta2": beta2,
        "batch_size": batch_size,
        "run": run,
        "final_loss": safe_float(payload.get("final_loss")),
        "cosine_similarity_emp_with_adam": safe_float(
            payload.get("cosine_similarity_emp_with_adam")
        ),
        "similarities": similarities,
    }


def nanmean(values):
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def nanstd(values):
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanstd(arr))


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    all_records = []
    ignored_files = []
    for json_path in sorted(input_dir.glob("*.json")):
        record = load_run_record(json_path)
        if record is None:
            ignored_files.append(json_path.name)
            continue
        all_records.append(record)

    if not all_records:
        raise RuntimeError(
            f"No run JSON files found in {input_dir} matching '<optimizer>_<beta2>_<batch>_<run>.json'"
        )

    optimizers = sorted({record["optimizer"] for record in all_records})
    if args.optimizer is None:
        if len(optimizers) != 1:
            raise RuntimeError(
                "Multiple optimizers detected. Provide --optimizer to select one: "
                + ", ".join(optimizers)
            )
        optimizer = optimizers[0]
    else:
        optimizer = args.optimizer

    records = [record for record in all_records if record["optimizer"] == optimizer]
    if not records:
        raise RuntimeError(f"No records found for optimizer '{optimizer}'")

    if args.drop_bad_runs:
        records_for_summary = [
            record
            for record in records
            if math.isnan(record["final_loss"])
            or record["final_loss"] <= args.bad_run_loss_threshold
        ]
    else:
        records_for_summary = records

    if not records_for_summary:
        raise RuntimeError(
            "No records left for merged summary after applying --drop_bad_runs "
            f"with threshold {args.bad_run_loss_threshold}"
        )

    beta2_values = sorted({record["beta2"] for record in records_for_summary})
    batch_size_values = sorted({record["batch_size"] for record in records_for_summary})

    fisher_types = set()
    for record in records_for_summary:
        fisher_types.update(record["similarities"].keys())
    fisher_types = sorted(fisher_types)

    grouped = defaultdict(list)
    for record in records_for_summary:
        grouped[(record["beta2"], record["batch_size"])].append(record)

    result_entries = {}
    for beta2 in beta2_values:
        for batch_size in batch_size_values:
            key_tuple = (beta2, batch_size)
            runs = grouped.get(key_tuple, [])

            mean_similarities = {}
            std_similarities = {}
            for fisher_type in fisher_types:
                fisher_values = [
                    run["similarities"].get(fisher_type, np.nan) for run in runs
                ]
                mean_similarities[fisher_type] = nanmean(fisher_values)
                std_similarities[fisher_type] = nanstd(fisher_values)

            emp_with_adam_values = [
                run.get("cosine_similarity_emp_with_adam", np.nan) for run in runs
            ]
            mean_cosine_similarity_emp_with_adam = nanmean(emp_with_adam_values)
            std_cosine_similarity_emp_with_adam = nanstd(emp_with_adam_values)

            result_entries[str(key_tuple)] = {
                "mean_similarities": mean_similarities,
                "std_similarities": std_similarities,
                "mean_cosine_similarity_emp_with_adam": mean_cosine_similarity_emp_with_adam,
                "std_cosine_similarity_emp_with_adam": std_cosine_similarity_emp_with_adam,
            }

    n_runs_per_config = {}
    for beta2 in beta2_values:
        for batch_size in batch_size_values:
            n_runs_per_config[str((beta2, batch_size))] = len(
                grouped.get((beta2, batch_size), [])
            )

    # Prepare individual loss records
    loss_records = []
    for record in records:
        loss_records.append(
            {
                "optimizer": record["optimizer"],
                "beta2": record["beta2"],
                "batch_size": record["batch_size"],
                "run": record["run"],
                "final_loss": record["final_loss"],
            }
        )

    summary = {
        "optimizer": optimizer,
        "drop_bad_runs": args.drop_bad_runs,
        "bad_run_loss_threshold": args.bad_run_loss_threshold,
        "n_runs_total": len(records),
        "n_runs_used_in_summary": len(records_for_summary),
        "n_runs_dropped": len(records) - len(records_for_summary),
        "beta2_values": beta2_values,
        "batch_size_values": batch_size_values,
        "results": result_entries,
        "n_runs_per_config": n_runs_per_config,
    }

    if args.output is None:
        output_path = input_dir / f"results_{optimizer}.json"
    else:
        output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote summary JSON: {output_path}")

    # Write losses JSON file
    losses_output_path = output_path.parent / f"losses_{optimizer}.json"
    losses_summary = {
        "optimizer": optimizer,
        "losses": loss_records,
    }
    with open(losses_output_path, "w") as handle:
        json.dump(losses_summary, handle, indent=2)
    print(f"Wrote losses JSON: {losses_output_path}")

    if ignored_files:
        print(
            "Ignored non-run JSON files: "
            + ", ".join(ignored_files[:10])
            + (" ..." if len(ignored_files) > 10 else "")
        )


if __name__ == "__main__":
    main()
