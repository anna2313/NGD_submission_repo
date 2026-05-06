import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Supported run file name formats:
# 1) <optimizer>_<beta2>_<batch>_<run>_<lr>.json
# 2) <optimizer>_<beta2>_<batch>_<lr>.json
# 3) <optimizer>_<beta2>_<batch>_<run>.json (lr taken from payload when available)
RUN_FILE_PATTERNS = [
    re.compile(
        r"^(?P<optimizer>[^_]+)_(?P<beta2>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_(?P<batch>\d+)_(?P<run>\d+)_(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.json$"
    ),
    re.compile(
        r"^(?P<optimizer>[^_]+)_(?P<beta2>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_(?P<batch>\d+)_(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.json$"
    ),
    re.compile(
        r"^(?P<optimizer>[^_]+)_(?P<beta2>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_(?P<batch>\d+)_(?P<run>\d+)\.json$"
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find the best learning rate per (optimizer, beta2, batch_size) "
            "using averaged final loss across repeated runs."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="sinexperiment/results",
        help="Directory containing per-run JSON files",
    )
    parser.add_argument(
        "--optimizer",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Only process files for these optimizers (space-separated). "
            "Default: use all optimizers"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "This is a directory. Output JSON will be written to this directory with a name like "
            "best_learning_rates_<optimizer>.json. Default: write to input directory."
        ),
    )
    parser.add_argument(
        "--loss-threshold",
        type=float,
        default=0.01,
        help=(
            "Threshold for reporting high-loss best configurations " "(default: 0.01)"
        ),
    )
    return parser.parse_args()


def safe_float(value):
    if value is None:
        return np.nan
    if isinstance(value, (float, int)):
        return float(value)
    return float(value)


def try_get_learning_rate(payload):
    for key in ("learning_rate", "lr"):
        if key in payload:
            return safe_float(payload.get(key))

    hyper = payload.get("hyperparameters")
    if isinstance(hyper, dict):
        for key in ("learning_rate", "lr"):
            if key in hyper:
                return safe_float(hyper.get(key))

    return np.nan


def load_run_record(json_path):
    match = None
    for pattern in RUN_FILE_PATTERNS:
        candidate = pattern.match(json_path.name)
        if candidate is not None:
            match = candidate
            break

    if match is None:
        return None

    with open(json_path, "r") as handle:
        payload = json.load(handle)

    optimizer = match.group("optimizer")
    beta2 = float(match.group("beta2"))
    batch_size = int(match.group("batch"))

    if "lr" in match.groupdict() and match.group("lr") is not None:
        learning_rate = float(match.group("lr"))
    else:
        learning_rate = try_get_learning_rate(payload)

    return {
        "optimizer": optimizer,
        "beta2": beta2,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": safe_float(payload.get("final_loss")),
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
            "No run JSON files found in "
            f"{input_dir} matching supported filename formats"
        )

    if args.optimizer is None:
        records = all_records
        optimizer_label = "all"
    else:
        selected_optimizers = set(args.optimizer)
        records = [
            record
            for record in all_records
            if record["optimizer"] in selected_optimizers
        ]
        optimizer_label = ", ".join(sorted(selected_optimizers))

    if not records:
        raise RuntimeError(f"No records found for optimizer '{optimizer_label}'")

    missing_lr_count = sum(1 for record in records if np.isnan(record["learning_rate"]))

    per_config_runs = defaultdict(list)
    for record in records:
        if np.isnan(record["learning_rate"]):
            continue
        key = (
            record["optimizer"],
            record["batch_size"],
            record["beta2"],
            record["learning_rate"],
        )
        per_config_runs[key].append(record["final_loss"])

    if not per_config_runs:
        raise RuntimeError(
            "No records with a valid learning rate were found. "
            "Please include learning rate in filename or payload."
        )

    per_triple_best = {}
    grouped_by_triple = defaultdict(list)

    for key, losses in per_config_runs.items():
        opt, batch_size, beta2, learning_rate = key
        avg_final_loss = nanmean(losses)
        std_final_loss = nanstd(losses)
        n_runs = len(losses)

        triple_key = (opt, beta2, batch_size)
        grouped_by_triple[triple_key].append(
            {
                "learning_rate": learning_rate,
                "avg_final_loss": avg_final_loss,
                "std_final_loss": std_final_loss,
                "n_runs": n_runs,
            }
        )

    for triple_key, candidates in grouped_by_triple.items():
        valid_candidates = [
            candidate
            for candidate in candidates
            if not np.isnan(candidate["avg_final_loss"])
        ]
        if valid_candidates:
            best_candidate = min(
                valid_candidates, key=lambda candidate: candidate["avg_final_loss"]
            )
        else:
            best_candidate = max(candidates, key=lambda candidate: candidate["n_runs"])

        per_triple_best[str(triple_key)] = {
            "optimizer": triple_key[0],
            "beta2": triple_key[1],
            "batch_size": triple_key[2],
            "best_learning_rate": best_candidate["learning_rate"],
            "best_avg_final_loss": best_candidate["avg_final_loss"],
            "best_std_final_loss": best_candidate["std_final_loss"],
            "best_lr_n_runs": best_candidate["n_runs"],
        }

    best_learning_rates = sorted(
        per_triple_best.values(),
        key=lambda row: (row["optimizer"], row["beta2"], row["batch_size"]),
    )

    above_threshold = []
    for row in best_learning_rates:
        if np.isnan(row["best_avg_final_loss"]):
            continue
        if row["best_avg_final_loss"] > args.loss_threshold:
            above_threshold.append(
                {
                    "optimizer": row["optimizer"],
                    "beta2": row["beta2"],
                    "batch_size": row["batch_size"],
                    "best_avg_final_loss": row["best_avg_final_loss"],
                    "best_learning_rate": row["best_learning_rate"],
                }
            )

    output_payload = {
        "optimizer_filter": args.optimizer,
        "loss_threshold": args.loss_threshold,
        "n_records_total": len(records),
        "n_records_missing_learning_rate": missing_lr_count,
        "best_learning_rates": best_learning_rates,
        "best_configs_above_threshold": above_threshold,
    }

    output_path = None
    if args.output is None:
        output_path = input_dir
    else:
        output_path = Path(args.output)
    if args.optimizer is None:
        output_path = Path.joinpath(output_path, "best_learning_rates.json")
    elif len(args.optimizer) == 1:
        output_path = Path.joinpath(
            output_path, f"best_learning_rates_{args.optimizer[0]}.json"
        )
    else:
        output_path = Path.joinpath(
            output_path, "best_learning_rates_selected_optimizers.json"
        )

    with open(output_path, "w") as handle:
        json.dump(output_payload, handle, indent=2)

    print(f"Wrote summary JSON: {output_path}")
    print(
        "Configurations with best_avg_final_loss above "
        f"{args.loss_threshold}: {len(above_threshold)}"
    )
    if above_threshold:
        for row in above_threshold:
            print(
                "  "
                f"optimizer={row['optimizer']}, "
                f"beta2={row['beta2']}, "
                f"batch_size={row['batch_size']}, "
                f"best_avg_final_loss={row['best_avg_final_loss']}, "
                f"best_learning_rate={row['best_learning_rate']}"
            )
    if ignored_files:
        print(
            "Ignored non-run JSON files: "
            + ", ".join(ignored_files[:10])
            + (" ..." if len(ignored_files) > 10 else "")
        )


if __name__ == "__main__":
    main()
