import argparse
import random
import shutil
import sys
from pathlib import Path
import torch
import os

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_generator import generate_normal_sin_data


def parser():
    parser = argparse.ArgumentParser(description="Initialize experiment environment")

    # Data generation parameters
    parser.add_argument(
        "--theta",
        nargs="+",
        type=float,
        default=[0.1],
        help="True parameter values (space-separated)",
    )
    parser.add_argument(
        "--sigma",
        nargs="+",
        type=float,
        default=[0.01],
        help="Noise standard deviations (space-separated)",
    )
    parser.add_argument(
        "--number_of_datapoints",
        type=int,
        default=300,
        help="Number of training datapoints",
    )
    parser.add_argument(
        "--number_of_test_points",
        type=int,
        default=300,
        help="Number of test datapoints",
    )
    parser.add_argument(
        "--where_to_save",
        type=str,
        default="sinexperiment/data",
        help="Directory to save the generated data",
    )
    parser.add_argument(
        "--regenerate_data",
        action="store_true",
        help="Whether to generate synthetic data (default: False)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Seeds (only set if specified)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    args = parser()
    theta = np.array(args.theta)
    sigma = np.array(args.sigma)
    number_of_datapoints = args.number_of_datapoints
    number_of_test_points = args.number_of_test_points
    where_to_save = args.where_to_save

    # Clean and create necessary directories
    if args.regenerate_data:
        for rel_path in [
            "sinexperiment/results",
            "sinexperiment/logs",
            "sinexperiment/data",
        ]:
            abs_path = PROJECT_ROOT / rel_path
            if abs_path.exists() and abs_path.is_dir():
                shutil.rmtree(abs_path)
            abs_path.mkdir(parents=True, exist_ok=True)
    else:
        for rel_path in [
            "sinexperiment/results",
            "sinexperiment/logs",
        ]:
            abs_path = PROJECT_ROOT / rel_path
            if abs_path.exists() and abs_path.is_dir():
                shutil.rmtree(abs_path)
            abs_path.mkdir(parents=True, exist_ok=True)

    # Generate and save data for each configuration
    if args.regenerate_data:
        print("Generating data...")
        generate_normal_sin_data(
            theta=theta,
            sigma=sigma,
            number_of_datapoints=number_of_datapoints,
            output_file=f"{where_to_save}/training_data.csv",
        )
        generate_normal_sin_data(
            theta=theta,
            sigma=sigma,
            number_of_datapoints=number_of_test_points,
            output_file=f"{where_to_save}/test_data.csv",
        )
        print("Data generation complete.")
    else:
        if (
            not (PROJECT_ROOT / where_to_save / "training_data.csv").exists()
            or not (PROJECT_ROOT / where_to_save / "test_data.csv").exists()
        ):
            print(
                f"Warning: Data files not found in {where_to_save}. "
                "Run with --generate_data to create synthetic data."
            )
            print(
                f"Warning: Data files not found in {where_to_save}. "
                "Run with --generate_data to create synthetic data."
            )
        else:
            print(
                f"Data files already exist in {where_to_save}. "
                "Use --generate_data to overwrite them if needed."
            )
