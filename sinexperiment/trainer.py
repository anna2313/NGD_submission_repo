import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
import random
import argparse
import warnings
import sys
from pathlib import Path
import os
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimizers import (
    EFAdam,
    ReAdam,
    get_loss_and_per_sample_grads,
)

from data_generator import CustomDataset, generate_normal_sin_data
from torch.utils.data import DataLoader
from model import NeuralNetworkTwoHiddenReluEnd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils import compute_fisher_snapshot, to_json_number


def dataset_to_tensors(dataset, indices=None, device="cuda"):
    if indices is None:
        indices = range(len(dataset))  # all
    xy_train = [dataset[i] for i in indices]
    x = torch.stack([e[0] for e in xy_train]).to(device)
    y = torch.stack([torch.tensor(e[1]) for e in xy_train]).to(device)
    return x, y


def plot_datapoints(theta, outdim, train_data):
    colors = cm.rainbow(np.linspace(0, 1, outdim))
    for X, y in train_data:
        for i in range(y.size(0)):
            plt.scatter(
                theta @ X.detach().cpu().numpy(),
                y.detach().cpu().numpy()[i],
                color=colors[i],
            )
    plt.xlabel("prediction")
    plt.ylabel("true values")
    plt.title("Training data")
    plt.legend([f"y{i+1}" for i in range(outdim)], loc="upper left")
    plt.show()


def setup_data(args, theta, sigma, where_to_save="sinexperiment/data"):
    """Generate synthetic data and create data loaders.

    Args:
        args: Parsed command-line arguments
        theta: Numpy array of true parameter values
        sigma: Numpy array of noise standard deviations
        where_to_save: Directory to save the generated data

    Returns:
        tuple: (train_loader, test_dataloader, Xdim, outdim)
    """
    number_of_datapoints = args.number_of_datapoints
    number_of_test_points = args.number_of_test_points
    batch_size = args.batch_size

    Xdim = len(theta)

    # Generate synthetic data (only if regenerate_data is True)
    if args.regenerate_data:
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

    train_data = CustomDataset(f"{where_to_save}/training_data.csv", Xdim)
    test_data = CustomDataset(f"{where_to_save}/test_data.csv", Xdim)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    outdim = sigma.size

    return train_loader, test_dataloader, Xdim, outdim


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a neural network on synthetic data"
    )

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
        default=20,
        help="Number of test datapoints",
    )
    parser.add_argument(
        "--regenerate_data",
        action="store_true",
        help="Regenerate synthetic data (default: reuse existing data files)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", "--lr", type=float, default=0.0, help="Learning rate"
    )

    # Optimizer parameters
    parser.add_argument(
        "--optimizer_name",
        type=str,
        default="Adam",
        choices=["Adam", "EFAdam", "ReAdam"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.999, help="Beta2 for Adam optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (L2 regularization)",
    )

    # Initialization parameters
    parser.add_argument(
        "--initialization_type",
        type=str,
        default="random",
        choices=["random", "from_pt"],
        help="How to initialize model parameters: random or load from a .pt file",
    )
    parser.add_argument(
        "--initialization_model_path",
        type=str,
        default=None,
        help="Path to .pt model file used when --initialization_type=from_pt",
    )

    # Seeds
    parser.add_argument(
        "--torch_seed",
        type=int,
        default=None,
        help="Random seed for PyTorch (None = no seed)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for Python random module (None = no seed)",
    )
    parser.add_argument(
        "--np_seed",
        type=int,
        default=None,
        help="Random seed for NumPy (None = no seed)",
    )

    # Fisher-related (for compatibility with other trainers)
    parser.add_argument(
        "--fisher_type",
        type=str,
        default=None,
        help="Fisher information type (for compatibility)",
    )
    parser.add_argument(
        "--fisher_shape",
        type=str,
        default=None,
        help="Fisher information shape (for compatibility)",
    )
    parser.add_argument(
        "--scheduler_period",
        type=int,
        default=None,
        help="Scheduler period (for compatibility)",
    )
    parser.add_argument(
        "--fisher_save_period",
        type=int,
        default=10,
        help="Save Fisher approximations every N epochs (default: 10)",
    )
    parser.add_argument(
        "--moving_average_fishers_on",
        action="store_true",
        help="Whether to compute moving averages of Fisher approximations (default: False)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (default: 'cuda' if available, else 'cpu')",
    )

    # Output
    parser.add_argument("--plot", action="store_true", help="Plot training data")
    parser.add_argument(
        "--save_model_parameters",
        action="store_true",
        help="Save model parameters (.pt) at the end of the run",
    )
    parser.add_argument(
        "--name_of_json_output_file",
        type=str,
        default=None,
        help="JSON output file name",
    )
    parser.add_argument(
        "--hyperparameter_sweep",
        action="store_true",
        help="Indicates if this run is part of a hyperparameter sweep (for compatibility)",
    )
    parser.add_argument(
        "--loss_threshold",
        type=float,
        default=0.01,
        help="Loss threshold for reporting if final loss is above threshold",
    )

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        warnings.warn(
            "Ignoring unrecognized command-line arguments: " + " ".join(unknown_args),
            stacklevel=2,
        )
    return args


def resolve_device(device_arg):
    """Validate requested device and fall back safely when unavailable."""
    try:
        device = torch.device(device_arg)
    except (TypeError, RuntimeError):
        warnings.warn(
            f"Invalid device '{device_arg}'. Falling back to CPU.",
            stacklevel=2,
        )
        return torch.device("cpu")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA device requested but CUDA is not available. Falling back to CPU.",
                stacklevel=2,
            )
            return torch.device("cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            warnings.warn(
                f"CUDA device index {device.index} is out of range. Falling back to CPU.",
                stacklevel=2,
            )
            return torch.device("cpu")

    if device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            warnings.warn(
                "MPS device requested but MPS is not available. Falling back to CPU.",
                stacklevel=2,
            )
            return torch.device("cpu")

    return device


def main(args):
    # Seeds (only set if specified)
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
    if args.random_seed is not None:
        random.seed(args.random_seed)
    if args.np_seed is not None:
        np.random.seed(args.np_seed)

    # Loss and optimizer parameters
    base_lr = args.learning_rate
    batch_size = args.batch_size
    optimizer_name = args.optimizer_name
    betas = (args.beta1, args.beta2)
    sigma = np.array(args.sigma)
    theta = np.array(args.theta)
    weight_decay = args.weight_decay
    epochs = args.epochs
    device = resolve_device(args.device)

    # Generate data and create data loaders
    train_loader, _, Xdim, outdim = setup_data(args, theta, sigma)

    # Check that dataset size is divisible by batch size to avoid issues with per-sample gradients.
    dataset_size = len(train_loader.dataset)
    if dataset_size % batch_size != 0:
        raise ValueError(
            f"Dataset size ({dataset_size}) is not divisible by batch size ({batch_size}). Please adjust the number of datapoints or batch size."
        )

    train_batches = len(train_loader)
    log_interval = max(1, train_batches // 5)
    print("Run configuration:")
    print(
        f"  optimizer={optimizer_name} lr={base_lr} betas={betas} weight_decay={weight_decay}"
    )
    print(f"  epochs={epochs} batch_size={batch_size} train_batches={train_batches}")
    print(f"  Xdim={Xdim} outdim={outdim}")

    dim = Xdim
    # print(f"Input dimension: {dim}, Output dimension: {outdim}")
    model = NeuralNetworkTwoHiddenReluEnd(dim, outdim).to(device)

    # Initialize model parameters
    if args.initialization_type == "random":
        # Random initialization for all linear layers
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.uniform_(module.bias, a=-0.01, b=0.01)
    elif args.initialization_type == "from_pt":
        if args.initialization_model_path is None:
            raise ValueError(
                "Initialization type 'from_pt' requires --initialization_model_path"
            )
        init_path = Path(args.initialization_model_path)
        if not init_path.exists():
            raise FileNotFoundError(f"Initialization model file not found: {init_path}")
        try:
            state_dict = torch.load(init_path, map_location=device, weights_only=True)
        except TypeError:
            # Backward compatibility with older PyTorch versions.
            state_dict = torch.load(init_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model parameters from {init_path}")
    else:
        raise ValueError(f"Unknown initialization type: {args.initialization_type}")

    # Model parameters
    # for name, param in model.named_parameters():
    #     print(f"Parameter {name}: {param.data}")

    lr = base_lr
    criterion = nn.MSELoss(reduction="mean")
    if optimizer_name == "ReAdam":
        optimizer = ReAdam(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            batch_size=batch_size,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "EFAdam":
        optimizer = EFAdam(
            model.parameters(),
            lr=lr,
            betas=tuple(betas),
            batch_size=batch_size,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=tuple(betas), weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = ExponentialLR(optimizer, gamma=0.99)
    scheduler_period = args.scheduler_period
    if scheduler_period is not None and scheduler_period <= 0:
        raise ValueError("--scheduler_period must be a positive integer when provided")
    if args.fisher_save_period <= 0:
        raise ValueError("--fisher_save_period must be a positive integer")

    print(f"Using optimizer: {optimizer.__class__.__name__}")
    print("Using scheduler: ExponentialLR(gamma=0.99)")

    if args.plot:
        pre_inputs, pre_targets = [], []
        for inputs_p, targets_p in train_loader:
            pre_inputs.append(inputs_p)
            pre_targets.append(targets_p)
        pre_inputs = torch.cat(pre_inputs, dim=0).detach().cpu().numpy()
        pre_targets = torch.cat(pre_targets, dim=0).detach().cpu().numpy()
        x_vals_pre = pre_inputs[:, 0]
        sort_idx_pre = np.argsort(x_vals_pre)
        true_curve = np.sin(x_vals_pre[sort_idx_pre] / theta[0])

        fig_pre, axes_pre = plt.subplots(1, outdim, figsize=(6 * outdim, 4))
        if outdim == 1:
            axes_pre = [axes_pre]
        for d in range(outdim):
            axes_pre[d].scatter(
                x_vals_pre,
                pre_targets[:, d],
                alpha=0.5,
                label="Training data",
                color="steelblue",
                s=20,
            )
            axes_pre[d].plot(
                x_vals_pre[sort_idx_pre],
                true_curve,
                color="green",
                linewidth=1.5,
                label=f"sin(x/θ), θ={theta[0]}",
            )
            axes_pre[d].set_xlabel("x")
            axes_pre[d].set_ylabel(f"y{d+1}")
            axes_pre[d].set_title(f"Training data before training (output {d+1})")
            axes_pre[d].legend()
            axes_pre[d].grid(True, alpha=0.3)
        plt.tight_layout()
        pre_plot_path = "training_data.png"
        plt.savefig(pre_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig_pre)
        print(f"Saved training data plot to {pre_plot_path}")

    # Training
    epoch_losses = []
    exp_avg_sq = 0.0
    fisher_snapshots = []
    MA_emp = 0 if args.moving_average_fishers_on else None
    MA_adam = 0 if args.moving_average_fishers_on else None
    first = True  # Flag to indicate if this is the first snapshot when using moving averages

    total_steps = epochs * train_batches
    for epoch in range(epochs):
        model.train()
        epoch_loss_sum = 0.0
        optimizer.zero_grad()
        if args.moving_average_fishers_on:
            computed_fisher = compute_fisher_snapshot(
                exp_avg_sq=exp_avg_sq,
                snapshot_epoch=epoch + 1,
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                device=device,
                dataset_size=dataset_size,
                beta2=args.beta2,
                MA_emp=MA_emp,
                MA_adam=MA_adam,
                first=first,
            )
            first = False  # After the first snapshot, set this to False so that EMA formula is used for updates
            MA_emp = computed_fisher["MA_emp_value"]
            MA_adam = computed_fisher["MA_adam_value"]
            if (epoch + 1) % args.fisher_save_period == 0 or (epoch + 1) == epochs:
                fisher_snapshots.append(computed_fisher)
        else:
            if (epoch + 1) % args.fisher_save_period == 0 or (epoch + 1) == epochs:
                fisher_snapshots.append(
                    compute_fisher_snapshot(
                        exp_avg_sq=exp_avg_sq,
                        snapshot_epoch=epoch + 1,
                        model=model,
                        train_loader=train_loader,
                        criterion=criterion,
                        device=device,
                        dataset_size=dataset_size,
                        beta2=args.beta2,
                    )
                )
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if isinstance(optimizer, EFAdam):
                loss, per_sample_grads = get_loss_and_per_sample_grads(
                    model, criterion, (inputs, targets)
                )

                # Manually set the .grad attribute for each parameter
                standard_grads = [g.mean(dim=0) for g in per_sample_grads]
                for p, g in zip(model.parameters(), standard_grads):
                    p.grad = g

                # Finally do the step
                optimizer.step(per_sample_grads)

            else:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # Backward to compute gradients
                loss.backward()

                # Finally do the step
                optimizer.step()

            exp_avg_sq_parts = []
            for param in model.parameters():
                param_state = optimizer.state.get(param, {})
                if "exp_avg_sq" in param_state:
                    exp_avg_sq_parts.append(param_state["exp_avg_sq"].reshape(-1))
            if exp_avg_sq_parts:
                exp_avg_sq = torch.cat(exp_avg_sq_parts).clone()
            epoch_loss_sum += loss.item()

            if (i % log_interval == 0) or (i == train_batches - 1):
                current_step = epoch * train_batches + (i + 1)
                progress = current_step / total_steps
                bar_width = 30
                filled = int(bar_width * progress)
                bar = "=" * filled + "-" * (bar_width - filled)
                current_epoch_loss = epoch_loss_sum / (i + 1)
                print(
                    f"progress [{bar}] {current_step}/{total_steps} loss={current_epoch_loss:.6f}",
                    end="\r",
                    flush=True,
                )

        epoch_losses.append(epoch_loss_sum / train_batches)

        if scheduler_period is None or (epoch + 1) % scheduler_period == 0:
            scheduler.step()

    final_loss = float(np.mean(epoch_losses[-5:]))

    if not fisher_snapshots:
        fisher_snapshots.append(
            compute_fisher_snapshot(
                snapshot_epoch=epoch + 1,
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                device=device,
                dataset_size=dataset_size,
                beta2=args.beta2,
            )
        )
    final_fisher_snapshot = fisher_snapshots[-1]

    print()

    # Compare exp_avg_sq with each Fisher approximation
    print("\nexp_avg_sq vs Fisher approximations:")
    cos_exp_ind = final_fisher_snapshot["fisher_dict"]["independent"]
    cos_exp_emp = final_fisher_snapshot["fisher_dict"]["empirical"]
    cos_exp_mix = final_fisher_snapshot["fisher_dict"]["mix_data"]
    cos_exp_adam = final_fisher_snapshot["fisher_dict"]["adam"]
    print(
        f"  exp_avg_sq vs independent: {cos_exp_ind:.6f}"
        if cos_exp_ind is not None
        else "  exp_avg_sq vs independent: N/A (zero vector)"
    )
    print(
        f"  exp_avg_sq vs empirical: {cos_exp_emp:.6f}"
        if cos_exp_emp is not None
        else "  exp_avg_sq vs empirical: N/A (zero vector)"
    )
    print(
        f"  exp_avg_sq vs mix_data: {cos_exp_mix:.6f}"
        if cos_exp_mix is not None
        else "  exp_avg_sq vs mix_data: N/A (zero vector)"
    )
    print(
        f"  exp_avg_sq vs adam: {cos_exp_adam:.6f}"
        if cos_exp_adam is not None
        else "  exp_avg_sq vs adam: N/A (zero vector)"
    )
    if args.moving_average_fishers_on:
        cos_exp_MA_emp = final_fisher_snapshot["fisher_dict"]["MA_emp"]
        cos_exp_MA_adam = final_fisher_snapshot["fisher_dict"]["MA_adam"]
        print(
            f"  exp_avg_sq vs MA_emp: {cos_exp_MA_emp:.6f}"
            if cos_exp_MA_emp is not None
            else "  exp_avg_sq vs MA_emp: N/A (zero vector)"
        )
        print(
            f"  exp_avg_sq vs MA_adam: {cos_exp_MA_adam:.6f}"
            if cos_exp_MA_adam is not None
            else "  exp_avg_sq vs MA_adam: N/A (zero vector)"
        )

    fisher_dict = final_fisher_snapshot["fisher_dict"]

    if args.plot:
        model.eval()
        all_inputs, all_targets, all_preds = [], [], []
        with torch.no_grad():
            for inputs_p, targets_p in train_loader:
                inputs_p, targets_p = inputs_p.to(device), targets_p.to(device)
                all_inputs.append(inputs_p)
                all_targets.append(targets_p)
                all_preds.append(model(inputs_p))
        all_inputs = torch.cat(all_inputs, dim=0).detach().cpu().numpy()
        all_targets = torch.cat(all_targets, dim=0).detach().cpu().numpy()
        all_preds = torch.cat(all_preds, dim=0).detach().cpu().numpy()

        x_vals = all_inputs[:, 0]
        sort_idx = np.argsort(x_vals)

        fig, axes = plt.subplots(1, outdim, figsize=(6 * outdim, 4))
        if outdim == 1:
            axes = [axes]
        for d in range(outdim):
            axes[d].scatter(
                x_vals,
                all_targets[:, d],
                alpha=0.5,
                label="Training data",
                color="steelblue",
                s=20,
            )
            axes[d].scatter(
                x_vals[sort_idx],
                all_preds[sort_idx, d],
                alpha=0.8,
                label="Model fit",
                color="orange",
                s=10,
            )
            axes[d].set_xlabel("x")
            axes[d].set_ylabel(f"y{d+1}")
            axes[d].set_title(f"Model fit vs training data (output {d+1})")
            axes[d].legend()
            axes[d].grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = "model_fit.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved model fit plot to {plot_path}")

    if args.save_model_parameters:
        model_file_name = (
            f"{args.optimizer_name}_{args.batch_size}_{args.beta2}_model.pt"
        )
        model_path = Path("sinexperiment/") / model_file_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model parameters to {model_path}")

    return fisher_dict, final_loss, fisher_snapshots


if __name__ == "__main__":
    # Training loop
    args = parse_args()
    fisher_dict, final_loss, fisher_snapshots = main(args)
    loss_above_threshold = final_loss > args.loss_threshold
    print(
        "Final loss threshold check: "
        f"final_loss={final_loss:.6f}, "
        f"threshold={args.loss_threshold}, "
        f"above_threshold={loss_above_threshold}"
    )

    # Save results to JSON file
    if args.name_of_json_output_file == None:
        file_path = os.path.join(
            "sinexperiment/results",
            f"{args.optimizer_name}_{args.batch_size}_{args.beta2}.json",
        )
    else:
        file_path = os.path.join("sinexperiment/results", args.name_of_json_output_file)
    with open(file_path, "w") as f:
        new_dict = {}
        for key, value in fisher_dict.items():
            new_dict[key] = {"cosine_similarity_with_exp_avg_sq": to_json_number(value)}

        fisher_approximations_over_time = []
        for snapshot in fisher_snapshots:
            fisher_approximations_over_time.append(
                {
                    "epoch": snapshot["epoch"],
                    "fisher_approximations_dist": snapshot["fisher_dict"],
                    "fisher_approximations_vs": snapshot["fisher_dict_vs"],
                }
            )

        if args.hyperparameter_sweep == True:
            json.dump(
                {
                    "final_loss": to_json_number(final_loss),
                    "fisher_approximations": new_dict,
                    "fisher_approximations_over_time": fisher_approximations_over_time,
                    "cosine_similarity_emp_with_adam": to_json_number(
                        fisher_snapshots[-1]["fisher_dict_vs"]["emp_vs_adam"]
                    ),
                    "optimizer_name": args.optimizer_name,
                    "batch_size": args.batch_size,
                    "beta2": to_json_number(args.beta2),
                },
                f,
                indent=4,
                default=to_json_number,
            )
        else:
            if args.moving_average_fishers_on:
                json.dump(
                    {
                        "final_loss": to_json_number(final_loss),
                        "fisher_approximations": new_dict,
                        "fisher_approximations_over_time": fisher_approximations_over_time,
                        "cosine_similarity_emp_with_adam": to_json_number(
                            fisher_snapshots[-1]["fisher_dict_vs"]["emp_vs_adam"]
                        ),
                        "cosine_similarity_MA_emp_with_MA_adam": to_json_number(
                            fisher_snapshots[-1]["fisher_dict_vs"][
                                "MA_emp_vs_MA_adam_cos_sim"
                            ]
                        ),
                    },
                    f,
                    indent=4,
                    default=to_json_number,
                )
            else:
                json.dump(
                    {
                        "final_loss": to_json_number(final_loss),
                        "fisher_approximations": new_dict,
                        "fisher_approximations_over_time": fisher_approximations_over_time,
                        "cosine_similarity_emp_with_adam": to_json_number(
                            fisher_snapshots[-1]["fisher_dict_vs"]["emp_vs_adam"]
                        ),
                    },
                    f,
                    indent=4,
                    default=to_json_number,
                )
