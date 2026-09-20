"""Heterogeneous-batch Fisher-merging demo (WS3b).

Two LeNet models share a random init and are trained with AdamW on disjoint
MNIST shards, optionally differing in batch size and shard size. They are then
merged coordinatewise as
theta = (w_A * theta_A + w_B * theta_B) / (w_A + w_B) with importance weights

- uniform:    plain parameter averaging,
- probe:      per-sample squared-gradient empirical Fisher on each model's
              own shard (the quantity Fisher merging asks for),
- theory raw: probe Fisher rescaled by the exact batch and finite-EMA factors
              predicted for the raw accumulator near stationarity,
- raw:        the model's raw exp_avg_sq accumulator (Squisher reuse),
- corrected:  bias-corrected exp_avg_sq * m(n-1)/(n-m) per model (Theorem 1).

Every Fisher-based scheme is also reported with dataset-size weighting. The
mean convention treats the two source tasks equally; the sum convention scales
each source by its shard size and corresponds to accumulated likelihood
precision. This distinction matters when shards are unbalanced and is separate
from the batch-size correction.

Prediction: near convergence exp_avg_sq targets ~F/m, so raw weights skew
toward the small-batch model by the batch-size ratio - a bias no lambda can
absorb because it is relative between models. The corrected weights should
match the probe weights, and the raw-merged model should track the
small-batch endpoint instead of the balanced merge.

Runs on CPU/MPS in minutes. Results JSON per seed under
merging_experiments/results/.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torchvision
from torch import nn
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from merging_experiments.models import LeNet, ResNet18  # noqa: E402
from merging_experiments.provenance import collect_runtime_provenance  # noqa: E402

MIN_STEPS = 1000  # floor on optimizer steps -- beta2=0.999's EMA has an effective ~1,000-step
                   # averaging window (1/(1-beta2)), so a run given too few steps under a fixed
                   # example budget would have exp_avg_sq dominated by burn-in rather than a
                   # converged, near-stationary accumulator. Matches Squisher's own MIN_ITERS
                   # fix for the same reason (and its own TUNE_BATCH=256 hits this same floor
                   # under the same 256,000-example budget). When this binds, the model
                   # processes more examples than examples_per_model nominally specifies --
                   # examples_processed_a/b in the saved JSON reflects the real total.


def parse_args():
    parser = argparse.ArgumentParser(description="Heterogeneous-batch Fisher-merging demo.")
    parser.add_argument("--batch_size_a", type=int, default=32)
    parser.add_argument("--batch_size_b", type=int, default=512)
    parser.add_argument("--microbatch_size", type=int, default=None,
                     help="If set and smaller than batch_size, split each optimizer step into "
                          "sequential chunks of this size, accumulating gradients before stepping "
                          "-- avoids holding the full batch's activations in GPU memory at once, "
                          "with no change to the training result.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--examples_per_model",
        type=int,
        default=None,
        help=(
            "If set, replace the shared step count by an equal example budget "
            "for each model. The budget must be divisible by both batch sizes."
        ),
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--probe_size", type=int, default=512)
    parser.add_argument(
        "--shard_size_a",
        type=int,
        default=None,
        help="Number of MNIST training examples assigned to model A (default: half).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data_dir", default="merging_experiments/data")
    parser.add_argument("--lenet_width", type=int, default=1)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist",
                     help="mnist -> LeNet on 1x28x28; cifar10 -> ResNet18 on 3x32x32.")
    parser.add_argument("--merge_eps", type=float, default=1e-12)
    parser.add_argument("--output_dir", default="merging_experiments/results")
    return parser.parse_args()


def resolve_training_steps(batch_size_a, batch_size_b, steps, examples_per_model=None):
    if min(batch_size_a, batch_size_b, steps) <= 0:
        raise ValueError("Batch sizes and steps must be positive")
    if examples_per_model is None:
        steps_a = steps_b = steps
    else:
        if examples_per_model <= 0:
            raise ValueError("examples_per_model must be positive")
        if examples_per_model % batch_size_a or examples_per_model % batch_size_b:
            raise ValueError("examples_per_model must be divisible by both batch sizes")
        steps_a = examples_per_model // batch_size_a
        steps_b = examples_per_model // batch_size_b
    return max(steps_a, MIN_STEPS), max(steps_b, MIN_STEPS)


def build_datasets(data_dir, dataset):
    if dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return train, test


def tensorize(dataset, indices, device):
    xs = torch.stack([dataset[i][0] for i in indices]).to(device)
    ys = torch.tensor([dataset[i][1] for i in indices], dtype=torch.long, device=device)
    return xs, ys


def train_model(model, xs, ys, batch_size, args, generator, steps):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, args.beta2),
        weight_decay=args.weight_decay,
    )
    n = xs.size(0)
    # Split each optimizer step into smaller forward/backward chunks when the full
    # batch would not fit on the GPU (e.g. batch_size=8000 through ResNet18). This
    # is mathematically identical to one large-batch step: each chunk's loss is
    # scaled by its share of the full batch, so the SUMMED gradient across chunks
    # equals the true batch-mean gradient -- exp_avg_sq ends up seeing exactly what
    # it would have seen from one un-chunked step. Only affects memory usage, never
    # the result. When microbatch_size is None or >= batch_size, this is a no-op:
    # the loop runs exactly once, identical to the original code.
    micro = args.microbatch_size or batch_size
    model.train()
    for _ in range(steps):
        idx = torch.randint(n, (batch_size,), generator=generator, device=xs.device)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, batch_size, micro):
            sub_idx = idx[start : start + micro]
            loss = nn.functional.cross_entropy(model(xs[sub_idx]), ys[sub_idx]) * (len(sub_idx) / batch_size)
            loss.backward()
        optimizer.step()
    return optimizer


def probe_fisher(model, xs, ys, probe_size, generator):
    """Per-sample squared-gradient empirical Fisher (mean convention) on a probe subset."""
    model.eval()
    idx = torch.randperm(xs.size(0), generator=generator, device=xs.device)[:probe_size]
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    grad_mean = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    for i in idx:
        model.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(xs[i : i + 1]), ys[i : i + 1])
        loss.backward()
        for name, p in model.named_parameters():
            fisher[name] += p.grad.pow(2)
            grad_mean[name] += p.grad
    model.zero_grad(set_to_none=True)
    k = len(idx)
    return (
        {name: v / k for name, v in fisher.items()},
        {name: g / k for name, g in grad_mean.items()},
    )


def accumulator_importances(model, optimizer, batch_size, shard_size, steps, beta2):
    """Raw and Theorem-1-corrected exp_avg_sq importances, plus diagnostics."""
    raw = {}
    corrected = {}
    exp_avg_sq_norm = 0.0
    exp_avg_norm = 0.0
    factor = batch_size * (shard_size - 1) / (shard_size - batch_size)
    bias_correction2 = 1.0 - beta2**steps
    for name, p in model.named_parameters():
        state = optimizer.state[p]
        v = state["exp_avg_sq"].detach().clone()
        raw[name] = v
        corrected[name] = v / bias_correction2 * factor
        exp_avg_sq_norm += float(v.pow(2).sum())
        exp_avg_norm += float(state["exp_avg"].detach().pow(2).sum())
    return raw, corrected, {
        "correction_factor": factor,
        "second_moment_bias_correction": bias_correction2,
        "exp_avg_sq_norm": exp_avg_sq_norm**0.5,
        "exp_avg_norm": exp_avg_norm**0.5,
    }


def scale_importances(importances, scalar):
    return {name: value * scalar for name, value in importances.items()}

def build_model(dataset, width):
    if dataset == "cifar10":
        return ResNet18(num_classes=10)
    return LeNet(width=width)


def merge_models(model_a, model_b, weights_a, weights_b, eps, dataset, width):
    merged = build_model(dataset, width).to(next(model_a.parameters()).device)
    merged.load_state_dict(model_a.state_dict())
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    share_sum = 0.0
    share_count = 0
    with torch.no_grad():
        for name, p in merged.named_parameters():
            w_a = weights_a[name] + eps
            w_b = weights_b[name] + eps
            p.copy_((w_a * params_a[name] + w_b * params_b[name]) / (w_a + w_b))
            share = w_a / (w_a + w_b)
            share_sum += float(share.sum())
            share_count += share.numel()
    return merged, share_sum / share_count


def evaluate(model, xs, ys, batch_size=1000):
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, xs.size(0), batch_size):
            logits = model(xs[start : start + batch_size])
            correct += int((logits.argmax(dim=1) == ys[start : start + batch_size]).sum())
    return correct / xs.size(0)


def flat(d):
    return torch.cat([v.flatten() for _, v in sorted(d.items())])


def main():
    args = parse_args()
    steps_a, steps_b = resolve_training_steps(
        args.batch_size_a,
        args.batch_size_b,
        args.steps,
        args.examples_per_model,
    )
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train, test = build_datasets(args.data_dir, args.dataset)
    n_train = len(train)
    perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(args.seed))
    shard_size_a = args.shard_size_a if args.shard_size_a is not None else n_train // 2
    if not 0 < shard_size_a < n_train:
        raise ValueError(f"shard_size_a must be between 1 and {n_train - 1}")
    shard_size_b = n_train - shard_size_a
    if args.batch_size_a >= shard_size_a or args.batch_size_b >= shard_size_b:
        raise ValueError("Each batch size must be smaller than its shard size")
    xs_a, ys_a = tensorize(train, perm[:shard_size_a].tolist(), device)
    xs_b, ys_b = tensorize(train, perm[shard_size_a:].tolist(), device)
    xs_test, ys_test = tensorize(test, range(len(test)), device)

    base = build_model(args.dataset, args.lenet_width).to(device)
    model_a = copy.deepcopy(base)
    model_b = copy.deepcopy(base)

    gen = torch.Generator(device=device).manual_seed(args.seed + 1)
    opt_a = train_model(model_a, xs_a, ys_a, args.batch_size_a, args, gen, steps_a)
    opt_b = train_model(model_b, xs_b, ys_b, args.batch_size_b, args, gen, steps_b)

    probe_gen = torch.Generator(device=device).manual_seed(args.seed + 2)
    fisher_a, grad_a = probe_fisher(model_a, xs_a, ys_a, args.probe_size, probe_gen)
    fisher_b, grad_b = probe_fisher(model_b, xs_b, ys_b, args.probe_size, probe_gen)
    raw_a, corr_a, diag_a = accumulator_importances(
        model_a, opt_a, args.batch_size_a, shard_size_a, steps_a, args.beta2
    )
    raw_b, corr_b, diag_b = accumulator_importances(
        model_b, opt_b, args.batch_size_b, shard_size_b, steps_b, args.beta2
    )

    uniform_a = {name: torch.ones_like(p) for name, p in model_a.named_parameters()}
    uniform_b = {name: torch.ones_like(p) for name, p in model_b.named_parameters()}
    theory_raw_a = {
        name: value * diag_a["second_moment_bias_correction"] / diag_a["correction_factor"]
        for name, value in fisher_a.items()
    }
    theory_raw_b = {
        name: value * diag_b["second_moment_bias_correction"] / diag_b["correction_factor"]
        for name, value in fisher_b.items()
    }
    nscaled_a = scale_importances(raw_a, shard_size_a)
    nscaled_b = scale_importances(raw_b, shard_size_b)
    mscaled_a = {name: value / diag_a["correction_factor"] * args.batch_size_a for name, value in corr_a.items()}
    mscaled_b = {name: value / diag_b["correction_factor"] * args.batch_size_b for name, value in corr_b.items()}
    fisher_sum_a = scale_importances(fisher_a, shard_size_a)
    fisher_sum_b = scale_importances(fisher_b, shard_size_b)
    theory_raw_sum_a = scale_importances(theory_raw_a, shard_size_a)
    theory_raw_sum_b = scale_importances(theory_raw_b, shard_size_b)
    raw_sum_a = scale_importances(raw_a, shard_size_a)
    raw_sum_b = scale_importances(raw_b, shard_size_b)
    corr_sum_a = scale_importances(corr_a, shard_size_a)
    corr_sum_b = scale_importances(corr_b, shard_size_b)
    schemes = {
        "uniform": (uniform_a, uniform_b),
        "probe_fisher": (fisher_a, fisher_b),
        "probe_theory_raw": (theory_raw_a, theory_raw_b),
        "squisher_raw": (raw_a, raw_b),
        "squisher_nscaled": (nscaled_a, nscaled_b),
        "squisher_mscaled": (mscaled_a, mscaled_b),
        "squisher_corrected": (corr_a, corr_b),
        "probe_fisher_sum": (fisher_sum_a, fisher_sum_b),
        "probe_theory_raw_sum": (theory_raw_sum_a, theory_raw_sum_b),
        "squisher_raw_sum": (raw_sum_a, raw_sum_b),
        "squisher_corrected_sum": (corr_sum_a, corr_sum_b),
    }

    results = {
        "config": {
            **vars(args),
            "steps_a": steps_a,
            "steps_b": steps_b,
            "examples_processed_a": steps_a * args.batch_size_a,
            "examples_processed_b": steps_b * args.batch_size_b,
            "shard_size_a_resolved": shard_size_a,
            "shard_size_b_resolved": shard_size_b,
            "importance_conventions": {
                "mean": "equal source weighting",
                "sum": "mean empirical Fisher multiplied by source shard size",
            },
        },
        "endpoint_accuracy": {
            "model_a": evaluate(model_a, xs_test, ys_test),
            "model_b": evaluate(model_b, xs_test, ys_test),
        },
        "diagnostics": {
            "model_a": {**diag_a, "probe_fisher_norm": float(flat(fisher_a).norm()),
                        "probe_grad_norm": float(flat(grad_a).norm()),
                        "raw_to_probe_norm_ratio": float(flat(raw_a).norm() / flat(fisher_a).norm()),
                        "corrected_to_probe_norm_ratio": float(flat(corr_a).norm() / flat(fisher_a).norm()),
                        "corrected_to_probe_cosine": float(nn.functional.cosine_similarity(
                            flat(corr_a), flat(fisher_a), dim=0))},
            "model_b": {**diag_b, "probe_fisher_norm": float(flat(fisher_b).norm()),
                        "probe_grad_norm": float(flat(grad_b).norm()),
                        "raw_to_probe_norm_ratio": float(flat(raw_b).norm() / flat(fisher_b).norm()),
                        "corrected_to_probe_norm_ratio": float(flat(corr_b).norm() / flat(fisher_b).norm()),
                        "corrected_to_probe_cosine": float(nn.functional.cosine_similarity(
                            flat(corr_b), flat(fisher_b), dim=0))},
        },
        "merges": {},
        "status": "ok",
        "command": [
            "python",
            "merging_experiments/fisher_merging_demo.py",
            *sys.argv[1:],
        ],
        "provenance": collect_runtime_provenance(PROJECT_ROOT),
    }
    for scheme, (w_a, w_b) in schemes.items():
        merged, share_a = merge_models(model_a, model_b, w_a, w_b, args.merge_eps, args.dataset, args.lenet_width)
        merged.to(device)
        results["merges"][scheme] = {
            "test_accuracy": evaluate(merged, xs_test, ys_test),
            "mean_weight_share_model_a": share_a,
        }
        print(f"{scheme:20s} acc={results['merges'][scheme]['test_accuracy']:.4f} "
              f"share_a={share_a:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_tag = (
        f"e{args.examples_per_model}"
        if args.examples_per_model is not None
        else f"s{args.steps}"
    )
    dataset_tag = "" if args.dataset == "mnist" else f"{args.dataset}_"
    output_path = output_dir / (
        f"merge_{dataset_tag}m{args.batch_size_a}v{args.batch_size_b}_{budget_tag}"
        f"{f'_na{args.shard_size_a}' if args.shard_size_a is not None else ''}"
        f"_seed{args.seed}.json"
    )
    with open(output_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
