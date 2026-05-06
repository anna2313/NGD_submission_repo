"""ASDL-backed diagonal empirical Fisher helpers.

The direct ASDL Fisher path computes E[g_i^2] without materializing the full
per-sample gradient matrix. That is the expensive part of EFAdam and Fisher
snapshot evaluation for scalar MSE/CrossEntropy losses.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterable

import torch
from torch import nn


@contextmanager
def disable_tf32():
    """Disable CUDA TF32 inside numerical-equivalence-sensitive Fisher code."""
    if not torch.cuda.is_available():
        yield
        return

    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def asdl_loss_type(loss_fn: nn.Module):
    from asdl_master.asdl import LOSS_CROSS_ENTROPY, LOSS_MSE

    if isinstance(loss_fn, nn.CrossEntropyLoss):
        return LOSS_CROSS_ENTROPY
    if isinstance(loss_fn, nn.MSELoss):
        return LOSS_MSE
    return None


def supports_asdl_diag_empirical_fisher(
    model_or_loss_fn, loss_fn: nn.Module | None = None
) -> bool:
    if loss_fn is None:
        model = None
        loss_fn = model_or_loss_fn
    else:
        model = model_or_loss_fn

    return hasattr(loss_fn, "reduction")


def _named_module_lookup(model: nn.Module) -> dict[str, nn.Module]:
    return dict(model.named_modules())


def _collect_diag_by_parameter_order(model: nn.Module) -> list[torch.Tensor]:
    modules = _named_module_lookup(model)
    diag_values = []

    for name, param in model.named_parameters():
        module_name, _, param_name = name.rpartition(".")
        module = modules[module_name]
        fisher = getattr(module, "fisher", None)
        if fisher is None or not fisher.has_diag:
            raise RuntimeError(
                f"ASDL did not produce a diagonal Fisher for parameter '{name}'."
            )
        diag = getattr(fisher.diag, param_name, None)
        if diag is None:
            raise RuntimeError(f"ASDL diagonal Fisher is missing parameter '{name}'.")
        if diag.shape != param.shape:
            raise RuntimeError(
                f"ASDL diagonal Fisher for '{name}' has shape {tuple(diag.shape)}, "
                f"expected {tuple(param.shape)}."
            )
        diag_values.append(diag.detach().clone())

    return diag_values


def diag_empirical_fisher_asdl(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> list[torch.Tensor]:
    """Return per-parameter diagonal empirical Fisher tensors in parameter order."""
    from asdl_master.asdl import FISHER_EMP, SHAPE_DIAG, FisherConfig, get_fisher_maker

    if not supports_asdl_diag_empirical_fisher(model, loss_fn):
        raise TypeError(
            "ASDL diagonal empirical Fisher is not enabled for this model/loss combination."
        )

    with disable_tf32():
        model.zero_grad(set_to_none=True)
        config = FisherConfig(
            fisher_type=FISHER_EMP,
            fisher_shapes=[SHAPE_DIAG],
            loss_type=asdl_loss_type(loss_fn),
            data_size=len(inputs),
        )
        fisher_maker = get_fisher_maker(model, config)
        output = fisher_maker.setup_model_call(model, inputs)
        fisher_maker.setup_loss_call(loss_fn, output, targets)
        fisher_maker.forward_and_backward()
        diag_values = _collect_diag_by_parameter_order(model)
        model.zero_grad(set_to_none=True)
    return diag_values


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def flat_diag_empirical_fisher_asdl(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return flatten_tensors(diag_empirical_fisher_asdl(model, loss_fn, inputs, targets))


def flat_gradient_sum(
    model: nn.Module,
    loss_fn: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return sum_i grad(loss_i) for mean-reduction scalar losses."""
    with disable_tf32():
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(inputs), targets) * len(inputs)
        loss.backward()
        parts = []
        for param in model.parameters():
            if param.grad is None:
                parts.append(torch.zeros_like(param).reshape(-1))
            else:
                parts.append(param.grad.detach().clone().reshape(-1))
        model.zero_grad(set_to_none=True)
    return torch.cat(parts)
