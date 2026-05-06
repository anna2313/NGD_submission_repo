import torch
from torch.optim import Adam
from torch.func import vmap, grad


class EFAdam(Adam):
    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs.pop("batch_size")
        super(EFAdam, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, per_sample_grads, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_index = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Here is the key difference for EF-Adam
                param_per_sample_grads = per_sample_grads[param_index]
                normalizer = (
                    # 1 / self.batch_size
                    1  # param_per_sample_grads.shape[0] ** 0.5
                )
                exp_avg_sq.mul_(beta2).add_(
                    (param_per_sample_grads.pow(2).mean(dim=0) / normalizer),
                    alpha=1 - beta2,
                )
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group["eps"])

                step_size = group["lr"] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
                param_index += 1

        return loss


class EFAdam_memory_efficient(Adam):
    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs.pop("batch_size")
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, diag_fisher, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_index = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Here is the key difference for EF-Adam
                # --- NEW EF-ADAM LOGIC ---
                # diag_fisher already contains the mean of the squared per-sample gradients
                # (depending on how ASDL normalizes, you may not need the normalizer at all now)
                param_diag_fisher = diag_fisher[param_index]

                exp_avg_sq.mul_(beta2).add_(
                    param_diag_fisher,  # No need for .pow(2).mean(dim=0) here anymore!
                    alpha=1 - beta2,
                )

                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group["eps"])

                step_size = group["lr"] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
                param_index += 1

        return loss


class ReAdam(Adam):
    """Minor re-implementation to control the behavior where necessary"""

    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs.pop("batch_size")
        super(ReAdam, self).__init__(*args, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        param_index = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Here weould be the key difference for EF-Adam
                # Scal by batch_size because we have B^2 summands instead of B for EF.
                # This leads essentially to a learning-rate scaling!
                # normalizer = 1 / (self.batch_size)
                normalizer = 1
                exp_avg_sq.mul_(beta2).add_((grad.pow(2) / normalizer), alpha=1 - beta2)

                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(group["eps"])

                step_size = group["lr"] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
                param_index += 1

        return loss


def get_loss_and_per_sample_grads(model, loss_fn, batch):
    inputs, targets = batch

    def compute_loss_for_grad(params, buffers, sample, target):
        output = torch.func.functional_call(
            model, (params, buffers), (sample.unsqueeze(0),)
        )
        loss = loss_fn(output, target.unsqueeze(0))
        return loss, loss

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    per_sample_grads_tuple, per_sample_losses = vmap(
        grad(compute_loss_for_grad, has_aux=True),
        in_dims=(None, None, 0, 0),
        randomness="same",
    )(params, buffers, inputs, targets)
    per_sample_grads = list(per_sample_grads_tuple.values())
    loss = per_sample_losses.mean()

    return loss, per_sample_grads


def get_loss_and_per_sample_grads_for_tracking(
    model,
    loss_fn,
    batch,
    enable_logging=False,
    log_file="per_sample_per_target_gradients.txt",
):
    inputs, targets = batch
    grads = torch.zeros(
        inputs.size(0),
        sum(p.numel() for p in model.parameters()),
        device=inputs.device,
    )

    for input_idx in range(inputs.size(0)):
        model_output = model(inputs[input_idx].unsqueeze(0))
        loss = loss_fn(
            model_output,
            targets[input_idx].unsqueeze(0),
        )
        grad = torch.autograd.grad(loss, model.parameters())
        grads[input_idx, :] = torch.cat([g.view(-1).clone() for g in grad])

    # Save gradients to file (optional)
    if enable_logging:
        with open(log_file, "a") as f:
            f.write(f"Gradient shape: {grads.shape}\n")
            f.write(f"Gradients:\n{grads}\n")
            f.write("-" * 80 + "\n")

    return loss, grads


def get_loss_and_per_sample_per_target_grads(
    model,
    loss_fn,
    batch,
    enable_logging=False,
    log_file="per_sample_per_target_gradients.txt",
):
    inputs, targets = batch
    grads = torch.zeros(
        inputs.size(0),
        targets.size(1),
        sum(p.numel() for p in model.parameters()),
        device=inputs.device,
    )

    for input_idx in range(inputs.size(0)):
        for target_idx in range(targets.size(1)):
            model_output = model(inputs[input_idx].unsqueeze(0))
            # Use the correct target for this input sample and target dimension
            loss = loss_fn(
                model_output[:, target_idx : target_idx + 1],
                targets[input_idx, target_idx : target_idx + 1].unsqueeze(0),
            )
            grad = torch.autograd.grad(loss, model.parameters())
            grads[input_idx, target_idx, :] = torch.cat(
                [g.view(-1).clone() for g in grad]
            )

    # Save gradients to file (optional)
    if enable_logging:
        with open(log_file, "a") as f:
            f.write(f"Gradient shape: {grads.shape}\n")
            f.write(f"Gradients:\n{grads}\n")
            f.write("-" * 80 + "\n")

    return loss, grads


import torch
from torch.optim import Optimizer


class CustomNGD(Optimizer):
    def __init__(self, named_params, lr=0.01):
        # Extract parameters and names
        param_groups = [
            {
                "params": list(named_params.values()),
                "lr": lr,
                "names": list(named_params.keys()),
            }
        ]
        super(CustomNGD, self).__init__(param_groups, defaults={"lr": lr})

        self.named_params = named_params

    def step(self):
        """
        Performs a single optimization step."""
        for group in self.param_groups:
            names = group["names"]
            for name, param in zip(names, group["params"]):
                if param.grad is None:
                    continue

                if name == "fisher.weight":
                    continue

                # Gradient descent step
                grad = torch.clone(param.grad)
                param.data -= (
                    group["lr"]
                    * grad.detach()
                    / (self.named_params["fisher.weight"] + 1e-8)
                )

        for group in self.param_groups:
            names = group["names"]
            for name, param in zip(names, group["params"]):
                if name == "linear.weight":
                    continue

                param.data = torch.zeros(size=param.data.size())

    def fisher(self, scale):
        """
        Fisher calculation"""

        # self.named_params['fisher.weight'] += self.named_params['linear.weight'].grad * self.named_params['linear.weight'].grad/scale

        for group in self.param_groups:
            names = group["names"]
            for name, param in zip(names, group["params"]):
                if name == "linear.weight":
                    continue

                param.data += (
                    self.named_params["linear.weight"].grad
                    * self.named_params["linear.weight"].grad
                    / scale
                )


def diag_emp_f(X, y, model, loss_fn, optimizer):
    scale = X.size(0)
    for i in range(X.size(0)):
        optimizer.zero_grad()
        pred = model(X[i, :])
        loss = loss_fn(pred, y[i, :])
        loss.backward()
        optimizer.fisher(scale=scale)


def diag_ind_f(X, y, model, loss_fn, optimizer):
    scale = X.size(0)
    for i in range(X.size(0)):
        for j in range(y.size(0)):
            optimizer.zero_grad()
            pred = model(X[i, :])
            loss = loss_fn(pred[j], y[i, j])
            loss.backward()
            optimizer.fisher(scale=scale)


def diag_adam_f(X, y, model, loss_fn, optimizer):
    scale = X.size(0) * X.size(0)
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, y)
    loss.backward()
    optimizer.fisher(scale=scale)
