import numpy as np
import torch
from optimizers import get_loss_and_per_sample_grads_for_tracking as get_grads
from asdl_fisher import diag_empirical_fisher_asdl


def cosine_similarity(
    vec1, vec2, label1="Vector 1", label2="Vector 2", zero_threshold=1e-9
):
    """
    Takes two vectors (torch tensors or numpy arrays), detaches if needed,
    prints them, and calculates/prints their cosine similarity.

    Args:
        vec1: First vector (torch.Tensor or numpy.ndarray)
        vec2: Second vector (torch.Tensor or numpy.ndarray)
        label1: Label for first vector (default: "Vector 1")
        label2: Label for second vector (default: "Vector 2")
        zero_threshold: Threshold below which vectors are considered zero (default: 1e-9)

    Returns:
        float: Cosine similarity between the two vectors, or None if either vector is near zero
    """
    # Convert to numpy if torch tensor
    if isinstance(vec1, torch.Tensor):
        vec1_np = vec1.detach().cpu().numpy().flatten()
    else:
        vec1_np = np.asarray(vec1).flatten()

    if isinstance(vec2, torch.Tensor):
        vec2_np = vec2.detach().cpu().numpy().flatten()
    else:
        vec2_np = np.asarray(vec2).flatten()

    # Check norms first to avoid division by zero
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np)

    if norm1 < zero_threshold or norm2 < zero_threshold:
        print(
            f"\nWarning: Vector norm too small (norm1={norm1:.2e}, norm2={norm2:.2e})"
        )
        return None

    # Calculate cosine similarity
    dot_product = np.dot(vec1_np, vec2_np)
    cosine_sim = dot_product / (norm1 * norm2)

    return cosine_sim


def compute_fisher_snapshot(
    exp_avg_sq,
    snapshot_epoch,
    train_loader,
    model,
    criterion,
    device,
    dataset_size,
    beta2,
    first=None,
    MA_emp=None,
    MA_adam=None,
):
    if MA_emp is None and MA_adam is not None:
        raise ValueError(
            "MA_adam provided without MA_emp. Both should be provided together or both should be None."
        )
    if MA_adam is None and MA_emp is not None:
        raise ValueError(
            "MA_emp provided without MA_adam. Both should be provided together or both should be None."
        )
    if first is None and (MA_emp is not None or MA_adam is not None):
        raise ValueError(
            "First snapshot indicator (first) is None while MA values are provided. Please set 'first' to True for the initial snapshot."
        )
    emp_estimate = 0.0
    adam_estimate = 0.0
    first_est = True
    for inputs_full, targets_full in train_loader:
        inputs_full, targets_full = inputs_full.to(device), targets_full.to(device)
        _, grads = get_grads(model, criterion, (inputs_full, targets_full))
        if first_est:
            emp_estimate = (grads**2).sum(dim=0) / dataset_size
            adam_estimate = grads.sum(dim=0) / dataset_size
            first_est = False
        else:
            emp_estimate += (grads**2).sum(dim=0) / dataset_size
            adam_estimate += grads.sum(dim=0) / dataset_size

        # Explicitly free batch memory to prevent accumulation on GPU
        del grads, inputs_full, targets_full
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adam_estimate = adam_estimate**2

    fisher_independent = cosine_similarity(
        exp_avg_sq, emp_estimate, label1="exp_avg_sq", label2="emp_estimate"
    )
    fisher_emp = fisher_independent
    fisher_mix_data = cosine_similarity(
        exp_avg_sq, adam_estimate, label1="exp_avg_sq", label2="adam_estimate"
    )
    fisher_adam = fisher_mix_data
    emp_vs_adam = cosine_similarity(
        emp_estimate, adam_estimate, label1="emp_estimate", label2="adam_estimate"
    )
    if MA_emp is not None and MA_adam is not None:
        # On first snapshot, initialize MA values directly; on subsequent snapshots, apply EMA formula
        MA_emp_new = (
            emp_estimate * (1 - beta2)
            if first
            else beta2 * MA_emp + (1 - beta2) * emp_estimate
        )
        MA_adam_new = (
            adam_estimate * (1 - beta2)
            if first
            else beta2 * MA_adam + (1 - beta2) * adam_estimate
        )
        fisher_MA_emp = cosine_similarity(
            exp_avg_sq, MA_emp_new, label1="exp_avg_sq", label2="MA_emp"
        )
        fisher_MA_adam = cosine_similarity(
            exp_avg_sq, MA_adam_new, label1="exp_avg_sq", label2="MA_adam"
        )
        MA_emp_vs_MA_adam_cos_sim = cosine_similarity(
            MA_emp_new, MA_adam_new, label1="MA_emp", label2="MA_adam"
        )
        fisher_dict_local = {
            "independent": fisher_independent,
            "empirical": fisher_emp,
            "mix_data": fisher_mix_data,
            "adam": fisher_adam,
            "MA_emp": fisher_MA_emp,
            "MA_adam": fisher_MA_adam,
        }
        fisher_dict_vs = {
            "emp_vs_adam": emp_vs_adam,
            "MA_emp_vs_MA_adam_cos_sim": MA_emp_vs_MA_adam_cos_sim,
        }
        return {
            "epoch": snapshot_epoch,
            "MA_emp_value": MA_emp_new,
            "MA_adam_value": MA_adam_new,
            "fisher_dict": fisher_dict_local,
            "fisher_dict_vs": fisher_dict_vs,
        }
    else:
        fisher_dict_local = {
            "independent": fisher_independent,
            "empirical": fisher_emp,
            "mix_data": fisher_mix_data,
            "adam": fisher_adam,
        }
        fisher_dict_vs = {"emp_vs_adam": emp_vs_adam}
        return {
            "epoch": snapshot_epoch,
            "fisher_dict": fisher_dict_local,
            "fisher_dict_vs": fisher_dict_vs,
        }


def compute_fisher_snapshot_memory_efficient(
    exp_avg_sq,
    snapshot_epoch,
    train_loader,
    model,
    criterion,
    device,
    dataset_size,
    beta2,
    first=None,
    MA_emp=None,
    MA_adam=None,
):
    if MA_emp is None and MA_adam is not None:
        raise ValueError(
            "MA_adam provided without MA_emp. Both should be provided together or both should be None."
        )
    if MA_adam is None and MA_emp is not None:
        raise ValueError(
            "MA_emp provided without MA_adam. Both should be provided together or both should be None."
        )
    if first is None and (MA_emp is not None or MA_adam is not None):
        raise ValueError(
            "First snapshot indicator (first) is None while MA values are provided. Please set 'first' to True for the initial snapshot."
        )
    emp_estimate = 0.0
    adam_estimate = 0.0
    first_est = True
    for inputs_full, targets_full in train_loader:
        inputs_full, targets_full = inputs_full.to(device), targets_full.to(device)
        # Compute gradients from summed loss so p.grad contains sum over the batch
        model.zero_grad()
        outputs = model(inputs_full)
        loss = criterion(outputs, targets_full)
        loss.backward()

        # 1. Extract and flatten gradients for this specific batch (before ASDL clears grads)
        current_grads = []
        ref_param = None
        for p in model.parameters():
            ref_param = p
            break
        for p in model.parameters():
            if p.grad is not None:
                current_grads.append(p.grad.detach().reshape(-1).clone())
            else:
                device = (
                    ref_param.device if ref_param is not None else torch.device("cpu")
                )
                dtype = ref_param.dtype if ref_param is not None else torch.float32
                current_grads.append(torch.zeros(p.numel(), device=device, dtype=dtype))

        # 2. Concatenate all flattened layers into a single vector
        if len(current_grads) > 0:
            current_flat_grads = torch.cat(current_grads)
        else:
            current_flat_grads = torch.tensor(
                [], device=ref_param.device if ref_param is not None else "cpu"
            )

        # Compute ASDL diag fisher (this internally does its own forward/backward and will clear grads)
        diag_fisher = diag_empirical_fisher_asdl(
            model=model, loss_fn=criterion, inputs=inputs_full, targets=targets_full
        )

        # diag_fisher is a list of per-parameter tensors; flatten to a single vector
        if isinstance(diag_fisher, list) and len(diag_fisher) > 0:
            flat_diag = torch.cat([d.reshape(-1).detach().clone() for d in diag_fisher])
        elif isinstance(diag_fisher, torch.Tensor):
            flat_diag = diag_fisher.reshape(-1).detach().clone()
        else:
            flat_diag = torch.tensor([], device=current_flat_grads.device)

        if first_est:
            # Use .clone() so we don't accidentally modify or lose the original tensor
            emp_estimate = flat_diag * len(inputs_full) / dataset_size
            adam_estimate = current_flat_grads.clone() * len(inputs_full) / dataset_size
            # Debug info for first batch to diagnose zero vectors
            first_est = False
        else:
            # Accumulate the gradients over the batches
            emp_estimate += flat_diag * len(inputs_full) / dataset_size
            adam_estimate += current_flat_grads * len(inputs_full) / dataset_size

    adam_estimate = adam_estimate**2

    fisher_independent = cosine_similarity(
        exp_avg_sq, emp_estimate, label1="exp_avg_sq", label2="emp_estimate"
    )
    fisher_emp = fisher_independent
    fisher_mix_data = cosine_similarity(
        exp_avg_sq, adam_estimate, label1="exp_avg_sq", label2="adam_estimate"
    )
    fisher_adam = fisher_mix_data
    emp_vs_adam = cosine_similarity(
        emp_estimate, adam_estimate, label1="emp_estimate", label2="adam_estimate"
    )
    if MA_emp is not None and MA_adam is not None:
        # On first snapshot, initialize MA values directly; on subsequent snapshots, apply EMA formula
        MA_emp_new = (
            emp_estimate * (1 - beta2)
            if first
            else beta2 * MA_emp + (1 - beta2) * emp_estimate
        )
        MA_adam_new = (
            adam_estimate * (1 - beta2)
            if first
            else beta2 * MA_adam + (1 - beta2) * adam_estimate
        )
        fisher_MA_emp = cosine_similarity(
            exp_avg_sq, MA_emp_new, label1="exp_avg_sq", label2="MA_emp"
        )
        fisher_MA_adam = cosine_similarity(
            exp_avg_sq, MA_adam_new, label1="exp_avg_sq", label2="MA_adam"
        )
        MA_emp_vs_MA_adam_cos_sim = cosine_similarity(
            MA_emp_new, MA_adam_new, label1="MA_emp", label2="MA_adam"
        )
        fisher_dict_local = {
            "independent": fisher_independent,
            "empirical": fisher_emp,
            "mix_data": fisher_mix_data,
            "adam": fisher_adam,
            "MA_emp": fisher_MA_emp,
            "MA_adam": fisher_MA_adam,
        }
        fisher_dict_vs = {
            "emp_vs_adam": emp_vs_adam,
            "MA_emp_vs_MA_adam_cos_sim": MA_emp_vs_MA_adam_cos_sim,
        }
        return {
            "epoch": snapshot_epoch,
            "MA_emp_value": MA_emp_new,
            "MA_adam_value": MA_adam_new,
            "fisher_dict": fisher_dict_local,
            "fisher_dict_vs": fisher_dict_vs,
        }
    else:
        fisher_dict_local = {
            "independent": fisher_independent,
            "empirical": fisher_emp,
            "mix_data": fisher_mix_data,
            "adam": fisher_adam,
        }
        fisher_dict_vs = {"emp_vs_adam": emp_vs_adam}
        return {
            "epoch": snapshot_epoch,
            "fisher_dict": fisher_dict_local,
            "fisher_dict_vs": fisher_dict_vs,
        }


def to_json_number(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
