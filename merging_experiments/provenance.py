"""Runtime provenance recorded with maintained experiment outputs."""

import hashlib
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

import torch

try:
    import torchvision
except ImportError:  # pragma: no cover - synthetic-only installations
    torchvision = None


def _git_output(project_root, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_runtime_provenance(project_root):
    project_root = Path(project_root).resolve()
    revision = _git_output(project_root, "rev-parse", "HEAD")
    status = _git_output(project_root, "status", "--porcelain")
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except (AssertionError, RuntimeError):
            gpu_name = None
    lockfile = project_root / "uv.lock"
    return {
        "git_revision": revision,
        "git_dirty": None if status is None else bool(status),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "torchvision_version": (
            torchvision.__version__ if torchvision is not None else None
        ),
        "uv_lock_sha256": sha256_file(lockfile) if lockfile.exists() else None,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
