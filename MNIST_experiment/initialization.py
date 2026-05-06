import random
import shutil
import sys
from pathlib import Path
import torch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    # Seeds (only set if specified)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Clean and create necessary directories
    for rel_path in [
        "MNIST_experiment/results",
        "MNIST_experiment/logs",
    ]:
        abs_path = PROJECT_ROOT / rel_path
        if abs_path.exists() and abs_path.is_dir():
            shutil.rmtree(abs_path)
        abs_path.mkdir(parents=True, exist_ok=True)
