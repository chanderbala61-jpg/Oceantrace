"""
OceanTrace - Shared Utilities Module

Provides random seed reproducibility, logging configuration, and path helpers.
"""

import os
import random
import torch
import numpy as np


def set_seed(seed: int = 42):
    """Sets fixed random seed across Python, NumPy, and PyTorch for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"-> Global random seed set to: {seed}")


def ensure_dirs(*dir_paths):
    """Ensures directories exist on filesystem."""
    for path in dir_paths:
        os.makedirs(path, exist_ok=True)
