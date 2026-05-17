"""
General utility functions for reproducibility and project setup.
"""

import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """
    Create all required output directories.
    """

    output_dirs = [
        config["outputs"]["processed_dir"],
        config["outputs"]["outputs_dir"],
        config["outputs"]["plots_dir"],
    ]

    for output_dir in output_dirs:
        Path(output_dir).mkdir(parents=True, exist_ok=True)


def save_dataframe(df, path: str) -> None:
    """
    Save a pandas DataFrame to CSV and create parent directories if needed.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)