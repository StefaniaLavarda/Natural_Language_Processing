"""
Configuration utilities.

This module loads the project YAML configuration file.
"""

from pathlib import Path
from typing import Dict, Any

import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Project configuration dictionary.
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config