"""
Command-line entry point for the project.

Run with:

python run_experiment.py
python run_experiment.py --config config.yaml
"""

import argparse

from src.config import load_config
from src.experiment import Experiment
from src.utils import set_seed, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NLP reasoning experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    ensure_output_dirs(config)
    set_seed(config["project"]["seed"])

    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
