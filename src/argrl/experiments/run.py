"""CLI: PYTHONPATH=src uv run python -m argrl.experiments.run --config <yaml>"""

from __future__ import annotations
import argparse
from pathlib import Path
from argrl.experiments.config import load_config
from argrl.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_experiment(load_config(args.config), str(args.config))


if __name__ == "__main__":
    main()
