"""CLI for config-driven simulation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from warehouse_sim.simulation.runner import run_experiment_from_path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for config-driven experiments."""

    parser = argparse.ArgumentParser(description="Run a config-driven warehouse simulation experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write-plots", action="store_true")
    parser.add_argument("--write-observation-dataset", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run an experiment from a TOML config."""

    args = build_parser().parse_args(argv)
    result, written = run_experiment_from_path(
        config_path=args.config,
        output_dir_override=args.output_dir,
        force_write_plots=args.write_plots or None,
        force_write_observation_dataset=args.write_observation_dataset or None,
    )

    print(f"Experiment: {args.config}")
    print(f"Policy: {result.policy_name}")
    print(f"Tasks completed: {result.metrics.tasks_completed}")
    print(f"Output directory: {(args.output_dir or args.config.parent).resolve() if args.output_dir else 'configured output dir'}")
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
