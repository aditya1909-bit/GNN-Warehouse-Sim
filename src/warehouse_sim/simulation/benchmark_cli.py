"""CLI for benchmark comparisons across scenarios and policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from warehouse_sim.simulation.benchmark import run_benchmark_from_path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for benchmark runs."""

    parser = argparse.ArgumentParser(description="Run a policy benchmark across scenario configs.")
    parser.add_argument("--config", type=Path, required=True, help="Benchmark TOML manifest.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--write-plots", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run a benchmark from a TOML manifest."""

    args = build_parser().parse_args(argv)
    written = run_benchmark_from_path(
        benchmark_config_path=args.config,
        benchmark_root_override=args.output_dir,
        force_write_plots=args.write_plots or None,
    )
    print(f"Benchmark: {args.config}")
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
