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
    parser.add_argument("--parallel-workers", type=str, default=None, help='Worker count or "auto".')
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--use-mps-for-learned-policies", action="store_true")
    parser.add_argument("--scenario", action="append", default=None, help="Scenario name filter; repeatable.")
    parser.add_argument("--policy", action="append", default=None, help="Policy filter; repeatable.")
    parser.add_argument("--seed", type=int, action="append", default=None, help="Seed filter; repeatable.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run a benchmark from a TOML manifest."""

    args = build_parser().parse_args(argv)
    written = run_benchmark_from_path(
        benchmark_config_path=args.config,
        benchmark_root_override=args.output_dir,
        force_write_plots=args.write_plots or None,
        parallel_workers_override=None if args.parallel_workers in {None, "auto"} else int(args.parallel_workers),
        resume_override=args.resume,
        fail_fast_override=args.fail_fast or None,
        use_mps_for_learned_policies_override=args.use_mps_for_learned_policies or None,
        scenario_filters=None if args.scenario is None else tuple(args.scenario),
        policy_filters=None if args.policy is None else tuple(args.policy),
        seed_filters=None if args.seed is None else tuple(args.seed),
    )
    print(f"Benchmark: {args.config}")
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
