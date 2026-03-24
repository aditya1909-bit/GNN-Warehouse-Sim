"""CLI entrypoint for the stage-1 stochastic demand generator."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from warehouse_sim.demand.config import (
    DemandGenerationConfig,
    DemandGenerationError,
    DemandValidationError,
    TaskMetadataConfig,
)
from warehouse_sim.demand.generator import generate_task_demand, write_task_demand_csv

DEFAULT_OUTPUT_PATH = Path("data/task_demand.csv")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for synthetic warehouse task-demand generation."""

    parser = argparse.ArgumentParser(description="Synthetic warehouse demand generator")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--horizon-seconds",
        type=float,
        default=28_800.0,
        help="Shift duration",
    )
    parser.add_argument(
        "--mean-interval",
        type=float,
        default=10.0,
        help="Base mean interarrival time",
    )
    parser.add_argument("--rush-start", type=float, default=1_800.0, help="Morning rush start")
    parser.add_argument("--rush-end", type=float, default=7_200.0, help="Morning rush end")
    parser.add_argument(
        "--rush-multiplier",
        type=float,
        default=2.0,
        help="Rate multiplier during rush",
    )
    parser.add_argument("--lunch-start", type=float, default=14_400.0, help="Lunch break start")
    parser.add_argument("--lunch-end", type=float, default=16_200.0, help="Lunch break end")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-tasks", type=int, default=200)
    parser.add_argument(
        "--include-task-metadata",
        action="store_true",
        help="Append optional task metadata columns after the legacy CSV schema.",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        default=list(TaskMetadataConfig().task_types),
        help="Candidate task types used when --include-task-metadata is enabled.",
    )
    parser.add_argument(
        "--source-zones",
        nargs="+",
        default=list(TaskMetadataConfig().source_zones),
        help="Candidate source zones used when --include-task-metadata is enabled.",
    )
    parser.add_argument(
        "--destination-zones",
        nargs="+",
        default=list(TaskMetadataConfig().destination_zones),
        help="Candidate destination zones used when --include-task-metadata is enabled.",
    )
    parser.add_argument(
        "--priorities",
        nargs="+",
        type=int,
        default=list(TaskMetadataConfig().priorities),
        help="Candidate integer priorities used when --include-task-metadata is enabled.",
    )
    parser.add_argument(
        "--service-duration-low",
        type=float,
        default=TaskMetadataConfig().service_duration_low,
        help="Lower bound for sampled service durations in seconds.",
    )
    parser.add_argument(
        "--service-duration-high",
        type=float,
        default=TaskMetadataConfig().service_duration_high,
        help="Upper bound for sampled service durations in seconds.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity for structured generator logs.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the generator CLI while preserving the legacy defaults."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        config = DemandGenerationConfig(
            horizon_seconds=args.horizon_seconds,
            mean_interval=args.mean_interval,
            rush_start=args.rush_start,
            rush_end=args.rush_end,
            rush_multiplier=args.rush_multiplier,
            lunch_start=args.lunch_start,
            lunch_end=args.lunch_end,
            seed=args.seed,
            min_tasks=args.min_tasks,
        )
        metadata_config = None
        if args.include_task_metadata:
            metadata_config = TaskMetadataConfig(
                task_types=tuple(args.task_types),
                source_zones=tuple(args.source_zones),
                destination_zones=tuple(args.destination_zones),
                priorities=tuple(args.priorities),
                service_duration_low=args.service_duration_low,
                service_duration_high=args.service_duration_high,
            )

        result = generate_task_demand(config=config, metadata_config=metadata_config)
        write_task_demand_csv(
            output_path=args.output,
            records=result.records,
            include_metadata=metadata_config is not None,
        )
    except (DemandValidationError, DemandGenerationError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {len(result.records)} tasks to {args.output}")
    print(f"Shift horizon: {config.horizon_seconds:.0f} sec")
    if result.summary.observed_mean_interarrival is None:
        print("Observed mean interarrival: n/a")
        print("95th percentile interarrival: n/a")
    else:
        print(
            "Observed mean interarrival: "
            f"{result.summary.observed_mean_interarrival:.3f} sec"
        )
        print(f"95th percentile interarrival: {result.summary.interarrival_p95:.3f} sec")


if __name__ == "__main__":
    main()

