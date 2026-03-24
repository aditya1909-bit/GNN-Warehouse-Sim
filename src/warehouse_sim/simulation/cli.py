"""CLI for the first baseline warehouse simulation."""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from warehouse_sim.agents import RobotSpec
from warehouse_sim.demand import DemandGenerationConfig, generate_task_demand
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import NodeType, SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.policies import (
    FIFODispatchPolicy,
    NearestRobotTaskPolicy,
    NearestTaskForIdleRobotPolicy,
    RandomDispatchPolicy,
)
from warehouse_sim.simulation.engine import run_simulation
from warehouse_sim.simulation.models import SimulationConfig
from warehouse_sim.tasks import DemandTaskAdapterConfig, tasks_from_demand_records


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the baseline simulation."""

    parser = argparse.ArgumentParser(description="Run the baseline warehouse simulation.")
    parser.add_argument(
        "--policy",
        choices=("fifo", "random", "nearest_robot_task", "nearest_task_for_idle_robot"),
        default="fifo",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--horizon-seconds", type=float, default=3_600.0)
    parser.add_argument("--mean-interval", type=float, default=120.0)
    parser.add_argument("--min-tasks", type=int, default=0)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the baseline simulation CLI."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    regime_boundary = args.horizon_seconds

    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=args.rows,
            columns=args.columns,
            special_node_types={
                (0, 0): NodeType.STORAGE,
                (args.rows - 1, args.columns - 1): NodeType.DROPOFF,
                (args.rows - 1, 0): NodeType.STAGING,
            },
            zone_labels={
                (0, 0): "storage_zone",
                (args.rows - 1, args.columns - 1): "dropoff_zone",
                (args.rows - 1, 0): "staging_zone",
            },
        )
    )
    environment = WarehouseEnvironment(graph=graph)
    demand = generate_task_demand(
        DemandGenerationConfig(
            horizon_seconds=args.horizon_seconds,
            mean_interval=args.mean_interval,
            seed=args.seed,
            min_tasks=args.min_tasks,
            rush_start=regime_boundary,
            rush_end=regime_boundary,
            lunch_start=regime_boundary,
            lunch_end=regime_boundary,
        )
    )
    tasks = tasks_from_demand_records(
        records=demand.records,
        environment=environment,
        config=DemandTaskAdapterConfig(
            default_pickup_zone="storage_zone",
            default_dropoff_zone="dropoff_zone",
        ),
    )
    robots = tuple(
        RobotSpec(robot_id=f"robot_{index + 1}", initial_node=environment.default_node_for_zone("staging_zone").node_id)
        for index in range(args.robots)
    )
    result = run_simulation(
        environment=environment,
        tasks=tasks,
        robots=robots,
        dispatch_policy=_build_policy(args.policy, args.seed),
        config=SimulationConfig(horizon_seconds=args.horizon_seconds),
    )

    print(f"Policy: {result.policy_name}")
    print(f"Tasks generated: {result.metrics.tasks_generated}")
    print(f"Tasks completed: {result.metrics.tasks_completed}")
    print(f"Tasks unassigned: {result.metrics.tasks_unassigned}")
    print(f"Makespan: {result.metrics.makespan:.3f} sec")
    print(f"Average waiting time: {_format_metric(result.metrics.average_waiting_time)} sec")
    print(f"Average turnaround time: {_format_metric(result.metrics.average_turnaround_time)} sec")
    print(
        "Average travel distance per task: "
        f"{_format_metric(result.metrics.average_travel_distance_per_task)}"
    )
    print(f"Average queue length: {result.metrics.average_queue_length:.3f}")
    print(f"Throughput: {result.metrics.throughput_per_hour:.3f} tasks/hour")


def _build_policy(policy_name: str, seed: int):
    if policy_name == "fifo":
        return FIFODispatchPolicy()
    if policy_name == "random":
        return RandomDispatchPolicy(seed=seed)
    if policy_name == "nearest_robot_task":
        return NearestRobotTaskPolicy()
    return NearestTaskForIdleRobotPolicy()


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
