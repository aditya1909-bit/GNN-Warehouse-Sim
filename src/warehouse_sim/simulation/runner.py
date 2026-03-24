"""Experiment runner utilities for config-driven simulations."""

from __future__ import annotations

from pathlib import Path

from warehouse_sim.agents import RobotSpec
from warehouse_sim.config import ExperimentConfig, load_experiment_config
from warehouse_sim.demand import DemandGenerationConfig, generate_task_demand
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import NodeType, SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.metrics import (
    write_default_plots,
    write_observation_dataset,
    write_simulation_report,
)
from warehouse_sim.policies import (
    FIFODispatchPolicy,
    LinearScoringDispatchPolicy,
    NearestRobotTaskPolicy,
    NearestTaskForIdleRobotPolicy,
    RandomDispatchPolicy,
)
from warehouse_sim.simulation.engine import run_simulation
from warehouse_sim.simulation.models import SimulationConfig, SimulationResult
from warehouse_sim.tasks import DemandTaskAdapterConfig, tasks_from_demand_records


def run_experiment_from_config(
    config: ExperimentConfig,
    output_dir_override: Path | None = None,
    force_write_plots: bool | None = None,
    force_write_observation_dataset: bool | None = None,
) -> tuple[SimulationResult, dict[str, Path]]:
    """Build and run a simulation experiment from a loaded config."""

    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=config.layout.rows,
            columns=config.layout.columns,
            edge_length=config.layout.edge_length,
            travel_speed=config.layout.travel_speed,
            blocked_cells=config.layout.blocked_cells,
            directed_edges=config.layout.directed_edges,
            special_node_types={
                config.layout.storage_cell: NodeType.STORAGE,
                config.layout.dropoff_cell: NodeType.DROPOFF,
                config.layout.staging_cell: NodeType.STAGING,
            },
            zone_labels={
                config.layout.storage_cell: "storage_zone",
                config.layout.dropoff_cell: "dropoff_zone",
                config.layout.staging_cell: "staging_zone",
            },
        )
    )
    environment = WarehouseEnvironment(graph=graph)

    demand = generate_task_demand(
        DemandGenerationConfig(
            horizon_seconds=config.demand.horizon_seconds,
            mean_interval=config.demand.mean_interval,
            seed=config.demand.seed,
            min_tasks=config.demand.min_tasks,
            rush_start=_window_start(config.demand.rush_start, config.demand.horizon_seconds),
            rush_end=_window_end(config.demand.rush_end, config.demand.horizon_seconds),
            rush_multiplier=config.demand.rush_multiplier,
            lunch_start=_window_start(config.demand.lunch_start, config.demand.horizon_seconds),
            lunch_end=_window_end(config.demand.lunch_end, config.demand.horizon_seconds),
        )
    )
    tasks = tasks_from_demand_records(
        records=demand.records,
        environment=environment,
        config=DemandTaskAdapterConfig(
            default_pickup_zone=config.tasks.default_pickup_zone,
            default_dropoff_zone=config.tasks.default_dropoff_zone,
            default_task_type=config.tasks.default_task_type,
            default_priority=config.tasks.default_priority,
            default_service_time_estimate=config.tasks.default_service_time_estimate,
            task_id_prefix=config.tasks.task_id_prefix,
        ),
    )
    robots = tuple(
        RobotSpec(
            robot_id=f"robot_{index + 1}",
            initial_node=environment.default_node_for_zone(config.robots.initial_zone).node_id,
            speed_multiplier=config.robots.speed_multiplier,
        )
        for index in range(config.robots.count)
    )
    result = run_simulation(
        environment=environment,
        tasks=tasks,
        robots=robots,
        dispatch_policy=_build_policy(config),
        config=SimulationConfig(
            horizon_seconds=config.simulation.horizon_seconds,
            continue_until_all_tasks_complete=config.simulation.continue_until_all_tasks_complete,
        ),
    )

    output_dir = output_dir_override or config.reporting.output_dir
    written_paths = write_simulation_report(output_dir=output_dir, result=result, experiment_name=config.name)
    write_plots = config.reporting.write_plots if force_write_plots is None else force_write_plots
    if write_plots:
        queue_plot, robot_plot = write_default_plots(output_dir=output_dir, result=result)
        written_paths["queue_plot"] = queue_plot
        written_paths["robot_utilization_plot"] = robot_plot
    write_dataset = (
        config.reporting.write_observation_dataset
        if force_write_observation_dataset is None
        else force_write_observation_dataset
    )
    if write_dataset:
        written_paths.update(
            write_observation_dataset(
                output_dir=output_dir,
                environment=environment,
                result=result,
                experiment_name=config.name,
            )
        )
    return result, written_paths


def run_experiment_from_path(
    config_path: Path,
    output_dir_override: Path | None = None,
    force_write_plots: bool | None = None,
    force_write_observation_dataset: bool | None = None,
) -> tuple[SimulationResult, dict[str, Path]]:
    """Load a config and run the corresponding experiment."""

    config = load_experiment_config(config_path)
    return run_experiment_from_config(
        config=config,
        output_dir_override=output_dir_override,
        force_write_plots=force_write_plots,
        force_write_observation_dataset=force_write_observation_dataset,
    )


def _build_policy(config: ExperimentConfig):
    policy_name = config.simulation.policy
    if policy_name == "fifo":
        return FIFODispatchPolicy()
    if policy_name == "random":
        return RandomDispatchPolicy(seed=config.demand.seed)
    if policy_name == "nearest_robot_task":
        return NearestRobotTaskPolicy()
    if policy_name == "nearest_task_for_idle_robot":
        return NearestTaskForIdleRobotPolicy()
    if policy_name == "linear_assignment_model":
        assert config.policy_model is not None
        return LinearScoringDispatchPolicy(
            weights=config.policy_model.weights,
            bias=config.policy_model.bias,
        )
    raise ValueError(f"Unknown policy: {policy_name}")


def _window_start(value: float | None, horizon_seconds: float) -> float:
    if value is None:
        return horizon_seconds
    return value


def _window_end(value: float | None, horizon_seconds: float) -> float:
    if value is None:
        return horizon_seconds
    return value
