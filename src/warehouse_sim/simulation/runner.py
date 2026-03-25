"""Experiment runner utilities for config-driven simulations."""

from __future__ import annotations

from pathlib import Path

from warehouse_sim.agents import RobotSpec
from warehouse_sim.config import ExperimentConfig, load_experiment_config
from warehouse_sim.demand import DemandGenerationConfig, generate_task_demand
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import NodeType, SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.integrated.engine import run_integrated_simulation
from warehouse_sim.integrated.policies import (
    EndToEndMacroArtifactPolicy,
    OptimalMAPFCoordinatorPolicy,
    PrioritizedSIPPCoordinatorPolicy,
    RandomMacroPolicy,
)
from warehouse_sim.learning.artifacts import load_dispatch_model_artifact
from warehouse_sim.learning.graph_model import load_graph_dispatch_model
from warehouse_sim.metrics import (
    write_default_plots,
    write_observation_dataset,
    write_simulation_report,
)
from warehouse_sim.policies import (
    ArtifactScoringDispatchPolicy,
    CongestionAwareNearestRobotTaskPolicy,
    FIFODispatchPolicy,
    GraphDispatchArtifactPolicy,
    LinearScoringDispatchPolicy,
    NearestRobotTaskPolicy,
    NearestTaskForIdleRobotPolicy,
    RandomDispatchPolicy,
)
from warehouse_sim.simulation.engine import run_simulation
from warehouse_sim.simulation.models import (
    CoordinationMode,
    CoordinationRuntimeConfig,
    ExecutionModel,
    SimulationConfig,
    SimulationResult,
)
from warehouse_sim.tasks import DemandTaskAdapterConfig, Task, tasks_from_demand_records


def run_experiment_from_config(
    config: ExperimentConfig,
    output_dir_override: Path | None = None,
    force_write_plots: bool | None = None,
    force_write_observation_dataset: bool | None = None,
) -> tuple[SimulationResult, dict[str, Path]]:
    """Build and run a simulation experiment from a loaded config."""

    environment, tasks, robots, simulation_config = build_experiment_inputs(config)
    if simulation_config.coordination_mode == CoordinationMode.INTEGRATED:
        result = run_integrated_simulation(
            environment=environment,
            tasks=tasks,
            robots=robots,
            coordinator_policy=_build_integrated_policy(config),
            config=simulation_config,
        )
    else:
        result = run_simulation(
            environment=environment,
            tasks=tasks,
            robots=robots,
            dispatch_policy=_build_policy(config),
            config=simulation_config,
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
                dataset_metadata={
                    "scenario_name": config.name,
                    "run_id": f"{config.name}__{config.simulation.policy}__seed_{config.demand.seed}",
                    "demand_seed": config.demand.seed,
                    "execution_model": config.simulation.execution_model,
                    "coordination_mode": config.simulation.coordination_mode,
                },
            )
        )
    return result, written_paths


def build_experiment_inputs(
    config: ExperimentConfig,
) -> tuple[WarehouseEnvironment, tuple[Task, ...], tuple[RobotSpec, ...], SimulationConfig]:
    """Build the reusable environment, task, robot, and simulation inputs for an experiment."""

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
    return (
        environment,
        tasks,
        robots,
        SimulationConfig(
            horizon_seconds=config.simulation.horizon_seconds,
            continue_until_all_tasks_complete=config.simulation.continue_until_all_tasks_complete,
            coordination_mode=CoordinationMode(config.simulation.coordination_mode),
            execution_model=ExecutionModel(config.simulation.execution_model),
            coordination=(
                None
                if config.coordination is None
                else CoordinationRuntimeConfig(
                    motion_model=config.coordination.motion_model,
                    control_dt=config.coordination.control_dt,
                    replan_period=config.coordination.replan_period,
                    robot_radius=config.coordination.robot_radius,
                    collision_clearance=config.coordination.collision_clearance,
                    k_shortest_paths=config.coordination.k_shortest_paths,
                    max_route_options_per_pair=config.coordination.max_route_options_per_pair,
                )
            ),
        ),
    )


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
    if policy_name == "congestion_aware_nearest_robot_task":
        return CongestionAwareNearestRobotTaskPolicy()
    if policy_name == "linear_assignment_model":
        assert config.policy_model is not None
        return LinearScoringDispatchPolicy(
            weights=config.policy_model.weights,
            bias=config.policy_model.bias,
        )
    if policy_name == "trained_linear_model":
        assert config.policy_model is not None
        assert config.policy_model.artifact_path is not None
        artifact = load_dispatch_model_artifact(config.policy_model.artifact_path)
        if artifact.model_type != "grouped_linear":
            raise ValueError(
                f"Expected grouped_linear artifact for trained_linear_model, got {artifact.model_type}"
            )
        return ArtifactScoringDispatchPolicy(artifact=artifact, policy_name=policy_name)
    if policy_name == "trained_mlp_model":
        assert config.policy_model is not None
        assert config.policy_model.artifact_path is not None
        artifact = load_dispatch_model_artifact(config.policy_model.artifact_path)
        if artifact.model_type != "grouped_mlp":
            raise ValueError(f"Expected grouped_mlp artifact for trained_mlp_model, got {artifact.model_type}")
        return ArtifactScoringDispatchPolicy(artifact=artifact, policy_name=policy_name)
    if policy_name == "trained_graph_dispatch_model":
        assert config.policy_model is not None
        assert config.policy_model.artifact_path is not None
        loaded = load_graph_dispatch_model(config.policy_model.artifact_path)
        return GraphDispatchArtifactPolicy(
            model=loaded.model,
            candidate_feature_names=tuple(loaded.artifact.parameters["candidate_feature_names"]),
            node_feature_names=tuple(loaded.artifact.parameters["node_feature_names"]),
            edge_feature_names=tuple(loaded.artifact.parameters["edge_feature_names"]),
        )
    raise ValueError(f"Unknown policy: {policy_name}")


def _build_integrated_policy(config: ExperimentConfig):
    policy_name = config.simulation.policy
    if policy_name == "prioritized_sipp_coordinator":
        return PrioritizedSIPPCoordinatorPolicy()
    if policy_name == "optimal_mapf_coordinator":
        return OptimalMAPFCoordinatorPolicy()
    if policy_name == "random_macro":
        return RandomMacroPolicy(seed=config.demand.seed)
    if policy_name == "trained_end_to_end_macro_ppo":
        assert config.policy_model is not None
        assert config.policy_model.artifact_path is not None
        from warehouse_sim.learning.integrated_rl import load_end_to_end_macro_model

        loaded = load_end_to_end_macro_model(config.policy_model.artifact_path)
        return EndToEndMacroArtifactPolicy(loaded.model)
    raise ValueError(f"Unknown integrated policy: {policy_name}")


def _window_start(value: float | None, horizon_seconds: float) -> float:
    if value is None:
        return horizon_seconds
    return value


def _window_end(value: float | None, horizon_seconds: float) -> float:
    if value is None:
        return horizon_seconds
    return value
