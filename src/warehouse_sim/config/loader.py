"""TOML-based experiment configuration loader."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from warehouse_sim.config.models import (
    CoordinationConfig,
    ConfigValidationError,
    DemandConfig,
    ExperimentConfig,
    LayoutConfig,
    PolicyModelConfig,
    ReportingConfig,
    RobotsConfig,
    SimulationRunConfig,
    TasksConfig,
)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment config from a TOML file."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    try:
        layout = raw["layout"]
        demand = raw["demand"]
        robots = raw["robots"]
        tasks = raw.get("tasks", {})
        simulation = raw.get("simulation", {})
        coordination = raw.get("coordination")
        reporting = raw.get("reporting", {})
        policy_model = raw.get("policy_model")
    except KeyError as exc:
        raise ConfigValidationError(f"Missing required config section: {exc.args[0]}") from exc

    return ExperimentConfig(
        name=str(raw["name"]),
        layout=LayoutConfig(
            rows=int(layout["rows"]),
            columns=int(layout["columns"]),
            edge_length=float(layout.get("edge_length", 1.0)),
            travel_speed=float(layout.get("travel_speed", 1.0)),
            storage_cell=_coordinate(layout.get("storage_cell", [0, 0])),
            dropoff_cell=_coordinate(
                layout.get("dropoff_cell", [layout["rows"] - 1, layout["columns"] - 1])
            ),
            staging_cell=_coordinate(layout.get("staging_cell", [layout["rows"] - 1, 0])),
            blocked_cells=tuple(_coordinate(cell) for cell in layout.get("blocked_cells", [])),
            directed_edges=tuple(
                (_coordinate(edge[0]), _coordinate(edge[1]))
                for edge in layout.get("directed_edges", [])
            ),
        ),
        demand=DemandConfig(
            horizon_seconds=float(demand["horizon_seconds"]),
            mean_interval=float(demand["mean_interval"]),
            seed=int(demand.get("seed", 7)),
            min_tasks=int(demand.get("min_tasks", 0)),
            rush_start=_optional_float(demand.get("rush_start")),
            rush_end=_optional_float(demand.get("rush_end")),
            rush_multiplier=float(demand.get("rush_multiplier", 1.0)),
            lunch_start=_optional_float(demand.get("lunch_start")),
            lunch_end=_optional_float(demand.get("lunch_end")),
        ),
        robots=RobotsConfig(
            count=int(robots["count"]),
            speed_multiplier=float(robots.get("speed_multiplier", 1.0)),
            initial_zone=str(robots.get("initial_zone", "staging_zone")),
        ),
        tasks=TasksConfig(
            default_pickup_zone=str(tasks.get("default_pickup_zone", "storage_zone")),
            default_dropoff_zone=str(tasks.get("default_dropoff_zone", "dropoff_zone")),
            default_task_type=str(tasks.get("default_task_type", "pick")),
            default_priority=int(tasks.get("default_priority", 1)),
            default_service_time_estimate=float(tasks.get("default_service_time_estimate", 60.0)),
            task_id_prefix=str(tasks.get("task_id_prefix", "task")),
        ),
        simulation=SimulationRunConfig(
            policy=str(simulation.get("policy", "fifo")),
            horizon_seconds=_optional_float(simulation.get("horizon_seconds")),
            continue_until_all_tasks_complete=bool(
                simulation.get("continue_until_all_tasks_complete", True)
            ),
            coordination_mode=str(simulation.get("coordination_mode", "dispatch")),
            execution_model=str(simulation.get("execution_model", "idealized")),
        ),
        reporting=ReportingConfig(
            output_dir=Path(str(reporting.get("output_dir", "outputs/default_experiment"))),
            write_plots=bool(reporting.get("write_plots", False)),
            write_observation_dataset=bool(reporting.get("write_observation_dataset", False)),
        ),
        coordination=None
        if coordination is None
        else CoordinationConfig(
            control_dt=float(coordination.get("control_dt", 0.25)),
            replan_period=float(coordination.get("replan_period", 1.0)),
            robot_radius=float(coordination.get("robot_radius", 0.2)),
            collision_clearance=float(coordination.get("collision_clearance", 0.05)),
            k_shortest_paths=int(coordination.get("k_shortest_paths", 3)),
            max_route_options_per_pair=int(coordination.get("max_route_options_per_pair", 3)),
        ),
        policy_model=None
        if policy_model is None
        else PolicyModelConfig(
            bias=float(policy_model.get("bias", 0.0)),
            weights={str(key): float(value) for key, value in policy_model.get("weights", {}).items()},
            artifact_path=None
            if policy_model.get("artifact_path") is None
            else (path.parent / Path(str(policy_model["artifact_path"]))).resolve(),
        ),
    )


def _coordinate(value: Any) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ConfigValidationError(f"Expected coordinate pair, got {value!r}")
    return int(value[0]), int(value[1])


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
