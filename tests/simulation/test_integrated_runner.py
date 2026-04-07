"""Tests for integrated coordination experiment execution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from warehouse_sim.config import load_experiment_config
from warehouse_sim.integrated.geometry import inflate_obstacles, segment_has_line_of_sight
from warehouse_sim.simulation import run_experiment_from_path
from warehouse_sim.simulation.runner import build_experiment_inputs, run_experiment_from_config


def test_run_integrated_experiment_from_config_path(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "scenarios" / "integrated_narrow_bottleneck.toml"

    result, written = run_experiment_from_path(
        config_path=config_path,
        output_dir_override=tmp_path,
        force_write_plots=False,
        force_write_observation_dataset=False,
    )

    assert result.policy_name == "prioritized_sipp_coordinator"
    assert result.metrics.safety_violations_total == 0
    assert written["summary"].exists()
    assert written["robot_trajectories"].exists()
    assert written["macro_decisions"].exists()
    assert written["planner_plans"].exists()


def test_run_integrated_optimal_mapf_policy(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "scenarios" / "integrated_narrow_bottleneck.toml"
    config = load_experiment_config(config_path)
    config = replace(
        config,
        simulation=replace(config.simulation, policy="optimal_mapf_coordinator"),
        reporting=replace(config.reporting, output_dir=tmp_path),
    )

    result, written = run_experiment_from_config(
        config=config,
        output_dir_override=tmp_path,
        force_write_plots=False,
        force_write_observation_dataset=False,
    )

    assert result.policy_name == "optimal_mapf_coordinator"
    assert result.metrics.safety_violations_total == 0
    assert result.planner_plans
    assert all(plan.planner_name == "optimal_mapf_joint_search" for plan in result.planner_plans)
    assert written["planner_plans"].exists()


def test_run_integrated_free_space_motion_mode(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "scenarios" / "integrated_high_fleet_density.toml"
    config = load_experiment_config(config_path)
    assert config.coordination is not None
    config = replace(
        config,
        coordination=replace(config.coordination, motion_model="free_space"),
        reporting=replace(config.reporting, output_dir=tmp_path),
    )

    result, written = run_experiment_from_config(
        config=config,
        output_dir_override=tmp_path,
        force_write_plots=False,
        force_write_observation_dataset=False,
    )

    assert result.metrics.safety_violations_total == 0
    assert result.robot_trajectories
    assert any(getattr(record, "start_x", None) is not None for record in result.robot_trajectories)
    assert written["robot_trajectories"].exists()


@pytest.mark.parametrize(
    "scenario_name",
    (
        "integrated_obstacle_slalom",
        "integrated_blocked_cross_aisle",
        "integrated_unseen_geometry_generalization",
    ),
)
def test_obstacle_aware_integrated_scenarios_avoid_obstacle_intersections(
    tmp_path: Path,
    scenario_name: str,
) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "scenarios" / f"{scenario_name}.toml"
    config = load_experiment_config(config_path)
    environment, _tasks, _robots, _simulation_config = build_experiment_inputs(config)
    inflated_obstacles = inflate_obstacles(
        environment.obstacles(),
        margin=config.coordination.robot_radius + config.coordination.collision_clearance,  # type: ignore[union-attr]
    )

    result, _written = run_experiment_from_config(
        config=replace(config, reporting=replace(config.reporting, output_dir=tmp_path / scenario_name)),
        output_dir_override=tmp_path / scenario_name,
        force_write_plots=False,
        force_write_observation_dataset=False,
    )

    assert result.metrics.safety_violations_total == 0
    assert result.robot_trajectories
    assert all(
        segment_has_line_of_sight(
            (record.start_x, record.start_y),
            (record.end_x, record.end_y),
            obstacles=inflated_obstacles,
        )
        for record in result.robot_trajectories
        if record.start_x is not None and record.start_y is not None
        and record.end_x is not None and record.end_y is not None
    )
