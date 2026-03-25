"""Tests for observation-driven scoring policies."""

from __future__ import annotations

from pathlib import Path

import pytest

from warehouse_sim.agents import RobotSpec, RobotState
from warehouse_sim.config import load_experiment_config
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.policies import (
    CandidateScoringError,
    CongestionObservation,
    DispatchContextBuilder,
    LinearScoringDispatchPolicy,
    ResourceReservationObservation,
    SUPPORTED_CANDIDATE_FEATURES,
    build_candidate_assignment_observations,
)
from warehouse_sim.simulation import run_experiment_from_path
from warehouse_sim.tasks import Task


def _environment() -> WarehouseEnvironment:
    return WarehouseEnvironment(build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=1, columns=4)))


def test_build_candidate_assignment_observations_exposes_supported_features() -> None:
    environment = _environment()
    context = DispatchContextBuilder(environment).build(
        current_time=3.0,
        robot_states=(
            RobotState.from_spec(RobotSpec(robot_id="robot_1", initial_node="r0_c0")),
        ),
        pending_tasks=(
            Task(task_id="task_1", release_time=1.0, pickup_node="r0_c1", dropoff_node="r0_c3"),
        ),
        congestion_observation=CongestionObservation(
            execution_model="reserved_edges",
            edge_reservations=(
                ResourceReservationObservation(resource_id="r0_c0->r0_c1", reserved_until=4.0),
            ),
        ),
        execution_model="reserved_edges",
    )

    candidates = build_candidate_assignment_observations(context)

    assert len(candidates) == 1
    assert candidates[0].feature("travel_to_pickup_time") == 1.0
    assert candidates[0].feature("task_age") == 2.0
    assert candidates[0].feature("active_reserved_edge_count") == 1.0
    assert candidates[0].feature("pickup_node_inbound_degree") == 2.0
    assert candidates[0].feature("dropoff_node_outbound_degree") == 1.0
    assert candidates[0].feature("travel_to_pickup_mean_transit_count") == 0.0
    assert candidates[0].feature("pickup_to_dropoff_max_transit_count") == 4.0
    assert candidates[0].feature("pickup_to_dropoff_mean_arc_traversal_count") == 3.5
    assert candidates[0].feature("estimated_pickup_congestion_delay") == 1.0
    assert candidates[0].feature("estimated_pickup_blocked_segments") == 1.0
    assert "estimated_dropoff_congestion_delay" in SUPPORTED_CANDIDATE_FEATURES
    assert "travel_to_pickup_mean_arc_traversal_count" in SUPPORTED_CANDIDATE_FEATURES
    assert "average_robot_time_until_available" in SUPPORTED_CANDIDATE_FEATURES


def test_linear_scoring_policy_prefers_highest_scored_candidate() -> None:
    environment = _environment()
    context = DispatchContextBuilder(environment).build(
        current_time=5.0,
        robot_states=(
            RobotState.from_spec(RobotSpec(robot_id="robot_1", initial_node="r0_c0")),
            RobotState.from_spec(RobotSpec(robot_id="robot_2", initial_node="r0_c3")),
        ),
        pending_tasks=(
            Task(task_id="younger_left_task", release_time=4.0, pickup_node="r0_c0", dropoff_node="r0_c1"),
            Task(task_id="older_right_task", release_time=0.0, pickup_node="r0_c3", dropoff_node="r0_c2"),
        ),
    )

    policy = LinearScoringDispatchPolicy(
        weights={
            "travel_to_pickup_time": -5.0,
            "task_age": 1.0,
        }
    )

    decision = policy.select_assignment_from_context(context)

    assert decision is not None
    assert decision.robot_id == "robot_2"
    assert decision.task_id == "older_right_task"


def test_linear_scoring_policy_accepts_new_congestion_features() -> None:
    policy = LinearScoringDispatchPolicy(
        weights={
            "estimated_pickup_congestion_delay": -1.0,
            "estimated_dropoff_blocked_segments": -2.0,
        }
    )

    assert policy is not None


def test_linear_scoring_policy_rejects_unknown_weights() -> None:
    with pytest.raises(CandidateScoringError):
        LinearScoringDispatchPolicy(weights={"not_a_real_feature": 1.0})


def test_load_linear_assignment_config_and_run(tmp_path: Path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "linear_assignment_experiment.toml"
    config = load_experiment_config(config_path)

    assert config.simulation.policy == "linear_assignment_model"
    assert config.policy_model is not None
    assert config.policy_model.weights["travel_to_pickup_time"] == -2.0

    result, written = run_experiment_from_path(
        config_path=config_path,
        output_dir_override=tmp_path,
        force_write_plots=False,
        force_write_observation_dataset=True,
    )

    assert result.policy_name == "linear_assignment_model"
    assert written["dispatch_observations"].exists()
