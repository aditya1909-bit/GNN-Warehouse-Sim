"""Observation-driven candidate scoring utilities and simple policy models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from warehouse_sim.candidate_features import SUPPORTED_CANDIDATE_FEATURES
from warehouse_sim.graph import WarehouseEdge
from warehouse_sim.learning.artifacts import DispatchModelArtifact
from warehouse_sim.policies.base import DispatchDecision, DispatchPolicy
from warehouse_sim.policies.observation import (
    CongestionObservation,
    DispatchContext,
    RobotObservation,
    TaskObservation,
)


class CandidateScoringError(ValueError):
    """Raised when a candidate-scoring model is invalid."""


@dataclass(frozen=True)
class CandidateAssignmentObservation:
    """Flattened candidate robot-task features at a dispatch decision."""

    robot_id: str
    task_id: str
    robot_current_node: str
    robot_current_zone: str | None
    task_pickup_node: str
    task_dropoff_node: str
    task_source_zone: str | None
    task_destination_zone: str | None
    feature_values: dict[str, float]

    def feature(self, name: str) -> float:
        """Return a scalar feature value by name."""

        try:
            return self.feature_values[name]
        except KeyError as exc:
            raise CandidateScoringError(f"Unknown candidate feature: {name}") from exc

    def linear_score(self, weights: dict[str, float], bias: float = 0.0) -> float:
        """Compute a linear score from named feature weights."""

        validate_candidate_weights(weights)
        score = bias
        for feature_name, weight in weights.items():
            score += weight * self.feature(feature_name)
        return score


class LinearScoringDispatchPolicy(DispatchPolicy):
    """Observation-driven dispatch policy using a linear candidate scorer."""

    name = "linear_assignment_model"

    def __init__(self, weights: dict[str, float], bias: float = 0.0) -> None:
        validate_candidate_weights(weights)
        if not isfinite(bias):
            raise CandidateScoringError("bias must be finite.")
        self._weights = dict(weights)
        self._bias = bias

    def select_assignment_from_context(self, context: DispatchContext) -> DispatchDecision | None:
        """Score all idle-robot and ready-task candidates and pick the best."""

        candidates = build_candidate_assignment_observations(context)
        if not candidates:
            return None

        scores = np.asarray(
            [candidate.linear_score(self._weights, self._bias) for candidate in candidates],
            dtype=float,
        )
        best_candidate = _select_best_candidate(candidates, scores)
        return DispatchDecision(robot_id=best_candidate.robot_id, task_id=best_candidate.task_id)

    def select_assignment(self, idle_robots, ready_tasks, environment):  # type: ignore[override]
        raise CandidateScoringError(
            "LinearScoringDispatchPolicy requires dispatch contexts and should not use the legacy selection path."
        )


class ArtifactScoringDispatchPolicy(DispatchPolicy):
    """Dispatch policy backed by a trained artifact scorer."""

    def __init__(self, artifact: DispatchModelArtifact, policy_name: str) -> None:
        self.name = policy_name
        self._artifact = artifact

    def select_assignment_from_context(self, context: DispatchContext) -> DispatchDecision | None:
        candidates = build_candidate_assignment_observations(context)
        if not candidates:
            return None

        feature_matrix = np.asarray(
            [
                [candidate.feature(feature_name) for feature_name in self._artifact.feature_names]
                for candidate in candidates
            ],
            dtype=float,
        )
        scores = self._artifact.score_matrix(feature_matrix)
        best_candidate = _select_best_candidate(candidates, scores)
        return DispatchDecision(robot_id=best_candidate.robot_id, task_id=best_candidate.task_id)

    def select_assignment(self, idle_robots, ready_tasks, environment):  # type: ignore[override]
        raise CandidateScoringError(
            "ArtifactScoringDispatchPolicy requires dispatch contexts and should not use the legacy selection path."
        )


def build_candidate_assignment_observations(
    context: DispatchContext,
) -> tuple[CandidateAssignmentObservation, ...]:
    """Build candidate robot-task observations for a dispatch decision."""

    robot_by_id = {robot.robot_id: robot for robot in context.robot_observations}
    task_by_id = {task.task_id: task for task in context.task_observations if task.is_ready}
    candidates: list[CandidateAssignmentObservation] = []

    for robot_state in context.idle_robots:
        robot_observation = robot_by_id[robot_state.spec.robot_id]
        for task in context.ready_tasks:
            task_observation = task_by_id[task.task_id]
            pickup_path = context.environment.shortest_path(
                robot_state.current_node,
                task.pickup_node,
                weight="travel_time",
            )
            pickup_edges = context.environment.shortest_path_edges(
                robot_state.current_node,
                task.pickup_node,
                weight="travel_time",
            )
            travel_to_pickup_time = (
                context.environment.path_travel_time(pickup_path) / robot_state.spec.speed_multiplier
            )
            estimated_pickup_delay, estimated_pickup_blocked_segments = _estimate_congestion(
                path_edges=pickup_edges,
                path_nodes=pickup_path,
                congestion_observation=context.congestion_observation,
                start_time=context.current_time,
                speed_multiplier=robot_state.spec.speed_multiplier,
            )
            dropoff_path = context.environment.shortest_path(
                task.pickup_node,
                task.dropoff_node,
                weight="travel_time",
            )
            dropoff_edges = context.environment.shortest_path_edges(
                task.pickup_node,
                task.dropoff_node,
                weight="travel_time",
            )
            estimated_dropoff_delay, estimated_dropoff_blocked_segments = _estimate_congestion(
                path_edges=dropoff_edges,
                path_nodes=dropoff_path,
                congestion_observation=context.congestion_observation,
                start_time=(
                    context.current_time
                    + travel_to_pickup_time
                    + estimated_pickup_delay
                    + task.service_time_estimate
                ),
                speed_multiplier=robot_state.spec.speed_multiplier,
            )
            candidates.append(
                CandidateAssignmentObservation(
                    robot_id=robot_state.spec.robot_id,
                    task_id=task.task_id,
                    robot_current_node=robot_observation.current_node,
                    robot_current_zone=robot_observation.current_zone,
                    task_pickup_node=task_observation.pickup_node,
                    task_dropoff_node=task_observation.dropoff_node,
                    task_source_zone=task_observation.source_zone,
                    task_destination_zone=task_observation.destination_zone,
                    feature_values=_candidate_feature_values(
                        context=context,
                        robot_observation=robot_observation,
                        task_observation=task_observation,
                        travel_to_pickup_time=travel_to_pickup_time,
                        travel_to_pickup_distance=context.environment.path_distance(pickup_path),
                        estimated_pickup_delay=estimated_pickup_delay,
                        estimated_dropoff_delay=estimated_dropoff_delay,
                        estimated_pickup_blocked_segments=estimated_pickup_blocked_segments,
                        estimated_dropoff_blocked_segments=estimated_dropoff_blocked_segments,
                    ),
                )
            )
    return tuple(candidates)


def validate_candidate_weights(weights: dict[str, float]) -> None:
    """Validate a named set of candidate feature weights."""

    for feature_name, weight in weights.items():
        if feature_name not in SUPPORTED_CANDIDATE_FEATURES:
            raise CandidateScoringError(f"Unsupported candidate feature weight: {feature_name}")
        if not isfinite(weight):
            raise CandidateScoringError(f"Weight for {feature_name} must be finite.")


def _candidate_feature_values(
    context: DispatchContext,
    robot_observation: RobotObservation,
    task_observation: TaskObservation,
    travel_to_pickup_time: float,
    travel_to_pickup_distance: float,
    estimated_pickup_delay: float,
    estimated_dropoff_delay: float,
    estimated_pickup_blocked_segments: int,
    estimated_dropoff_blocked_segments: int,
) -> dict[str, float]:
    return {
        "travel_to_pickup_time": travel_to_pickup_time,
        "travel_to_pickup_distance": travel_to_pickup_distance,
        "pickup_to_dropoff_time": task_observation.pickup_to_dropoff_travel_time,
        "pickup_to_dropoff_distance": task_observation.pickup_to_dropoff_distance,
        "task_age": task_observation.age,
        "task_priority": float(task_observation.priority),
        "task_service_time_estimate": task_observation.service_time_estimate,
        "robot_speed_multiplier": robot_observation.speed_multiplier,
        "robot_completed_task_count": float(robot_observation.completed_task_count),
        "robot_total_busy_time": robot_observation.total_busy_time,
        "robot_total_idle_time": robot_observation.total_idle_time,
        "robot_total_travel_time": robot_observation.total_travel_time,
        "robot_total_travel_distance": robot_observation.total_travel_distance,
        "pending_task_count": float(context.global_observation.pending_task_count),
        "ready_task_count": float(context.global_observation.ready_task_count),
        "future_task_count": float(context.global_observation.future_task_count),
        "idle_robot_count": float(context.global_observation.idle_robot_count),
        "busy_robot_count": float(context.global_observation.busy_robot_count),
        "mean_ready_task_age": context.global_observation.mean_ready_task_age,
        "average_robot_time_until_available": context.global_observation.average_robot_time_until_available,
        "active_reserved_edge_count": float(context.global_observation.active_reserved_edge_count),
        "active_reserved_node_count": float(context.global_observation.active_reserved_node_count),
        "estimated_pickup_congestion_delay": estimated_pickup_delay,
        "estimated_dropoff_congestion_delay": estimated_dropoff_delay,
        "estimated_pickup_blocked_segments": float(estimated_pickup_blocked_segments),
        "estimated_dropoff_blocked_segments": float(estimated_dropoff_blocked_segments),
    }


def _descending_string_key(value: str) -> str:
    return "".join(chr(255 - ord(char)) for char in value)


def _select_best_candidate(
    candidates: tuple[CandidateAssignmentObservation, ...],
    scores: np.ndarray,
) -> CandidateAssignmentObservation:
    best_index = max(
        range(len(candidates)),
        key=lambda index: (
            float(scores[index]),
            -candidates[index].feature("travel_to_pickup_time"),
            candidates[index].feature("task_age"),
            _descending_string_key(candidates[index].robot_id),
            _descending_string_key(candidates[index].task_id),
        ),
    )
    return candidates[best_index]


def _estimate_congestion(
    *,
    path_edges: tuple[WarehouseEdge, ...],
    path_nodes: tuple[str, ...],
    congestion_observation: CongestionObservation,
    start_time: float,
    speed_multiplier: float,
) -> tuple[float, int]:
    if congestion_observation.execution_model == "idealized":
        return 0.0, 0
    if congestion_observation.execution_model == "reserved_edges":
        edge_reserved_until = {
            _parse_arc_id(reservation.resource_id): reservation.reserved_until
            for reservation in congestion_observation.edge_reservations
        }
        current_time = start_time
        total_delay = 0.0
        blocked_segments = 0
        for edge in path_edges:
            departure_time = max(current_time, edge_reserved_until.get((edge.source, edge.target), 0.0))
            if departure_time > current_time:
                total_delay += departure_time - current_time
                blocked_segments += 1
            current_time = departure_time + (edge.travel_time / speed_multiplier)
        return total_delay, blocked_segments
    if congestion_observation.execution_model == "reserved_nodes":
        node_reserved_until = {
            reservation.resource_id: reservation.reserved_until
            for reservation in congestion_observation.node_reservations
        }
        current_time = start_time
        total_delay = 0.0
        blocked_segments = 0
        for edge in path_edges:
            earliest_arrival = current_time + (edge.travel_time / speed_multiplier)
            entry_time = max(earliest_arrival, node_reserved_until.get(edge.target, 0.0))
            if entry_time > earliest_arrival:
                total_delay += entry_time - earliest_arrival
                blocked_segments += 1
            current_time = entry_time
        return total_delay, blocked_segments
    raise CandidateScoringError(
        f"Unsupported congestion observation execution_model: {congestion_observation.execution_model}"
    )


def _parse_arc_id(value: str) -> tuple[str, str]:
    source, target = value.split("->", maxsplit=1)
    return source, target
