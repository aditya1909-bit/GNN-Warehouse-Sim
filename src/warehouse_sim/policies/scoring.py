"""Observation-driven candidate scoring utilities and simple policy models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from warehouse_sim.policies.base import DispatchDecision, DispatchPolicy
from warehouse_sim.policies.observation import DispatchContext, RobotObservation, TaskObservation


SUPPORTED_CANDIDATE_FEATURES = (
    "travel_to_pickup_time",
    "travel_to_pickup_distance",
    "pickup_to_dropoff_time",
    "pickup_to_dropoff_distance",
    "task_age",
    "task_priority",
    "task_service_time_estimate",
    "robot_speed_multiplier",
    "robot_completed_task_count",
    "robot_total_busy_time",
    "robot_total_idle_time",
    "robot_total_travel_time",
    "robot_total_travel_distance",
    "pending_task_count",
    "ready_task_count",
    "future_task_count",
    "idle_robot_count",
    "busy_robot_count",
    "mean_ready_task_age",
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

        best_candidate: CandidateAssignmentObservation | None = None
        best_ranking: tuple[float, float, float, str, str] | None = None
        for candidate in candidates:
            score = candidate.linear_score(self._weights, self._bias)
            ranking = (
                score,
                -candidate.feature("travel_to_pickup_time"),
                candidate.feature("task_age"),
                _descending_string_key(candidate.robot_id),
                _descending_string_key(candidate.task_id),
            )
            if best_ranking is None or ranking > best_ranking:
                best_ranking = ranking
                best_candidate = candidate

        assert best_candidate is not None
        return DispatchDecision(robot_id=best_candidate.robot_id, task_id=best_candidate.task_id)

    def select_assignment(self, idle_robots, ready_tasks, environment):  # type: ignore[override]
        raise CandidateScoringError(
            "LinearScoringDispatchPolicy requires dispatch contexts and should not use the legacy selection path."
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
                        travel_to_pickup_time=(
                            context.environment.travel_time(robot_state.current_node, task.pickup_node)
                            / robot_state.spec.speed_multiplier
                        ),
                        travel_to_pickup_distance=context.environment.distance(
                            robot_state.current_node,
                            task.pickup_node,
                        ),
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
    }


def _descending_string_key(value: str) -> str:
    return "".join(chr(255 - ord(char)) for char in value)
