"""Observation-driven candidate scoring utilities and simple policy models."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np

from warehouse_sim.candidate_features import SUPPORTED_CANDIDATE_FEATURES
from warehouse_sim.graph import WarehouseEdge
from warehouse_sim.learning.artifacts import DispatchModelArtifact
from warehouse_sim.learning.graph_data import build_graph_dispatch_example_from_context
from warehouse_sim.policies.base import DispatchDecision, DispatchPolicy
from warehouse_sim.policies.observation import (
    CongestionObservation,
    DispatchContext,
    RobotObservation,
    TaskObservation,
)
from warehouse_sim.utils.battery import (
    battery_enabled,
    estimate_charge_action,
    estimate_task_action,
    nearest_charging_option,
)
from warehouse_sim.utils.dependencies import require_dependency

if TYPE_CHECKING:
    from warehouse_sim.learning.graph_model import GraphDispatchScorer
else:
    GraphDispatchScorer = Any


class CandidateScoringError(ValueError):
    """Raised when a candidate-scoring model is invalid."""


@dataclass(frozen=True)
class CandidateAssignmentObservation:
    """Flattened candidate robot-task features at a dispatch decision."""

    robot_id: str
    action_type: str
    task_id: str | None
    charging_node_id: str | None
    robot_current_node: str
    robot_current_zone: str | None
    task_pickup_node: str | None
    task_dropoff_node: str | None
    task_source_zone: str | None
    task_destination_zone: str | None
    feature_values: dict[str, float]

    @property
    def candidate_id(self) -> str:
        if self.action_type == "charge":
            return f"charge::{self.charging_node_id or ''}"
        return self.task_id or ""

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
    policy_score_label = "linear_model_score"

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
        return _dispatch_decision_for_candidate(best_candidate)

    def select_assignment(self, idle_robots, ready_tasks, environment):  # type: ignore[override]
        raise CandidateScoringError(
            "LinearScoringDispatchPolicy requires dispatch contexts and should not use the legacy selection path."
        )

    def score_assignment_candidates_from_context(self, context, candidates):  # type: ignore[override]
        return tuple(candidate.linear_score(self._weights, self._bias) for candidate in candidates)


class ArtifactScoringDispatchPolicy(DispatchPolicy):
    """Dispatch policy backed by a trained artifact scorer."""

    def __init__(self, artifact: DispatchModelArtifact, policy_name: str) -> None:
        self.name = policy_name
        self._artifact = artifact
        self.policy_score_label = f"{artifact.model_type}_score"

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
        return _dispatch_decision_for_candidate(best_candidate)

    def select_assignment(self, idle_robots, ready_tasks, environment):  # type: ignore[override]
        raise CandidateScoringError(
            "ArtifactScoringDispatchPolicy requires dispatch contexts and should not use the legacy selection path."
        )

    def score_assignment_candidates_from_context(self, context, candidates):  # type: ignore[override]
        feature_matrix = np.asarray(
            [
                [candidate.feature(feature_name) for feature_name in self._artifact.feature_names]
                for candidate in candidates
            ],
            dtype=float,
        )
        scores = self._artifact.score_matrix(feature_matrix)
        return tuple(float(value) for value in scores)


class GraphDispatchArtifactPolicy(DispatchPolicy):
    """Dispatch policy backed by a trained PyG graph-conditioned scorer."""

    name = "trained_graph_dispatch_model"
    policy_score_label = "graph_dispatch_logit"

    def __init__(
        self,
        *,
        model: GraphDispatchScorer,
        candidate_feature_names: tuple[str, ...],
        node_feature_names: tuple[str, ...],
        edge_feature_names: tuple[str, ...],
    ) -> None:
        self._model = model
        self._candidate_feature_names = candidate_feature_names
        self._node_feature_names = node_feature_names
        self._edge_feature_names = edge_feature_names

    def _device(self):
        torch = require_dependency("torch", feature="trained graph-dispatch policy inference")
        return next(self._model.parameters(), torch.empty((), device="cpu")).device

    def select_assignment_from_context(self, context: DispatchContext) -> DispatchDecision | None:
        candidates = build_candidate_assignment_observations(context)
        if not candidates:
            return None

        torch = require_dependency("torch", feature="trained graph-dispatch policy inference")
        example = build_graph_dispatch_example_from_context(
            context,
            dispatch_index=0,
            candidate_feature_names=self._candidate_feature_names,
            node_feature_names=self._node_feature_names,
            edge_feature_names=self._edge_feature_names,
            dispatch_group_id="live::dispatch",
        )
        device = self._device()
        with torch.no_grad():
            logits, _ = self._model(
                node_features=torch.tensor(example.node_features, dtype=torch.float32, device=device),
                edge_index=torch.tensor(example.edge_index, dtype=torch.long, device=device),
                edge_features=torch.tensor(example.edge_features, dtype=torch.float32, device=device),
                candidate_features=torch.tensor(example.candidate_features, dtype=torch.float32, device=device),
            )
        scores = logits.detach().cpu().numpy()
        best_candidate = _select_best_candidate(candidates, scores)
        return _dispatch_decision_for_candidate(best_candidate)

    def select_assignment(self, idle_robots, ready_tasks, environment):  # type: ignore[override]
        raise CandidateScoringError(
            "GraphDispatchArtifactPolicy requires dispatch contexts and should not use the legacy selection path."
        )

    def score_assignment_candidates_from_context(self, context, candidates):  # type: ignore[override]
        torch = require_dependency("torch", feature="trained graph-dispatch policy inference")
        example = build_graph_dispatch_example_from_context(
            context,
            dispatch_index=0,
            candidate_feature_names=self._candidate_feature_names,
            node_feature_names=self._node_feature_names,
            edge_feature_names=self._edge_feature_names,
            dispatch_group_id="live::dispatch",
        )
        device = self._device()
        with torch.no_grad():
            logits, _ = self._model(
                node_features=torch.tensor(example.node_features, dtype=torch.float32, device=device),
                edge_index=torch.tensor(example.edge_index, dtype=torch.long, device=device),
                edge_features=torch.tensor(example.edge_features, dtype=torch.float32, device=device),
                candidate_features=torch.tensor(example.candidate_features, dtype=torch.float32, device=device),
            )
        scores = logits.detach().cpu().numpy()
        return tuple(float(value) for value in scores)


def build_candidate_assignment_observations(
    context: DispatchContext,
) -> tuple[CandidateAssignmentObservation, ...]:
    """Build candidate robot-task observations for a dispatch decision."""

    robot_by_id = {robot.robot_id: robot for robot in context.robot_observations}
    task_by_id = {task.task_id: task for task in context.task_observations if task.is_ready}
    node_feature_by_id = {node.node_id: node for node in context.graph_features.nodes}
    arc_feature_by_edge = {
        (arc.source_id, arc.target_id): arc
        for arc in context.graph_features.arcs
    }
    candidates: list[CandidateAssignmentObservation] = []

    for robot_state in context.idle_robots:
        robot_observation = robot_by_id[robot_state.spec.robot_id]
        charge_candidate = _build_charge_candidate(
            context=context,
            robot_state=robot_state,
            robot_observation=robot_observation,
            node_feature_by_id=node_feature_by_id,
            arc_feature_by_edge=arc_feature_by_edge,
        )
        if charge_candidate is not None:
            candidates.append(charge_candidate)
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
            battery_estimate = estimate_task_action(
                context.environment,
                robot_node=robot_state.current_node,
                pickup_node=task.pickup_node,
                dropoff_node=task.dropoff_node,
                battery_level=robot_state.battery_level,
                speed_multiplier=robot_state.spec.speed_multiplier,
                battery_config=context.battery_config,
            )
            if battery_enabled(context.battery_config) and (
                battery_estimate is None or not battery_estimate.charger_reachable_after_action
            ):
                continue
            candidates.append(
                CandidateAssignmentObservation(
                    robot_id=robot_state.spec.robot_id,
                    action_type="task",
                    task_id=task.task_id,
                    charging_node_id=None,
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
                        pickup_path=pickup_path,
                        pickup_edges=pickup_edges,
                        dropoff_path=dropoff_path,
                        dropoff_edges=dropoff_edges,
                        node_feature_by_id=node_feature_by_id,
                        arc_feature_by_edge=arc_feature_by_edge,
                        estimated_pickup_delay=estimated_pickup_delay,
                        estimated_dropoff_delay=estimated_dropoff_delay,
                        estimated_pickup_blocked_segments=estimated_pickup_blocked_segments,
                        estimated_dropoff_blocked_segments=estimated_dropoff_blocked_segments,
                        battery_estimate=battery_estimate,
                        is_charge_action=False,
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
    task_observation: TaskObservation | None,
    travel_to_pickup_time: float,
    travel_to_pickup_distance: float,
    pickup_path: tuple[str, ...],
    pickup_edges: tuple[WarehouseEdge, ...],
    dropoff_path: tuple[str, ...],
    dropoff_edges: tuple[WarehouseEdge, ...],
    node_feature_by_id: dict[str, object],
    arc_feature_by_edge: dict[tuple[str, str], object],
    estimated_pickup_delay: float,
    estimated_dropoff_delay: float,
    estimated_pickup_blocked_segments: int,
    estimated_dropoff_blocked_segments: int,
    battery_estimate,
    is_charge_action: bool,
) -> dict[str, float]:
    pickup_node_id = (
        task_observation.pickup_node
        if task_observation is not None
        else pickup_path[-1]
    )
    dropoff_node_id = (
        task_observation.dropoff_node
        if task_observation is not None
        else (dropoff_path[-1] if dropoff_path else pickup_path[-1])
    )
    pickup_node_features = node_feature_by_id[pickup_node_id]
    dropoff_node_features = node_feature_by_id[dropoff_node_id]
    pickup_path_features = _path_structure_features(
        path_nodes=pickup_path,
        path_edges=pickup_edges,
        node_feature_by_id=node_feature_by_id,
        arc_feature_by_edge=arc_feature_by_edge,
    )
    dropoff_path_features = _path_structure_features(
        path_nodes=dropoff_path,
        path_edges=dropoff_edges,
        node_feature_by_id=node_feature_by_id,
        arc_feature_by_edge=arc_feature_by_edge,
    )
    return {
        "travel_to_pickup_time": travel_to_pickup_time,
        "travel_to_pickup_distance": travel_to_pickup_distance,
        "pickup_to_dropoff_time": (
            0.0 if task_observation is None else task_observation.pickup_to_dropoff_travel_time
        ),
        "pickup_to_dropoff_distance": (
            0.0 if task_observation is None else task_observation.pickup_to_dropoff_distance
        ),
        "pickup_node_inbound_degree": float(pickup_node_features.inbound_degree),
        "pickup_node_outbound_degree": float(pickup_node_features.outbound_degree),
        "dropoff_node_inbound_degree": float(dropoff_node_features.inbound_degree),
        "dropoff_node_outbound_degree": float(dropoff_node_features.outbound_degree),
        "travel_to_pickup_mean_transit_count": pickup_path_features["mean_transit_count"],
        "travel_to_pickup_max_transit_count": pickup_path_features["max_transit_count"],
        "travel_to_pickup_mean_arc_traversal_count": pickup_path_features["mean_arc_traversal_count"],
        "travel_to_pickup_max_arc_traversal_count": pickup_path_features["max_arc_traversal_count"],
        "pickup_to_dropoff_mean_transit_count": dropoff_path_features["mean_transit_count"],
        "pickup_to_dropoff_max_transit_count": dropoff_path_features["max_transit_count"],
        "pickup_to_dropoff_mean_arc_traversal_count": dropoff_path_features["mean_arc_traversal_count"],
        "pickup_to_dropoff_max_arc_traversal_count": dropoff_path_features["max_arc_traversal_count"],
        "task_age": 0.0 if task_observation is None else task_observation.age,
        "task_priority": 0.0 if task_observation is None else float(task_observation.priority),
        "task_service_time_estimate": 0.0 if task_observation is None else task_observation.service_time_estimate,
        "due_time_remaining": (
            0.0
            if task_observation is None or task_observation.due_time_remaining is None
            else task_observation.due_time_remaining
        ),
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
        "battery_fraction": robot_observation.battery_fraction,
        "estimated_action_energy": 0.0 if battery_estimate is None else battery_estimate.estimated_action_energy,
        "post_action_battery_fraction": 1.0 if battery_estimate is None else battery_estimate.post_action_battery_fraction,
        "charger_reachable_after_action": (
            1.0
            if battery_estimate is None or battery_estimate.charger_reachable_after_action
            else 0.0
        ),
        "is_charge_action": 1.0 if is_charge_action else 0.0,
    }


def _path_structure_features(
    *,
    path_nodes: tuple[str, ...],
    path_edges: tuple[WarehouseEdge, ...],
    node_feature_by_id: dict[str, object],
    arc_feature_by_edge: dict[tuple[str, str], object],
) -> dict[str, float]:
    interior_nodes = path_nodes[1:-1]
    node_transit_counts = [
        float(node_feature_by_id[node_id].shortest_path_transit_count)
        for node_id in interior_nodes
    ]
    arc_traversal_counts = [
        float(arc_feature_by_edge[(edge.source, edge.target)].shortest_path_traversal_count)
        for edge in path_edges
    ]
    return {
        "mean_transit_count": (
            sum(node_transit_counts) / len(node_transit_counts) if node_transit_counts else 0.0
        ),
        "max_transit_count": max(node_transit_counts, default=0.0),
        "mean_arc_traversal_count": (
            sum(arc_traversal_counts) / len(arc_traversal_counts) if arc_traversal_counts else 0.0
        ),
        "max_arc_traversal_count": max(arc_traversal_counts, default=0.0),
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
            _descending_string_key(candidates[index].candidate_id),
        ),
    )
    return candidates[best_index]


def _build_charge_candidate(
    *,
    context: DispatchContext,
    robot_state,
    robot_observation: RobotObservation,
    node_feature_by_id: dict[str, object],
    arc_feature_by_edge: dict[tuple[str, str], object],
) -> CandidateAssignmentObservation | None:
    if not battery_enabled(context.battery_config):
        return None
    charge_option = nearest_charging_option(
        context.environment,
        source_node=robot_state.current_node,
        speed_multiplier=robot_state.spec.speed_multiplier,
        battery_config=context.battery_config,
    )
    if charge_option is None:
        return None
    battery_estimate = estimate_charge_action(
        context.environment,
        robot_node=robot_state.current_node,
        charging_node_id=charge_option.charging_node_id,
        battery_level=robot_state.battery_level,
        speed_multiplier=robot_state.spec.speed_multiplier,
        battery_config=context.battery_config,
    )
    if not battery_estimate.charger_reachable_after_action:
        return None
    if (
        charge_option.charging_node_id == robot_state.current_node
        and robot_observation.battery_fraction >= 1.0 - 1e-9
    ):
        return None
    pickup_path = context.environment.shortest_path(
        robot_state.current_node,
        charge_option.charging_node_id,
        weight="travel_time",
    )
    pickup_edges = context.environment.shortest_path_edges(
        robot_state.current_node,
        charge_option.charging_node_id,
        weight="travel_time",
    )
    estimated_pickup_delay, estimated_pickup_blocked_segments = _estimate_congestion(
        path_edges=pickup_edges,
        path_nodes=pickup_path,
        congestion_observation=context.congestion_observation,
        start_time=context.current_time,
        speed_multiplier=robot_state.spec.speed_multiplier,
    )
    charge_node = charge_option.charging_node_id
    charge_zone = context.environment.zone_for_node(charge_node)
    return CandidateAssignmentObservation(
        robot_id=robot_state.spec.robot_id,
        action_type="charge",
        task_id=None,
        charging_node_id=charge_node,
        robot_current_node=robot_observation.current_node,
        robot_current_zone=robot_observation.current_zone,
        task_pickup_node=charge_node,
        task_dropoff_node=charge_node,
        task_source_zone=charge_zone,
        task_destination_zone=charge_zone,
        feature_values=_candidate_feature_values(
            context=context,
            robot_observation=robot_observation,
            task_observation=None,
            travel_to_pickup_time=charge_option.travel_time,
            travel_to_pickup_distance=charge_option.distance,
            pickup_path=pickup_path,
            pickup_edges=pickup_edges,
            dropoff_path=(charge_node,),
            dropoff_edges=(),
            node_feature_by_id=node_feature_by_id,
            arc_feature_by_edge=arc_feature_by_edge,
            estimated_pickup_delay=estimated_pickup_delay,
            estimated_dropoff_delay=0.0,
            estimated_pickup_blocked_segments=estimated_pickup_blocked_segments,
            estimated_dropoff_blocked_segments=0,
            battery_estimate=battery_estimate,
            is_charge_action=True,
        ),
    )


def _dispatch_decision_for_candidate(candidate: CandidateAssignmentObservation) -> DispatchDecision:
    if candidate.action_type == "charge":
        assert candidate.charging_node_id is not None
        return DispatchDecision.for_charge(
            robot_id=candidate.robot_id,
            charging_node_id=candidate.charging_node_id,
        )
    assert candidate.task_id is not None
    return DispatchDecision.for_task(robot_id=candidate.robot_id, task_id=candidate.task_id)


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
