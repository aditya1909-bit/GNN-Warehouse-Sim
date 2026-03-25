"""Graph-conditioned dispatch datasets and live graph example builders."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from warehouse_sim.candidate_features import SUPPORTED_CANDIDATE_FEATURES
from warehouse_sim.graph.models import NodeType
from warehouse_sim.learning.datasets import _parse_scalar
from warehouse_sim.learning.features import validate_candidate_feature_names
from warehouse_sim.simulation.models import DispatchArcObservationRecord, DispatchNodeObservationRecord

if TYPE_CHECKING:
    from warehouse_sim.policies.observation import DispatchContext
    from warehouse_sim.policies.base import DispatchDecision


NODE_TYPE_FEATURE_NAMES = tuple(f"node_type_{node_type.value}" for node_type in NodeType)
DEFAULT_GRAPH_NODE_FEATURES = (
    "x",
    "y",
    "inbound_degree",
    "outbound_degree",
    "shortest_path_transit_count",
    *NODE_TYPE_FEATURE_NAMES,
    "is_robot_occupied",
    "robot_count",
    "is_ready_task_pickup",
    "is_ready_task_dropoff",
    "is_reserved_node",
    "reserved_time_remaining",
)
DEFAULT_GRAPH_EDGE_FEATURES = (
    "distance",
    "travel_time",
    "shortest_path_traversal_count",
    "is_reserved_arc",
    "reserved_time_remaining",
)
DEFAULT_GRAPH_CANDIDATE_FEATURES = tuple(SUPPORTED_CANDIDATE_FEATURES)


@dataclass(frozen=True)
class GraphDispatchExample:
    """One dispatch-event graph example for offline or live scoring."""

    dispatch_group_id: str
    dispatch_index: int
    decision_time: float
    node_ids: tuple[str, ...]
    edge_ids: tuple[tuple[str, str], ...]
    edge_index: np.ndarray
    node_features: np.ndarray
    edge_features: np.ndarray
    candidate_features: np.ndarray
    labels: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_features.shape[0])


@dataclass(frozen=True)
class GraphDispatchDataset:
    """Collection of dispatch-level graph examples with grouping metadata."""

    source: Path
    examples: tuple[GraphDispatchExample, ...]

    @property
    def row_count(self) -> int:
        return len(self.examples)

    @property
    def group_ids(self) -> np.ndarray:
        return np.asarray([example.dispatch_group_id for example in self.examples], dtype=object)

    @property
    def dispatch_groups(self) -> tuple[str, ...]:
        return tuple(example.dispatch_group_id for example in self.examples)

    @property
    def group_count(self) -> int:
        return len(self.examples)

    def subset(self, indices: np.ndarray) -> "GraphDispatchDataset":
        return GraphDispatchDataset(
            source=self.source,
            examples=tuple(self.examples[index] for index in indices.tolist()),
        )

    def split_values(self, split_unit: str) -> np.ndarray:
        if split_unit == "dispatch_group":
            return np.asarray([example.dispatch_group_id for example in self.examples], dtype=object)
        if split_unit == "run":
            return np.asarray([str(example.metadata["run_id"]) for example in self.examples], dtype=object)
        if split_unit == "scenario":
            return np.asarray([str(example.metadata["scenario_name"]) for example in self.examples], dtype=object)
        raise ValueError(f"Unsupported split unit: {split_unit}")


def load_graph_dispatch_dataset(
    source: Path,
    *,
    candidate_feature_names: tuple[str, ...] | list[str] | None = None,
    node_feature_names: tuple[str, ...] | list[str] | None = None,
    edge_feature_names: tuple[str, ...] | list[str] | None = None,
) -> GraphDispatchDataset:
    """Load per-dispatch graph examples from exported observation datasets."""

    resolved_source = source.resolve()
    sources = _resolve_graph_dataset_sources(resolved_source)
    candidate_names = (
        DEFAULT_GRAPH_CANDIDATE_FEATURES
        if candidate_feature_names is None
        else validate_candidate_feature_names(candidate_feature_names)
    )
    resolved_node_features = tuple(node_feature_names or DEFAULT_GRAPH_NODE_FEATURES)
    resolved_edge_features = tuple(edge_feature_names or DEFAULT_GRAPH_EDGE_FEATURES)

    examples: list[GraphDispatchExample] = []
    for dataset_index, dataset_source in enumerate(sources):
        graph_nodes = _read_csv_rows(dataset_source.graph_nodes_path)
        graph_arcs = _read_csv_rows(dataset_source.graph_arcs_path)
        dispatch_rows = _read_csv_rows(dataset_source.dispatch_observations_path)
        dispatch_node_rows = _read_csv_rows(dataset_source.dispatch_node_observations_path)
        dispatch_arc_rows = _read_csv_rows(dataset_source.dispatch_arc_observations_path)
        manifest_payload = dataset_source.manifest_payload or {}
        run_id = str(manifest_payload.get("run_id") or f"dataset_{dataset_index}")
        scenario_name = str(
            manifest_payload.get("scenario_name")
            or manifest_payload.get("experiment_name")
            or run_id
        )

        dispatch_candidates: dict[int, list[dict[str, object]]] = {}
        for row in dispatch_rows:
            dispatch_candidates.setdefault(int(row["dispatch_index"]), []).append(row)
        dispatch_nodes: dict[int, list[dict[str, object]]] = {}
        for row in dispatch_node_rows:
            dispatch_nodes.setdefault(int(row["dispatch_index"]), []).append(row)
        dispatch_arcs: dict[int, list[dict[str, object]]] = {}
        for row in dispatch_arc_rows:
            dispatch_arcs.setdefault(int(row["dispatch_index"]), []).append(row)

        for dispatch_index, candidate_rows in sorted(dispatch_candidates.items()):
            group_id = f"{run_id}::dispatch_{dispatch_index}"
            example = build_graph_dispatch_example_from_tables(
                dispatch_group_id=group_id,
                dispatch_index=dispatch_index,
                decision_time=float(candidate_rows[0]["decision_time"]),
                graph_node_rows=graph_nodes,
                graph_arc_rows=graph_arcs,
                dispatch_node_rows=dispatch_nodes[dispatch_index],
                dispatch_arc_rows=dispatch_arcs[dispatch_index],
                candidate_rows=candidate_rows,
                candidate_feature_names=candidate_names,
                node_feature_names=resolved_node_features,
                edge_feature_names=resolved_edge_features,
                metadata={
                    "run_id": run_id,
                    "scenario_name": scenario_name,
                    "source_policy_name": str(manifest_payload.get("policy_name") or "unknown_policy"),
                    "selected_robot_id": candidate_rows[0]["selected_robot_id"],
                    "selected_task_id": candidate_rows[0]["selected_task_id"],
                },
            )
            examples.append(example)
    return GraphDispatchDataset(source=resolved_source, examples=tuple(examples))


def build_graph_dispatch_example_from_context(
    context: DispatchContext,
    *,
    dispatch_index: int,
    candidate_feature_names: tuple[str, ...] | list[str] | None = None,
    node_feature_names: tuple[str, ...] | list[str] | None = None,
    edge_feature_names: tuple[str, ...] | list[str] | None = None,
    decision: "DispatchDecision | None" = None,
    dispatch_group_id: str = "live::dispatch",
) -> GraphDispatchExample:
    """Build one live graph example directly from a dispatch context."""

    from warehouse_sim.policies.scoring import build_candidate_assignment_observations

    candidates = build_candidate_assignment_observations(context)
    labels = np.zeros(len(candidates), dtype=int)
    if decision is not None:
        for index, candidate in enumerate(candidates):
            if candidate.robot_id == decision.robot_id and candidate.task_id == decision.task_id:
                labels[index] = 1
                break

    candidate_rows = [
        {
            "dispatch_index": dispatch_index,
            "decision_time": context.current_time,
            "selected_robot_id": "" if decision is None else decision.robot_id,
            "selected_task_id": "" if decision is None else decision.task_id,
            "candidate_robot_id": candidate.robot_id,
            "candidate_task_id": candidate.task_id,
            "is_selected": bool(labels[index]),
            **candidate.feature_values,
        }
        for index, candidate in enumerate(candidates)
    ]

    graph_node_rows = [
        {
            "node_id": node.node_id,
            "x": node.x,
            "y": node.y,
            "node_type": node.node_type,
            "zone_id": node.zone_id,
            "inbound_degree": node.inbound_degree,
            "outbound_degree": node.outbound_degree,
            "shortest_path_transit_count": node.shortest_path_transit_count,
        }
        for node in context.graph_features.nodes
    ]
    graph_arc_rows = [
        {
            "source_id": arc.source_id,
            "target_id": arc.target_id,
            "distance": arc.distance,
            "travel_time": arc.travel_time,
            "shortest_path_traversal_count": arc.shortest_path_traversal_count,
        }
        for arc in context.graph_features.arcs
    ]
    dispatch_node_rows = [
        record.__dict__
        for record in build_dispatch_node_observation_records(
            context=context,
            dispatch_index=dispatch_index,
            decision=decision,
        )
    ]
    dispatch_arc_rows = [
        record.__dict__
        for record in build_dispatch_arc_observation_records(
            context=context,
            dispatch_index=dispatch_index,
        )
    ]
    return build_graph_dispatch_example_from_tables(
        dispatch_group_id=dispatch_group_id,
        dispatch_index=dispatch_index,
        decision_time=context.current_time,
        graph_node_rows=graph_node_rows,
        graph_arc_rows=graph_arc_rows,
        dispatch_node_rows=dispatch_node_rows,
        dispatch_arc_rows=dispatch_arc_rows,
        candidate_rows=candidate_rows,
        candidate_feature_names=tuple(candidate_feature_names or DEFAULT_GRAPH_CANDIDATE_FEATURES),
        node_feature_names=tuple(node_feature_names or DEFAULT_GRAPH_NODE_FEATURES),
        edge_feature_names=tuple(edge_feature_names or DEFAULT_GRAPH_EDGE_FEATURES),
        metadata={
            "run_id": dispatch_group_id.split("::", maxsplit=1)[0],
            "scenario_name": "live",
        },
    )


def build_graph_dispatch_example_from_tables(
    *,
    dispatch_group_id: str,
    dispatch_index: int,
    decision_time: float,
    graph_node_rows: list[dict[str, object]],
    graph_arc_rows: list[dict[str, object]],
    dispatch_node_rows: list[dict[str, object]],
    dispatch_arc_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    candidate_feature_names: tuple[str, ...],
    node_feature_names: tuple[str, ...],
    edge_feature_names: tuple[str, ...],
    metadata: dict[str, object],
) -> GraphDispatchExample:
    """Build one graph example from static and dispatch-level tables."""

    node_row_by_id = {str(row["node_id"]): row for row in graph_node_rows}
    dispatch_node_by_id = {str(row["node_id"]): row for row in dispatch_node_rows}
    ordered_node_ids = tuple(sorted(node_row_by_id))
    node_index_by_id = {node_id: index for index, node_id in enumerate(ordered_node_ids)}

    ordered_edge_rows = sorted(
        graph_arc_rows,
        key=lambda row: (str(row["source_id"]), str(row["target_id"])),
    )
    dispatch_arc_by_edge = {
        (str(row["source_id"]), str(row["target_id"])): row
        for row in dispatch_arc_rows
    }

    node_features = np.asarray(
        [
            [
                _node_feature_value(
                    node_row=node_row_by_id[node_id],
                    dispatch_row=dispatch_node_by_id[node_id],
                    feature_name=feature_name,
                )
                for feature_name in node_feature_names
            ]
            for node_id in ordered_node_ids
        ],
        dtype=float,
    )
    edge_index = np.asarray(
        [
            [node_index_by_id[str(row["source_id"])] for row in ordered_edge_rows],
            [node_index_by_id[str(row["target_id"])] for row in ordered_edge_rows],
        ],
        dtype=int,
    )
    edge_features = np.asarray(
        [
            [
                _edge_feature_value(
                    edge_row=row,
                    dispatch_row=dispatch_arc_by_edge[(str(row["source_id"]), str(row["target_id"]))],
                    feature_name=feature_name,
                )
                for feature_name in edge_feature_names
            ]
            for row in ordered_edge_rows
        ],
        dtype=float,
    )
    ordered_candidates = list(candidate_rows)
    candidate_features = np.asarray(
        [
            [float(row[feature_name]) for feature_name in candidate_feature_names]
            for row in ordered_candidates
        ],
        dtype=float,
    )
    labels = np.asarray(
        [1 if bool(row["is_selected"]) else 0 for row in ordered_candidates],
        dtype=int,
    )
    return GraphDispatchExample(
        dispatch_group_id=dispatch_group_id,
        dispatch_index=dispatch_index,
        decision_time=decision_time,
        node_ids=ordered_node_ids,
        edge_ids=tuple((str(row["source_id"]), str(row["target_id"])) for row in ordered_edge_rows),
        edge_index=edge_index,
        node_features=node_features,
        edge_features=edge_features,
        candidate_features=candidate_features,
        labels=labels,
        metadata={
            **dict(metadata),
            "candidate_robot_ids": tuple(str(row["candidate_robot_id"]) for row in ordered_candidates),
            "candidate_task_ids": tuple(str(row["candidate_task_id"]) for row in ordered_candidates),
        },
    )


def build_dispatch_node_observation_records(
    *,
    context: DispatchContext,
    dispatch_index: int,
    decision: "DispatchDecision | None" = None,
) -> tuple[DispatchNodeObservationRecord, ...]:
    """Build dynamic node records for one dispatch event."""

    robot_counts: dict[str, int] = {}
    for robot in context.robot_observations:
        robot_counts[robot.current_node] = robot_counts.get(robot.current_node, 0) + 1

    ready_pickups = {task.pickup_node for task in context.ready_tasks}
    ready_dropoffs = {task.dropoff_node for task in context.ready_tasks}
    selected_pickups: set[str] = set()
    selected_dropoffs: set[str] = set()
    if decision is not None:
        selected_task = next(task for task in context.ready_tasks if task.task_id == decision.task_id)
        selected_pickups.add(selected_task.pickup_node)
        selected_dropoffs.add(selected_task.dropoff_node)

    node_reserved_until = {
        reservation.resource_id: reservation.reserved_until
        for reservation in context.congestion_observation.node_reservations
    }
    records = [
        DispatchNodeObservationRecord(
            dispatch_index=dispatch_index,
            decision_time=context.current_time,
            node_id=node.node_id,
            is_robot_occupied=node.node_id in robot_counts,
            robot_count=robot_counts.get(node.node_id, 0),
            is_ready_task_pickup=node.node_id in ready_pickups,
            is_ready_task_dropoff=node.node_id in ready_dropoffs,
            is_selected_task_pickup=node.node_id in selected_pickups,
            is_selected_task_dropoff=node.node_id in selected_dropoffs,
            is_reserved_node=node.node_id in node_reserved_until,
            reserved_time_remaining=max(node_reserved_until.get(node.node_id, 0.0) - context.current_time, 0.0),
        )
        for node in context.graph_features.nodes
    ]
    return tuple(records)


def build_dispatch_arc_observation_records(
    *,
    context: DispatchContext,
    dispatch_index: int,
) -> tuple[DispatchArcObservationRecord, ...]:
    """Build dynamic directed-arc records for one dispatch event."""

    edge_reserved_until = {}
    for reservation in context.congestion_observation.edge_reservations:
        source_id, target_id = str(reservation.resource_id).split("->", maxsplit=1)
        edge_reserved_until[(source_id, target_id)] = reservation.reserved_until
    records = [
        DispatchArcObservationRecord(
            dispatch_index=dispatch_index,
            decision_time=context.current_time,
            source_id=arc.source_id,
            target_id=arc.target_id,
            is_reserved_arc=(arc.source_id, arc.target_id) in edge_reserved_until,
            reserved_time_remaining=max(
                edge_reserved_until.get((arc.source_id, arc.target_id), 0.0) - context.current_time,
                0.0,
            ),
        )
        for arc in context.graph_features.arcs
    ]
    return tuple(records)


@dataclass(frozen=True)
class _GraphDatasetSource:
    manifest_path: Path | None
    manifest_payload: dict[str, object] | None
    graph_nodes_path: Path
    graph_arcs_path: Path
    dispatch_observations_path: Path
    dispatch_node_observations_path: Path
    dispatch_arc_observations_path: Path


def _resolve_graph_dataset_sources(source: Path) -> tuple[_GraphDatasetSource, ...]:
    if source.is_file():
        if source.name != "dataset_manifest.json":
            raise ValueError("Graph dispatch datasets must be loaded from a manifest or directory.")
        return (_graph_dataset_source_from_manifest(source),)

    if not source.is_dir():
        raise ValueError(f"Dataset source does not exist: {source}")

    manifest_paths = tuple(sorted(source.glob("**/dataset_manifest.json")))
    if not manifest_paths:
        direct_manifest = source / "dataset_manifest.json"
        if direct_manifest.exists():
            manifest_paths = (direct_manifest,)
    if not manifest_paths:
        raise ValueError(f"Could not find graph dataset manifests under {source}")
    return tuple(_graph_dataset_source_from_manifest(path) for path in manifest_paths)


def _graph_dataset_source_from_manifest(path: Path) -> _GraphDatasetSource:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = payload["files"]
    return _GraphDatasetSource(
        manifest_path=path.resolve(),
        manifest_payload=payload,
        graph_nodes_path=(path.parent / str(files["graph_nodes"])).resolve(),
        graph_arcs_path=(path.parent / str(files["graph_arcs"])).resolve(),
        dispatch_observations_path=(path.parent / str(files["dispatch_observations"])).resolve(),
        dispatch_node_observations_path=(path.parent / str(files["dispatch_node_observations"])).resolve(),
        dispatch_arc_observations_path=(path.parent / str(files["dispatch_arc_observations"])).resolve(),
    )


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is empty: {path}")
        return [
            {key: _parse_scalar(value) for key, value in row.items() if key is not None}
            for row in reader
        ]


def _node_feature_value(
    *,
    node_row: dict[str, object],
    dispatch_row: dict[str, object],
    feature_name: str,
) -> float:
    if feature_name in {"x", "y", "inbound_degree", "outbound_degree", "shortest_path_transit_count"}:
        return float(node_row[feature_name])
    if feature_name.startswith("node_type_"):
        node_type_value = str(node_row["node_type"])
        return 1.0 if feature_name == f"node_type_{node_type_value}" else 0.0
    return float(dispatch_row[feature_name])


def _edge_feature_value(
    *,
    edge_row: dict[str, object],
    dispatch_row: dict[str, object],
    feature_name: str,
) -> float:
    if feature_name in {"distance", "travel_time", "shortest_path_traversal_count"}:
        return float(edge_row[feature_name])
    return float(dispatch_row[feature_name])
