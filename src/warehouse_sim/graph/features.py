"""Graph featurization helpers for future learned warehouse policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from warehouse_sim.graph.models import NodeType, PathNotFoundError, WarehouseGraph


@dataclass(frozen=True)
class GraphNodeFeature:
    """Static node-level features derived from a warehouse graph."""

    node_id: str
    x: int
    y: int
    node_type: NodeType
    zone_id: str | None
    inbound_degree: int
    outbound_degree: int
    shortest_path_transit_count: int


@dataclass(frozen=True)
class GraphArcFeature:
    """Directed travel arc emitted from the warehouse topology."""

    source_id: str
    target_id: str
    distance: float
    travel_time: float
    shortest_path_traversal_count: int


@dataclass(frozen=True)
class GraphFeatures:
    """Static graph feature payload suitable for policy observations."""

    nodes: tuple[GraphNodeFeature, ...]
    arcs: tuple[GraphArcFeature, ...]


def build_graph_features(
    graph: WarehouseGraph,
    zone_lookup: Callable[[str], str | None] | None = None,
) -> GraphFeatures:
    """Build static node and directed-arc features from a warehouse graph.

    Undirected graph edges are expanded into two directed travel arcs so future
    graph-learning code can consume a single, explicit message-passing view.
    """

    arcs = _build_arc_features(graph)
    inbound_degree: dict[str, int] = {node.node_id: 0 for node in graph.nodes()}
    outbound_degree: dict[str, int] = {node.node_id: 0 for node in graph.nodes()}
    for arc in arcs:
        outbound_degree[arc.source_id] += 1
        inbound_degree[arc.target_id] += 1
    node_transit_counts, arc_traversal_counts = _shortest_path_usage(graph)

    nodes = tuple(
        GraphNodeFeature(
            node_id=node.node_id,
            x=node.x,
            y=node.y,
            node_type=node.node_type,
            zone_id=zone_lookup(node.node_id) if zone_lookup is not None else node.zone,
            inbound_degree=inbound_degree[node.node_id],
            outbound_degree=outbound_degree[node.node_id],
            shortest_path_transit_count=node_transit_counts[node.node_id],
        )
        for node in sorted(graph.nodes(), key=lambda item: item.node_id)
    )
    enriched_arcs = tuple(
        GraphArcFeature(
            source_id=arc.source_id,
            target_id=arc.target_id,
            distance=arc.distance,
            travel_time=arc.travel_time,
            shortest_path_traversal_count=arc_traversal_counts[(arc.source_id, arc.target_id)],
        )
        for arc in arcs
    )
    return GraphFeatures(nodes=nodes, arcs=enriched_arcs)


def _build_arc_features(graph: WarehouseGraph) -> tuple[GraphArcFeature, ...]:
    arcs: list[GraphArcFeature] = []
    for edge in sorted(graph.edges(), key=lambda item: (item.source, item.target)):
        arcs.append(
            GraphArcFeature(
                source_id=edge.source,
                target_id=edge.target,
                distance=edge.distance,
                travel_time=edge.travel_time,
                shortest_path_traversal_count=0,
            )
        )
        if not edge.directed:
            arcs.append(
                GraphArcFeature(
                    source_id=edge.target,
                    target_id=edge.source,
                    distance=edge.distance,
                    travel_time=edge.travel_time,
                    shortest_path_traversal_count=0,
                )
            )
    return tuple(sorted(arcs, key=lambda item: (item.source_id, item.target_id)))


def _shortest_path_usage(
    graph: WarehouseGraph,
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    node_counts = {node.node_id: 0 for node in graph.nodes()}
    arc_counts: dict[tuple[str, str], int] = {}
    node_ids = [node.node_id for node in sorted(graph.nodes(), key=lambda item: item.node_id)]

    for source_id in node_ids:
        for target_id in node_ids:
            if source_id == target_id:
                continue
            try:
                path = graph.shortest_path(source_id, target_id, weight="travel_time")
            except PathNotFoundError:
                continue
            for node_id in path[1:-1]:
                node_counts[node_id] += 1
            for edge in graph.path_edges(path):
                key = (edge.source, edge.target)
                arc_counts[key] = arc_counts.get(key, 0) + 1

    for edge in _build_arc_features(graph):
        arc_counts.setdefault((edge.source_id, edge.target_id), 0)
    return node_counts, arc_counts
