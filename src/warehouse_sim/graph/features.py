"""Graph featurization helpers for future learned warehouse policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from warehouse_sim.graph.models import NodeType, WarehouseGraph


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


@dataclass(frozen=True)
class GraphArcFeature:
    """Directed travel arc emitted from the warehouse topology."""

    source_id: str
    target_id: str
    distance: float
    travel_time: float


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

    nodes = tuple(
        GraphNodeFeature(
            node_id=node.node_id,
            x=node.x,
            y=node.y,
            node_type=node.node_type,
            zone_id=zone_lookup(node.node_id) if zone_lookup is not None else node.zone,
            inbound_degree=inbound_degree[node.node_id],
            outbound_degree=outbound_degree[node.node_id],
        )
        for node in sorted(graph.nodes(), key=lambda item: item.node_id)
    )
    return GraphFeatures(nodes=nodes, arcs=arcs)


def _build_arc_features(graph: WarehouseGraph) -> tuple[GraphArcFeature, ...]:
    arcs: list[GraphArcFeature] = []
    for edge in sorted(graph.edges(), key=lambda item: (item.source, item.target)):
        arcs.append(
            GraphArcFeature(
                source_id=edge.source,
                target_id=edge.target,
                distance=edge.distance,
                travel_time=edge.travel_time,
            )
        )
        if not edge.directed:
            arcs.append(
                GraphArcFeature(
                    source_id=edge.target,
                    target_id=edge.source,
                    distance=edge.distance,
                    travel_time=edge.travel_time,
                )
            )
    return tuple(sorted(arcs, key=lambda item: (item.source_id, item.target_id)))
