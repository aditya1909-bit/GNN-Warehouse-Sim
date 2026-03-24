"""Core graph models and pathfinding for warehouse layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from heapq import heappop, heappush
from math import isfinite


class GraphValidationError(ValueError):
    """Raised when a warehouse graph or layout is invalid."""


class PathNotFoundError(LookupError):
    """Raised when no path exists between two graph nodes."""


class NodeType(StrEnum):
    """Supported node roles for warehouse layouts."""

    TRANSIT = "transit"
    STORAGE = "storage"
    PICK_STATION = "pick_station"
    DROPOFF = "dropoff"
    CHARGING = "charging"
    STAGING = "staging"
    OBSTACLE = "obstacle"


@dataclass(frozen=True)
class WarehouseNode:
    """A physical location in the warehouse graph."""

    node_id: str
    x: int
    y: int
    node_type: NodeType = NodeType.TRANSIT
    zone: str | None = None
    attributes: dict[str, str | int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise GraphValidationError("node_id must be non-empty.")


@dataclass(frozen=True)
class WarehouseEdge:
    """A traversable connection between two warehouse nodes."""

    source: str
    target: str
    distance: float
    travel_time: float
    directed: bool = False

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise GraphValidationError("Edge endpoints must be non-empty.")
        if self.source == self.target:
            raise GraphValidationError("Self-loop edges are not supported.")
        if not isfinite(self.distance) or self.distance <= 0:
            raise GraphValidationError("distance must be a finite value > 0.")
        if not isfinite(self.travel_time) or self.travel_time <= 0:
            raise GraphValidationError("travel_time must be a finite value > 0.")


class WarehouseGraph:
    """Directed-or-undirected warehouse topology with weighted shortest paths."""

    def __init__(self) -> None:
        self._nodes: dict[str, WarehouseNode] = {}
        self._adjacency: dict[str, dict[str, WarehouseEdge]] = {}
        self._edges: list[WarehouseEdge] = []

    def add_node(self, node: WarehouseNode) -> None:
        """Add a node to the graph."""

        if node.node_id in self._nodes:
            raise GraphValidationError(f"Duplicate node_id: {node.node_id}")
        self._nodes[node.node_id] = node
        self._adjacency[node.node_id] = {}

    def add_edge(self, edge: WarehouseEdge) -> None:
        """Add a weighted edge to the graph."""

        if edge.source not in self._nodes:
            raise GraphValidationError(f"Unknown source node: {edge.source}")
        if edge.target not in self._nodes:
            raise GraphValidationError(f"Unknown target node: {edge.target}")

        self._adjacency[edge.source][edge.target] = edge
        if not edge.directed:
            reverse = WarehouseEdge(
                source=edge.target,
                target=edge.source,
                distance=edge.distance,
                travel_time=edge.travel_time,
                directed=False,
            )
            self._adjacency[edge.target][edge.source] = reverse
        self._edges.append(edge)

    def node(self, node_id: str) -> WarehouseNode:
        """Fetch a node by id."""

        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise GraphValidationError(f"Unknown node_id: {node_id}") from exc

    def nodes(self) -> tuple[WarehouseNode, ...]:
        """Return all graph nodes."""

        return tuple(self._nodes.values())

    def edges(self) -> tuple[WarehouseEdge, ...]:
        """Return the explicitly added graph edges."""

        return tuple(self._edges)

    def neighbors(self, node_id: str) -> tuple[WarehouseNode, ...]:
        """Return outward neighbors of a node."""

        self.node(node_id)
        return tuple(self._nodes[neighbor_id] for neighbor_id in sorted(self._adjacency[node_id]))

    def shortest_path(self, source: str, target: str, weight: str = "travel_time") -> tuple[str, ...]:
        """Compute the shortest path between two nodes."""

        _, path = self._dijkstra(source=source, target=target, weight=weight)
        return path

    def shortest_path_length(
        self,
        source: str,
        target: str,
        weight: str = "travel_time",
    ) -> float:
        """Compute the shortest-path cost between two nodes."""

        distance, _ = self._dijkstra(source=source, target=target, weight=weight)
        return distance

    def travel_time(self, source: str, target: str) -> float:
        """Convenience alias for shortest travel time."""

        return self.shortest_path_length(source=source, target=target, weight="travel_time")

    def distance(self, source: str, target: str) -> float:
        """Convenience alias for shortest travel distance."""

        return self.shortest_path_length(source=source, target=target, weight="distance")

    def _dijkstra(self, source: str, target: str, weight: str) -> tuple[float, tuple[str, ...]]:
        if weight not in {"travel_time", "distance"}:
            raise GraphValidationError("weight must be 'travel_time' or 'distance'.")

        self.node(source)
        self.node(target)
        queue: list[tuple[float, str]] = [(0.0, source)]
        costs: dict[str, float] = {source: 0.0}
        previous: dict[str, str | None] = {source: None}

        while queue:
            current_cost, current_node = heappop(queue)
            if current_node == target:
                break
            if current_cost > costs[current_node]:
                continue

            for neighbor_id, edge in self._adjacency[current_node].items():
                edge_cost = getattr(edge, weight)
                next_cost = current_cost + edge_cost
                if next_cost < costs.get(neighbor_id, float("inf")):
                    costs[neighbor_id] = next_cost
                    previous[neighbor_id] = current_node
                    heappush(queue, (next_cost, neighbor_id))

        if target not in costs:
            raise PathNotFoundError(f"No path found from {source} to {target}.")

        path: list[str] = [target]
        while previous[path[-1]] is not None:
            path.append(previous[path[-1]] or "")
        path.reverse()
        return costs[target], tuple(path)

