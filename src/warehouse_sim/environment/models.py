"""Warehouse environment domain models."""

from __future__ import annotations

from dataclasses import dataclass

from warehouse_sim.graph import GraphValidationError, WarehouseEdge, WarehouseGraph, WarehouseNode


@dataclass(frozen=True)
class Zone:
    """A named logical grouping of warehouse nodes."""

    zone_id: str
    node_ids: tuple[str, ...]
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise GraphValidationError("zone_id must be non-empty.")
        if not self.node_ids:
            raise GraphValidationError("Zones must contain at least one node.")


@dataclass(frozen=True)
class ObstacleRectangle:
    """Axis-aligned obstacle rectangle in warehouse coordinates."""

    obstacle_id: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        if not self.obstacle_id:
            raise GraphValidationError("obstacle_id must be non-empty.")
        if self.min_x >= self.max_x:
            raise GraphValidationError("ObstacleRectangle min_x must be < max_x.")
        if self.min_y >= self.max_y:
            raise GraphValidationError("ObstacleRectangle min_y must be < max_y.")

    def inflate(self, margin: float) -> "ObstacleRectangle":
        """Return a rectangle expanded by a non-negative margin."""

        if margin < 0:
            raise GraphValidationError("ObstacleRectangle inflation margin must be >= 0.")
        return ObstacleRectangle(
            obstacle_id=f"{self.obstacle_id}__inflated_{margin:.6f}",
            min_x=self.min_x - margin,
            min_y=self.min_y - margin,
            max_x=self.max_x + margin,
            max_y=self.max_y + margin,
        )


class WarehouseEnvironment:
    """A warehouse environment backed by a topology graph and named zones."""

    def __init__(
        self,
        graph: WarehouseGraph,
        zones: tuple[Zone, ...] | None = None,
        obstacles: tuple[ObstacleRectangle, ...] | None = None,
    ) -> None:
        self.graph = graph
        resolved_zones = zones if zones is not None else _derive_zones_from_graph(graph)
        resolved_obstacles = obstacles if obstacles is not None else ()
        self._zones: dict[str, Zone] = {}
        self._node_to_zone: dict[str, str] = {}
        self._obstacles: tuple[ObstacleRectangle, ...] = tuple(resolved_obstacles)

        for zone in resolved_zones:
            if zone.zone_id in self._zones:
                raise GraphValidationError(f"Duplicate zone_id: {zone.zone_id}")
            for node_id in zone.node_ids:
                graph.node(node_id)
                if node_id in self._node_to_zone:
                    raise GraphValidationError(f"Node {node_id} belongs to multiple zones.")
                self._node_to_zone[node_id] = zone.zone_id
            self._zones[zone.zone_id] = zone

    def zones(self) -> tuple[Zone, ...]:
        """Return all zones."""

        return tuple(self._zones.values())

    def obstacles(self) -> tuple[ObstacleRectangle, ...]:
        """Return warehouse obstacle rectangles."""

        return self._obstacles

    def zone(self, zone_id: str) -> Zone:
        """Fetch a zone by id."""

        try:
            return self._zones[zone_id]
        except KeyError as exc:
            raise GraphValidationError(f"Unknown zone_id: {zone_id}") from exc

    def nodes_in_zone(self, zone_id: str) -> tuple[WarehouseNode, ...]:
        """Return the nodes assigned to a zone."""

        zone = self.zone(zone_id)
        return tuple(self.graph.node(node_id) for node_id in sorted(zone.node_ids))

    def default_node_for_zone(self, zone_id: str) -> WarehouseNode:
        """Select the default node to represent a zone.

        Stage 2 uses the lexicographically first node id as the default
        representative. Later stages can add richer selection policies.
        """

        return self.nodes_in_zone(zone_id)[0]

    def zone_for_node(self, node_id: str) -> str | None:
        """Return the zone assigned to a node, if any."""

        self.graph.node(node_id)
        return self._node_to_zone.get(node_id)

    def shortest_path(self, source: str, target: str, weight: str = "travel_time") -> tuple[str, ...]:
        """Delegate weighted shortest-path queries to the graph."""

        return self.graph.shortest_path(source=source, target=target, weight=weight)

    def shortest_path_edges(
        self,
        source: str,
        target: str,
        weight: str = "travel_time",
    ) -> tuple[WarehouseEdge, ...]:
        """Delegate explicit shortest-path edge queries to the graph."""

        return self.graph.shortest_path_edges(source=source, target=target, weight=weight)

    def path_distance(self, path: tuple[str, ...]) -> float:
        """Return the explicit distance of a materialized path."""

        return self.graph.path_distance(path)

    def path_travel_time(self, path: tuple[str, ...]) -> float:
        """Return the explicit travel time of a materialized path."""

        return self.graph.path_travel_time(path)

    def travel_time(self, source: str, target: str) -> float:
        """Return the minimum travel time between two nodes."""

        return self.graph.travel_time(source=source, target=target)

    def distance(self, source: str, target: str) -> float:
        """Return the minimum path distance between two nodes."""

        return self.graph.distance(source=source, target=target)


def _derive_zones_from_graph(graph: WarehouseGraph) -> tuple[Zone, ...]:
    grouped: dict[str, list[str]] = {}
    for node in graph.nodes():
        if node.zone is None:
            continue
        grouped.setdefault(node.zone, []).append(node.node_id)
    return tuple(Zone(zone_id=zone_id, node_ids=tuple(sorted(node_ids))) for zone_id, node_ids in sorted(grouped.items()))


def obstacle_rectangles_from_blocked_cells(
    blocked_cells: tuple[tuple[int, int], ...],
    *,
    edge_length: float,
) -> tuple[ObstacleRectangle, ...]:
    """Materialize blocked grid cells as square obstacle rectangles."""

    if edge_length <= 0:
        raise GraphValidationError("edge_length must be > 0 for obstacle rectangle generation.")
    half_extent = edge_length / 2.0
    obstacles: list[ObstacleRectangle] = []
    for row, column in blocked_cells:
        center_x = float(column) * edge_length
        center_y = float(row) * edge_length
        obstacles.append(
            ObstacleRectangle(
                obstacle_id=f"blocked_r{row}_c{column}",
                min_x=center_x - half_extent,
                min_y=center_y - half_extent,
                max_x=center_x + half_extent,
                max_y=center_y + half_extent,
            )
        )
    return tuple(obstacles)
