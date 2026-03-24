"""Warehouse environment domain models."""

from __future__ import annotations

from dataclasses import dataclass

from warehouse_sim.graph import GraphValidationError, WarehouseGraph, WarehouseNode


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


class WarehouseEnvironment:
    """A warehouse environment backed by a topology graph and named zones."""

    def __init__(self, graph: WarehouseGraph, zones: tuple[Zone, ...] | None = None) -> None:
        self.graph = graph
        resolved_zones = zones if zones is not None else _derive_zones_from_graph(graph)
        self._zones: dict[str, Zone] = {}
        self._node_to_zone: dict[str, str] = {}

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

