"""Execution-model helpers for route-aware and congestion-aware simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import WarehouseEdge
from warehouse_sim.policies.observation import CongestionObservation, ResourceReservationObservation
from warehouse_sim.simulation.models import ExecutionModel


@dataclass(frozen=True)
class RouteExecution:
    """Realized execution details for a single path leg."""

    path_nodes: tuple[str, ...]
    path_arcs: tuple[str, ...]
    distance: float
    ideal_travel_time: float
    realized_travel_time: float
    wait_time: float
    blocked_events: int


@dataclass(frozen=True)
class AssignmentExecution:
    """Realized execution details for a full robot-task assignment."""

    travel_to_pickup: RouteExecution
    travel_to_dropoff: RouteExecution
    pickup_arrival_time: float
    service_start_time: float
    completion_time: float
    congestion_delay_time: float
    blocked_traversal_events: int


@dataclass(frozen=True)
class ChargeExecution:
    """Realized execution details for a charging action."""

    travel_to_charger: RouteExecution
    arrival_time: float
    charging_start_time: float
    completion_time: float
    waiting_time: float
    charge_duration: float


class ResourceReservationTable:
    """Track future occupancy for simplified node/edge reservation models."""

    def __init__(self, execution_model: ExecutionModel) -> None:
        self.execution_model = execution_model
        self._edge_reserved_until: dict[tuple[str, str], float] = {}
        self._node_reserved_until: dict[str, float] = {}
        self._charger_reserved_until: dict[str, float] = {}

    def execute_assignment(
        self,
        *,
        environment: WarehouseEnvironment,
        execution_model: ExecutionModel,
        current_time: float,
        start_node: str,
        pickup_node: str,
        dropoff_node: str,
        service_time: float,
        speed_multiplier: float,
    ) -> AssignmentExecution:
        """Plan and reserve a realized robot assignment."""

        to_pickup = self._execute_leg(
            environment=environment,
            execution_model=execution_model,
            source=start_node,
            target=pickup_node,
            start_time=current_time,
            speed_multiplier=speed_multiplier,
        )
        pickup_arrival_time = current_time + to_pickup.realized_travel_time
        service_start_time = pickup_arrival_time
        service_end_time = service_start_time + service_time
        if execution_model == ExecutionModel.RESERVED_NODES:
            self._node_reserved_until[pickup_node] = max(
                self._node_reserved_until.get(pickup_node, 0.0),
                service_end_time,
            )
        to_dropoff = self._execute_leg(
            environment=environment,
            execution_model=execution_model,
            source=pickup_node,
            target=dropoff_node,
            start_time=service_end_time,
            speed_multiplier=speed_multiplier,
        )
        completion_time = service_end_time + to_dropoff.realized_travel_time
        if execution_model == ExecutionModel.RESERVED_NODES:
            self._node_reserved_until[dropoff_node] = max(
                self._node_reserved_until.get(dropoff_node, 0.0),
                completion_time,
            )

        congestion_delay_time = to_pickup.wait_time + to_dropoff.wait_time
        blocked_traversal_events = to_pickup.blocked_events + to_dropoff.blocked_events
        return AssignmentExecution(
            travel_to_pickup=to_pickup,
            travel_to_dropoff=to_dropoff,
            pickup_arrival_time=pickup_arrival_time,
            service_start_time=service_start_time,
            completion_time=completion_time,
            congestion_delay_time=congestion_delay_time,
            blocked_traversal_events=blocked_traversal_events,
        )

    def execute_charge(
        self,
        *,
        environment: WarehouseEnvironment,
        execution_model: ExecutionModel,
        current_time: float,
        start_node: str,
        charging_node: str,
        charge_duration: float,
        speed_multiplier: float,
    ) -> ChargeExecution:
        """Plan and reserve a realized robot charging action."""

        to_charger = self._execute_leg(
            environment=environment,
            execution_model=execution_model,
            source=start_node,
            target=charging_node,
            start_time=current_time,
            speed_multiplier=speed_multiplier,
        )
        arrival_time = current_time + to_charger.realized_travel_time
        charging_start_time = max(arrival_time, self._charger_reserved_until.get(charging_node, 0.0))
        waiting_time = max(charging_start_time - arrival_time, 0.0)
        completion_time = charging_start_time + charge_duration
        self._charger_reserved_until[charging_node] = completion_time
        self._node_reserved_until[charging_node] = max(self._node_reserved_until.get(charging_node, 0.0), completion_time)
        return ChargeExecution(
            travel_to_charger=to_charger,
            arrival_time=arrival_time,
            charging_start_time=charging_start_time,
            completion_time=completion_time,
            waiting_time=waiting_time,
            charge_duration=charge_duration,
        )

    def snapshot(self, current_time: float) -> CongestionObservation:
        """Expose currently active reservations for dispatch-time observations."""

        edge_reservations = tuple(
            ResourceReservationObservation(
                resource_id=_arc_id(source, target),
                reserved_until=reserved_until,
            )
            for (source, target), reserved_until in sorted(self._edge_reserved_until.items())
            if reserved_until > current_time
        )
        node_reservations = tuple(
            ResourceReservationObservation(
                resource_id=node_id,
                reserved_until=reserved_until,
            )
            for node_id, reserved_until in sorted(self._node_reserved_until.items())
            if reserved_until > current_time
        )
        return CongestionObservation(
            execution_model=self.execution_model.value,
            edge_reservations=edge_reservations,
            node_reservations=node_reservations,
        )

    def _execute_leg(
        self,
        *,
        environment: WarehouseEnvironment,
        execution_model: ExecutionModel,
        source: str,
        target: str,
        start_time: float,
        speed_multiplier: float,
    ) -> RouteExecution:
        path_nodes = environment.shortest_path(source=source, target=target, weight="travel_time")
        path_edges = environment.shortest_path_edges(source=source, target=target, weight="travel_time")
        ideal_travel_time = environment.path_travel_time(path_nodes) / speed_multiplier
        distance = environment.path_distance(path_nodes)
        path_arcs = tuple(_arc_id(edge.source, edge.target) for edge in path_edges)

        if execution_model == ExecutionModel.IDEALIZED or not path_edges:
            return RouteExecution(
                path_nodes=path_nodes,
                path_arcs=path_arcs,
                distance=distance,
                ideal_travel_time=ideal_travel_time,
                realized_travel_time=ideal_travel_time,
                wait_time=0.0,
                blocked_events=0,
            )
        if execution_model == ExecutionModel.RESERVED_EDGES:
            return self._execute_reserved_edges(
                path_nodes=path_nodes,
                path_edges=path_edges,
                start_time=start_time,
                speed_multiplier=speed_multiplier,
            )
        if execution_model == ExecutionModel.RESERVED_NODES:
            return self._execute_reserved_nodes(
                path_nodes=path_nodes,
                path_edges=path_edges,
                start_time=start_time,
                speed_multiplier=speed_multiplier,
            )
        raise ValueError(f"Unsupported execution_model: {execution_model}")

    def _execute_reserved_edges(
        self,
        *,
        path_nodes: tuple[str, ...],
        path_edges: Iterable[WarehouseEdge],
        start_time: float,
        speed_multiplier: float,
    ) -> RouteExecution:
        edge_sequence = tuple(path_edges)
        current_time = start_time
        blocked_events = 0
        wait_time = 0.0
        for edge in edge_sequence:
            resource_key = (edge.source, edge.target)
            departure_time = max(current_time, self._edge_reserved_until.get(resource_key, 0.0))
            if departure_time > current_time:
                wait_time += departure_time - current_time
                blocked_events += 1
            traversal_time = edge.travel_time / speed_multiplier
            arrival_time = departure_time + traversal_time
            self._edge_reserved_until[resource_key] = arrival_time
            current_time = arrival_time
        distance = sum(edge.distance for edge in edge_sequence)
        ideal_travel_time = sum(edge.travel_time for edge in edge_sequence) / speed_multiplier
        return RouteExecution(
            path_nodes=path_nodes,
            path_arcs=tuple(_arc_id(edge.source, edge.target) for edge in edge_sequence),
            distance=distance,
            ideal_travel_time=ideal_travel_time,
            realized_travel_time=current_time - start_time,
            wait_time=wait_time,
            blocked_events=blocked_events,
        )

    def _execute_reserved_nodes(
        self,
        *,
        path_nodes: tuple[str, ...],
        path_edges: Iterable[WarehouseEdge],
        start_time: float,
        speed_multiplier: float,
    ) -> RouteExecution:
        edge_sequence = tuple(path_edges)
        current_time = start_time
        blocked_events = 0
        wait_time = 0.0
        current_node = path_nodes[0]
        self._node_reserved_until[current_node] = max(self._node_reserved_until.get(current_node, 0.0), current_time)

        for edge in edge_sequence:
            traversal_time = edge.travel_time / speed_multiplier
            next_node = edge.target
            earliest_arrival = current_time + traversal_time
            entry_time = max(earliest_arrival, self._node_reserved_until.get(next_node, 0.0))
            if entry_time > earliest_arrival:
                wait_time += entry_time - earliest_arrival
                blocked_events += 1
            departure_time = entry_time - traversal_time
            self._node_reserved_until[current_node] = max(
                self._node_reserved_until.get(current_node, 0.0),
                departure_time,
            )
            self._node_reserved_until[next_node] = max(self._node_reserved_until.get(next_node, 0.0), entry_time)
            current_time = entry_time
            current_node = next_node

        distance = sum(edge.distance for edge in edge_sequence)
        ideal_travel_time = sum(edge.travel_time for edge in edge_sequence) / speed_multiplier
        return RouteExecution(
            path_nodes=path_nodes,
            path_arcs=tuple(_arc_id(edge.source, edge.target) for edge in edge_sequence),
            distance=distance,
            ideal_travel_time=ideal_travel_time,
            realized_travel_time=current_time - start_time,
            wait_time=wait_time,
            blocked_events=blocked_events,
        )


def _arc_id(source: str, target: str) -> str:
    return f"{source}->{target}"
