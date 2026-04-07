"""Free-space off-graph motion utilities for integrated coordination."""

from __future__ import annotations

from math import hypot

from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.integrated.geometry import (
    inflate_obstacles,
    polyline_distance,
    visibility_graph_shortest_path,
)
from warehouse_sim.integrated.models import TimedTraversal, TimedWaypoint
from warehouse_sim.integrated.planner import PlannedMacro
from warehouse_sim.integrated.planner import ConflictConstraint, _apply_node_constraints


class FreeSpaceOccupancyTable:
    """Continuous free-space segment occupancy for disc robots."""

    def __init__(self, *, robot_radius: float, collision_clearance: float) -> None:
        self.robot_radius = robot_radius
        self.collision_clearance = collision_clearance
        self._segments: list[TimedTraversal] = []
        self._node_times: dict[str, list[tuple[float, str]]] = {}

    def reserve(self, traversals: tuple[TimedTraversal, ...]) -> None:
        self._segments.extend(traversals)
        for traversal in traversals:
            self._node_times.setdefault(traversal.target_id, []).append((traversal.end_time, traversal.robot_id))

    def reserve_node_time(self, *, node_id: str, time: float, robot_id: str) -> None:
        self._node_times.setdefault(node_id, []).append((time, robot_id))

    def future_traversals(self, current_time: float) -> tuple[TimedTraversal, ...]:
        return tuple(
            sorted(
                (segment for segment in self._segments if segment.end_time > current_time),
                key=lambda item: (item.start_time, item.robot_id, item.source_id, item.target_id),
            )
        )

    def future_node_times(self, current_time: float) -> tuple[TimedWaypoint, ...]:
        points: list[TimedWaypoint] = []
        for node_id, timestamps in self._node_times.items():
            for time, _robot_id in timestamps:
                if time <= current_time:
                    continue
                points.append(TimedWaypoint(node_id=node_id, time=time))
        return tuple(sorted(points, key=lambda item: (item.time, item.node_id)))

    def clone(self) -> "FreeSpaceOccupancyTable":
        cloned = FreeSpaceOccupancyTable(
            robot_radius=self.robot_radius,
            collision_clearance=self.collision_clearance,
        )
        cloned._segments = list(self._segments)
        cloned._node_times = {key: list(value) for key, value in self._node_times.items()}
        return cloned

    def node_arrival_delay(self, *, node_id: str, arrival_time: float) -> float:
        clearance_time = max(self.robot_radius * 2.0 + self.collision_clearance, 1e-6)
        current_arrival = arrival_time
        while True:
            next_arrival = current_arrival
            for occupied_time, _robot_id in self._node_times.get(node_id, []):
                if abs(current_arrival - occupied_time) < clearance_time:
                    next_arrival = max(next_arrival, occupied_time + clearance_time)
            if abs(next_arrival - current_arrival) < 1e-9:
                return current_arrival
            current_arrival = next_arrival

    def segment_delay(
        self,
        *,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        departure_time: float,
        traversal_time: float,
    ) -> float:
        current_departure = departure_time
        while True:
            next_departure = current_departure
            candidate = _make_segment(
                robot_id="candidate",
                source_id="candidate_start",
                target_id="candidate_end",
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                start_time=current_departure,
                traversal_time=traversal_time,
            )
            for reserved in self._segments:
                if _free_space_segments_conflict(
                    candidate,
                    reserved,
                    threshold=self.robot_radius * 2.0 + self.collision_clearance,
                ):
                    next_departure = max(next_departure, reserved.end_time + 1e-6)
            if abs(next_departure - current_departure) < 1e-9:
                return current_departure
            current_departure = next_departure


def estimate_free_space_completion_time(
    environment: WarehouseEnvironment,
    *,
    route_nodes: tuple[str, ...],
    speed_multiplier: float,
) -> float:
    """Estimate direct free-space travel time over a node sequence."""

    total_distance = 0.0
    for source_id, target_id in zip(route_nodes, route_nodes[1:]):
        total_distance += free_space_distance(environment, source_id=source_id, target_id=target_id)
    return total_distance / max(_base_travel_speed(environment) * speed_multiplier, 1e-6)


def estimate_obstacle_aware_completion_time(
    environment: WarehouseEnvironment,
    *,
    route_nodes: tuple[str, ...],
    speed_multiplier: float,
    robot_radius: float,
    collision_clearance: float,
) -> float | None:
    """Estimate obstacle-aware free-space travel time over a node sequence."""

    total_distance = 0.0
    inflated_obstacles = _inflated_environment_obstacles(
        environment,
        robot_radius=robot_radius,
        collision_clearance=collision_clearance,
    )
    for source_id, target_id in zip(route_nodes, route_nodes[1:]):
        leg_points = _obstacle_aware_leg_points(
            environment,
            source_id=source_id,
            target_id=target_id,
            inflated_obstacles=inflated_obstacles,
        )
        if leg_points is None:
            return None
        total_distance += polyline_distance(leg_points)
    return total_distance / max(_base_travel_speed(environment) * speed_multiplier, 1e-6)


def plan_free_space_candidate(
    environment: WarehouseEnvironment,
    *,
    robot_id: str,
    start_time: float,
    speed_multiplier: float,
    occupancy_table: FreeSpaceOccupancyTable,
    candidate,
    service_time: float = 0.0,
    constraints: tuple[ConflictConstraint, ...] = (),
) -> PlannedMacro | None:
    """Realize a macro candidate as direct free-space segments between node positions."""

    current_time = start_time
    traversals: list[TimedTraversal] = []
    reserved_node_times: list[tuple[str, float]] = []
    blocked_events = 0
    wait_time = 0.0
    pickup_arrival_time: float | None = None
    speed = max(_base_travel_speed(environment) * speed_multiplier, 1e-6)

    for index, (source_id, target_id) in enumerate(zip(candidate.route_nodes, candidate.route_nodes[1:])):
        start_x, start_y = node_position(environment, node_id=source_id)
        end_x, end_y = node_position(environment, node_id=target_id)
        distance = free_space_distance(environment, source_id=source_id, target_id=target_id)
        if distance <= 1e-9:
            if candidate.pickup_node == target_id and pickup_arrival_time is None:
                pickup_arrival_time = current_time
                current_time += service_time
                reserved_node_times.append((target_id, current_time))
            continue
        traversal_time = distance / speed
        earliest_departure = occupancy_table.segment_delay(
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
            departure_time=current_time,
            traversal_time=traversal_time,
        )
        departure_time, arrival_time = _apply_free_space_constraints(
            robot_id=robot_id,
            source_id=source_id,
            target_id=target_id,
            earliest_departure=earliest_departure,
            traversal_time=traversal_time,
            occupancy_table=occupancy_table,
            constraints=constraints,
        )
        if departure_time - current_time > 1e-9:
            blocked_events += 1
            wait_time += departure_time - current_time
        phase = "travel_to_pickup"
        if pickup_arrival_time is not None:
            phase = "travel_to_dropoff"
        elif candidate.pickup_node == target_id:
            pickup_arrival_time = arrival_time
        traversals.append(
            TimedTraversal(
                robot_id=robot_id,
                source_id=source_id,
                target_id=target_id,
                start_time=departure_time,
                end_time=arrival_time,
                distance=distance,
                travel_time=traversal_time,
                task_id=candidate.task_id,
                phase=phase,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
            )
        )
        current_time = arrival_time
        if candidate.pickup_node == target_id and pickup_arrival_time is not None:
            current_time += service_time
            reserved_node_times.append((target_id, current_time))

    completion_time = traversals[-1].end_time if traversals else start_time
    return PlannedMacro(
        task_id=candidate.task_id,
        route_nodes=candidate.route_nodes,
        traversals=tuple(traversals),
        completion_time=completion_time,
        pickup_arrival_time=pickup_arrival_time,
        blocked_events=blocked_events,
        wait_time=wait_time,
        reserved_node_times=tuple(reserved_node_times),
    )


def plan_obstacle_aware_free_space_candidate(
    environment: WarehouseEnvironment,
    *,
    robot_id: str,
    start_time: float,
    speed_multiplier: float,
    occupancy_table: FreeSpaceOccupancyTable,
    candidate,
    robot_radius: float,
    collision_clearance: float,
    service_time: float = 0.0,
    constraints: tuple[ConflictConstraint, ...] = (),
) -> PlannedMacro | None:
    """Realize a macro candidate as an obstacle-aware continuous free-space path."""

    current_time = start_time
    traversals: list[TimedTraversal] = []
    reserved_node_times: list[tuple[str, float]] = []
    blocked_events = 0
    wait_time = 0.0
    pickup_arrival_time: float | None = None
    speed = max(_base_travel_speed(environment) * speed_multiplier, 1e-6)
    inflated_obstacles = _inflated_environment_obstacles(
        environment,
        robot_radius=robot_radius,
        collision_clearance=collision_clearance,
    )

    for leg_index, (source_id, target_id) in enumerate(zip(candidate.route_nodes, candidate.route_nodes[1:])):
        leg_points = _obstacle_aware_leg_points(
            environment,
            source_id=source_id,
            target_id=target_id,
            inflated_obstacles=inflated_obstacles,
        )
        if leg_points is None:
            return None
        if len(leg_points) == 1:
            if candidate.pickup_node == target_id and pickup_arrival_time is None:
                pickup_arrival_time = current_time
                current_time += service_time
                reserved_node_times.append((target_id, current_time))
            continue

        leg_phase = "travel_to_dropoff" if pickup_arrival_time is not None else "travel_to_pickup"
        for point_index, (start_point, end_point) in enumerate(zip(leg_points, leg_points[1:])):
            segment_source_id = _free_space_point_id(
                source_id=source_id,
                target_id=target_id,
                robot_id=robot_id,
                leg_index=leg_index,
                point_index=point_index,
                point_count=len(leg_points),
            )
            segment_target_id = _free_space_point_id(
                source_id=source_id,
                target_id=target_id,
                robot_id=robot_id,
                leg_index=leg_index,
                point_index=point_index + 1,
                point_count=len(leg_points),
            )
            distance = hypot(end_point[0] - start_point[0], end_point[1] - start_point[1])
            if distance <= 1e-9:
                continue
            traversal_time = distance / speed
            earliest_departure = occupancy_table.segment_delay(
                start_x=start_point[0],
                start_y=start_point[1],
                end_x=end_point[0],
                end_y=end_point[1],
                departure_time=current_time,
                traversal_time=traversal_time,
            )
            departure_time, arrival_time = _apply_free_space_constraints(
                robot_id=robot_id,
                source_id=segment_source_id,
                target_id=segment_target_id,
                earliest_departure=earliest_departure,
                traversal_time=traversal_time,
                occupancy_table=occupancy_table,
                constraints=constraints,
            )
            if departure_time - current_time > 1e-9:
                blocked_events += 1
                wait_time += departure_time - current_time
            traversals.append(
                TimedTraversal(
                    robot_id=robot_id,
                    source_id=segment_source_id,
                    target_id=segment_target_id,
                    start_time=departure_time,
                    end_time=arrival_time,
                    distance=distance,
                    travel_time=traversal_time,
                    task_id=candidate.task_id,
                    phase=leg_phase,
                    start_x=start_point[0],
                    start_y=start_point[1],
                    end_x=end_point[0],
                    end_y=end_point[1],
                )
            )
            current_time = arrival_time

        if candidate.pickup_node == target_id and pickup_arrival_time is None:
            pickup_arrival_time = current_time
            current_time += service_time
            reserved_node_times.append((target_id, current_time))

    completion_time = traversals[-1].end_time if traversals else start_time
    return PlannedMacro(
        task_id=candidate.task_id,
        route_nodes=candidate.route_nodes,
        traversals=tuple(traversals),
        completion_time=completion_time,
        pickup_arrival_time=pickup_arrival_time,
        blocked_events=blocked_events,
        wait_time=wait_time,
        reserved_node_times=tuple(reserved_node_times),
    )


def detect_free_space_collision_events(
    traversals: tuple[TimedTraversal, ...],
    *,
    robot_radius: float,
    collision_clearance: float,
) -> tuple[tuple[float, str, str | None, str, str], ...]:
    """Detect pairwise free-space conflicts across moving segments."""

    events: list[tuple[float, str, str | None, str, str]] = []
    ordered = sorted(traversals, key=lambda item: (item.start_time, item.robot_id))
    threshold = robot_radius * 2.0 + collision_clearance
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left.robot_id == right.robot_id:
                continue
            if _free_space_segments_conflict(left, right, threshold=threshold):
                events.append(
                    (
                        max(left.start_time, right.start_time),
                        left.robot_id,
                        right.robot_id,
                        "free_space_conflict",
                        f"{left.source_id}->{left.target_id}|{right.source_id}->{right.target_id}",
                    )
                )
    return tuple(events)


def free_space_distance(environment: WarehouseEnvironment, *, source_id: str, target_id: str) -> float:
    start_x, start_y = node_position(environment, node_id=source_id)
    end_x, end_y = node_position(environment, node_id=target_id)
    return hypot(end_x - start_x, end_y - start_y)


def node_position(environment: WarehouseEnvironment, *, node_id: str) -> tuple[float, float]:
    node = environment.graph.node(node_id)
    scale = _coordinate_scale(environment)
    return float(node.x) * scale, float(node.y) * scale


def _base_travel_speed(environment: WarehouseEnvironment) -> float:
    edges = environment.graph.edges()
    if not edges:
        return 1.0
    edge = edges[0]
    return edge.distance / max(edge.travel_time, 1e-6)


def _coordinate_scale(environment: WarehouseEnvironment) -> float:
    edges = environment.graph.edges()
    if not edges:
        return 1.0
    edge = edges[0]
    source = environment.graph.node(edge.source)
    target = environment.graph.node(edge.target)
    geometry = hypot(float(target.x) - float(source.x), float(target.y) - float(source.y))
    if geometry <= 1e-9:
        return edge.distance
    return edge.distance / geometry


def _inflated_environment_obstacles(
    environment: WarehouseEnvironment,
    *,
    robot_radius: float,
    collision_clearance: float,
):
    return inflate_obstacles(
        environment.obstacles(),
        margin=robot_radius + collision_clearance,
    )


def _obstacle_aware_leg_points(
    environment: WarehouseEnvironment,
    *,
    source_id: str,
    target_id: str,
    inflated_obstacles,
):
    return visibility_graph_shortest_path(
        node_position(environment, node_id=source_id),
        node_position(environment, node_id=target_id),
        obstacles=inflated_obstacles,
    )


def _free_space_point_id(
    *,
    source_id: str,
    target_id: str,
    robot_id: str,
    leg_index: int,
    point_index: int,
    point_count: int,
) -> str:
    if point_index == 0:
        return source_id
    if point_index == point_count - 1:
        return target_id
    return f"{robot_id}::{source_id}::{target_id}::leg_{leg_index}::wp_{point_index}"


def _apply_free_space_constraints(
    *,
    robot_id: str,
    source_id: str,
    target_id: str,
    earliest_departure: float,
    traversal_time: float,
    occupancy_table: FreeSpaceOccupancyTable,
    constraints: tuple[ConflictConstraint, ...],
) -> tuple[float, float]:
    departure_time = earliest_departure
    arrival_time = departure_time + traversal_time
    while True:
        updated_arrival = occupancy_table.node_arrival_delay(node_id=target_id, arrival_time=departure_time + traversal_time)
        updated_arrival = _apply_node_constraints(
            robot_id=robot_id,
            node_id=target_id,
            arrival_time=updated_arrival,
            constraints=constraints,
        )
        updated_departure = updated_arrival - traversal_time
        if abs(updated_departure - departure_time) < 1e-9 and abs(updated_arrival - arrival_time) < 1e-9:
            return updated_departure, updated_arrival
        departure_time = updated_departure
        arrival_time = updated_arrival


def _make_segment(
    *,
    robot_id: str,
    source_id: str,
    target_id: str,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    start_time: float,
    traversal_time: float,
) -> TimedTraversal:
    return TimedTraversal(
        robot_id=robot_id,
        source_id=source_id,
        target_id=target_id,
        start_time=start_time,
        end_time=start_time + traversal_time,
        distance=hypot(end_x - start_x, end_y - start_y),
        travel_time=traversal_time,
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
    )


def _free_space_segments_conflict(left: TimedTraversal, right: TimedTraversal, *, threshold: float) -> bool:
    if left.start_x is None or left.start_y is None or left.end_x is None or left.end_y is None:
        return False
    if right.start_x is None or right.start_y is None or right.end_x is None or right.end_y is None:
        return False
    overlap_start = max(left.start_time, right.start_time)
    overlap_end = min(left.end_time, right.end_time)
    if overlap_end <= overlap_start + 1e-9:
        return False

    left_velocity = (
        (left.end_x - left.start_x) / max(left.travel_time, 1e-6),
        (left.end_y - left.start_y) / max(left.travel_time, 1e-6),
    )
    right_velocity = (
        (right.end_x - right.start_x) / max(right.travel_time, 1e-6),
        (right.end_y - right.start_y) / max(right.travel_time, 1e-6),
    )
    left_position = (
        left.start_x + left_velocity[0] * (overlap_start - left.start_time),
        left.start_y + left_velocity[1] * (overlap_start - left.start_time),
    )
    right_position = (
        right.start_x + right_velocity[0] * (overlap_start - right.start_time),
        right.start_y + right_velocity[1] * (overlap_start - right.start_time),
    )
    relative_position = (
        left_position[0] - right_position[0],
        left_position[1] - right_position[1],
    )
    relative_velocity = (
        left_velocity[0] - right_velocity[0],
        left_velocity[1] - right_velocity[1],
    )
    interval = overlap_end - overlap_start
    velocity_norm = relative_velocity[0] ** 2 + relative_velocity[1] ** 2
    if velocity_norm <= 1e-12:
        return relative_position[0] ** 2 + relative_position[1] ** 2 < threshold**2
    best_time = -(
        relative_position[0] * relative_velocity[0] + relative_position[1] * relative_velocity[1]
    ) / velocity_norm
    best_time = min(max(best_time, 0.0), interval)
    dx = relative_position[0] + relative_velocity[0] * best_time
    dy = relative_position[1] + relative_velocity[1] * best_time
    return dx * dx + dy * dy < threshold**2
