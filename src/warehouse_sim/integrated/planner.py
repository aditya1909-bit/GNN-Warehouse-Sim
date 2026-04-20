"""Continuous-time prioritized planning utilities for integrated coordination."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import inf
from typing import TYPE_CHECKING

from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import PathNotFoundError, WarehouseEdge
from warehouse_sim.integrated.models import MacroCandidate, TimedTraversal, TimedWaypoint

if TYPE_CHECKING:
    from warehouse_sim.agents import RobotState
    from warehouse_sim.integrated.models import IntegratedObservation
    from warehouse_sim.simulation.models import SimulationConfig
    from warehouse_sim.tasks import Task


_TIME_EPSILON = 1e-9


@dataclass(frozen=True)
class PlannedMacro:
    """A feasible continuous-time macro plan."""

    task_id: str | None
    route_nodes: tuple[str, ...]
    traversals: tuple[TimedTraversal, ...]
    completion_time: float
    pickup_arrival_time: float | None
    blocked_events: int
    wait_time: float
    reserved_node_times: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ConflictConstraint:
    """One additional low-level constraint used by exact conflict-based planning."""

    robot_id: str
    constraint_type: str
    start_time: float
    end_time: float
    source_id: str | None = None
    target_id: str | None = None
    node_id: str | None = None


@dataclass(frozen=True)
class TraversalConflict:
    """Earliest detected conflict between two planned traversals."""

    time: float
    conflict_type: str
    left_robot_id: str
    right_robot_id: str
    location_id: str
    left_source_id: str | None = None
    left_target_id: str | None = None
    right_source_id: str | None = None
    right_target_id: str | None = None
    left_start_time: float = 0.0
    left_end_time: float = 0.0
    right_start_time: float = 0.0
    right_end_time: float = 0.0


@dataclass(frozen=True)
class ExactMAPFSolution:
    """Exact joint solution over the current macro candidate surface."""

    chosen_indices: tuple[int, ...]
    planned_routes: dict[str, PlannedMacro]
    assigned_task_count: int
    objective_cost: float
    makespan: float
    planner_name: str = "optimal_mapf_joint_search"


@dataclass(frozen=True)
class _CBSNode:
    constraints: tuple[ConflictConstraint, ...]
    planned_routes: dict[str, PlannedMacro]
    chosen_candidate_indices: dict[str, int]
    objective_cost: float
    makespan: float


class ContinuousOccupancyTable:
    """Continuous-time edge and node occupancy windows."""

    def __init__(self, *, robot_radius: float, collision_clearance: float) -> None:
        self.robot_radius = robot_radius
        self.collision_clearance = collision_clearance
        self._edge_intervals: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
        self._node_times: dict[str, list[tuple[float, str]]] = {}

    def reserve(self, traversals: tuple[TimedTraversal, ...]) -> None:
        for traversal in traversals:
            self._edge_intervals.setdefault((traversal.source_id, traversal.target_id), []).append(
                (traversal.start_time, traversal.end_time, traversal.robot_id)
            )
            self._node_times.setdefault(traversal.target_id, []).append((traversal.end_time, traversal.robot_id))

    def reserve_node_time(self, *, node_id: str, time: float, robot_id: str) -> None:
        self._node_times.setdefault(node_id, []).append((time, robot_id))

    def clone(self) -> "ContinuousOccupancyTable":
        """Create a structural copy suitable for speculative planning search."""

        cloned = ContinuousOccupancyTable(
            robot_radius=self.robot_radius,
            collision_clearance=self.collision_clearance,
        )
        cloned._edge_intervals = {
            key: list(value)
            for key, value in self._edge_intervals.items()
        }
        cloned._node_times = {
            key: list(value)
            for key, value in self._node_times.items()
        }
        return cloned

    def future_traversals(self, current_time: float) -> tuple[TimedTraversal, ...]:
        traversals: list[TimedTraversal] = []
        for (source_id, target_id), windows in self._edge_intervals.items():
            for start_time, end_time, robot_id in windows:
                if end_time <= current_time:
                    continue
                traversals.append(
                    TimedTraversal(
                        robot_id=robot_id,
                        source_id=source_id,
                        target_id=target_id,
                        start_time=start_time,
                        end_time=end_time,
                        distance=0.0,
                        travel_time=end_time - start_time,
                    )
                )
        traversals.sort(key=lambda item: (item.start_time, item.robot_id, item.source_id, item.target_id))
        return tuple(traversals)

    def future_node_times(self, current_time: float) -> tuple[TimedWaypoint, ...]:
        points: list[TimedWaypoint] = []
        for node_id, timestamps in self._node_times.items():
            for time, _robot_id in timestamps:
                if time <= current_time:
                    continue
                points.append(TimedWaypoint(node_id=node_id, time=time))
        points.sort(key=lambda item: (item.time, item.node_id))
        return tuple(points)

    def edge_delay(
        self,
        *,
        edge: WarehouseEdge,
        departure_time: float,
        traversal_time: float,
    ) -> float:
        headway = _headway_time(
            distance=edge.distance,
            traversal_time=traversal_time,
            robot_radius=self.robot_radius,
            collision_clearance=self.collision_clearance,
        )
        current_departure = departure_time
        while True:
            next_departure = current_departure
            for start_time, end_time, _robot_id in self._edge_intervals.get((edge.source, edge.target), []):
                if abs(current_departure - start_time) < headway:
                    next_departure = max(next_departure, start_time + headway)
            for start_time, end_time, _robot_id in self._edge_intervals.get((edge.target, edge.source), []):
                if not (current_departure + traversal_time + headway <= start_time or current_departure >= end_time + headway):
                    next_departure = max(next_departure, end_time + headway)
            if next_departure == current_departure:
                return current_departure
            current_departure = next_departure

    def node_arrival_delay(self, *, node_id: str, arrival_time: float) -> float:
        clearance_time = max(self.robot_radius * 2.0 + self.collision_clearance, 1e-6)
        current_arrival = arrival_time
        while True:
            next_arrival = current_arrival
            for occupied_time, _robot_id in self._node_times.get(node_id, []):
                if abs(current_arrival - occupied_time) + _TIME_EPSILON < clearance_time:
                    next_arrival = max(next_arrival, occupied_time + clearance_time)
            if next_arrival == current_arrival:
                return current_arrival
            current_arrival = next_arrival


def k_shortest_paths(
    environment: WarehouseEnvironment,
    *,
    source: str,
    target: str,
    k: int,
    weight: str = "travel_time",
) -> tuple[tuple[str, ...], ...]:
    """Enumerate up to k simple paths in ascending total weight."""

    if k <= 0:
        return ()
    if source == target:
        return ((source,),)

    queue: list[tuple[float, tuple[str, ...]]] = [(0.0, (source,))]
    results: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    while queue and len(results) < k:
        cost, path = heappop(queue)
        current = path[-1]
        if current == target:
            if path not in seen:
                seen.add(path)
                results.append(path)
            continue
        for neighbor in environment.graph.neighbors(current):
            if neighbor.node_id in path:
                continue
            edge = environment.graph.edge(current, neighbor.node_id)
            edge_cost = getattr(edge, weight)
            heappush(queue, (cost + edge_cost, (*path, neighbor.node_id)))
    if not results:
        raise PathNotFoundError(f"No path found from {source} to {target}.")
    return tuple(results)


def generate_route_options(
    environment: WarehouseEnvironment,
    *,
    source: str,
    pickup_node: str,
    dropoff_node: str,
    k_shortest: int,
    max_route_options: int,
    task_id: str,
    service_time_estimate: float = 0.0,
) -> tuple[MacroCandidate, ...]:
    """Generate route macro options for a task."""

    first_leg = k_shortest_paths(environment, source=source, target=pickup_node, k=k_shortest)
    second_leg = k_shortest_paths(environment, source=pickup_node, target=dropoff_node, k=k_shortest)
    options: list[MacroCandidate] = []
    seen_paths: set[tuple[str, ...]] = set()

    for pickup_path in first_leg:
        for dropoff_path in second_leg:
            route_nodes = pickup_path[:-1] + dropoff_path
            if route_nodes in seen_paths:
                continue
            seen_paths.add(route_nodes)
            route_edges = tuple(zip(route_nodes, route_nodes[1:]))
            options.append(
                MacroCandidate(
                    macro_type="task_route",
                    task_id=task_id,
                    route_nodes=route_nodes,
                    route_edges=route_edges,
                    estimated_completion_time=environment.path_travel_time(route_nodes) + service_time_estimate,
                    service_time_estimate=service_time_estimate,
                    pickup_node=pickup_node,
                    dropoff_node=dropoff_node,
                )
            )
            if len(options) >= max_route_options:
                return tuple(options)
    return tuple(options)


def plan_route_candidate(
    environment: WarehouseEnvironment,
    *,
    robot_id: str,
    start_time: float,
    speed_multiplier: float,
    occupancy_table: ContinuousOccupancyTable,
    candidate: MacroCandidate,
    service_time: float = 0.0,
    constraints: tuple[ConflictConstraint, ...] = (),
) -> PlannedMacro | None:
    """Find the earliest continuous-time feasible realization for a route macro."""

    current_time = start_time
    traversals: list[TimedTraversal] = []
    reserved_node_times: list[tuple[str, float]] = []
    blocked_events = 0
    wait_time = 0.0
    pickup_arrival_time: float | None = None
    for index, (source_id, target_id) in enumerate(candidate.route_edges):
        edge = environment.graph.edge(source_id, target_id)
        traversal_time = edge.travel_time / max(speed_multiplier, 1e-6)
        earliest_departure = occupancy_table.edge_delay(
            edge=edge,
            departure_time=current_time,
            traversal_time=traversal_time,
        )
        departure_time, arrival_time = _apply_constraints_to_traversal(
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
        current_time = arrival_time
        phase = "travel_to_pickup"
        if candidate.pickup_node is not None and source_id == candidate.pickup_node:
            phase = "travel_to_dropoff"
        elif candidate.pickup_node is not None and target_id == candidate.pickup_node:
            phase = "travel_to_pickup"
            pickup_arrival_time = arrival_time
        elif pickup_arrival_time is not None:
            phase = "travel_to_dropoff"
        traversals.append(
            TimedTraversal(
                robot_id=robot_id,
                source_id=source_id,
                target_id=target_id,
                start_time=departure_time,
                end_time=arrival_time,
                distance=edge.distance,
                travel_time=traversal_time,
                task_id=candidate.task_id,
                phase=phase,
            )
        )
        if index == len(candidate.route_edges) - 1 and pickup_arrival_time is None and candidate.pickup_node == target_id:
            pickup_arrival_time = arrival_time
        if candidate.pickup_node is not None and target_id == candidate.pickup_node and pickup_arrival_time is not None:
            current_time += service_time
            reserved_node_times.append((target_id, current_time))
    if candidate.route_edges and traversals[-1].target_id != candidate.route_nodes[-1]:
        return None
    if not candidate.route_edges and candidate.route_nodes:
        pickup_arrival_time = start_time if candidate.pickup_node == candidate.route_nodes[0] else None
        if pickup_arrival_time is not None:
            current_time += service_time
            reserved_node_times.append((candidate.route_nodes[0], current_time))
    completion_time = current_time
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


def plan_motion_candidate(
    environment: WarehouseEnvironment,
    *,
    robot_id: str,
    start_time: float,
    speed_multiplier: float,
    occupancy_table,
    candidate: MacroCandidate,
    service_time: float = 0.0,
    constraints: tuple[ConflictConstraint, ...] = (),
    motion_model: str = "graph_embedded",
) -> PlannedMacro | None:
    """Plan a candidate under the selected integrated motion model."""

    if motion_model == "free_space":
        from warehouse_sim.integrated.free_space import plan_free_space_candidate

        return plan_free_space_candidate(
            environment,
            robot_id=robot_id,
            start_time=start_time,
            speed_multiplier=speed_multiplier,
            occupancy_table=occupancy_table,
            candidate=candidate,
            service_time=service_time,
            constraints=constraints,
        )
    if motion_model == "obstacle_aware_free_space":
        from warehouse_sim.integrated.free_space import plan_obstacle_aware_free_space_candidate

        if not hasattr(occupancy_table, "robot_radius") or not hasattr(occupancy_table, "collision_clearance"):
            raise ValueError("obstacle_aware_free_space requires a free-space occupancy table.")
        return plan_obstacle_aware_free_space_candidate(
            environment,
            robot_id=robot_id,
            start_time=start_time,
            speed_multiplier=speed_multiplier,
            occupancy_table=occupancy_table,
            candidate=candidate,
            robot_radius=float(occupancy_table.robot_radius),
            collision_clearance=float(occupancy_table.collision_clearance),
            service_time=service_time,
            constraints=constraints,
        )
    return plan_route_candidate(
        environment,
        robot_id=robot_id,
        start_time=start_time,
        speed_multiplier=speed_multiplier,
        occupancy_table=occupancy_table,
        candidate=candidate,
        service_time=service_time,
        constraints=constraints,
    )


def detect_collision_events(
    traversals: tuple[TimedTraversal, ...],
    *,
    robot_radius: float,
    collision_clearance: float,
) -> tuple[tuple[float, str, str | None, str, str], ...]:
    """Detect explicit safety violations in a set of traversals."""

    headway_cache: dict[tuple[str, str, float], float] = {}
    events: list[tuple[float, str, str | None, str, str]] = []
    ordered = sorted(traversals, key=lambda item: (item.start_time, item.robot_id))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left.robot_id == right.robot_id:
                continue
            if right.start_time > left.end_time + 10.0:
                break
            key = (left.source_id, left.target_id, max(left.distance, 1e-6))
            headway = headway_cache.setdefault(
                key,
                _headway_time(
                    distance=max(left.distance, right.distance),
                    traversal_time=max(left.travel_time, right.travel_time),
                    robot_radius=robot_radius,
                    collision_clearance=collision_clearance,
                ),
            )
            if left.source_id == right.source_id and left.target_id == right.target_id:
                if abs(left.start_time - right.start_time) + _TIME_EPSILON < headway:
                    events.append((max(left.start_time, right.start_time), left.robot_id, right.robot_id, "same_edge_conflict", f"{left.source_id}->{left.target_id}"))
            if left.source_id == right.target_id and left.target_id == right.source_id:
                if not (left.end_time + headway <= right.start_time or right.end_time + headway <= left.start_time):
                    events.append((max(left.start_time, right.start_time), left.robot_id, right.robot_id, "opposite_edge_conflict", f"{left.source_id}<->{left.target_id}"))
            if left.target_id == right.target_id and abs(left.end_time - right.end_time) + _TIME_EPSILON < max(robot_radius * 2.0 + collision_clearance, 1e-6):
                events.append((max(left.end_time, right.end_time), left.robot_id, right.robot_id, "node_conflict", left.target_id))
    return tuple(events)


def _headway_time(
    *,
    distance: float,
    traversal_time: float,
    robot_radius: float,
    collision_clearance: float,
) -> float:
    return traversal_time * ((robot_radius * 2.0 + collision_clearance) / max(distance, 1e-6))


def solve_exact_mapf_macro_plan(
    environment: WarehouseEnvironment,
    *,
    observation: "IntegratedObservation",
    robot_states: tuple["RobotState", ...],
    occupancy_table: ContinuousOccupancyTable,
    current_time: float,
    config: "SimulationConfig",
    tasks: tuple["Task", ...],
) -> ExactMAPFSolution | None:
    """Solve the current integrated routing subproblem exactly over the current macro set.

    The optimality claim here is intentionally bounded: it is exact over the current
    dispatch/replan epoch's finite macro candidates and continuous-time conflict model.
    It is not a global warehouse-task optimality guarantee across future task releases.
    """

    task_by_id = {task.task_id: task for task in tasks}
    robot_index_by_id = {robot_id: index for index, robot_id in enumerate(observation.robot_ids)}
    chosen_indices = [0 for _ in observation.robot_ids]
    best_solution: ExactMAPFSolution | None = None

    def recurse(
        remaining_robot_ids: tuple[str, ...],
        current_occupancy: ContinuousOccupancyTable,
        used_tasks: set[str],
        planned_routes: dict[str, PlannedMacro],
        assigned_task_count: int,
    ) -> None:
        nonlocal best_solution
        if not remaining_robot_ids:
            objective_cost = sum(planned.completion_time - current_time for planned in planned_routes.values())
            makespan = max((planned.completion_time for planned in planned_routes.values()), default=current_time) - current_time
            candidate_solution = ExactMAPFSolution(
                chosen_indices=tuple(chosen_indices),
                planned_routes=dict(planned_routes),
                assigned_task_count=assigned_task_count,
                objective_cost=objective_cost,
                makespan=makespan,
            )
            ranking = (
                -candidate_solution.assigned_task_count,
                candidate_solution.objective_cost,
                candidate_solution.makespan,
                candidate_solution.chosen_indices,
            )
            incumbent = (
                float("inf"),
                float("inf"),
                float("inf"),
                (),
            ) if best_solution is None else (
                -best_solution.assigned_task_count,
                best_solution.objective_cost,
                best_solution.makespan,
                best_solution.chosen_indices,
            )
            if best_solution is None or ranking < incumbent:
                best_solution = candidate_solution
            return

        robot_id = remaining_robot_ids[0]
        robot_index = robot_index_by_id[robot_id]
        robot_state = robot_states[robot_index]
        candidates = observation.macro_candidates[robot_index]
        for candidate_index, candidate in enumerate(candidates):
            if candidate.task_id is not None and candidate.task_id in used_tasks:
                continue
            chosen_indices[robot_index] = candidate_index
            if candidate.macro_type == "wait":
                recurse(
                    remaining_robot_ids[1:],
                    current_occupancy.clone(),
                    set(used_tasks),
                    dict(planned_routes),
                    assigned_task_count,
                )
                continue
            if candidate.macro_type not in {"task_route", "charge_route"}:
                continue
            service_time = 0.0
            if candidate.macro_type in {"task_route", "charge_route"}:
                service_time = candidate.service_time_estimate
            planned = plan_motion_candidate(
                environment,
                robot_id=robot_state.spec.robot_id,
                start_time=current_time,
                speed_multiplier=robot_state.spec.speed_multiplier,
                occupancy_table=current_occupancy,
                candidate=candidate,
                service_time=service_time,
                motion_model=config.coordination.motion_model,  # type: ignore[union-attr]
            )
            if planned is None:
                continue
            partial_cost = sum(item.completion_time - current_time for item in planned_routes.values()) + (
                planned.completion_time - current_time
            )
            if (
                best_solution is not None
                and assigned_task_count + (1 if candidate.task_id is not None else 0) == best_solution.assigned_task_count
                and partial_cost > best_solution.objective_cost + 1e-9
            ):
                continue
            next_occupancy = current_occupancy.clone()
            next_occupancy.reserve(planned.traversals)
            for node_id, time in planned.reserved_node_times:
                next_occupancy.reserve_node_time(node_id=node_id, time=time, robot_id=robot_id)
            next_used_tasks = set(used_tasks)
            if candidate.task_id is not None:
                next_used_tasks.add(candidate.task_id)
            next_planned_routes = dict(planned_routes)
            next_planned_routes[robot_id] = planned
            recurse(
                remaining_robot_ids[1:],
                next_occupancy,
                next_used_tasks,
                next_planned_routes,
                assigned_task_count + (1 if candidate.task_id is not None else 0),
            )

    recurse(
        tuple(
            robot_id
            for robot_id, candidates in zip(observation.robot_ids, observation.macro_candidates, strict=True)
            if candidates and candidates[0].macro_type != "continue_current_plan"
        ),
        occupancy_table.clone(),
        set(),
        {},
        0,
    )
    return best_solution


def _solve_task_assignment(observation: "IntegratedObservation") -> tuple[int, ...] | None:
    idle_robot_indices = [
        index
        for index, candidates in enumerate(observation.macro_candidates)
        if candidates and candidates[0].macro_type != "continue_current_plan"
    ]
    chosen = [0 for _ in observation.macro_candidates]
    best_signature: tuple[int, float, tuple[int, ...]] | None = None

    per_robot_options: list[list[tuple[str | None, int, float]]] = []
    for robot_index in idle_robot_indices:
        options = [(None, 0, observation.macro_candidates[robot_index][0].estimated_completion_time)]
        grouped: dict[str, tuple[int, float]] = {}
        for candidate_index, candidate in enumerate(observation.macro_candidates[robot_index]):
            if candidate.task_id is None:
                continue
            current = grouped.get(candidate.task_id)
            ranking = (candidate.estimated_completion_time, candidate_index)
            if current is None or ranking < (current[1], current[0]):
                grouped[candidate.task_id] = (candidate_index, candidate.estimated_completion_time)
        for task_id, (candidate_index, estimate) in sorted(grouped.items()):
            options.append((task_id, candidate_index, estimate))
        per_robot_options.append(options)

    def recurse(position: int, used_tasks: set[str], assigned_count: int, estimate_sum: float) -> None:
        nonlocal best_signature
        if position >= len(idle_robot_indices):
            signature = (-assigned_count, estimate_sum, tuple(chosen))
            if best_signature is None or signature < best_signature:
                best_signature = signature
            return
        robot_index = idle_robot_indices[position]
        for task_id, candidate_index, estimate in per_robot_options[position]:
            if task_id is not None and task_id in used_tasks:
                continue
            chosen[robot_index] = candidate_index
            if task_id is None:
                recurse(position + 1, used_tasks, assigned_count, estimate_sum + estimate)
            else:
                used_tasks.add(task_id)
                recurse(position + 1, used_tasks, assigned_count + 1, estimate_sum + estimate)
                used_tasks.remove(task_id)

    recurse(0, set(), 0, 0.0)
    return None if best_signature is None else best_signature[2]


def _build_cbs_node(
    environment: WarehouseEnvironment,
    *,
    observation: "IntegratedObservation",
    robot_states: tuple["RobotState", ...],
    occupancy_table: ContinuousOccupancyTable,
    current_time: float,
    constraints: tuple[ConflictConstraint, ...],
    allowed_indices: dict[str, tuple[int, ...]],
    task_by_id: dict[str, "Task"],
    reuse_node: _CBSNode | None = None,
    replanned_robot_id: str | None = None,
) -> _CBSNode | None:
    planned_routes = {} if reuse_node is None else dict(reuse_node.planned_routes)
    chosen_candidate_indices = {} if reuse_node is None else dict(reuse_node.chosen_candidate_indices)

    for robot_index, robot_id in enumerate(observation.robot_ids):
        if robot_id not in allowed_indices:
            continue
        if replanned_robot_id is not None and robot_id != replanned_robot_id and reuse_node is not None:
            continue
        planned = _best_constrained_plan_for_robot(
            environment=environment,
            robot_state=robot_states[robot_index],
            candidates=observation.macro_candidates[robot_index],
            allowed_indices=allowed_indices[robot_id],
            occupancy_table=occupancy_table,
            constraints=tuple(constraint for constraint in constraints if constraint.robot_id == robot_id),
            current_time=current_time,
            task_by_id=task_by_id,
        )
        if planned is None:
            return None
        planned_macro, chosen_index = planned
        planned_routes[robot_id] = planned_macro
        chosen_candidate_indices[robot_id] = chosen_index

    objective_cost = sum(planned.completion_time - current_time for planned in planned_routes.values())
    makespan = max((planned.completion_time for planned in planned_routes.values()), default=current_time) - current_time
    return _CBSNode(
        constraints=constraints,
        planned_routes=planned_routes,
        chosen_candidate_indices=chosen_candidate_indices,
        objective_cost=objective_cost,
        makespan=makespan,
    )


def _best_constrained_plan_for_robot(
    environment: WarehouseEnvironment,
    *,
    robot_state: "RobotState",
    candidates: tuple[MacroCandidate, ...],
    allowed_indices: tuple[int, ...],
    occupancy_table: ContinuousOccupancyTable,
    constraints: tuple[ConflictConstraint, ...],
    current_time: float,
    task_by_id: dict[str, "Task"],
) -> tuple[PlannedMacro, int] | None:
    best: tuple[tuple[float, float, int], PlannedMacro, int] | None = None
    for candidate_index in allowed_indices:
        candidate = candidates[candidate_index]
        if candidate.task_id is None:
            continue
        planned = plan_route_candidate(
            environment,
            robot_id=robot_state.spec.robot_id,
            start_time=current_time,
            speed_multiplier=robot_state.spec.speed_multiplier,
            occupancy_table=occupancy_table,
            candidate=candidate,
            service_time=task_by_id[candidate.task_id].service_time_estimate,
            constraints=constraints,
        )
        if planned is None:
            continue
        ranking = (planned.completion_time, planned.wait_time, candidate_index)
        if best is None or ranking < best[0]:
            best = (ranking, planned, candidate_index)
    if best is None:
        return None
    return best[1], best[2]


def _first_traversal_conflict(
    traversals: tuple[TimedTraversal, ...],
    *,
    robot_radius: float,
    collision_clearance: float,
) -> TraversalConflict | None:
    ordered = sorted(traversals, key=lambda item: (item.start_time, item.robot_id, item.source_id, item.target_id))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left.robot_id == right.robot_id:
                continue
            if right.start_time > left.end_time + 10.0:
                break
            headway = _headway_time(
                distance=max(left.distance, right.distance, 1e-6),
                traversal_time=max(left.travel_time, right.travel_time),
                robot_radius=robot_radius,
                collision_clearance=collision_clearance,
            )
            if left.source_id == right.source_id and left.target_id == right.target_id:
                if abs(left.start_time - right.start_time) + _TIME_EPSILON < headway:
                    return TraversalConflict(
                        time=max(left.start_time, right.start_time),
                        conflict_type="same_edge_conflict",
                        left_robot_id=left.robot_id,
                        right_robot_id=right.robot_id,
                        location_id=f"{left.source_id}->{left.target_id}",
                        left_source_id=left.source_id,
                        left_target_id=left.target_id,
                        right_source_id=right.source_id,
                        right_target_id=right.target_id,
                        left_start_time=left.start_time,
                        left_end_time=left.end_time,
                        right_start_time=right.start_time,
                        right_end_time=right.end_time,
                    )
            if left.source_id == right.target_id and left.target_id == right.source_id:
                if not (
                    left.end_time + headway <= right.start_time or right.end_time + headway <= left.start_time
                ):
                    return TraversalConflict(
                        time=max(left.start_time, right.start_time),
                        conflict_type="opposite_edge_conflict",
                        left_robot_id=left.robot_id,
                        right_robot_id=right.robot_id,
                        location_id=f"{left.source_id}<->{left.target_id}",
                        left_source_id=left.source_id,
                        left_target_id=left.target_id,
                        right_source_id=right.source_id,
                        right_target_id=right.target_id,
                        left_start_time=left.start_time,
                        left_end_time=left.end_time,
                        right_start_time=right.start_time,
                        right_end_time=right.end_time,
                    )
            clearance_time = max(robot_radius * 2.0 + collision_clearance, 1e-6)
            if left.target_id == right.target_id and abs(left.end_time - right.end_time) + _TIME_EPSILON < clearance_time:
                return TraversalConflict(
                    time=max(left.end_time, right.end_time),
                    conflict_type="node_conflict",
                    left_robot_id=left.robot_id,
                    right_robot_id=right.robot_id,
                    location_id=left.target_id,
                    left_source_id=left.source_id,
                    left_target_id=left.target_id,
                    right_source_id=right.source_id,
                    right_target_id=right.target_id,
                    left_start_time=left.start_time,
                    left_end_time=left.end_time,
                    right_start_time=right.start_time,
                    right_end_time=right.end_time,
                )
    return None


def _branch_constraints_for_conflict(
    conflict: TraversalConflict,
    *,
    robot_radius: float,
    collision_clearance: float,
) -> tuple[ConflictConstraint, ConflictConstraint]:
    if conflict.conflict_type == "same_edge_conflict":
        headway = _headway_time(
            distance=1.0,
            traversal_time=max(conflict.left_end_time - conflict.left_start_time, conflict.right_end_time - conflict.right_start_time, 1e-6),
            robot_radius=robot_radius,
            collision_clearance=collision_clearance,
        )
        return (
            ConflictConstraint(
                robot_id=conflict.left_robot_id,
                constraint_type="edge_departure",
                source_id=conflict.left_source_id,
                target_id=conflict.left_target_id,
                start_time=max(conflict.right_start_time - headway, 0.0),
                end_time=conflict.right_start_time + headway,
            ),
            ConflictConstraint(
                robot_id=conflict.right_robot_id,
                constraint_type="edge_departure",
                source_id=conflict.right_source_id,
                target_id=conflict.right_target_id,
                start_time=max(conflict.left_start_time - headway, 0.0),
                end_time=conflict.left_start_time + headway,
            ),
        )
    if conflict.conflict_type == "opposite_edge_conflict":
        headway = max(robot_radius * 2.0 + collision_clearance, 1e-6)
        return (
            ConflictConstraint(
                robot_id=conflict.left_robot_id,
                constraint_type="edge_occupancy",
                source_id=conflict.left_source_id,
                target_id=conflict.left_target_id,
                start_time=max(conflict.right_start_time - headway, 0.0),
                end_time=conflict.right_end_time + headway,
            ),
            ConflictConstraint(
                robot_id=conflict.right_robot_id,
                constraint_type="edge_occupancy",
                source_id=conflict.right_source_id,
                target_id=conflict.right_target_id,
                start_time=max(conflict.left_start_time - headway, 0.0),
                end_time=conflict.left_end_time + headway,
            ),
        )
    clearance_time = max(robot_radius * 2.0 + collision_clearance, 1e-6)
    return (
        ConflictConstraint(
            robot_id=conflict.left_robot_id,
            constraint_type="node_arrival",
            node_id=conflict.location_id,
            start_time=max(conflict.right_end_time - clearance_time, 0.0),
            end_time=conflict.right_end_time + clearance_time,
        ),
        ConflictConstraint(
            robot_id=conflict.right_robot_id,
            constraint_type="node_arrival",
            node_id=conflict.location_id,
            start_time=max(conflict.left_end_time - clearance_time, 0.0),
            end_time=conflict.left_end_time + clearance_time,
        ),
    )


def _apply_constraints_to_traversal(
    *,
    robot_id: str,
    source_id: str,
    target_id: str,
    earliest_departure: float,
    traversal_time: float,
    occupancy_table: ContinuousOccupancyTable,
    constraints: tuple[ConflictConstraint, ...],
) -> tuple[float, float]:
    departure_time = earliest_departure
    arrival_time = departure_time + traversal_time
    while True:
        updated_departure = _apply_edge_constraints(
            robot_id=robot_id,
            source_id=source_id,
            target_id=target_id,
            departure_time=departure_time,
            traversal_time=traversal_time,
            constraints=constraints,
        )
        arrival_time = occupancy_table.node_arrival_delay(node_id=target_id, arrival_time=updated_departure + traversal_time)
        updated_arrival = _apply_node_constraints(
            robot_id=robot_id,
            node_id=target_id,
            arrival_time=arrival_time,
            constraints=constraints,
        )
        updated_departure = _apply_edge_constraints(
            robot_id=robot_id,
            source_id=source_id,
            target_id=target_id,
            departure_time=updated_arrival - traversal_time,
            traversal_time=traversal_time,
            constraints=constraints,
        )
        if abs(updated_departure - departure_time) < 1e-9 and abs(updated_arrival - arrival_time) < 1e-9:
            return updated_departure, updated_arrival
        departure_time = updated_departure
        arrival_time = updated_arrival


def _apply_edge_constraints(
    *,
    robot_id: str,
    source_id: str,
    target_id: str,
    departure_time: float,
    traversal_time: float,
    constraints: tuple[ConflictConstraint, ...],
) -> float:
    current_departure = departure_time
    while True:
        next_departure = current_departure
        for constraint in constraints:
            if constraint.robot_id != robot_id:
                continue
            if constraint.constraint_type == "edge_departure":
                if constraint.source_id != source_id or constraint.target_id != target_id:
                    continue
                if constraint.start_time - 1e-9 <= current_departure <= constraint.end_time + 1e-9:
                    next_departure = max(next_departure, constraint.end_time)
            elif constraint.constraint_type == "edge_occupancy":
                same_undirected_edge = {
                    constraint.source_id,
                    constraint.target_id,
                } == {source_id, target_id}
                if not same_undirected_edge:
                    continue
                arrival_time = current_departure + traversal_time
                if not (
                    arrival_time <= constraint.start_time + 1e-9
                    or current_departure >= constraint.end_time - 1e-9
                ):
                    next_departure = max(next_departure, constraint.end_time)
        if abs(next_departure - current_departure) < 1e-9:
            return current_departure
        current_departure = next_departure


def _apply_node_constraints(
    *,
    robot_id: str,
    node_id: str,
    arrival_time: float,
    constraints: tuple[ConflictConstraint, ...],
) -> float:
    current_arrival = arrival_time
    while True:
        next_arrival = current_arrival
        for constraint in constraints:
            if constraint.robot_id != robot_id or constraint.constraint_type != "node_arrival":
                continue
            if constraint.node_id != node_id:
                continue
            if constraint.start_time - 1e-9 <= current_arrival <= constraint.end_time + 1e-9:
                next_arrival = max(next_arrival, constraint.end_time)
        if abs(next_arrival - current_arrival) < 1e-9:
            return current_arrival
        current_arrival = next_arrival
