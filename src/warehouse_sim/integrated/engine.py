"""Integrated coordination engine with continuous-time planning."""

from __future__ import annotations

from dataclasses import dataclass

from warehouse_sim.agents import RobotSpec, RobotState
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.integrated.free_space import (
    continuous_route_options,
    FreeSpaceOccupancyTable,
    detect_free_space_collision_events,
)
from warehouse_sim.integrated.models import (
    CollisionEventRecord,
    IntegratedObservation,
    IntegratedRobotTrajectoryRecord,
    MacroCandidate,
    MacroDecisionRecord,
    PlannerPlanRecord,
)
from warehouse_sim.integrated.planner import (
    ContinuousOccupancyTable,
    detect_collision_events,
    generate_route_options,
    plan_motion_candidate,
)
from warehouse_sim.integrated.policies import IntegratedCoordinatorPolicy
from warehouse_sim.metrics.collector import compute_simulation_metrics
from warehouse_sim.simulation.models import (
    ChargingExecution,
    ExecutionModel,
    QueueSnapshot,
    SimulationConfig,
    SimulationResult,
    TaskExecution,
)
from warehouse_sim.tasks import Task
from warehouse_sim.utils.battery import (
    battery_enabled,
    estimate_charge_action,
    estimate_task_action,
    nearest_charging_option,
    travel_energy,
)


@dataclass(frozen=True)
class _ActivePlan:
    action_type: str
    task: Task | None
    assigned_at: float
    pickup_arrival_time: float | None
    completion_time: float
    traversals: tuple
    blocked_events: int
    wait_time: float
    charging_node_id: str | None = None
    energy_before: float = 0.0


def run_integrated_simulation(
    *,
    environment: WarehouseEnvironment,
    tasks: tuple[Task, ...],
    robots: tuple[RobotSpec, ...],
    coordinator_policy: IntegratedCoordinatorPolicy,
    config: SimulationConfig,
) -> SimulationResult:
    """Run the integrated coordination stack."""

    if config.coordination is None:
        raise ValueError("Integrated simulation requires coordination settings.")

    robot_states = tuple(RobotState.from_spec(robot) for robot in robots)
    task_by_id = {task.task_id: task for task in tasks}
    released_task_ids: set[str] = set()
    claimed_task_ids: set[str] = set()
    completed_task_ids: set[str] = set()
    active_plans: dict[str, _ActivePlan] = {}
    occupancy = _build_occupancy_table(config)

    current_time = 0.0
    next_replan_time = 0.0
    decision_index = 0
    plan_index = 0
    executions: list[TaskExecution] = []
    charging_executions: list[ChargingExecution] = []
    queue_snapshots: list[QueueSnapshot] = []
    robot_trajectories: list[IntegratedRobotTrajectoryRecord] = []
    macro_decisions: list[MacroDecisionRecord] = []
    collision_events: list[CollisionEventRecord] = []
    planner_plans: list[PlannerPlanRecord] = []
    _record_queue_snapshot(queue_snapshots, current_time, tasks, released_task_ids, completed_task_ids, active_plans)

    while True:
        _release_ready_tasks(tasks, current_time, released_task_ids)
        _finalize_completed_plans(
            current_time=current_time,
            robot_states=robot_states,
            active_plans=active_plans,
            executions=executions,
            charging_executions=charging_executions,
            completed_task_ids=completed_task_ids,
            environment=environment,
            battery_config=config.battery,
        )

        should_replan = current_time + 1e-9 >= next_replan_time
        if should_replan:
            observation = build_integrated_observation(
                environment=environment,
                robot_states=robot_states,
                tasks=tasks,
                released_task_ids=released_task_ids,
                claimed_task_ids=claimed_task_ids,
                completed_task_ids=completed_task_ids,
                active_plans=active_plans,
                occupancy=occupancy,
                current_time=current_time,
                config=config,
            )
            output = coordinator_policy.plan_joint_macros(
                observation,
                environment=environment,
                occupancy=occupancy,
                robot_states=robot_states,
                tasks=tasks,
                current_time=current_time,
                config=config,
            ) or coordinator_policy.select_macros(observation)
            planner_name = output.planner_name or getattr(coordinator_policy, "planner_name", "prioritized_sipp")
            pre_resolution_conflict_count = _count_pre_resolution_conflicts(
                environment=environment,
                observation=observation,
                output=output,
                robot_states=robot_states,
                occupancy=occupancy,
                current_time=current_time,
                config=config,
            )
            used_tasks: set[str] = set()
            for robot_index, robot in enumerate(robot_states):
                candidates = observation.macro_candidates[robot_index]
                chosen_index = output.chosen_indices[robot_index] if robot_index < len(output.chosen_indices) else 0
                if chosen_index < 0 or chosen_index >= len(candidates):
                    chosen_index = 0
                candidate = candidates[chosen_index]
                if candidate.task_id is not None and candidate.task_id in used_tasks:
                    candidate = candidates[0]
                    chosen_index = 0
                (
                    selection_rank,
                    best_candidate_estimated_completion,
                    selected_completion_gap,
                ) = _macro_selection_diagnostics(candidates, chosen_index)
                macro_decisions.append(
                    MacroDecisionRecord(
                        decision_index=decision_index,
                        decision_time=current_time,
                        robot_id=robot.spec.robot_id,
                        macro_type=candidate.macro_type,
                        task_id=candidate.task_id,
                        charging_node=candidate.charging_node,
                        route_nodes=candidate.route_nodes,
                        route_edges=tuple(f"{source}->{target}" for source, target in candidate.route_edges),
                        estimated_completion_time=candidate.estimated_completion_time,
                        selected_by_policy=coordinator_policy.name,
                        candidate_count=len(candidates),
                        selected_rank_by_estimated_completion=selection_rank,
                        best_candidate_estimated_completion=best_candidate_estimated_completion,
                        selected_completion_gap=selected_completion_gap,
                    )
                )
                decision_index += 1
                if candidate.macro_type not in {"task_route", "charge_route"}:
                    continue
                if robot.spec.robot_id in active_plans:
                    continue
                task = None if candidate.task_id is None else task_by_id[candidate.task_id]
                service_time = candidate.service_time_estimate
                if candidate.macro_type == "charge_route":
                    if config.battery is None or candidate.charging_node is None:
                        continue
                planned = output.planned_routes.get(robot.spec.robot_id)
                if planned is None:
                    planned = plan_motion_candidate(
                        environment,
                        robot_id=robot.spec.robot_id,
                        start_time=current_time,
                        speed_multiplier=robot.spec.speed_multiplier,
                        occupancy_table=occupancy,
                        candidate=candidate,
                        service_time=service_time,
                        motion_model=config.coordination.motion_model,
                    )
                if planned is None:
                    planner_plans.append(
                        PlannerPlanRecord(
                            plan_index=plan_index,
                            plan_time=current_time,
                            robot_id=robot.spec.robot_id,
                            task_id=None if task is None else task.task_id,
                            priority_rank=robot_index,
                            path_nodes=candidate.route_nodes,
                            path_edges=tuple(f"{source}->{target}" for source, target in candidate.route_edges),
                            planned_start_time=current_time,
                            planned_end_time=current_time,
                            planner_name=planner_name,
                            status="failed",
                            pre_resolution_conflict_count=pre_resolution_conflict_count,
                        )
                    )
                    plan_index += 1
                    continue
                occupancy.reserve(planned.traversals)
                for node_id, time in planned.reserved_node_times:
                    occupancy.reserve_node_time(node_id=node_id, time=time, robot_id=robot.spec.robot_id)
                if task is not None:
                    used_tasks.add(task.task_id)
                    claimed_task_ids.add(task.task_id)
                active_plans[robot.spec.robot_id] = _ActivePlan(
                    action_type=candidate.macro_type,
                    task=task,
                    assigned_at=current_time,
                    pickup_arrival_time=planned.pickup_arrival_time,
                    completion_time=planned.completion_time,
                    traversals=planned.traversals,
                    blocked_events=planned.blocked_events,
                    wait_time=planned.wait_time,
                    charging_node_id=candidate.charging_node,
                    energy_before=robot.battery_level,
                )
                robot.available_time = planned.completion_time
                planner_plans.append(
                    PlannerPlanRecord(
                        plan_index=plan_index,
                        plan_time=current_time,
                        robot_id=robot.spec.robot_id,
                        task_id=None if task is None else task.task_id,
                        priority_rank=robot_index,
                        path_nodes=planned.route_nodes,
                        path_edges=tuple(
                            f"{traversal.source_id}->{traversal.target_id}" for traversal in planned.traversals
                        ),
                        planned_start_time=planned.traversals[0].start_time if planned.traversals else current_time,
                        planned_end_time=planned.completion_time,
                        planner_name=planner_name,
                        status="planned",
                        pre_resolution_conflict_count=pre_resolution_conflict_count,
                        wait_insertion_count=planned.blocked_events,
                        wait_insertion_time=planned.wait_time,
                    )
                )
                plan_index += 1
                for traversal in planned.traversals:
                    robot_trajectories.append(
                        IntegratedRobotTrajectoryRecord(
                            robot_id=robot.spec.robot_id,
                            task_id=None if task is None else task.task_id,
                            phase=traversal.phase,
                            source_id=traversal.source_id,
                            target_id=traversal.target_id,
                            start_time=traversal.start_time,
                            end_time=traversal.end_time,
                            distance=traversal.distance,
                            travel_time=traversal.travel_time,
                            start_x=traversal.start_x,
                            start_y=traversal.start_y,
                            end_x=traversal.end_x,
                            end_y=traversal.end_y,
                        )
                    )
            for event in _detect_motion_collisions(
                traversals=tuple(
                    traversal
                    for record in active_plans.values()
                    for traversal in record.traversals
                    if traversal.end_time >= current_time
                ),
                config=config,
            ):
                collision_events.append(
                    CollisionEventRecord(
                        time=event[0],
                        robot_id=event[1],
                        other_robot_id=event[2],
                        event_type=event[3],
                        location_id=event[4],
                    )
                )
            next_replan_time = current_time + config.coordination.replan_period

        _record_queue_snapshot(queue_snapshots, current_time, tasks, released_task_ids, completed_task_ids, active_plans)
        next_time = _next_integrated_event_time(
            current_time=current_time,
            tasks=tasks,
            released_task_ids=released_task_ids,
            completed_task_ids=completed_task_ids,
            robot_states=robot_states,
            next_replan_time=next_replan_time,
            config=config,
        )
        if next_time is None:
            break
        current_time = next_time

    finished_at = max([current_time, *(robot.available_time for robot in robot_states)], default=current_time)
    _finalize_completed_plans(
        current_time=finished_at,
        robot_states=robot_states,
        active_plans=active_plans,
        executions=executions,
        charging_executions=charging_executions,
        completed_task_ids=completed_task_ids,
        environment=environment,
        battery_config=config.battery,
    )
    _record_queue_snapshot(queue_snapshots, finished_at, tasks, released_task_ids, completed_task_ids, active_plans)
    result = SimulationResult(
        policy_name=coordinator_policy.name,
        started_at=0.0,
        finished_at=finished_at,
        tasks_generated=len(tasks),
        robot_states=robot_states,
        executions=tuple(executions),
        dispatch_traces=(),
        dispatch_node_observations=(),
        dispatch_arc_observations=(),
        unassigned_tasks=tuple(
            task for task in tasks if task.task_id not in completed_task_ids and task.task_id not in claimed_task_ids
        ),
        queue_snapshots=tuple(queue_snapshots),
        metrics=None,  # type: ignore[arg-type]
        charging_executions=tuple(charging_executions),
        robot_trajectories=tuple(robot_trajectories),
        macro_decisions=tuple(macro_decisions),
        collision_events=tuple(collision_events),
        planner_plans=tuple(planner_plans),
    )
    metrics = compute_simulation_metrics(result)
    return SimulationResult(
        policy_name=result.policy_name,
        started_at=result.started_at,
        finished_at=result.finished_at,
        tasks_generated=result.tasks_generated,
        robot_states=result.robot_states,
        executions=result.executions,
        dispatch_traces=result.dispatch_traces,
        dispatch_node_observations=result.dispatch_node_observations,
        dispatch_arc_observations=result.dispatch_arc_observations,
        unassigned_tasks=result.unassigned_tasks,
        queue_snapshots=result.queue_snapshots,
        metrics=metrics,
        charging_executions=result.charging_executions,
        robot_trajectories=result.robot_trajectories,
        macro_decisions=result.macro_decisions,
        collision_events=result.collision_events,
        planner_plans=result.planner_plans,
    )


def build_integrated_observation(
    *,
    environment: WarehouseEnvironment,
    robot_states: tuple[RobotState, ...],
    tasks: tuple[Task, ...],
    released_task_ids: set[str],
    claimed_task_ids: set[str],
    completed_task_ids: set[str],
    active_plans: dict[str, _ActivePlan],
    occupancy: ContinuousOccupancyTable,
    current_time: float,
    config: SimulationConfig,
) -> IntegratedObservation:
    """Build the centralized observation used by integrated policies."""

    node_ids = tuple(node.node_id for node in environment.graph.nodes())
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edge_index = tuple(
        (node_index[edge.source], node_index[edge.target])
        for node_id in node_ids
        for neighbor in environment.graph.neighbors(node_id)
        for edge in (environment.graph.edge(node_id, neighbor.node_id),)
    )
    occupied_edges = {
        (traversal.source_id, traversal.target_id): traversal
        for traversal in occupancy.future_traversals(current_time)
    }
    node_features = tuple(
        (
            float(environment.graph.node(node_id).x),
            float(environment.graph.node(node_id).y),
            1.0 if any(robot.current_node == node_id and robot.available_time <= current_time for robot in robot_states) else 0.0,
            float(sum(task.pickup_node == node_id and task.task_id in released_task_ids and task.task_id not in claimed_task_ids and task.task_id not in completed_task_ids for task in tasks)),
            float(sum(task.dropoff_node == node_id and task.task_id in released_task_ids and task.task_id not in completed_task_ids for task in tasks)),
        )
        for node_id in node_ids
    )
    edge_features = tuple(
        (
            float(environment.graph.edge(node_ids[source_index], node_ids[target_index]).distance),
            float(environment.graph.edge(node_ids[source_index], node_ids[target_index]).travel_time),
            1.0 if (node_ids[source_index], node_ids[target_index]) in occupied_edges else 0.0,
        )
        for source_index, target_index in edge_index
    )
    robot_features = tuple(
        _robot_feature_row(environment, robot, active_plans.get(robot.spec.robot_id), current_time)
        for robot in robot_states
    )
    active_task_ids = {task_id for task_id in claimed_task_ids if task_id not in completed_task_ids}
    task_ids = tuple(task.task_id for task in tasks if task.task_id in released_task_ids and task.task_id not in completed_task_ids)
    task_features = tuple(
        (
            task.release_time,
            max(current_time - task.release_time, 0.0),
            float(task.priority),
            task.service_time_estimate,
            1.0 if task.task_id in active_task_ids else 0.0,
        )
        for task in tasks
        if task.task_id in task_ids
    )
    macro_candidates = tuple(
        _build_robot_macro_candidates(
            environment=environment,
            robot=robot,
            tasks=tasks,
            released_task_ids=released_task_ids,
            claimed_task_ids=claimed_task_ids,
            completed_task_ids=completed_task_ids,
            active_plan=active_plans.get(robot.spec.robot_id),
            current_time=current_time,
            config=config,
        )
        for robot in robot_states
    )
    return IntegratedObservation(
        current_time=current_time,
        graph_node_ids=node_ids,
        edge_index=edge_index,
        node_features=node_features,
        edge_features=edge_features,
        robot_features=robot_features,
        task_features=task_features,
        robot_ids=tuple(robot.spec.robot_id for robot in robot_states),
        task_ids=task_ids,
        macro_candidates=macro_candidates,
    )


def _macro_selection_diagnostics(
    candidates: tuple[MacroCandidate, ...],
    chosen_index: int,
) -> tuple[int, float, float]:
    if not candidates:
        return 0, 0.0, 0.0
    estimated_pairs = sorted(
        ((candidate.estimated_completion_time, index) for index, candidate in enumerate(candidates)),
        key=lambda item: (item[0], item[1]),
    )
    rank_by_index = {index: rank + 1 for rank, (_value, index) in enumerate(estimated_pairs)}
    best_estimated_completion = float(estimated_pairs[0][0])
    chosen_estimated_completion = float(candidates[chosen_index].estimated_completion_time)
    return (
        rank_by_index.get(chosen_index, len(candidates)),
        best_estimated_completion,
        chosen_estimated_completion - best_estimated_completion,
    )


def _count_pre_resolution_conflicts(
    *,
    environment: WarehouseEnvironment,
    observation: IntegratedObservation,
    output,
    robot_states: tuple[RobotState, ...],
    occupancy,
    current_time: float,
    config: SimulationConfig,
) -> int:
    if config.coordination is None:
        return 0
    naive_traversals = []
    for robot_index, robot in enumerate(robot_states):
        candidates = observation.macro_candidates[robot_index]
        if not candidates:
            continue
        chosen_index = output.chosen_indices[robot_index] if robot_index < len(output.chosen_indices) else 0
        if chosen_index < 0 or chosen_index >= len(candidates):
            chosen_index = 0
        candidate = candidates[chosen_index]
        if candidate.macro_type not in {"task_route", "charge_route"}:
            continue
        planned = plan_motion_candidate(
            environment,
            robot_id=robot.spec.robot_id,
            start_time=current_time,
            speed_multiplier=robot.spec.speed_multiplier,
            occupancy_table=occupancy.clone(),
            candidate=candidate,
            service_time=candidate.service_time_estimate,
            motion_model=config.coordination.motion_model,
        )
        if planned is None:
            continue
        naive_traversals.extend(planned.traversals)
    return len(_detect_motion_collisions(traversals=tuple(naive_traversals), config=config))


def _build_robot_macro_candidates(
    *,
    environment: WarehouseEnvironment,
    robot: RobotState,
    tasks: tuple[Task, ...],
    released_task_ids: set[str],
    claimed_task_ids: set[str],
    completed_task_ids: set[str],
    active_plan: _ActivePlan | None,
    current_time: float,
    config: SimulationConfig,
) -> tuple[MacroCandidate, ...]:
    if active_plan is not None and active_plan.completion_time > current_time:
        return (
            MacroCandidate(
                macro_type="continue_current_plan",
                task_id=None if active_plan.task is None else active_plan.task.task_id,
                route_nodes=tuple(
                    [active_plan.traversals[0].source_id, *(traversal.target_id for traversal in active_plan.traversals)]
                ) if active_plan.traversals else (robot.current_node,),
                route_edges=tuple((traversal.source_id, traversal.target_id) for traversal in active_plan.traversals),
                estimated_completion_time=active_plan.completion_time,
                pickup_node=None if active_plan.task is None else active_plan.task.pickup_node,
                dropoff_node=None if active_plan.task is None else active_plan.task.dropoff_node,
                charging_node=active_plan.charging_node_id,
            ),
        )

    candidates: list[MacroCandidate] = []
    assert config.coordination is not None
    if not (
        config.battery is not None
        and config.battery.enabled
        and robot.spec.battery_capacity > 0
        and robot.battery_level / max(robot.spec.battery_capacity, 1e-9) <= config.battery.dispatch_charge_threshold
    ):
        candidates.append(
            MacroCandidate(
                macro_type="wait",
                estimated_completion_time=current_time + config.coordination.control_dt if config.coordination else current_time,
            )
        )
    for task in sorted(tasks, key=lambda item: (item.release_time, item.task_id)):
        if task.task_id not in released_task_ids or task.task_id in claimed_task_ids or task.task_id in completed_task_ids:
            continue
        if battery_enabled(config.battery):
            battery_estimate = estimate_task_action(
                environment,
                robot_node=robot.current_node,
                pickup_node=task.pickup_node,
                dropoff_node=task.dropoff_node,
                battery_level=robot.battery_level,
                speed_multiplier=robot.spec.speed_multiplier,
                battery_config=config.battery,
            )
            if battery_estimate is None or not battery_estimate.charger_reachable_after_action:
                continue
        if config.coordination.motion_model in {"free_space", "obstacle_aware_free_space"}:
            route_nodes = (robot.current_node, task.pickup_node, task.dropoff_node)
            route_options = continuous_route_options(
                environment,
                route_nodes=route_nodes,
                speed_multiplier=robot.spec.speed_multiplier,
                robot_radius=config.coordination.robot_radius,
                collision_clearance=config.coordination.collision_clearance,
                max_leg_paths=config.coordination.k_shortest_paths,
                max_route_options=config.coordination.max_route_options_per_pair,
            )
            for option in route_options:
                candidates.append(
                    MacroCandidate(
                        macro_type="task_route",
                        task_id=task.task_id,
                        route_nodes=route_nodes,
                        route_edges=tuple(zip(route_nodes, route_nodes[1:])),
                        estimated_completion_time=option.travel_time + task.service_time_estimate,
                        service_time_estimate=task.service_time_estimate,
                        pickup_node=task.pickup_node,
                        dropoff_node=task.dropoff_node,
                        leg_points=option.leg_points,
                    )
                )
        else:
            candidates.extend(
                generate_route_options(
                    environment,
                    source=robot.current_node,
                    pickup_node=task.pickup_node,
                    dropoff_node=task.dropoff_node,
                    k_shortest=config.coordination.k_shortest_paths,
                    max_route_options=config.coordination.max_route_options_per_pair,
                    task_id=task.task_id,
                    service_time_estimate=task.service_time_estimate,
                )
            )
    if battery_enabled(config.battery):
        charge_option = nearest_charging_option(
            environment,
            source_node=robot.current_node,
            speed_multiplier=robot.spec.speed_multiplier,
            battery_config=config.battery,
        )
        if charge_option is not None:
            battery_estimate = estimate_charge_action(
                environment,
                robot_node=robot.current_node,
                charging_node_id=charge_option.charging_node_id,
                battery_level=robot.battery_level,
                speed_multiplier=robot.spec.speed_multiplier,
                battery_config=config.battery,
            )
            if battery_estimate.charger_reachable_after_action and (battery_estimate.charge_duration_to_full or 0.0) > 1e-9:
                charge_duration = battery_estimate.charge_duration_to_full or 0.0
                route_nodes = (
                    (robot.current_node,)
                    if robot.current_node == charge_option.charging_node_id
                    else (robot.current_node, charge_option.charging_node_id)
                )
                if config.coordination.motion_model in {"free_space", "obstacle_aware_free_space"}:
                    route_options = continuous_route_options(
                        environment,
                        route_nodes=route_nodes,
                        speed_multiplier=robot.spec.speed_multiplier,
                        robot_radius=config.coordination.robot_radius,
                        collision_clearance=config.coordination.collision_clearance,
                        max_leg_paths=config.coordination.k_shortest_paths,
                        max_route_options=config.coordination.max_route_options_per_pair,
                    )
                    if route_options:
                        for option in route_options:
                            candidates.append(
                                MacroCandidate(
                                    macro_type="charge_route",
                                    route_nodes=route_nodes,
                                    route_edges=tuple(zip(route_nodes, route_nodes[1:])),
                                    estimated_completion_time=option.travel_time + charge_duration,
                                    service_time_estimate=charge_duration,
                                    pickup_node=charge_option.charging_node_id,
                                    dropoff_node=charge_option.charging_node_id,
                                    charging_node=charge_option.charging_node_id,
                                    leg_points=option.leg_points,
                                )
                            )
                else:
                    candidates.append(
                        MacroCandidate(
                            macro_type="charge_route",
                            route_nodes=route_nodes,
                            route_edges=tuple(zip(route_nodes, route_nodes[1:])),
                            estimated_completion_time=charge_option.travel_time + charge_duration,
                            service_time_estimate=charge_duration,
                            pickup_node=charge_option.charging_node_id,
                            dropoff_node=charge_option.charging_node_id,
                            charging_node=charge_option.charging_node_id,
                        )
                    )
    return tuple(candidates)


def _robot_feature_row(
    environment: WarehouseEnvironment,
    robot: RobotState,
    active_plan: _ActivePlan | None,
    current_time: float,
) -> tuple[float, ...]:
    node = environment.graph.node(robot.current_node)
    return (
        float(node.x),
        float(node.y),
        max(robot.available_time - current_time, 0.0),
        robot.spec.speed_multiplier,
        1.0 if active_plan is not None and active_plan.completion_time > current_time else 0.0,
        float(len(robot.completed_task_ids)),
    )


def _release_ready_tasks(tasks: tuple[Task, ...], current_time: float, released_task_ids: set[str]) -> None:
    for task in tasks:
        if task.release_time <= current_time:
            released_task_ids.add(task.task_id)


def _finalize_completed_plans(
    *,
    current_time: float,
    robot_states: tuple[RobotState, ...],
    active_plans: dict[str, _ActivePlan],
    executions: list[TaskExecution],
    charging_executions: list[ChargingExecution],
    completed_task_ids: set[str],
    environment: WarehouseEnvironment,
    battery_config,
) -> None:
    completed_robot_ids = [
        robot_id for robot_id, plan in active_plans.items() if plan.completion_time <= current_time + 1e-9
    ]
    for robot_id in completed_robot_ids:
        plan = active_plans.pop(robot_id)
        robot = next(robot for robot in robot_states if robot.spec.robot_id == robot_id)
        if plan.action_type == "charge_route":
            assert battery_enabled(battery_config)
            assert plan.charging_node_id is not None
            charge_estimate = estimate_charge_action(
                environment,
                robot_node=plan.traversals[0].source_id if plan.traversals else plan.charging_node_id,
                charging_node_id=plan.charging_node_id,
                battery_level=plan.energy_before,
                speed_multiplier=robot.spec.speed_multiplier,
                battery_config=battery_config,
            )
            travel_distance = sum(item.distance for item in plan.traversals)
            travel_time = sum(item.travel_time for item in plan.traversals)
            arrival_time = plan.pickup_arrival_time or plan.assigned_at
            charge_duration = max(plan.completion_time - arrival_time, 0.0)
            energy_after_travel = max(plan.energy_before - charge_estimate.estimated_action_energy, 0.0)
            robot.current_node = plan.charging_node_id
            robot.available_time = plan.completion_time
            robot.total_busy_time += plan.completion_time - plan.assigned_at
            robot.total_travel_time += travel_time
            robot.total_travel_distance += travel_distance
            robot.total_congestion_delay += plan.wait_time
            robot.blocked_traversal_events += plan.blocked_events
            robot.total_energy_consumed += charge_estimate.estimated_action_energy
            robot.total_energy_charged += max(float(battery_config.capacity) - energy_after_travel, 0.0)
            robot.total_charging_time += charge_duration
            robot.charging_events += 1
            robot.battery_level = float(battery_config.capacity)
            charging_executions.append(
                ChargingExecution(
                    robot_id=robot_id,
                    charging_node_id=plan.charging_node_id,
                    started_at=plan.assigned_at,
                    arrival_time=arrival_time,
                    charging_start_time=arrival_time,
                    completion_time=plan.completion_time,
                    travel_time=travel_time,
                    travel_distance=travel_distance,
                    charge_duration=charge_duration,
                    waiting_time=plan.wait_time,
                    energy_before=plan.energy_before,
                    energy_after=float(battery_config.capacity),
                )
            )
            continue
        assert plan.task is not None
        travel_to_pickup = tuple(
            traversal for traversal in plan.traversals if traversal.phase == "travel_to_pickup"
        )
        travel_to_dropoff = tuple(
            traversal for traversal in plan.traversals if traversal.phase == "travel_to_dropoff"
        )
        pickup_distance = sum(item.distance for item in travel_to_pickup)
        dropoff_distance = sum(item.distance for item in travel_to_dropoff)
        pickup_time = sum(item.travel_time for item in travel_to_pickup)
        dropoff_time = sum(item.travel_time for item in travel_to_dropoff)
        service_start_time = plan.pickup_arrival_time
        assert service_start_time is not None
        robot.current_node = plan.task.dropoff_node
        robot.available_time = plan.completion_time
        robot.total_busy_time += plan.completion_time - plan.assigned_at
        robot.total_travel_time += pickup_time + dropoff_time
        robot.total_travel_distance += pickup_distance + dropoff_distance
        robot.total_congestion_delay += plan.wait_time
        robot.blocked_traversal_events += plan.blocked_events
        robot.completed_task_ids.append(plan.task.task_id)
        completed_task_ids.add(plan.task.task_id)
        if battery_enabled(battery_config):
            consumed = travel_energy(pickup_distance + dropoff_distance, battery_config) + float(battery_config.service_energy)
            robot.total_energy_consumed += consumed
            robot.battery_level = max(robot.battery_level - consumed, 0.0)
            if robot.battery_level <= 1e-9:
                robot.battery_depletion_events += 1
        executions.append(
            TaskExecution(
                task_id=plan.task.task_id,
                robot_id=robot_id,
                release_time=plan.task.release_time,
                assigned_at=plan.assigned_at,
                pickup_arrival_time=plan.pickup_arrival_time,
                service_start_time=service_start_time,
                completion_time=plan.completion_time,
                waiting_time=plan.assigned_at - plan.task.release_time,
                turnaround_time=plan.completion_time - plan.task.release_time,
                execution_model=ExecutionModel.CONTINUOUS,
                travel_to_pickup_time=pickup_time,
                travel_to_pickup_distance=pickup_distance,
                travel_to_pickup_ideal_time=pickup_time,
                travel_to_pickup_wait_time=plan.wait_time,
                travel_to_pickup_blocked_events=plan.blocked_events,
                travel_to_pickup_path_nodes=tuple(
                    [travel_to_pickup[0].source_id, *(item.target_id for item in travel_to_pickup)]
                ) if travel_to_pickup else (plan.task.pickup_node,),
                travel_to_pickup_path_arcs=tuple(
                    f"{item.source_id}->{item.target_id}" for item in travel_to_pickup
                ),
                travel_to_dropoff_time=dropoff_time,
                travel_to_dropoff_distance=dropoff_distance,
                travel_to_dropoff_ideal_time=dropoff_time,
                travel_to_dropoff_wait_time=0.0,
                travel_to_dropoff_blocked_events=0,
                travel_to_dropoff_path_nodes=tuple(
                    [travel_to_dropoff[0].source_id, *(item.target_id for item in travel_to_dropoff)]
                ) if travel_to_dropoff else (plan.task.dropoff_node,),
                travel_to_dropoff_path_arcs=tuple(
                    f"{item.source_id}->{item.target_id}" for item in travel_to_dropoff
                ),
                congestion_delay_time=plan.wait_time,
                blocked_traversal_events=plan.blocked_events,
                task_due_time=plan.task.due_time,
                task_tardiness=(
                    0.0
                    if plan.task.due_time is None
                    else max(plan.completion_time - plan.task.due_time, 0.0)
                ),
                completed_on_time=(
                    True
                    if plan.task.due_time is None
                    else plan.completion_time <= plan.task.due_time + 1e-9
                ),
            )
        )


def _record_queue_snapshot(
    snapshots: list[QueueSnapshot],
    current_time: float,
    tasks: tuple[Task, ...],
    released_task_ids: set[str],
    completed_task_ids: set[str],
    active_plans: dict[str, _ActivePlan],
) -> None:
    ready_tasks = sum(
        task.task_id in released_task_ids
        and task.task_id not in completed_task_ids
        and all(plan.task is None or plan.task.task_id != task.task_id for plan in active_plans.values())
        for task in tasks
    )
    future_tasks = sum(task.release_time > current_time for task in tasks if task.task_id not in completed_task_ids)
    snapshot = QueueSnapshot(
        time=current_time,
        ready_tasks=ready_tasks,
        future_tasks=future_tasks,
        busy_robots=len(active_plans),
        completed_tasks=len(completed_task_ids),
    )
    if snapshots and abs(snapshots[-1].time - snapshot.time) < 1e-9:
        snapshots[-1] = snapshot
    else:
        snapshots.append(snapshot)


def _next_integrated_event_time(
    *,
    current_time: float,
    tasks: tuple[Task, ...],
    released_task_ids: set[str],
    completed_task_ids: set[str],
    robot_states: tuple[RobotState, ...],
    next_replan_time: float,
    config: SimulationConfig,
) -> float | None:
    future_releases = [
        task.release_time
        for task in tasks
        if task.task_id not in released_task_ids and task.release_time > current_time + 1e-9
    ]
    active_robot_times = [
        robot.available_time
        for robot in robot_states
        if robot.available_time > current_time + 1e-9
    ]
    ready_incomplete_tasks = [
        task
        for task in tasks
        if task.task_id in released_task_ids and task.task_id not in completed_task_ids
    ]
    if not future_releases and not active_robot_times and not ready_incomplete_tasks:
        return None
    next_times = [
        time
        for time in [next_replan_time]
        if time > current_time + 1e-9 and (ready_incomplete_tasks or future_releases or active_robot_times)
    ]
    next_times.extend(future_releases)
    next_times.extend(active_robot_times)
    if not next_times:
        return None
    next_time = min(next_times)
    if config.horizon_seconds is not None and next_time > config.horizon_seconds and not config.continue_until_all_tasks_complete:
        return None
    return next_time


def _build_occupancy_table(config: SimulationConfig):
    assert config.coordination is not None
    if config.coordination.motion_model in {"free_space", "obstacle_aware_free_space"}:
        return FreeSpaceOccupancyTable(
            robot_radius=config.coordination.robot_radius,
            collision_clearance=config.coordination.collision_clearance,
        )
    return ContinuousOccupancyTable(
        robot_radius=config.coordination.robot_radius,
        collision_clearance=config.coordination.collision_clearance,
    )


def _detect_motion_collisions(*, traversals: tuple, config: SimulationConfig) -> tuple[tuple[float, str, str | None, str, str], ...]:
    assert config.coordination is not None
    if config.coordination.motion_model in {"free_space", "obstacle_aware_free_space"}:
        return detect_free_space_collision_events(
            traversals,
            robot_radius=config.coordination.robot_radius,
            collision_clearance=config.coordination.collision_clearance,
        )
    return detect_collision_events(
        traversals,
        robot_radius=config.coordination.robot_radius,
        collision_clearance=config.coordination.collision_clearance,
    )
