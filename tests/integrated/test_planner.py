"""Tests for integrated continuous-time planning utilities."""

from __future__ import annotations

from warehouse_sim.agents import RobotSpec, RobotState
from warehouse_sim.environment import WarehouseEnvironment, obstacle_rectangles_from_blocked_cells
from warehouse_sim.graph import SyntheticGridLayoutConfig, build_synthetic_grid_layout
from warehouse_sim.integrated.free_space import (
    FreeSpaceOccupancyTable,
    detect_free_space_collision_events,
    plan_obstacle_aware_free_space_candidate,
    plan_free_space_candidate,
)
from warehouse_sim.integrated.geometry import inflate_obstacles, segment_has_line_of_sight
from warehouse_sim.integrated.engine import build_integrated_observation
from warehouse_sim.integrated.models import MacroCandidate, TimedTraversal
from warehouse_sim.integrated.planner import (
    ContinuousOccupancyTable,
    detect_collision_events,
    generate_route_options,
    plan_route_candidate,
    solve_exact_mapf_macro_plan,
)
from warehouse_sim.simulation.models import CoordinationMode, CoordinationRuntimeConfig, ExecutionModel, SimulationConfig
from warehouse_sim.tasks import Task


def test_generate_route_options_and_plan_candidate() -> None:
    environment = WarehouseEnvironment(build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=2, columns=3)))
    occupancy = ContinuousOccupancyTable(robot_radius=0.2, collision_clearance=0.05)
    options = generate_route_options(
        environment,
        source="r1_c0",
        pickup_node="r0_c0",
        dropoff_node="r1_c2",
        k_shortest=2,
        max_route_options=2,
        task_id="task_1",
    )

    planned = plan_route_candidate(
        environment,
        robot_id="robot_1",
        start_time=0.0,
        speed_multiplier=1.0,
        occupancy_table=occupancy,
        candidate=options[0],
        service_time=1.0,
    )

    assert options
    assert planned is not None
    assert planned.pickup_arrival_time is not None
    assert planned.completion_time > planned.pickup_arrival_time


def test_detect_same_edge_and_opposite_direction_conflicts() -> None:
    environment = WarehouseEnvironment(build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=1, columns=3)))
    occupancy = ContinuousOccupancyTable(robot_radius=0.2, collision_clearance=0.05)
    options_left = generate_route_options(
        environment,
        source="r0_c0",
        pickup_node="r0_c1",
        dropoff_node="r0_c2",
        k_shortest=1,
        max_route_options=1,
        task_id="task_left",
    )
    options_right = generate_route_options(
        environment,
        source="r0_c2",
        pickup_node="r0_c1",
        dropoff_node="r0_c0",
        k_shortest=1,
        max_route_options=1,
        task_id="task_right",
    )
    left = plan_route_candidate(
        environment,
        robot_id="robot_left",
        start_time=0.0,
        speed_multiplier=1.0,
        occupancy_table=occupancy,
        candidate=options_left[0],
        service_time=0.0,
    )
    assert left is not None
    raw_conflicts = detect_collision_events(
        left.traversals
        + (
            type(left.traversals[0])(
                robot_id="robot_other",
                source_id=left.traversals[0].source_id,
                target_id=left.traversals[0].target_id,
                start_time=left.traversals[0].start_time,
                end_time=left.traversals[0].end_time,
                distance=left.traversals[0].distance,
                travel_time=left.traversals[0].travel_time,
            ),
        ),
        robot_radius=0.2,
        collision_clearance=0.05,
    )
    opposite = plan_route_candidate(
        environment,
        robot_id="robot_right",
        start_time=0.0,
        speed_multiplier=1.0,
        occupancy_table=ContinuousOccupancyTable(robot_radius=0.2, collision_clearance=0.05),
        candidate=options_right[0],
        service_time=0.0,
    )
    assert opposite is not None
    opposite_conflicts = detect_collision_events(
        (
            left.traversals[0],
            opposite.traversals[-1],
        ),
        robot_radius=0.2,
        collision_clearance=0.05,
    )

    assert any(event[3] == "same_edge_conflict" for event in raw_conflicts)
    assert any(event[3] in {"opposite_edge_conflict", "node_conflict"} for event in opposite_conflicts)


def test_solve_exact_mapf_macro_plan_returns_conflict_free_joint_solution() -> None:
    environment = WarehouseEnvironment(build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=1, columns=3)))
    tasks = (
        Task(
            task_id="task_left_to_right",
            release_time=0.0,
            pickup_node="r0_c1",
            dropoff_node="r0_c2",
            service_time_estimate=0.0,
        ),
        Task(
            task_id="task_right_to_left",
            release_time=0.0,
            pickup_node="r0_c1",
            dropoff_node="r0_c0",
            service_time_estimate=0.0,
        ),
    )
    robots = (
        RobotState.from_spec(RobotSpec(robot_id="robot_left", initial_node="r0_c0")),
        RobotState.from_spec(RobotSpec(robot_id="robot_right", initial_node="r0_c2")),
    )
    config = SimulationConfig(
        coordination_mode=CoordinationMode.INTEGRATED,
        execution_model=ExecutionModel.IDEALIZED,
        coordination=CoordinationRuntimeConfig(
            control_dt=0.25,
            replan_period=1.0,
            robot_radius=0.2,
            collision_clearance=0.05,
            k_shortest_paths=2,
            max_route_options_per_pair=2,
        ),
    )
    occupancy = ContinuousOccupancyTable(robot_radius=0.2, collision_clearance=0.05)
    observation = build_integrated_observation(
        environment=environment,
        robot_states=robots,
        tasks=tasks,
        released_task_ids={task.task_id for task in tasks},
        claimed_task_ids=set(),
        completed_task_ids=set(),
        active_plans={},
        occupancy=occupancy,
        current_time=0.0,
        config=config,
    )

    solution = solve_exact_mapf_macro_plan(
        environment,
        observation=observation,
        robot_states=robots,
        occupancy_table=occupancy,
        current_time=0.0,
        config=config,
        tasks=tasks,
    )

    assert solution is not None
    assert solution.assigned_task_count == 2
    assert solution.planner_name == "optimal_mapf_joint_search"
    conflicts = detect_collision_events(
        tuple(
            traversal
            for planned in solution.planned_routes.values()
            for traversal in planned.traversals
        ),
        robot_radius=0.2,
        collision_clearance=0.05,
    )
    assert not conflicts


def test_detect_free_space_crossing_conflict() -> None:
    conflicts = detect_free_space_collision_events(
        (
            TimedTraversal(
                robot_id="robot_a",
                source_id="r0_c0",
                target_id="r1_c1",
                start_time=0.0,
                end_time=1.0,
                distance=1.4142,
                travel_time=1.0,
                start_x=0.0,
                start_y=0.0,
                end_x=1.0,
                end_y=1.0,
            ),
            TimedTraversal(
                robot_id="robot_b",
                source_id="r0_c1",
                target_id="r1_c0",
                start_time=0.0,
                end_time=1.0,
                distance=1.4142,
                travel_time=1.0,
                start_x=1.0,
                start_y=0.0,
                end_x=0.0,
                end_y=1.0,
            ),
        ),
        robot_radius=0.2,
        collision_clearance=0.05,
    )

    assert conflicts
    assert conflicts[0][3] == "free_space_conflict"


def test_plan_free_space_candidate_delays_conflicting_segment() -> None:
    environment = WarehouseEnvironment(build_synthetic_grid_layout(SyntheticGridLayoutConfig(rows=2, columns=2)))
    occupancy = FreeSpaceOccupancyTable(robot_radius=0.2, collision_clearance=0.05)
    first = plan_free_space_candidate(
        environment,
        robot_id="robot_a",
        start_time=0.0,
        speed_multiplier=1.0,
        occupancy_table=occupancy,
        candidate=MacroCandidate(
            macro_type="task_route",
            task_id="task_a",
            route_nodes=("r0_c0", "r1_c1"),
            route_edges=(("r0_c0", "r1_c1"),),
        ),
    )
    assert first is not None
    occupancy.reserve(first.traversals)
    second = plan_free_space_candidate(
        environment,
        robot_id="robot_b",
        start_time=0.0,
        speed_multiplier=1.0,
        occupancy_table=occupancy,
        candidate=MacroCandidate(
            macro_type="task_route",
            task_id="task_b",
            route_nodes=("r0_c1", "r1_c0"),
            route_edges=(("r0_c1", "r1_c0"),),
        ),
    )

    assert second is not None
    assert second.traversals[0].start_time >= first.traversals[0].end_time


def test_plan_obstacle_aware_free_space_candidate_reroutes_around_blocked_cell() -> None:
    graph = build_synthetic_grid_layout(
        SyntheticGridLayoutConfig(
            rows=4,
            columns=4,
            blocked_cells=((1, 1),),
        )
    )
    environment = WarehouseEnvironment(
        graph,
        obstacles=obstacle_rectangles_from_blocked_cells(((1, 1),), edge_length=1.0),
    )
    inflated_obstacles = inflate_obstacles(environment.obstacles(), margin=0.25)
    occupancy = FreeSpaceOccupancyTable(robot_radius=0.2, collision_clearance=0.05)

    planned = plan_obstacle_aware_free_space_candidate(
        environment,
        robot_id="robot_a",
        start_time=0.0,
        speed_multiplier=1.0,
        occupancy_table=occupancy,
        candidate=MacroCandidate(
            macro_type="task_route",
            task_id="task_a",
            route_nodes=("r0_c0", "r2_c2"),
            route_edges=(("r0_c0", "r2_c2"),),
        ),
        robot_radius=0.2,
        collision_clearance=0.05,
        service_time=0.0,
    )

    assert planned is not None
    assert len(planned.traversals) > 1
    assert all(
        segment_has_line_of_sight(
            (traversal.start_x, traversal.start_y),
            (traversal.end_x, traversal.end_y),
            obstacles=inflated_obstacles,
        )
        for traversal in planned.traversals
        if traversal.start_x is not None and traversal.start_y is not None
        and traversal.end_x is not None and traversal.end_y is not None
    )
