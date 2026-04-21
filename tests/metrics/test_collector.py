"""Tests for simulation metric aggregation."""

from __future__ import annotations

from warehouse_sim.agents import RobotSpec, RobotState
from warehouse_sim.integrated.models import PlannerPlanRecord
from warehouse_sim.metrics.collector import compute_simulation_metrics
from warehouse_sim.metrics.models import SimulationMetrics
from warehouse_sim.simulation.models import ExecutionModel, QueueSnapshot, SimulationResult, TaskExecution


def test_compute_simulation_metrics_emits_lateness_and_planner_conflict_metrics() -> None:
    robot = RobotState.from_spec(RobotSpec(robot_id="robot_1", initial_node="r0_c0"))
    robot.available_time = 20.0
    robot.total_busy_time = 15.0
    robot.completed_task_ids.extend(["task_1", "task_2"])

    executions = (
        TaskExecution(
            task_id="task_1",
            robot_id="robot_1",
            release_time=0.0,
            assigned_at=1.0,
            pickup_arrival_time=3.0,
            service_start_time=4.0,
            completion_time=10.0,
            waiting_time=1.0,
            turnaround_time=10.0,
            execution_model=ExecutionModel.IDEALIZED,
            travel_to_pickup_time=2.0,
            travel_to_pickup_distance=2.0,
            travel_to_pickup_ideal_time=2.0,
            travel_to_pickup_wait_time=0.0,
            travel_to_pickup_blocked_events=0,
            travel_to_pickup_path_nodes=("r0_c0", "r0_c1"),
            travel_to_pickup_path_arcs=("r0_c0->r0_c1",),
            travel_to_dropoff_time=4.0,
            travel_to_dropoff_distance=4.0,
            travel_to_dropoff_ideal_time=4.0,
            travel_to_dropoff_wait_time=0.0,
            travel_to_dropoff_blocked_events=0,
            travel_to_dropoff_path_nodes=("r0_c1", "r0_c2"),
            travel_to_dropoff_path_arcs=("r0_c1->r0_c2",),
            congestion_delay_time=0.0,
            blocked_traversal_events=0,
            task_due_time=12.0,
            task_tardiness=0.0,
            completed_on_time=True,
        ),
        TaskExecution(
            task_id="task_2",
            robot_id="robot_1",
            release_time=0.0,
            assigned_at=2.0,
            pickup_arrival_time=5.0,
            service_start_time=6.0,
            completion_time=18.0,
            waiting_time=2.0,
            turnaround_time=18.0,
            execution_model=ExecutionModel.IDEALIZED,
            travel_to_pickup_time=3.0,
            travel_to_pickup_distance=3.0,
            travel_to_pickup_ideal_time=3.0,
            travel_to_pickup_wait_time=0.0,
            travel_to_pickup_blocked_events=0,
            travel_to_pickup_path_nodes=("r0_c0", "r1_c0"),
            travel_to_pickup_path_arcs=("r0_c0->r1_c0",),
            travel_to_dropoff_time=6.0,
            travel_to_dropoff_distance=6.0,
            travel_to_dropoff_ideal_time=6.0,
            travel_to_dropoff_wait_time=0.0,
            travel_to_dropoff_blocked_events=0,
            travel_to_dropoff_path_nodes=("r1_c0", "r2_c0"),
            travel_to_dropoff_path_arcs=("r1_c0->r2_c0",),
            congestion_delay_time=1.5,
            blocked_traversal_events=2,
            task_due_time=14.0,
            task_tardiness=4.0,
            completed_on_time=False,
        ),
    )
    planner_plans = (
        PlannerPlanRecord(
            plan_index=0,
            plan_time=5.0,
            robot_id="robot_1",
            task_id="task_1",
            priority_rank=0,
            path_nodes=("r0_c0", "r0_c1"),
            path_edges=("r0_c0->r0_c1",),
            planned_start_time=5.0,
            planned_end_time=8.0,
            planner_name="prioritized_sipp",
            status="planned",
            pre_resolution_conflict_count=3,
            wait_insertion_count=1,
            wait_insertion_time=2.0,
        ),
        PlannerPlanRecord(
            plan_index=1,
            plan_time=5.0,
            robot_id="robot_2",
            task_id="task_2",
            priority_rank=1,
            path_nodes=("r1_c0", "r2_c0"),
            path_edges=("r1_c0->r2_c0",),
            planned_start_time=5.0,
            planned_end_time=9.0,
            planner_name="prioritized_sipp",
            status="planned",
            pre_resolution_conflict_count=3,
            wait_insertion_count=2,
            wait_insertion_time=4.0,
        ),
        PlannerPlanRecord(
            plan_index=2,
            plan_time=9.0,
            robot_id="robot_1",
            task_id="task_2",
            priority_rank=0,
            path_nodes=("r0_c1", "r0_c2"),
            path_edges=("r0_c1->r0_c2",),
            planned_start_time=9.0,
            planned_end_time=13.0,
            planner_name="prioritized_sipp",
            status="planned",
            pre_resolution_conflict_count=2,
            wait_insertion_count=3,
            wait_insertion_time=6.0,
        ),
    )
    placeholder_metrics = SimulationMetrics(
        tasks_generated=0,
        tasks_completed=0,
        tasks_unassigned=0,
        average_waiting_time=None,
        average_turnaround_time=None,
        average_travel_distance_per_task=None,
        realized_travel_time_total=0.0,
        realized_travel_distance_total=0.0,
        congestion_delay_total=0.0,
        average_congestion_delay_per_completed_task=None,
        blocked_traversal_events_total=0,
        total_energy_consumed=0.0,
        total_energy_charged=0.0,
        total_charging_time=0.0,
        charging_events_total=0,
        battery_depletion_incidents_total=0,
        average_queue_length=0.0,
        throughput_per_hour=0.0,
        makespan=0.0,
        robot_metrics=(),
    )
    result = SimulationResult(
        policy_name="prioritized_sipp_coordinator",
        started_at=0.0,
        finished_at=20.0,
        tasks_generated=2,
        robot_states=(robot,),
        executions=executions,
        dispatch_traces=(),
        dispatch_node_observations=(),
        dispatch_arc_observations=(),
        unassigned_tasks=(),
        queue_snapshots=(
            QueueSnapshot(time=0.0, ready_tasks=1, future_tasks=1, busy_robots=0, completed_tasks=0),
            QueueSnapshot(time=10.0, ready_tasks=0, future_tasks=0, busy_robots=1, completed_tasks=1),
            QueueSnapshot(time=20.0, ready_tasks=0, future_tasks=0, busy_robots=0, completed_tasks=2),
        ),
        metrics=placeholder_metrics,
        planner_plans=planner_plans,
    )

    metrics = compute_simulation_metrics(result)

    assert metrics.on_time_completion_rate == 0.5
    assert metrics.mean_tardiness == 2.0
    assert metrics.p95_tardiness == 4.0
    assert metrics.overdue_task_count == 1
    assert metrics.path_conflicts_before_resolution_total == 5
    assert metrics.sipp_wait_insertions_total == 6
    assert metrics.planner_wait_time_total == 12.0
