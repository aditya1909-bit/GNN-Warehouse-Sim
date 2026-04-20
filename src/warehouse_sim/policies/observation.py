"""Observation and featurization contracts for dispatch policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from warehouse_sim.agents import RobotState
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import GraphFeatures, build_graph_features
from warehouse_sim.tasks import Task
from warehouse_sim.utils.battery import battery_enabled, battery_fraction

if TYPE_CHECKING:
    from warehouse_sim.simulation.models import BatteryRuntimeConfig


@dataclass(frozen=True)
class ResourceReservationObservation:
    """Active reservation on a simplified node or directed edge resource."""

    resource_id: str
    reserved_until: float


@dataclass(frozen=True)
class CongestionObservation:
    """Dispatch-time summary of currently active resource reservations."""

    execution_model: str
    edge_reservations: tuple[ResourceReservationObservation, ...] = ()
    node_reservations: tuple[ResourceReservationObservation, ...] = ()

    @property
    def active_reserved_edge_count(self) -> int:
        return len(self.edge_reservations)

    @property
    def active_reserved_node_count(self) -> int:
        return len(self.node_reservations)


@dataclass(frozen=True)
class RobotObservation:
    """Feature view of a robot at a dispatch decision point."""

    robot_id: str
    current_node: str
    current_zone: str | None
    available_time: float
    time_until_available: float
    speed_multiplier: float
    completed_task_count: int
    total_busy_time: float
    total_idle_time: float
    total_travel_time: float
    total_travel_distance: float
    battery_level: float
    battery_fraction: float
    total_charging_time: float
    total_energy_consumed: float
    total_energy_charged: float
    is_idle: bool


@dataclass(frozen=True)
class TaskObservation:
    """Feature view of a pending task at a dispatch decision point."""

    task_id: str
    release_time: float
    age: float
    time_until_release: float
    priority: int
    service_time_estimate: float
    due_time_remaining: float | None
    pickup_node: str
    dropoff_node: str
    source_zone: str | None
    destination_zone: str | None
    pickup_to_dropoff_distance: float
    pickup_to_dropoff_travel_time: float
    is_ready: bool


@dataclass(frozen=True)
class GlobalObservation:
    """Global simulation features available to a dispatch policy."""

    current_time: float
    pending_task_count: int
    ready_task_count: int
    future_task_count: int
    idle_robot_count: int
    busy_robot_count: int
    mean_ready_task_age: float
    max_robot_available_time: float
    average_robot_time_until_available: float
    execution_model: str
    active_reserved_edge_count: int
    active_reserved_node_count: int


@dataclass(frozen=True)
class DispatchContext:
    """Full dispatch-time context for baseline and future learned policies."""

    current_time: float
    environment: WarehouseEnvironment
    graph_features: GraphFeatures
    idle_robots: tuple[RobotState, ...]
    busy_robots: tuple[RobotState, ...]
    ready_tasks: tuple[Task, ...]
    future_tasks: tuple[Task, ...]
    robot_observations: tuple[RobotObservation, ...]
    task_observations: tuple[TaskObservation, ...]
    global_observation: GlobalObservation
    congestion_observation: CongestionObservation
    battery_config: "BatteryRuntimeConfig | None" = None


class DispatchContextBuilder:
    """Build reusable dispatch contexts from simulation state.

    The static graph featurization is computed once per environment and reused
    across dispatch decisions within a simulation run.
    """

    def __init__(self, environment: WarehouseEnvironment) -> None:
        self._environment = environment
        self._graph_features = build_graph_features(
            environment.graph,
            zone_lookup=environment.zone_for_node,
        )

    def build(
        self,
        current_time: float,
        robot_states: tuple[RobotState, ...],
        pending_tasks: tuple[Task, ...],
        congestion_observation: CongestionObservation | None = None,
        execution_model: str = "idealized",
        battery_config: "BatteryRuntimeConfig | None" = None,
    ) -> DispatchContext:
        """Build the dispatch context for the current simulation time."""

        if current_time < 0:
            raise ValueError("current_time must be >= 0.")

        ready_tasks = tuple(task for task in pending_tasks if task.release_time <= current_time)
        future_tasks = tuple(task for task in pending_tasks if task.release_time > current_time)
        idle_robots = tuple(
            sorted(
                (robot for robot in robot_states if robot.available_time <= current_time),
                key=lambda item: item.spec.robot_id,
            )
        )
        busy_robots = tuple(
            sorted(
                (robot for robot in robot_states if robot.available_time > current_time),
                key=lambda item: item.spec.robot_id,
            )
        )
        robot_observations = tuple(
            self._build_robot_observation(
                current_time=current_time,
                robot=robot,
                battery_config=battery_config,
            )
            for robot in sorted(robot_states, key=lambda item: item.spec.robot_id)
        )
        task_observations = tuple(
            self._build_task_observation(current_time=current_time, task=task)
            for task in pending_tasks
        )
        global_observation = self._build_global_observation(
            current_time=current_time,
            ready_tasks=ready_tasks,
            future_tasks=future_tasks,
            idle_robots=idle_robots,
            busy_robots=busy_robots,
            robot_states=robot_states,
            execution_model=execution_model,
            congestion_observation=congestion_observation or CongestionObservation(execution_model=execution_model),
        )
        return DispatchContext(
            current_time=current_time,
            environment=self._environment,
            graph_features=self._graph_features,
            idle_robots=idle_robots,
            busy_robots=busy_robots,
            ready_tasks=ready_tasks,
            future_tasks=future_tasks,
            robot_observations=robot_observations,
            task_observations=task_observations,
            global_observation=global_observation,
            congestion_observation=congestion_observation or CongestionObservation(execution_model=execution_model),
            battery_config=battery_config,
        )

    def _build_robot_observation(
        self,
        current_time: float,
        robot: RobotState,
        battery_config: "BatteryRuntimeConfig | None" = None,
    ) -> RobotObservation:
        return RobotObservation(
            robot_id=robot.spec.robot_id,
            current_node=robot.current_node,
            current_zone=self._environment.zone_for_node(robot.current_node),
            available_time=robot.available_time,
            time_until_available=max(robot.available_time - current_time, 0.0),
            speed_multiplier=robot.spec.speed_multiplier,
            completed_task_count=len(robot.completed_task_ids),
            total_busy_time=robot.total_busy_time,
            total_idle_time=robot.total_idle_time,
            total_travel_time=robot.total_travel_time,
            total_travel_distance=robot.total_travel_distance,
            battery_level=robot.battery_level,
            battery_fraction=(
                battery_fraction(battery_level=robot.battery_level, battery_config=battery_config)
                if battery_enabled(battery_config)
                else 1.0
            ),
            total_charging_time=robot.total_charging_time,
            total_energy_consumed=robot.total_energy_consumed,
            total_energy_charged=robot.total_energy_charged,
            is_idle=robot.available_time <= current_time,
        )

    def _build_task_observation(self, current_time: float, task: Task) -> TaskObservation:
        return TaskObservation(
            task_id=task.task_id,
            release_time=task.release_time,
            age=max(current_time - task.release_time, 0.0),
            time_until_release=max(task.release_time - current_time, 0.0),
            priority=task.priority,
            service_time_estimate=task.service_time_estimate,
            due_time_remaining=None if task.due_time is None else task.due_time - current_time,
            pickup_node=task.pickup_node,
            dropoff_node=task.dropoff_node,
            source_zone=task.source_zone or self._environment.zone_for_node(task.pickup_node),
            destination_zone=task.destination_zone or self._environment.zone_for_node(task.dropoff_node),
            pickup_to_dropoff_distance=self._environment.distance(task.pickup_node, task.dropoff_node),
            pickup_to_dropoff_travel_time=self._environment.travel_time(task.pickup_node, task.dropoff_node),
            is_ready=task.release_time <= current_time,
        )

    @staticmethod
    def _build_global_observation(
        current_time: float,
        ready_tasks: tuple[Task, ...],
        future_tasks: tuple[Task, ...],
        idle_robots: tuple[RobotState, ...],
        busy_robots: tuple[RobotState, ...],
        robot_states: tuple[RobotState, ...],
        execution_model: str,
        congestion_observation: CongestionObservation,
    ) -> GlobalObservation:
        mean_ready_task_age = 0.0
        if ready_tasks:
            mean_ready_task_age = sum(current_time - task.release_time for task in ready_tasks) / len(ready_tasks)

        max_robot_available_time = max((robot.available_time for robot in robot_states), default=current_time)
        average_robot_time_until_available = 0.0
        if robot_states:
            average_robot_time_until_available = sum(
                max(robot.available_time - current_time, 0.0) for robot in robot_states
            ) / len(robot_states)
        return GlobalObservation(
            current_time=current_time,
            pending_task_count=len(ready_tasks) + len(future_tasks),
            ready_task_count=len(ready_tasks),
            future_task_count=len(future_tasks),
            idle_robot_count=len(idle_robots),
            busy_robot_count=len(busy_robots),
            mean_ready_task_age=mean_ready_task_age,
            max_robot_available_time=max_robot_available_time,
            average_robot_time_until_available=average_robot_time_until_available,
            execution_model=execution_model,
            active_reserved_edge_count=congestion_observation.active_reserved_edge_count,
            active_reserved_node_count=congestion_observation.active_reserved_node_count,
        )
