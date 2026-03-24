"""Non-learning baseline dispatch policies."""

from __future__ import annotations

import random

from warehouse_sim.agents import RobotState
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.policies.base import DispatchDecision, DispatchPolicy
from warehouse_sim.policies.scoring import build_candidate_assignment_observations
from warehouse_sim.tasks import Task


class FIFODispatchPolicy(DispatchPolicy):
    """Assign the earliest released task to the lexicographically first idle robot."""

    name = "fifo"

    def select_assignment(
        self,
        idle_robots: tuple[RobotState, ...],
        ready_tasks: tuple[Task, ...],
        environment: WarehouseEnvironment,
    ) -> DispatchDecision | None:
        if not idle_robots or not ready_tasks:
            return None
        robot = sorted(idle_robots, key=lambda item: item.spec.robot_id)[0]
        task = sorted(ready_tasks, key=lambda item: (item.release_time, item.task_id))[0]
        return DispatchDecision(robot_id=robot.spec.robot_id, task_id=task.task_id)


class RandomDispatchPolicy(DispatchPolicy):
    """Choose a random idle-robot and ready-task pair with deterministic seeding."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select_assignment(
        self,
        idle_robots: tuple[RobotState, ...],
        ready_tasks: tuple[Task, ...],
        environment: WarehouseEnvironment,
    ) -> DispatchDecision | None:
        if not idle_robots or not ready_tasks:
            return None
        robot = self._rng.choice(sorted(idle_robots, key=lambda item: item.spec.robot_id))
        task = self._rng.choice(sorted(ready_tasks, key=lambda item: (item.release_time, item.task_id)))
        return DispatchDecision(robot_id=robot.spec.robot_id, task_id=task.task_id)


class NearestRobotTaskPolicy(DispatchPolicy):
    """Choose the globally closest robot-task pairing by travel time to pickup."""

    name = "nearest_robot_task"

    def select_assignment(
        self,
        idle_robots: tuple[RobotState, ...],
        ready_tasks: tuple[Task, ...],
        environment: WarehouseEnvironment,
    ) -> DispatchDecision | None:
        if not idle_robots or not ready_tasks:
            return None

        best_pair: tuple[float, float, str, str] | None = None
        for robot in idle_robots:
            for task in ready_tasks:
                travel_time = (
                    environment.travel_time(robot.current_node, task.pickup_node)
                    / robot.spec.speed_multiplier
                )
                ranking = (
                    travel_time,
                    task.release_time,
                    robot.spec.robot_id,
                    task.task_id,
                )
                if best_pair is None or ranking < best_pair:
                    best_pair = ranking

        assert best_pair is not None
        return DispatchDecision(robot_id=best_pair[2], task_id=best_pair[3])


class NearestTaskForIdleRobotPolicy(DispatchPolicy):
    """Choose the nearest task for the lexicographically first idle robot."""

    name = "nearest_task_for_idle_robot"

    def select_assignment(
        self,
        idle_robots: tuple[RobotState, ...],
        ready_tasks: tuple[Task, ...],
        environment: WarehouseEnvironment,
    ) -> DispatchDecision | None:
        if not idle_robots or not ready_tasks:
            return None

        robot = sorted(idle_robots, key=lambda item: item.spec.robot_id)[0]
        task = min(
            ready_tasks,
            key=lambda item: (
                environment.travel_time(robot.current_node, item.pickup_node)
                / robot.spec.speed_multiplier,
                item.release_time,
                item.task_id,
            ),
        )
        return DispatchDecision(robot_id=robot.spec.robot_id, task_id=task.task_id)


class CongestionAwareNearestRobotTaskPolicy(DispatchPolicy):
    """Greedy baseline that penalizes routes with predicted blocking."""

    name = "congestion_aware_nearest_robot_task"

    def __init__(self, blocking_penalty: float = 1.0) -> None:
        self._blocking_penalty = blocking_penalty

    def select_assignment_from_context(self, context):  # type: ignore[override]
        candidates = build_candidate_assignment_observations(context)
        if not candidates:
            return None

        best_candidate = min(
            candidates,
            key=lambda candidate: (
                candidate.feature("travel_to_pickup_time")
                + candidate.feature("pickup_to_dropoff_time")
                + candidate.feature("estimated_pickup_congestion_delay")
                + candidate.feature("estimated_dropoff_congestion_delay")
                + self._blocking_penalty
                * (
                    candidate.feature("estimated_pickup_blocked_segments")
                    + candidate.feature("estimated_dropoff_blocked_segments")
                ),
                candidate.feature("task_age") * -1.0,
                candidate.robot_id,
                candidate.task_id,
            ),
        )
        return DispatchDecision(robot_id=best_candidate.robot_id, task_id=best_candidate.task_id)

    def select_assignment(self, idle_robots, ready_tasks, environment):  # type: ignore[override]
        raise RuntimeError(
            "CongestionAwareNearestRobotTaskPolicy requires dispatch contexts and cannot use the legacy selection path."
        )
