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
        return DispatchDecision.for_task(robot_id=robot.spec.robot_id, task_id=task.task_id)

    def select_assignment_from_context(self, context):  # type: ignore[override]
        candidates = build_candidate_assignment_observations(context)
        if not candidates:
            return None
        urgent_charge, task_candidates = _split_dispatch_candidates(context, candidates)
        if urgent_charge:
            return DispatchDecision.for_charge(
                robot_id=urgent_charge[0].robot_id,
                charging_node_id=urgent_charge[0].charging_node_id or "",
            )
        if not task_candidates:
            return None
        best_candidate = min(
            task_candidates,
            key=lambda candidate: (
                -candidate.feature("task_age"),
                candidate.task_id or "",
                candidate.robot_id,
            ),
        )
        return DispatchDecision.for_task(robot_id=best_candidate.robot_id, task_id=best_candidate.task_id or "")


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
        return DispatchDecision.for_task(robot_id=robot.spec.robot_id, task_id=task.task_id)

    def select_assignment_from_context(self, context):  # type: ignore[override]
        candidates = build_candidate_assignment_observations(context)
        if not candidates:
            return None
        urgent_charge, task_candidates = _split_dispatch_candidates(context, candidates)
        pool = urgent_charge or task_candidates
        if not pool:
            return None
        chosen = self._rng.choice(pool)
        if chosen.action_type == "charge":
            return DispatchDecision.for_charge(
                robot_id=chosen.robot_id,
                charging_node_id=chosen.charging_node_id or "",
            )
        return DispatchDecision.for_task(robot_id=chosen.robot_id, task_id=chosen.task_id or "")


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
        return DispatchDecision.for_task(robot_id=best_pair[2], task_id=best_pair[3])

    def select_assignment_from_context(self, context):  # type: ignore[override]
        candidates = build_candidate_assignment_observations(context)
        if not candidates:
            return None
        urgent_charge, task_candidates = _split_dispatch_candidates(context, candidates)
        if urgent_charge:
            chosen = min(
                urgent_charge,
                key=lambda candidate: (
                    candidate.feature("travel_to_pickup_time"),
                    candidate.robot_id,
                    candidate.charging_node_id or "",
                ),
            )
            return DispatchDecision.for_charge(
                robot_id=chosen.robot_id,
                charging_node_id=chosen.charging_node_id or "",
            )
        if not task_candidates:
            return None
        chosen = min(
            task_candidates,
            key=lambda candidate: (
                candidate.feature("travel_to_pickup_time"),
                candidate.feature("task_age") * -1.0,
                candidate.robot_id,
                candidate.task_id or "",
            ),
        )
        return DispatchDecision.for_task(robot_id=chosen.robot_id, task_id=chosen.task_id or "")


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
        return DispatchDecision.for_task(robot_id=robot.spec.robot_id, task_id=task.task_id)

    def select_assignment_from_context(self, context):  # type: ignore[override]
        candidates = build_candidate_assignment_observations(context)
        if not candidates:
            return None
        urgent_charge, task_candidates = _split_dispatch_candidates(context, candidates)
        first_robot_id = min(candidate.robot_id for candidate in candidates)
        first_robot_charge = [candidate for candidate in urgent_charge if candidate.robot_id == first_robot_id]
        if first_robot_charge:
            chosen = min(
                first_robot_charge,
                key=lambda candidate: (
                    candidate.feature("travel_to_pickup_time"),
                    candidate.charging_node_id or "",
                ),
            )
            return DispatchDecision.for_charge(
                robot_id=chosen.robot_id,
                charging_node_id=chosen.charging_node_id or "",
            )
        first_robot_tasks = [candidate for candidate in task_candidates if candidate.robot_id == first_robot_id]
        if not first_robot_tasks:
            return None
        chosen = min(
            first_robot_tasks,
            key=lambda candidate: (
                candidate.feature("travel_to_pickup_time"),
                candidate.feature("task_age") * -1.0,
                candidate.task_id or "",
            ),
        )
        return DispatchDecision.for_task(robot_id=chosen.robot_id, task_id=chosen.task_id or "")


class CongestionAwareNearestRobotTaskPolicy(DispatchPolicy):
    """Greedy baseline that penalizes routes with predicted blocking."""

    name = "congestion_aware_nearest_robot_task"

    def __init__(self, blocking_penalty: float = 1.0) -> None:
        self._blocking_penalty = blocking_penalty

    def select_assignment_from_context(self, context):  # type: ignore[override]
        candidates = build_candidate_assignment_observations(context)
        if not candidates:
            return None

        urgent_charge, task_candidates = _split_dispatch_candidates(context, candidates)
        if urgent_charge:
            chosen = min(
                urgent_charge,
                key=lambda candidate: (
                    candidate.feature("travel_to_pickup_time"),
                    candidate.robot_id,
                    candidate.charging_node_id or "",
                ),
            )
            return DispatchDecision.for_charge(
                robot_id=chosen.robot_id,
                charging_node_id=chosen.charging_node_id or "",
            )
        if not task_candidates:
            return None
        best_candidate = min(
            task_candidates,
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
                candidate.task_id or "",
            ),
        )
        return DispatchDecision.for_task(robot_id=best_candidate.robot_id, task_id=best_candidate.task_id or "")

    def select_assignment(self, idle_robots, ready_tasks, environment):  # type: ignore[override]
        raise RuntimeError(
            "CongestionAwareNearestRobotTaskPolicy requires dispatch contexts and cannot use the legacy selection path."
        )


def _split_dispatch_candidates(context, candidates):
    task_candidates = [candidate for candidate in candidates if candidate.action_type == "task"]
    if not context.battery_config or not context.battery_config.enabled:
        return [], task_candidates
    task_counts_by_robot: dict[str, int] = {}
    for candidate in task_candidates:
        task_counts_by_robot[candidate.robot_id] = task_counts_by_robot.get(candidate.robot_id, 0) + 1
    urgent_charge = [
        candidate
        for candidate in candidates
        if candidate.action_type == "charge"
        and (
            candidate.feature("battery_fraction") <= context.battery_config.dispatch_charge_threshold
            or task_counts_by_robot.get(candidate.robot_id, 0) == 0
        )
    ]
    return urgent_charge, task_candidates
