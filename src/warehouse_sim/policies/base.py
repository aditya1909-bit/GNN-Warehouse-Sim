"""Dispatch-policy interfaces for baseline warehouse simulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from warehouse_sim.agents import RobotState
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.policies.observation import DispatchContext
from warehouse_sim.tasks import Task


@dataclass(frozen=True)
class DispatchDecision:
    """Assignment decision returned by a dispatch policy."""

    robot_id: str
    action_type: str = "task"
    task_id: str | None = None
    charging_node_id: str | None = None

    def __post_init__(self) -> None:
        if self.action_type not in {"task", "charge"}:
            raise ValueError("DispatchDecision.action_type must be 'task' or 'charge'.")
        if self.action_type == "task" and not self.task_id:
            raise ValueError("DispatchDecision.task_id must be set for task actions.")
        if self.action_type == "charge" and not self.charging_node_id:
            raise ValueError("DispatchDecision.charging_node_id must be set for charge actions.")

    @classmethod
    def for_task(cls, *, robot_id: str, task_id: str) -> "DispatchDecision":
        return cls(robot_id=robot_id, action_type="task", task_id=task_id)

    @classmethod
    def for_charge(cls, *, robot_id: str, charging_node_id: str) -> "DispatchDecision":
        return cls(robot_id=robot_id, action_type="charge", charging_node_id=charging_node_id)


class DispatchPolicy(ABC):
    """Abstract base class for task-dispatch policies."""

    name: str

    def select_assignment_from_context(self, context: DispatchContext) -> DispatchDecision | None:
        """Select an assignment from a rich dispatch-time observation.

        Baseline policies can keep implementing ``select_assignment`` and rely on
        this adapter. Future learned policies can override this method and
        consume graph, robot, task, and global feature views directly.
        """

        return self.select_assignment(
            idle_robots=context.idle_robots,
            ready_tasks=context.ready_tasks,
            environment=context.environment,
        )

    @abstractmethod
    def select_assignment(
        self,
        idle_robots: tuple[RobotState, ...],
        ready_tasks: tuple[Task, ...],
        environment: WarehouseEnvironment,
    ) -> DispatchDecision | None:
        """Select the next robot-task assignment."""
