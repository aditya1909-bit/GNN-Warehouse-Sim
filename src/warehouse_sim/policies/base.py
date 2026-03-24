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
    task_id: str


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
