"""Integrated coordination policies over task-plus-route macros."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
from typing import TYPE_CHECKING

import torch

from warehouse_sim.integrated.models import IntegratedObservation
from warehouse_sim.integrated.planner import solve_exact_mapf_macro_plan

if TYPE_CHECKING:
    from warehouse_sim.agents import RobotState
    from warehouse_sim.environment import WarehouseEnvironment
    from warehouse_sim.integrated.planner import PlannedMacro
    from warehouse_sim.integrated.planner import ContinuousOccupancyTable
    from warehouse_sim.simulation.models import SimulationConfig
    from warehouse_sim.tasks import Task


@dataclass(frozen=True)
class IntegratedPolicyOutput:
    """Chosen macro indices and optional policy statistics."""

    chosen_indices: tuple[int, ...]
    log_prob: float = 0.0
    value: float | None = None
    planner_name: str | None = None
    planned_routes: dict[str, "PlannedMacro"] = field(default_factory=dict)


class IntegratedCoordinatorPolicy(ABC):
    """Abstract policy for integrated coordination."""

    name: str
    planner_name: str = "prioritized_sipp"

    @abstractmethod
    def select_macros(self, observation: IntegratedObservation) -> IntegratedPolicyOutput:
        """Select one macro index per robot in observation order."""

    def plan_joint_macros(
        self,
        observation: IntegratedObservation,
        *,
        environment: "WarehouseEnvironment",
        occupancy: "ContinuousOccupancyTable",
        robot_states: tuple["RobotState", ...],
        tasks: tuple["Task", ...],
        current_time: float,
        config: "SimulationConfig",
    ) -> IntegratedPolicyOutput | None:
        """Optionally return a fully planned joint realization for this epoch."""

        return None


class RandomMacroPolicy(IntegratedCoordinatorPolicy):
    """Weak randomized integrated baseline."""

    name = "random_macro"

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def select_macros(self, observation: IntegratedObservation) -> IntegratedPolicyOutput:
        chosen: list[int] = []
        used_tasks: set[str] = set()
        for candidates in observation.macro_candidates:
            feasible_indices = [
                index
                for index, candidate in enumerate(candidates)
                if candidate.task_id is None or candidate.task_id not in used_tasks
            ]
            index = self._rng.choice(feasible_indices) if feasible_indices else 0
            chosen.append(index)
            task_id = candidates[index].task_id
            if task_id is not None:
                used_tasks.add(task_id)
        return IntegratedPolicyOutput(chosen_indices=tuple(chosen))


class PrioritizedSIPPCoordinatorPolicy(IntegratedCoordinatorPolicy):
    """Greedy prioritized macro selector based on earliest completion."""

    name = "prioritized_sipp_coordinator"
    planner_name = "prioritized_sipp"

    def select_macros(self, observation: IntegratedObservation) -> IntegratedPolicyOutput:
        chosen: list[int] = []
        used_tasks: set[str] = set()
        for candidates in observation.macro_candidates:
            best_index = 0
            best_key = None
            for index, candidate in enumerate(candidates):
                if candidate.task_id is not None and candidate.task_id in used_tasks:
                    continue
                ranking = (
                    0 if candidate.macro_type == "task_route" else 1,
                    candidate.estimated_completion_time,
                    candidate.task_id or "",
                )
                if best_key is None or ranking < best_key:
                    best_key = ranking
                    best_index = index
            chosen.append(best_index)
            task_id = candidates[best_index].task_id
            if task_id is not None:
                used_tasks.add(task_id)
        return IntegratedPolicyOutput(chosen_indices=tuple(chosen))


class EndToEndMacroArtifactPolicy(IntegratedCoordinatorPolicy):
    """Artifact-backed integrated macro controller."""

    name = "trained_end_to_end_macro_ppo"
    planner_name = "trained_end_to_end_macro_ppo"

    def __init__(self, model) -> None:
        self._model = model

    def select_macros(self, observation: IntegratedObservation) -> IntegratedPolicyOutput:
        with torch.no_grad():
            return self._model.act(observation, greedy=True)


class OptimalMAPFCoordinatorPolicy(IntegratedCoordinatorPolicy):
    """Exact joint-search router over the current macro candidate set."""

    name = "optimal_mapf_coordinator"
    planner_name = "optimal_mapf_joint_search"

    def select_macros(self, observation: IntegratedObservation) -> IntegratedPolicyOutput:
        # Fallback if the engine asks for indices only.
        return PrioritizedSIPPCoordinatorPolicy().select_macros(observation)

    def plan_joint_macros(
        self,
        observation: IntegratedObservation,
        *,
        environment: "WarehouseEnvironment",
        occupancy: "ContinuousOccupancyTable",
        robot_states: tuple["RobotState", ...],
        tasks: tuple["Task", ...],
        current_time: float,
        config: "SimulationConfig",
    ) -> IntegratedPolicyOutput | None:
        solution = solve_exact_mapf_macro_plan(
            environment,
            observation=observation,
            robot_states=robot_states,
            occupancy_table=occupancy,
            current_time=current_time,
            config=config,
            tasks=tasks,
        )
        if solution is None:
            return None
        return IntegratedPolicyOutput(
            chosen_indices=solution.chosen_indices,
            planner_name=solution.planner_name,
            planned_routes=solution.planned_routes,
        )
