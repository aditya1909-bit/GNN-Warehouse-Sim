"""End-to-end macro PPO training and artifact loading for integrated coordination."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.nn import GATv2Conv

from warehouse_sim.config import ExperimentConfig, IntegratedRLTrainingConfig, load_experiment_config
from warehouse_sim.integrated.engine import (
    _ActivePlan,
    _build_occupancy_table,
    _count_pre_resolution_conflicts,
    _detect_motion_collisions,
    _finalize_completed_plans,
    _macro_selection_diagnostics,
    _next_integrated_event_time,
    _record_queue_snapshot,
    _release_ready_tasks,
    build_integrated_observation,
)
from warehouse_sim.integrated.models import CollisionEventRecord, IntegratedObservation, IntegratedPolicyStep, MacroDecisionRecord, PlannerPlanRecord
from warehouse_sim.integrated.planner import plan_motion_candidate
from warehouse_sim.integrated.policies import IntegratedPolicyOutput, PrioritizedSIPPCoordinatorPolicy
from warehouse_sim.metrics.collector import compute_simulation_metrics
from warehouse_sim.simulation.runner import build_experiment_inputs, run_experiment_from_config
from warehouse_sim.simulation.models import SimulationResult


@dataclass(frozen=True)
class EndToEndMacroArtifact:
    """Serializable artifact for integrated end-to-end macro PPO."""

    artifact_version: int
    model_type: str
    parameters: dict[str, object]
    metadata: dict[str, object]


def write_end_to_end_macro_artifact(artifact: EndToEndMacroArtifact, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "artifact_version": artifact.artifact_version,
                "model_type": artifact.model_type,
                "parameters": artifact.parameters,
                "metadata": artifact.metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_end_to_end_macro_artifact(path: Path) -> EndToEndMacroArtifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EndToEndMacroArtifact(
        artifact_version=int(payload["artifact_version"]),
        model_type=str(payload["model_type"]),
        parameters=dict(payload["parameters"]),
        metadata=dict(payload.get("metadata", {})),
    )


class EndToEndMacroPolicyNetwork(nn.Module):
    """Centralized macro controller over integrated observations."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        robot_dim: int,
        task_dim: int,
        macro_dim: int = 7,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.robot_encoder = nn.Sequential(nn.Linear(robot_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.task_encoder = nn.Sequential(nn.Linear(max(task_dim, 1), hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.macro_encoder = nn.Sequential(nn.Linear(macro_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.conv = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=hidden_dim, add_self_loops=False)
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_graph(self, observation: IntegratedObservation) -> torch.Tensor:
        device = self._device()
        node_features = torch.tensor(observation.node_features, dtype=torch.float32, device=device)
        edge_features = torch.tensor(observation.edge_features, dtype=torch.float32, device=device)
        edge_index = torch.tensor(observation.edge_index, dtype=torch.long, device=device).T.contiguous()
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)
        x = torch.relu(self.conv(x, edge_index, edge_attr))
        return x.mean(dim=0)

    def act(self, observation: IntegratedObservation, greedy: bool = False) -> IntegratedPolicyOutput:
        device = self._device()
        graph_embedding = self.encode_graph(observation)
        used_tasks: set[str] = set()
        chosen_indices: list[int] = []
        log_prob_total = torch.tensor(0.0, device=device)
        for robot_index, candidates in enumerate(observation.macro_candidates):
            robot_embedding = self.robot_encoder(
                torch.tensor(observation.robot_features[robot_index], dtype=torch.float32, device=device)
            )
            candidate_matrix = torch.stack(
                [self.macro_encoder(_macro_feature_tensor(observation, candidate, device=device)) for candidate in candidates]
            )
            repeated_graph = graph_embedding.unsqueeze(0).expand(candidate_matrix.shape[0], -1)
            repeated_robot = robot_embedding.unsqueeze(0).expand(candidate_matrix.shape[0], -1)
            logits = self.candidate_head(torch.cat([repeated_graph, repeated_robot, candidate_matrix], dim=-1)).squeeze(-1)
            mask = torch.tensor(
                [candidate.task_id is None or candidate.task_id not in used_tasks for candidate in candidates],
                dtype=torch.bool,
                device=device,
            )
            masked_logits = logits.masked_fill(~mask, float("-inf"))
            distribution = torch.distributions.Categorical(logits=masked_logits)
            index = int(torch.argmax(masked_logits).item()) if greedy else int(distribution.sample().item())
            chosen_indices.append(index)
            log_prob_total = log_prob_total + distribution.log_prob(torch.tensor(index, device=device))
            task_id = candidates[index].task_id
            if task_id is not None:
                used_tasks.add(task_id)
        value = self.value(observation, graph_embedding)
        return IntegratedPolicyOutput(
            chosen_indices=tuple(chosen_indices),
            log_prob=float(log_prob_total.item()),
            value=float(value.item()),
        )

    def evaluate(self, observation: IntegratedObservation, chosen_indices: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self._device()
        graph_embedding = self.encode_graph(observation)
        used_tasks: set[str] = set()
        log_prob_total = torch.tensor(0.0, device=device)
        entropy_total = torch.tensor(0.0, device=device)
        for robot_index, candidates in enumerate(observation.macro_candidates):
            robot_embedding = self.robot_encoder(
                torch.tensor(observation.robot_features[robot_index], dtype=torch.float32, device=device)
            )
            candidate_matrix = torch.stack(
                [self.macro_encoder(_macro_feature_tensor(observation, candidate, device=device)) for candidate in candidates]
            )
            repeated_graph = graph_embedding.unsqueeze(0).expand(candidate_matrix.shape[0], -1)
            repeated_robot = robot_embedding.unsqueeze(0).expand(candidate_matrix.shape[0], -1)
            logits = self.candidate_head(torch.cat([repeated_graph, repeated_robot, candidate_matrix], dim=-1)).squeeze(-1)
            mask = torch.tensor(
                [candidate.task_id is None or candidate.task_id not in used_tasks for candidate in candidates],
                dtype=torch.bool,
                device=device,
            )
            masked_logits = logits.masked_fill(~mask, float("-inf"))
            distribution = torch.distributions.Categorical(logits=masked_logits)
            index = chosen_indices[robot_index]
            log_prob_total = log_prob_total + distribution.log_prob(torch.tensor(index, device=device))
            entropy_total = entropy_total + distribution.entropy()
            task_id = candidates[index].task_id
            if task_id is not None:
                used_tasks.add(task_id)
        value = self.value(observation, graph_embedding)
        return log_prob_total, value, entropy_total

    def value(self, observation: IntegratedObservation, graph_embedding: torch.Tensor | None = None) -> torch.Tensor:
        device = self._device()
        if graph_embedding is None:
            graph_embedding = self.encode_graph(observation)
        robot_tensor = torch.tensor(observation.robot_features, dtype=torch.float32, device=device)
        robot_embedding = self.robot_encoder(robot_tensor).mean(dim=0)
        if observation.task_features:
            task_embedding = self.task_encoder(
                torch.tensor(observation.task_features, dtype=torch.float32, device=device)
            ).mean(dim=0)
        else:
            task_embedding = torch.zeros_like(graph_embedding)
        return self.value_head(torch.cat([graph_embedding, robot_embedding, task_embedding], dim=-1)).squeeze(-1)

    def _device(self) -> torch.device:
        return next(self.parameters()).device


@dataclass(frozen=True)
class LoadedEndToEndMacroModel:
    artifact: EndToEndMacroArtifact
    model: EndToEndMacroPolicyNetwork


def load_end_to_end_macro_model(path: Path, device: torch.device | str = "cpu") -> LoadedEndToEndMacroModel:
    artifact = load_end_to_end_macro_artifact(path)
    if artifact.model_type != "end_to_end_macro_ppo":
        raise ValueError(f"Expected end_to_end_macro_ppo artifact, got {artifact.model_type}")
    parameters = artifact.parameters
    model = EndToEndMacroPolicyNetwork(
        node_dim=int(parameters["node_dim"]),
        edge_dim=int(parameters["edge_dim"]),
        robot_dim=int(parameters["robot_dim"]),
        task_dim=int(parameters["task_dim"]),
        macro_dim=int(parameters.get("macro_dim", 7)),
        hidden_dim=int(parameters.get("hidden_dim", 64)),
    )
    state_path = path.parent / str(parameters["state_dict_path"])
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()
    return LoadedEndToEndMacroModel(artifact=artifact, model=model)


class IntegratedCoordinationRLEnv:
    """Replanning-boundary RL environment for integrated coordination."""

    def __init__(self, config: ExperimentConfig, seed: int) -> None:
        seeded_config = replace(config, demand=replace(config.demand, seed=seed))
        self._experiment_config = seeded_config
        self._environment, self._tasks, self._robots, self._simulation_config = build_experiment_inputs(seeded_config)
        self._robot_states = ()
        self._occupancy = None
        self._released_task_ids: set[str] = set()
        self._claimed_task_ids: set[str] = set()
        self._completed_task_ids: set[str] = set()
        self._active_plans = {}
        self._current_time = 0.0
        self._next_replan_time = 0.0
        self._macro_decisions: list[MacroDecisionRecord] = []
        self._planner_plans: list[PlannerPlanRecord] = []
        self._collision_events: list[CollisionEventRecord] = []
        self._queue_snapshots = []
        self._executions = []
        self._charging_executions = []

    def reset(self) -> IntegratedObservation:
        self._environment, self._tasks, self._robots, self._simulation_config = build_experiment_inputs(self._experiment_config)
        from warehouse_sim.agents import RobotState

        self._robot_states = tuple(RobotState.from_spec(robot) for robot in self._robots)
        self._occupancy = _build_occupancy_table(self._simulation_config)
        self._released_task_ids = set()
        self._claimed_task_ids = set()
        self._completed_task_ids = set()
        self._active_plans = {}
        self._current_time = 0.0
        self._next_replan_time = 0.0
        self._macro_decisions = []
        self._planner_plans = []
        self._collision_events = []
        self._queue_snapshots = []
        self._executions = []
        self._charging_executions = []
        _record_queue_snapshot(self._queue_snapshots, 0.0, self._tasks, self._released_task_ids, self._completed_task_ids, self._active_plans)
        _release_ready_tasks(self._tasks, 0.0, self._released_task_ids)
        return self._observation()

    def step(self, chosen_indices: tuple[int, ...], reward_config) -> tuple[IntegratedObservation | None, float, bool, dict[str, float]]:
        assert self._occupancy is not None
        before_completed = len(self._completed_task_ids)
        before_wait = sum(execution.waiting_time for execution in self._executions)
        before_delay = sum(execution.congestion_delay_time for execution in self._executions)
        before_safety = len(self._collision_events)
        decision_index = len(self._macro_decisions)
        plan_index = len(self._planner_plans)
        observation = self._observation()
        pre_resolution_conflict_count = _count_pre_resolution_conflicts(
            environment=self._environment,
            observation=observation,
            output=IntegratedPolicyOutput(chosen_indices=chosen_indices),
            robot_states=self._robot_states,
            occupancy=self._occupancy,
            current_time=self._current_time,
            config=self._simulation_config,
        )
        used_tasks: set[str] = set()
        for robot_index, robot in enumerate(self._robot_states):
            candidates = observation.macro_candidates[robot_index]
            chosen_index = chosen_indices[robot_index] if robot_index < len(chosen_indices) else 0
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
            self._macro_decisions.append(
                MacroDecisionRecord(
                    decision_index=decision_index,
                    decision_time=self._current_time,
                    robot_id=robot.spec.robot_id,
                    macro_type=candidate.macro_type,
                    task_id=candidate.task_id,
                    charging_node=candidate.charging_node,
                    route_nodes=candidate.route_nodes,
                    route_edges=tuple(f"{source}->{target}" for source, target in candidate.route_edges),
                    estimated_completion_time=candidate.estimated_completion_time,
                    selected_by_policy="trained_end_to_end_macro_ppo",
                    candidate_count=len(candidates),
                    selected_rank_by_estimated_completion=selection_rank,
                    best_candidate_estimated_completion=best_candidate_estimated_completion,
                    selected_completion_gap=selected_completion_gap,
                )
            )
            decision_index += 1
            if candidate.macro_type not in {"task_route", "charge_route"}:
                continue
            if robot.spec.robot_id in self._active_plans:
                continue
            task = None if candidate.task_id is None else next(task for task in self._tasks if task.task_id == candidate.task_id)
            service_time = candidate.service_time_estimate
            if candidate.macro_type == "charge_route":
                if self._simulation_config.battery is None or candidate.charging_node is None:
                    continue
            planned = plan_motion_candidate(
                self._environment,
                robot_id=robot.spec.robot_id,
                start_time=self._current_time,
                speed_multiplier=robot.spec.speed_multiplier,
                occupancy_table=self._occupancy,
                candidate=candidate,
                service_time=service_time,
                motion_model=self._simulation_config.coordination.motion_model,  # type: ignore[union-attr]
            )
            if planned is None:
                self._planner_plans.append(
                    PlannerPlanRecord(
                        plan_index=plan_index,
                        plan_time=self._current_time,
                        robot_id=robot.spec.robot_id,
                        task_id=None if task is None else task.task_id,
                        priority_rank=robot_index,
                        path_nodes=candidate.route_nodes,
                        path_edges=tuple(f"{source}->{target}" for source, target in candidate.route_edges),
                        planned_start_time=self._current_time,
                        planned_end_time=self._current_time,
                        planner_name="trained_end_to_end_macro_ppo",
                        status="failed",
                        pre_resolution_conflict_count=pre_resolution_conflict_count,
                    )
                )
                plan_index += 1
                continue
            self._occupancy.reserve(planned.traversals)
            for node_id, time in planned.reserved_node_times:
                self._occupancy.reserve_node_time(node_id=node_id, time=time, robot_id=robot.spec.robot_id)
            if task is not None:
                self._claimed_task_ids.add(task.task_id)
            self._active_plans[robot.spec.robot_id] = _ActivePlan(
                action_type=candidate.macro_type,
                task=task,
                assigned_at=self._current_time,
                pickup_arrival_time=planned.pickup_arrival_time,
                completion_time=planned.completion_time,
                traversals=planned.traversals,
                blocked_events=planned.blocked_events,
                wait_time=planned.wait_time,
                charging_node_id=candidate.charging_node,
                energy_before=robot.battery_level,
            )
            robot.available_time = planned.completion_time
            self._planner_plans.append(
                PlannerPlanRecord(
                    plan_index=plan_index,
                    plan_time=self._current_time,
                    robot_id=robot.spec.robot_id,
                    task_id=None if task is None else task.task_id,
                    priority_rank=robot_index,
                    path_nodes=planned.route_nodes,
                    path_edges=tuple(f"{traversal.source_id}->{traversal.target_id}" for traversal in planned.traversals),
                    planned_start_time=planned.traversals[0].start_time if planned.traversals else self._current_time,
                    planned_end_time=planned.completion_time,
                    planner_name="trained_end_to_end_macro_ppo",
                    status="planned",
                    pre_resolution_conflict_count=pre_resolution_conflict_count,
                    wait_insertion_count=planned.blocked_events,
                    wait_insertion_time=planned.wait_time,
                )
            )
            plan_index += 1
            if task is not None:
                used_tasks.add(task.task_id)
        for event in _detect_motion_collisions(
            traversals=tuple(traversal for plan in self._active_plans.values() for traversal in plan.traversals),
            config=self._simulation_config,
        ):
            self._collision_events.append(
                CollisionEventRecord(
                    time=event[0],
                    robot_id=event[1],
                    other_robot_id=event[2],
                    event_type=event[3],
                    location_id=event[4],
                )
            )
        self._next_replan_time = self._current_time + self._simulation_config.coordination.replan_period  # type: ignore[union-attr]

        next_time = _next_integrated_event_time(
            current_time=self._current_time,
            tasks=self._tasks,
            released_task_ids=self._released_task_ids,
            completed_task_ids=self._completed_task_ids,
            robot_states=self._robot_states,
            next_replan_time=self._next_replan_time,
            config=self._simulation_config,
        )
        if next_time is None:
            done = True
            self._finalize_all()
        else:
            self._current_time = next_time
            _release_ready_tasks(self._tasks, self._current_time, self._released_task_ids)
            _finalize_completed_plans(
                current_time=self._current_time,
                robot_states=self._robot_states,
                active_plans=self._active_plans,
                executions=self._executions,
                charging_executions=self._charging_executions,
                completed_task_ids=self._completed_task_ids,
                environment=self._environment,
                battery_config=self._simulation_config.battery,
            )
            _record_queue_snapshot(
                self._queue_snapshots,
                self._current_time,
                self._tasks,
                self._released_task_ids,
                self._completed_task_ids,
                self._active_plans,
            )
            done = False

        completed_delta = len(self._completed_task_ids) - before_completed
        wait_delta = sum(execution.waiting_time for execution in self._executions) - before_wait
        delay_delta = sum(execution.congestion_delay_time for execution in self._executions) - before_delay
        safety_delta = len(self._collision_events) - before_safety
        reward = (
            reward_config.task_completion * completed_delta
            + reward_config.waiting_time * wait_delta
            + reward_config.congestion_delay * delay_delta
            + reward_config.safety_violation * safety_delta
        )
        return (None if done else self._observation()), float(reward), done, {
            "completed_delta": float(completed_delta),
            "wait_delta": float(wait_delta),
            "delay_delta": float(delay_delta),
            "safety_delta": float(safety_delta),
        }

    def _observation(self) -> IntegratedObservation:
        assert self._occupancy is not None
        return build_integrated_observation(
            environment=self._environment,
            robot_states=self._robot_states,
            tasks=self._tasks,
            released_task_ids=self._released_task_ids,
            claimed_task_ids=self._claimed_task_ids,
            completed_task_ids=self._completed_task_ids,
            active_plans=self._active_plans,
            occupancy=self._occupancy,
            current_time=self._current_time,
            config=self._simulation_config,
        )

    def _finalize_all(self) -> None:
        finished_at = max([self._current_time, *(robot.available_time for robot in self._robot_states)], default=self._current_time)
        _finalize_completed_plans(
            current_time=finished_at,
            robot_states=self._robot_states,
            active_plans=self._active_plans,
            executions=self._executions,
            charging_executions=self._charging_executions,
            completed_task_ids=self._completed_task_ids,
            environment=self._environment,
            battery_config=self._simulation_config.battery,
        )
        self._current_time = finished_at

    def build_result(self) -> SimulationResult:
        self._finalize_all()
        result = SimulationResult(
            policy_name="trained_end_to_end_macro_ppo",
            started_at=0.0,
            finished_at=self._current_time,
            tasks_generated=len(self._tasks),
            robot_states=self._robot_states,
            executions=tuple(self._executions),
            dispatch_traces=(),
            dispatch_node_observations=(),
            dispatch_arc_observations=(),
            unassigned_tasks=tuple(task for task in self._tasks if task.task_id not in self._completed_task_ids),
            queue_snapshots=tuple(self._queue_snapshots),
            metrics=None,  # type: ignore[arg-type]
            charging_executions=tuple(self._charging_executions),
            robot_trajectories=(),
            macro_decisions=tuple(self._macro_decisions),
            collision_events=tuple(self._collision_events),
            planner_plans=tuple(self._planner_plans),
        )
        metrics = compute_simulation_metrics(result)
        return replace(result, metrics=metrics)


def run_integrated_rl_training_from_config(config: IntegratedRLTrainingConfig) -> dict[str, Path]:
    """Train a PPO macro controller from scratch on integrated scenarios."""

    scenarios = [load_experiment_config(path) for path in config.curriculum.scenario_configs]
    env = IntegratedCoordinationRLEnv(scenarios[0], config.curriculum.train_seeds[0])
    sample_observation = env.reset()
    model = EndToEndMacroPolicyNetwork(
        node_dim=len(sample_observation.node_features[0]),
        edge_dim=len(sample_observation.edge_features[0]) if sample_observation.edge_features else 3,
        robot_dim=len(sample_observation.robot_features[0]),
        task_dim=len(sample_observation.task_features[0]) if sample_observation.task_features else 5,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.ppo.learning_rate)
    warm_start_rows = _warm_start_model(model, scenarios, config)
    warm_start_validation_rows, warm_start_gate_payload = _evaluate_integrated_macro_model(model, scenarios, config)
    selected_state_dict = deepcopy(model.state_dict())
    selected_validation_rows = warm_start_validation_rows
    selected_gate_payload = warm_start_gate_payload
    selected_stage = "warm_start" if config.warm_start.epochs > 0 else "initial"
    transitions: list[IntegratedPolicyStep] = []
    training_rows: list[dict[str, object]] = []
    scenario_index = 0
    for episode in range(config.ppo.total_episodes):
        scenario = scenarios[scenario_index % len(scenarios)]
        seed = config.curriculum.train_seeds[episode % len(config.curriculum.train_seeds)]
        scenario_index += 1
        env = IntegratedCoordinationRLEnv(scenario, seed)
        observation = env.reset()
        done = False
        while not done:
            output = model.act(observation, greedy=False)
            next_observation, reward, done, info = env.step(output.chosen_indices, config.reward)
            transitions.append(
                IntegratedPolicyStep(
                    observation=observation,
                    chosen_indices=output.chosen_indices,
                    old_log_prob=output.log_prob,
                    reward=reward,
                    value=output.value or 0.0,
                    done=done,
                )
            )
            observation = next_observation or observation
        result = env.build_result()
        training_rows.append(
            {
                "episode": episode,
                "scenario_name": scenario.name,
                "seed": seed,
                "tasks_completed": result.metrics.tasks_completed,
                "throughput_per_hour": result.metrics.throughput_per_hour,
                "safety_violations_total": result.metrics.safety_violations_total,
            }
        )
        _ppo_update(model, optimizer, transitions, config)
        transitions.clear()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_validation_rows, final_gate_payload = _evaluate_integrated_macro_model(model, scenarios, config)
    if _is_better_gate_payload(final_gate_payload, selected_gate_payload):
        selected_state_dict = deepcopy(model.state_dict())
        selected_validation_rows = final_validation_rows
        selected_gate_payload = final_gate_payload
        selected_stage = "ppo_final"
    model.load_state_dict(selected_state_dict)
    state_path = output_dir / "end_to_end_macro_ppo.pt"
    torch.save(model.state_dict(), state_path)
    artifact = EndToEndMacroArtifact(
        artifact_version=1,
        model_type="end_to_end_macro_ppo",
        parameters={
            "node_dim": len(sample_observation.node_features[0]),
            "edge_dim": len(sample_observation.edge_features[0]) if sample_observation.edge_features else 3,
            "robot_dim": len(sample_observation.robot_features[0]),
            "task_dim": len(sample_observation.task_features[0]) if sample_observation.task_features else 5,
            "macro_dim": 7,
            "hidden_dim": 64,
            "state_dict_path": state_path.name,
        },
        metadata={
            "benchmark_gate": selected_gate_payload,
            "selected_checkpoint_stage": selected_stage,
            "candidate_gate_evaluations": {
                "warm_start": warm_start_gate_payload,
                "ppo_final": final_gate_payload,
            },
            "warm_start": {
                "epochs": config.warm_start.epochs,
                "teacher_policy": config.warm_start.teacher_policy,
            },
        },
    )
    artifact_path = write_end_to_end_macro_artifact(artifact, output_dir / "model_artifact.json")
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, checkpoint_path)
    training_metrics_path = output_dir / "training_metrics.csv"
    evaluation_path = output_dir / "evaluation_rollouts.json"
    gate_path = output_dir / "claim_gate.json"
    warm_start_metrics_path = output_dir / "warm_start_metrics.csv"
    _write_csv(training_metrics_path, training_rows)
    _write_csv(warm_start_metrics_path, warm_start_rows)
    evaluation_path.write_text(json.dumps(selected_validation_rows, indent=2), encoding="utf-8")
    gate_path.write_text(json.dumps(selected_gate_payload, indent=2), encoding="utf-8")
    return {
        "artifact": artifact_path,
        "checkpoint": checkpoint_path,
        "training_metrics": training_metrics_path,
        "warm_start_metrics": warm_start_metrics_path,
        "evaluation_rollouts": evaluation_path,
        "claim_gate": gate_path,
    }


def _warm_start_model(
    model: EndToEndMacroPolicyNetwork,
    scenarios: list[ExperimentConfig],
    config: IntegratedRLTrainingConfig,
) -> list[dict[str, object]]:
    if config.warm_start.epochs <= 0:
        return []
    teacher = _teacher_policy(config.warm_start.teacher_policy)
    optimizer = optim.Adam(model.parameters(), lr=config.warm_start.learning_rate)
    rows: list[dict[str, object]] = []
    for epoch in range(config.warm_start.epochs):
        epoch_losses: list[float] = []
        matched = 0
        total = 0
        for scenario in scenarios:
            for seed in config.curriculum.train_seeds:
                env = IntegratedCoordinationRLEnv(scenario, seed)
                observation = env.reset()
                done = False
                while not done:
                    teacher_output = teacher.select_macros(observation)
                    loss = _behavior_cloning_loss(model, observation, teacher_output.chosen_indices)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.item()))
                    greedy_indices = model.act(observation, greedy=True).chosen_indices
                    matched += sum(
                        int(predicted == target)
                        for predicted, target in zip(greedy_indices, teacher_output.chosen_indices, strict=True)
                    )
                    total += len(teacher_output.chosen_indices)
                    next_observation, _reward, done, _info = env.step(teacher_output.chosen_indices, config.reward)
                    observation = next_observation or observation
        rows.append(
            {
                "warm_start_epoch": epoch,
                "teacher_policy": config.warm_start.teacher_policy,
                "mean_bc_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0,
                "teacher_action_match_rate": matched / total if total else 0.0,
            }
        )
    return rows


def _ppo_update(
    model: EndToEndMacroPolicyNetwork,
    optimizer: optim.Optimizer,
    transitions: list[IntegratedPolicyStep],
    config: IntegratedRLTrainingConfig,
) -> None:
    if not transitions:
        return
    returns = []
    advantages = []
    running_return = 0.0
    running_advantage = 0.0
    next_value = 0.0
    for transition in reversed(transitions):
        running_return = transition.reward + config.ppo.gamma * running_return * (0.0 if transition.done else 1.0)
        delta = transition.reward + config.ppo.gamma * next_value * (0.0 if transition.done else 1.0) - transition.value
        running_advantage = delta + config.ppo.gamma * config.ppo.gae_lambda * running_advantage * (0.0 if transition.done else 1.0)
        returns.append(running_return)
        advantages.append(running_advantage)
        next_value = transition.value
    returns.reverse()
    advantages.reverse()
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    old_log_probs = torch.tensor([transition.old_log_prob for transition in transitions], dtype=torch.float32)

    for _ in range(config.ppo.ppo_epochs):
        losses = []
        for index, transition in enumerate(transitions):
            log_prob, value, entropy = model.evaluate(transition.observation, transition.chosen_indices)
            ratio = torch.exp(log_prob - old_log_probs[index])
            unclipped = ratio * advantages_tensor[index]
            clipped = torch.clamp(ratio, 1.0 - config.ppo.clip_epsilon, 1.0 + config.ppo.clip_epsilon) * advantages_tensor[index]
            actor_loss = -torch.minimum(unclipped, clipped)
            value_loss = torch.nn.functional.mse_loss(value, returns_tensor[index])
            loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
            losses.append(loss)
        optimizer.zero_grad()
        torch.stack(losses).mean().backward()
        optimizer.step()


def _evaluate_integrated_macro_model(
    model: EndToEndMacroPolicyNetwork,
    scenarios: list[ExperimentConfig],
    config: IntegratedRLTrainingConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    validation_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for seed in (config.curriculum.validation_seeds or config.curriculum.train_seeds[:1]):
            env = IntegratedCoordinationRLEnv(scenario, seed)
            observation = env.reset()
            done = False
            while not done:
                output = model.act(observation, greedy=True)
                observation, _reward, done, _info = env.step(output.chosen_indices, config.reward)
                if observation is None:
                    break
            result = env.build_result()
            validation_rows.append(
                {
                    "scenario_name": scenario.name,
                    "seed": seed,
                    "policy": "trained_end_to_end_macro_ppo",
                    "tasks_completed": result.metrics.tasks_completed,
                    "tasks_generated": result.metrics.tasks_generated,
                    "throughput_per_hour": result.metrics.throughput_per_hour,
                    "safety_violations_total": result.metrics.safety_violations_total,
                }
            )
            baseline_config = replace(
                scenario,
                simulation=replace(scenario.simulation, coordination_mode="integrated", policy="prioritized_sipp_coordinator", execution_model="idealized"),
            )
            baseline_result, _ = run_experiment_from_config(
                baseline_config,
                output_dir_override=config.output_dir / "baseline_eval" / scenario.name / f"seed_{seed}",
                force_write_plots=False,
                force_write_observation_dataset=False,
            )
            baseline_rows.append(
                {
                    "scenario_name": scenario.name,
                    "seed": seed,
                    "throughput_per_hour": baseline_result.metrics.throughput_per_hour,
                }
            )
    throughput_ratio = (
        sum(row["throughput_per_hour"] for row in validation_rows) / max(sum(row["throughput_per_hour"] for row in baseline_rows), 1e-6)
    )
    task_completion_rate = (
        sum(row["tasks_completed"] for row in validation_rows) / max(sum(row["tasks_generated"] for row in validation_rows), 1)
    )
    safety_violations = sum(row["safety_violations_total"] for row in validation_rows)
    gate_payload = {
        "max_safety_violations": config.benchmark_gate.max_safety_violations,
        "min_task_completion_rate": config.benchmark_gate.min_task_completion_rate,
        "min_throughput_ratio_vs_baseline": config.benchmark_gate.min_throughput_ratio_vs_baseline,
        "observed_safety_violations": safety_violations,
        "observed_task_completion_rate": task_completion_rate,
        "observed_throughput_ratio_vs_baseline": throughput_ratio,
        "claim_ready": (
            safety_violations <= config.benchmark_gate.max_safety_violations
            and task_completion_rate >= config.benchmark_gate.min_task_completion_rate
            and throughput_ratio >= config.benchmark_gate.min_throughput_ratio_vs_baseline
        ),
    }
    return validation_rows, gate_payload


def _teacher_policy(name: str):
    if name == "prioritized_sipp_coordinator":
        return PrioritizedSIPPCoordinatorPolicy()
    raise ValueError(f"Unsupported warm-start teacher policy: {name}")


def _behavior_cloning_loss(
    model: EndToEndMacroPolicyNetwork,
    observation: IntegratedObservation,
    chosen_indices: tuple[int, ...],
) -> torch.Tensor:
    graph_embedding = model.encode_graph(observation)
    used_tasks: set[str] = set()
    losses: list[torch.Tensor] = []
    for robot_index, candidates in enumerate(observation.macro_candidates):
        robot_embedding = model.robot_encoder(
            torch.tensor(observation.robot_features[robot_index], dtype=torch.float32)
        )
        candidate_matrix = torch.stack(
            [model.macro_encoder(_macro_feature_tensor(observation, candidate)) for candidate in candidates]
        )
        repeated_graph = graph_embedding.unsqueeze(0).expand(candidate_matrix.shape[0], -1)
        repeated_robot = robot_embedding.unsqueeze(0).expand(candidate_matrix.shape[0], -1)
        logits = model.candidate_head(torch.cat([repeated_graph, repeated_robot, candidate_matrix], dim=-1)).squeeze(-1)
        mask = torch.tensor(
            [candidate.task_id is None or candidate.task_id not in used_tasks for candidate in candidates],
            dtype=torch.bool,
        )
        masked_logits = logits.masked_fill(~mask, float("-inf"))
        target_index = int(chosen_indices[robot_index])
        losses.append(F.cross_entropy(masked_logits.unsqueeze(0), torch.tensor([target_index])))
        task_id = candidates[target_index].task_id
        if task_id is not None:
            used_tasks.add(task_id)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def _is_better_gate_payload(candidate: dict[str, object], incumbent: dict[str, object]) -> bool:
    return _gate_selection_key(candidate) > _gate_selection_key(incumbent)


def _gate_selection_key(payload: dict[str, object]) -> tuple[float, float, float, float]:
    return (
        1.0 if bool(payload.get("claim_ready", False)) else 0.0,
        -float(payload.get("observed_safety_violations", 0.0)),
        float(payload.get("observed_task_completion_rate", 0.0)),
        float(payload.get("observed_throughput_ratio_vs_baseline", 0.0)),
    )


def _macro_feature_tensor(
    observation: IntegratedObservation,
    candidate,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    route_time = 0.0
    if candidate.route_nodes:
        route_time = max(candidate.estimated_completion_time - observation.current_time, 0.0)
    macro_type = candidate.macro_type
    return torch.tensor(
        [
            1.0 if macro_type == "continue_current_plan" else 0.0,
            1.0 if macro_type == "wait" else 0.0,
            1.0 if macro_type == "task_route" else 0.0,
            route_time,
            float(len(candidate.route_nodes)),
            float(len(candidate.route_edges)),
            1.0 if candidate.task_id is not None else 0.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
