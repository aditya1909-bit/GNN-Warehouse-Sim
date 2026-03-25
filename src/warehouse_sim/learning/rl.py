"""Dispatch-event RL environment and masked PPO fine-tuning."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import optim

from warehouse_sim.agents import RobotSpec, RobotState
from warehouse_sim.config import (
    ExperimentConfig,
    RLFineTuningConfig,
    load_experiment_config,
)
from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.learning.artifacts import load_dispatch_model_artifact, write_dispatch_model_artifact
from warehouse_sim.learning.graph_data import (
    DEFAULT_GRAPH_CANDIDATE_FEATURES,
    DEFAULT_GRAPH_EDGE_FEATURES,
    DEFAULT_GRAPH_NODE_FEATURES,
    GraphDispatchExample,
    build_graph_dispatch_example_from_context,
)
from warehouse_sim.learning.graph_model import GraphDispatchActorCritic, load_graph_dispatch_model
from warehouse_sim.metrics.collector import compute_simulation_metrics
from warehouse_sim.policies.base import DispatchDecision
from warehouse_sim.policies.observation import DispatchContextBuilder
from warehouse_sim.policies.scoring import build_candidate_assignment_observations
from warehouse_sim.simulation.engine import (
    _assign_task,
    _build_dispatch_context,
    _next_event_time,
    _record_snapshot,
    _robot_by_id,
    _task_by_id,
)
from warehouse_sim.simulation.execution import ResourceReservationTable
from warehouse_sim.simulation.models import SimulationResult
from warehouse_sim.simulation.runner import build_experiment_inputs
from warehouse_sim.tasks import Task, TaskQueue


@dataclass(frozen=True)
class DispatchTransition:
    example: GraphDispatchExample
    action_index: int
    old_log_prob: float
    reward: float
    value: float
    done: bool


class DispatchEventRLEnv(gym.Env):
    """Gym-style environment that steps at dispatch events."""

    def __init__(
        self,
        *,
        environment: WarehouseEnvironment,
        tasks: tuple[Task, ...],
        robots: tuple[RobotSpec, ...],
        simulation_config,
        reward_weights,
        artifact,
    ) -> None:
        super().__init__()
        self._environment = environment
        self._tasks = tasks
        self._robots = robots
        self._simulation_config = simulation_config
        self._reward_weights = reward_weights
        self._artifact = artifact
        self._candidate_feature_names = tuple(artifact.parameters["candidate_feature_names"])
        self._node_feature_names = tuple(artifact.parameters["node_feature_names"])
        self._edge_feature_names = tuple(artifact.parameters["edge_feature_names"])
        self._context_builder = DispatchContextBuilder(environment)
        self._reservation_table = ResourceReservationTable(simulation_config.execution_model)
        self._queue: TaskQueue | None = None
        self._robot_states: tuple[RobotState, ...] | None = None
        self._current_time = 0.0
        self._executions = []
        self._dispatch_traces = []
        self._dispatch_node_observations = []
        self._dispatch_arc_observations = []
        self._snapshots = []
        self._context = None
        self._current_example = None
        self._current_candidates = None

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._queue = TaskQueue(self._tasks)
        self._robot_states = tuple(RobotState.from_spec(robot) for robot in self._robots)
        self._reservation_table = ResourceReservationTable(self._simulation_config.execution_model)
        self._current_time = 0.0
        self._executions = []
        self._dispatch_traces = []
        self._dispatch_node_observations = []
        self._dispatch_arc_observations = []
        self._snapshots = []
        _record_snapshot(self._current_time, self._queue, self._robot_states, self._executions, self._snapshots)
        done = self._advance_to_next_dispatch()
        assert self._current_example is not None or done
        observation = None if done else self._current_example
        info = {"action_mask": np.ones(observation.candidate_count, dtype=bool) if observation else np.zeros(0, dtype=bool)}
        return observation, info

    def step(self, action: int):
        if self._context is None or self._current_example is None or self._current_candidates is None:
            raise RuntimeError("Environment is not ready for a dispatch action.")
        if action < 0 or action >= len(self._current_candidates):
            raise ValueError(f"Invalid action index: {action}")

        decision_candidate = self._current_candidates[action]
        decision = DispatchDecision(
            robot_id=decision_candidate.robot_id,
            task_id=decision_candidate.task_id,
        )
        self._dispatch_traces.extend(
            _build_dispatch_trace_records(self._context, decision, len(self._executions))
        )
        self._dispatch_node_observations.extend(
            build_dispatch_graph_dispatch_node_records(self._context, len(self._executions), decision)
        )
        self._dispatch_arc_observations.extend(
            build_dispatch_graph_dispatch_arc_records(self._context, len(self._executions))
        )
        robot = _robot_by_id(self._robot_states, decision.robot_id)
        task = _task_by_id(self._context.ready_tasks, decision.task_id)
        self._queue.remove_task(task.task_id)
        execution = _assign_task(
            current_time=self._current_time,
            environment=self._environment,
            robot=robot,
            task=task,
            execution_model=self._simulation_config.execution_model,
            reservation_table=self._reservation_table,
        )
        self._executions.append(execution)
        reward = (
            self._reward_weights.task_completion * 1.0
            + self._reward_weights.waiting_time * execution.waiting_time
            + self._reward_weights.congestion_delay * execution.congestion_delay_time
            + self._reward_weights.blocked_events * execution.blocked_traversal_events
        )
        done = self._advance_to_next_dispatch()
        observation = None if done else self._current_example
        info = {
            "action_mask": np.ones(observation.candidate_count, dtype=bool) if observation else np.zeros(0, dtype=bool),
            "reward_components": {
                "task_completion": 1.0,
                "waiting_time": execution.waiting_time,
                "congestion_delay": execution.congestion_delay_time,
                "blocked_events": execution.blocked_traversal_events,
            },
        }
        return observation, reward, done, False, info

    def _advance_to_next_dispatch(self) -> bool:
        assert self._queue is not None
        assert self._robot_states is not None
        while True:
            context = _build_dispatch_context(
                context_builder=self._context_builder,
                queue=self._queue,
                current_time=self._current_time,
                robot_states=self._robot_states,
                config=self._simulation_config,
                reservation_table=self._reservation_table,
            )
            if context.ready_tasks and context.idle_robots:
                self._context = context
                self._current_candidates = build_candidate_assignment_observations(context)
                self._current_example = build_graph_dispatch_example_from_context(
                    context,
                    dispatch_index=len(self._executions),
                    candidate_feature_names=self._candidate_feature_names,
                    node_feature_names=self._node_feature_names,
                    edge_feature_names=self._edge_feature_names,
                    dispatch_group_id=f"episode::dispatch_{len(self._executions)}",
                )
                return False
            _record_snapshot(self._current_time, self._queue, self._robot_states, self._executions, self._snapshots)
            next_time = _next_event_time(
                current_time=self._current_time,
                queue=self._queue,
                robot_states=self._robot_states,
                config=self._simulation_config,
            )
            if next_time is None:
                self._context = None
                self._current_example = None
                self._current_candidates = None
                return True
            self._current_time = next_time

    def build_result(self) -> SimulationResult:
        assert self._robot_states is not None
        assert self._queue is not None
        finished_at = max([self._current_time, *(robot.available_time for robot in self._robot_states)], default=self._current_time)
        result = SimulationResult(
            policy_name="trained_graph_dispatch_model",
            started_at=0.0,
            finished_at=finished_at,
            tasks_generated=len(self._tasks),
            robot_states=self._robot_states,
            executions=tuple(self._executions),
            dispatch_traces=tuple(self._dispatch_traces),
            dispatch_node_observations=tuple(self._dispatch_node_observations),
            dispatch_arc_observations=tuple(self._dispatch_arc_observations),
            unassigned_tasks=tuple(self._queue.pending_tasks()),
            queue_snapshots=tuple(self._snapshots),
            metrics=None,  # type: ignore[arg-type]
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
        )


def run_rl_fine_tuning_from_config(config: RLFineTuningConfig) -> dict[str, Path]:
    """Run masked PPO fine-tuning over the configured scenario curriculum."""

    loaded = load_graph_dispatch_model(config.pretrained_artifact_path)
    actor_critic = GraphDispatchActorCritic(
        scorer=loaded.model,
        hidden_dim=int(loaded.artifact.parameters["hidden_dim"]),
    )
    optimizer = optim.Adam(actor_critic.parameters(), lr=config.ppo.learning_rate)
    train_configs = [load_experiment_config(path) for path in config.curriculum.scenario_configs]

    training_rows: list[dict[str, object]] = []
    episode_counter = 0
    for _ in range(config.ppo.total_episodes):
        trajectories: list[DispatchTransition] = []
        for scenario_config in train_configs:
            seed = config.curriculum.train_seeds[episode_counter % len(config.curriculum.train_seeds)]
            env = _env_from_experiment_config(scenario_config, seed, config, loaded.artifact)
            observation, _ = env.reset()
            done = observation is None
            while not done:
                logits, graph_embedding = actor_critic.forward_actor(
                    torch.tensor(observation.node_features, dtype=torch.float32),
                    torch.tensor(observation.edge_index, dtype=torch.long),
                    torch.tensor(observation.edge_features, dtype=torch.float32),
                    torch.tensor(observation.candidate_features, dtype=torch.float32),
                )
                distribution = torch.distributions.Categorical(logits=logits)
                action = int(distribution.sample().item())
                log_prob = float(distribution.log_prob(torch.tensor(action)).item())
                value = float(actor_critic.forward_value(graph_embedding).item())
                next_observation, reward, done, _, _ = env.step(action)
                trajectories.append(
                    DispatchTransition(
                        example=observation,
                        action_index=action,
                        old_log_prob=log_prob,
                        reward=reward,
                        value=value,
                        done=done,
                    )
                )
                observation = next_observation
            episode_result = env.build_result()
            training_rows.append(
                {
                    "episode": episode_counter,
                    "scenario_name": scenario_config.name,
                    "seed": seed,
                    "tasks_completed": episode_result.metrics.tasks_completed,
                    "average_waiting_time": episode_result.metrics.average_waiting_time,
                    "congestion_delay_total": episode_result.metrics.congestion_delay_total,
                }
            )
            episode_counter += 1
            if len(trajectories) >= config.ppo.rollout_horizon:
                break
        if trajectories:
            _ppo_update(actor_critic, optimizer, trajectories, config)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "actor_critic.pt"
    torch.save(actor_critic.state_dict(), checkpoint_path)
    updated_state_path = output_dir / "graph_dispatch_model.pt"
    torch.save(actor_critic.scorer.state_dict(), updated_state_path)
    artifact = loaded.artifact
    updated_artifact = artifact.__class__(
        artifact_version=artifact.artifact_version,
        model_type=artifact.model_type,
        objective=artifact.objective,
        feature_names=artifact.feature_names,
        parameters={**artifact.parameters, "state_dict_path": updated_state_path.name},
        metadata={
            **artifact.metadata,
            "rl_fine_tuning": {
                "episodes": episode_counter,
                "checkpoint_path": str(checkpoint_path.name),
            },
        },
    )
    artifact_path = write_dispatch_model_artifact(updated_artifact, output_dir / "model_artifact.json")
    metrics_path = output_dir / "training_metrics.csv"
    _write_csv(metrics_path, training_rows)
    evaluation_path = output_dir / "evaluation_rollouts.json"
    evaluation_payload = _evaluate_actor_critic(actor_critic, train_configs, config)
    evaluation_path.write_text(json.dumps(evaluation_payload, indent=2), encoding="utf-8")
    return {
        "artifact": artifact_path,
        "state_dict": updated_state_path,
        "checkpoint": checkpoint_path,
        "training_metrics": metrics_path,
        "evaluation_rollouts": evaluation_path,
    }


def _env_from_experiment_config(
    config: ExperimentConfig,
    seed: int,
    rl_config: RLFineTuningConfig,
    artifact,
) -> DispatchEventRLEnv:
    from dataclasses import replace

    seeded_config = replace(config, demand=replace(config.demand, seed=seed))
    environment, tasks, robots, simulation_config = build_experiment_inputs(seeded_config)
    return DispatchEventRLEnv(
        environment=environment,
        tasks=tasks,
        robots=robots,
        simulation_config=simulation_config,
        reward_weights=rl_config.reward,
        artifact=artifact,
    )


def _ppo_update(
    actor_critic: GraphDispatchActorCritic,
    optimizer: optim.Optimizer,
    trajectories: list[DispatchTransition],
    config: RLFineTuningConfig,
) -> None:
    returns = []
    advantages = []
    running_return = 0.0
    running_advantage = 0.0
    next_value = 0.0
    for transition in reversed(trajectories):
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

    for _ in range(config.ppo.ppo_epochs):
        losses = []
        for index, transition in enumerate(trajectories):
            logits, graph_embedding = actor_critic.forward_actor(
                torch.tensor(transition.example.node_features, dtype=torch.float32),
                torch.tensor(transition.example.edge_index, dtype=torch.long),
                torch.tensor(transition.example.edge_features, dtype=torch.float32),
                torch.tensor(transition.example.candidate_features, dtype=torch.float32),
            )
            distribution = torch.distributions.Categorical(logits=logits)
            log_prob = distribution.log_prob(torch.tensor(transition.action_index))
            ratio = torch.exp(log_prob - torch.tensor(transition.old_log_prob))
            unclipped = ratio * advantages_tensor[index]
            clipped = torch.clamp(ratio, 1.0 - config.ppo.clip_epsilon, 1.0 + config.ppo.clip_epsilon) * advantages_tensor[index]
            actor_loss = -torch.minimum(unclipped, clipped)
            value = actor_critic.forward_value(graph_embedding)
            value_loss = torch.nn.functional.mse_loss(value, torch.tensor(returns[index], dtype=torch.float32))
            loss = actor_loss + 0.5 * value_loss
            losses.append(loss)
        optimizer.zero_grad()
        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimizer.step()


def _evaluate_actor_critic(actor_critic, train_configs, config):
    payload = {"validation_rollouts": []}
    validation_seeds = config.curriculum.validation_seeds or config.curriculum.train_seeds[:1]
    for scenario_config in train_configs:
        for seed in validation_seeds:
            env = _env_from_experiment_config(scenario_config, seed, config, load_dispatch_model_artifact(config.pretrained_artifact_path))
            observation, _ = env.reset()
            done = observation is None
            while not done:
                with torch.no_grad():
                    logits, _ = actor_critic.forward_actor(
                        torch.tensor(observation.node_features, dtype=torch.float32),
                        torch.tensor(observation.edge_index, dtype=torch.long),
                        torch.tensor(observation.edge_features, dtype=torch.float32),
                        torch.tensor(observation.candidate_features, dtype=torch.float32),
                    )
                action = int(torch.argmax(logits).item())
                observation, _, done, _, _ = env.step(action)
            result = env.build_result()
            payload["validation_rollouts"].append(
                {
                    "scenario_name": scenario_config.name,
                    "seed": seed,
                    "tasks_completed": result.metrics.tasks_completed,
                    "average_waiting_time": result.metrics.average_waiting_time,
                    "congestion_delay_total": result.metrics.congestion_delay_total,
                }
            )
    return payload


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_dispatch_graph_dispatch_node_records(context, dispatch_index, decision):
    from warehouse_sim.learning.graph_data import build_dispatch_node_observation_records

    return build_dispatch_node_observation_records(
        context=context,
        dispatch_index=dispatch_index,
        decision=decision,
    )


def build_dispatch_graph_dispatch_arc_records(context, dispatch_index):
    from warehouse_sim.learning.graph_data import build_dispatch_arc_observation_records

    return build_dispatch_arc_observation_records(
        context=context,
        dispatch_index=dispatch_index,
    )


def _build_dispatch_trace_records(context, decision, dispatch_index):
    from warehouse_sim.simulation.engine import _build_dispatch_trace_records as _inner

    return _inner(context, decision, dispatch_index)
