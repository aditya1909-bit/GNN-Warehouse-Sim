"""End-to-end macro PPO training and artifact loading for integrated coordination."""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from dataclasses import dataclass, replace
from pathlib import Path
import random
import sys

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
from warehouse_sim.integrated.models import (
    BatchIntegratedObservation,
    CollisionEventRecord,
    IntegratedObservation,
    IntegratedPolicyStep,
    MacroDecisionRecord,
    PlannerPlanRecord,
)
from warehouse_sim.integrated.planner import plan_motion_candidate
from warehouse_sim.integrated.policies import (
    IntegratedPolicyOutput,
    OptimalMAPFCoordinatorPolicy,
    PrioritizedSIPPCoordinatorPolicy,
)
from warehouse_sim.metrics.collector import compute_simulation_metrics
from warehouse_sim.simulation.runner import build_experiment_inputs, run_experiment_from_config
from warehouse_sim.simulation.models import SimulationResult
from warehouse_sim.utils.progress import ProgressTracker


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
    if artifact.model_type == "conflict_graph_macro_ppo":
        loaded = load_conflict_graph_macro_model(path, device=device)
        return LoadedEndToEndMacroModel(artifact=artifact, model=loaded.model)
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


@dataclass(frozen=True)
class LoadedConflictGraphMacroModel:
    artifact: EndToEndMacroArtifact
    model: "ConflictGraphMacroPolicyNetwork"


class ConflictGraphMacroPolicyNetwork(nn.Module):
    """Conflict-aware macro controller over warehouse, robot, and macro graphs."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        robot_dim: int,
        task_dim: int,
        macro_dim: int,
        density_dim: int,
        robot_robot_edge_dim: int,
        robot_macro_edge_dim: int,
        macro_conflict_edge_dim: int,
        hidden_dim: int = 64,
        warehouse_message_passing_layers: int = 1,
        conflict_message_passing_layers: int = 2,
        dropout: float = 0.0,
        top_k_conflicting_robots: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k_conflicting_robots = top_k_conflicting_robots
        self.node_encoder = _mlp(node_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.edge_encoder = _mlp(edge_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.robot_encoder = _mlp(robot_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.task_encoder = _mlp(max(task_dim, 1), hidden_dim, hidden_dim, dropout=dropout)
        self.macro_encoder = _mlp(macro_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.density_encoder = _mlp(max(density_dim, 1), hidden_dim, hidden_dim, dropout=dropout)
        self.robot_robot_edge_encoder = _mlp(max(robot_robot_edge_dim, 1), hidden_dim, hidden_dim, dropout=dropout)
        self.robot_macro_edge_encoder = _mlp(max(robot_macro_edge_dim, 1), hidden_dim, hidden_dim, dropout=dropout)
        self.macro_conflict_edge_encoder = _mlp(max(macro_conflict_edge_dim, 1), hidden_dim, hidden_dim, dropout=dropout)
        self.warehouse_convs = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=1,
                    concat=False,
                    edge_dim=hidden_dim,
                    add_self_loops=False,
                    dropout=dropout,
                )
                for _ in range(warehouse_message_passing_layers)
            ]
        )
        self.robot_conflict_convs = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=1,
                    concat=False,
                    edge_dim=hidden_dim,
                    add_self_loops=False,
                    dropout=dropout,
                )
                for _ in range(conflict_message_passing_layers)
            ]
        )
        self.macro_conflict_convs = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=1,
                    concat=False,
                    edge_dim=hidden_dim,
                    add_self_loops=False,
                    dropout=dropout,
                )
                for _ in range(conflict_message_passing_layers)
            ]
        )
        self.robot_macro_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def encode_state(
        self,
        observation: IntegratedObservation,
    ) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], torch.Tensor]:
        device = self._device()
        graph_embedding = self._encode_warehouse_graph(observation, device=device)
        density_embedding = self._encode_density(observation, device=device)
        robot_embeddings = self._encode_robot_conflicts(observation, device=device)
        macro_embeddings, macro_slices = self._encode_macro_graph(
            observation,
            robot_embeddings=robot_embeddings,
            graph_embedding=graph_embedding,
            density_embedding=density_embedding,
            device=device,
        )
        return graph_embedding, robot_embeddings, macro_slices, macro_embeddings + density_embedding.unsqueeze(0)

    def act(self, observation: IntegratedObservation, greedy: bool = False) -> IntegratedPolicyOutput:
        graph_embedding, robot_embeddings, macro_slices, macro_embeddings = self.encode_state(observation)
        chosen_indices, log_prob_total, _entropy_total = self._decode_parallel_assignment(
            observation=observation,
            graph_embedding=graph_embedding,
            robot_embeddings=robot_embeddings,
            macro_slices=macro_slices,
            macro_embeddings=macro_embeddings,
            chosen_indices=None,
            greedy=greedy,
        )
        value = self.value(observation, state=(graph_embedding, robot_embeddings, macro_slices, macro_embeddings))
        return IntegratedPolicyOutput(
            chosen_indices=tuple(chosen_indices),
            log_prob=float(log_prob_total.item()),
            value=float(value.item()),
        )

    def evaluate(
        self,
        observation: IntegratedObservation,
        chosen_indices: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_embedding, robot_embeddings, macro_slices, macro_embeddings = self.encode_state(observation)
        chosen_indices, log_prob_total, entropy_total = self._decode_parallel_assignment(
            observation=observation,
            graph_embedding=graph_embedding,
            robot_embeddings=robot_embeddings,
            macro_slices=macro_slices,
            macro_embeddings=macro_embeddings,
            chosen_indices=chosen_indices,
            greedy=True,
        )
        value = self.value(observation, state=(graph_embedding, robot_embeddings, macro_slices, macro_embeddings))
        return log_prob_total, value, entropy_total

    def act_batch(
        self,
        observations: list[IntegratedObservation] | tuple[IntegratedObservation, ...] | BatchIntegratedObservation,
        greedy: bool = False,
    ) -> list[IntegratedPolicyOutput]:
        batch = _ensure_batch_integrated_observation(observations)
        outputs: list[IntegratedPolicyOutput] = []
        for observation in batch.observations:
            outputs.append(self.act(observation, greedy=greedy))
        return outputs

    def evaluate_batch(
        self,
        observations: list[IntegratedObservation] | tuple[IntegratedObservation, ...] | BatchIntegratedObservation,
        chosen_indices_batch: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = _ensure_batch_integrated_observation(observations)
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for observation, chosen_indices in zip(batch.observations, chosen_indices_batch, strict=True):
            log_prob, value, entropy = self.evaluate(observation, chosen_indices)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
        return torch.stack(log_probs), torch.stack(values), torch.stack(entropies)

    def value(
        self,
        observation: IntegratedObservation,
        state: tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        device = self._device()
        if state is None:
            state = self.encode_state(observation)
        graph_embedding, robot_embeddings, _macro_slices, macro_embeddings = state
        density_embedding = self._encode_density(observation, device=device)
        if observation.task_features:
            task_embedding = self.task_encoder(
                torch.tensor(observation.task_features, dtype=torch.float32, device=device)
            ).mean(dim=0)
        else:
            task_embedding = torch.zeros(self.hidden_dim, dtype=torch.float32, device=device)
        macro_pool = macro_embeddings.mean(dim=0) if macro_embeddings.numel() else torch.zeros_like(graph_embedding)
        robot_pool = robot_embeddings.mean(dim=0) if robot_embeddings.numel() else torch.zeros_like(graph_embedding)
        return self.value_head(
            torch.cat([graph_embedding + task_embedding, robot_pool, macro_pool, density_embedding], dim=-1)
        ).squeeze(-1)

    def _encode_warehouse_graph(self, observation: IntegratedObservation, *, device: torch.device) -> torch.Tensor:
        node_features = torch.tensor(observation.node_features, dtype=torch.float32, device=device)
        edge_features = torch.tensor(observation.edge_features, dtype=torch.float32, device=device)
        edge_index = torch.tensor(observation.edge_index, dtype=torch.long, device=device).T.contiguous()
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)
        for conv in self.warehouse_convs:
            x = torch.relu(conv(x, edge_index, edge_attr))
            x = self.dropout(x)
        return x.mean(dim=0)

    def _encode_density(self, observation: IntegratedObservation, *, device: torch.device) -> torch.Tensor:
        density = observation.global_density_features or (0.0,)
        return self.density_encoder(torch.tensor(density, dtype=torch.float32, device=device))

    def _encode_robot_conflicts(self, observation: IntegratedObservation, *, device: torch.device) -> torch.Tensor:
        robot_tensor = torch.tensor(observation.robot_features, dtype=torch.float32, device=device)
        embeddings = self.robot_encoder(robot_tensor)
        if not observation.robot_robot_conflict_edges:
            return embeddings
        kept_edges, kept_features = _top_k_conflict_edges(
            observation.robot_robot_conflict_edges,
            observation.robot_robot_conflict_features,
            top_k=self.top_k_conflicting_robots,
        )
        if not kept_edges:
            return embeddings
        edge_index = torch.tensor(kept_edges, dtype=torch.long, device=device).T.contiguous()
        edge_attr = self.robot_robot_edge_encoder(torch.tensor(kept_features, dtype=torch.float32, device=device))
        for conv in self.robot_conflict_convs:
            embeddings = torch.relu(conv(embeddings, edge_index, edge_attr))
            embeddings = self.dropout(embeddings)
        return embeddings

    def _encode_macro_graph(
        self,
        observation: IntegratedObservation,
        *,
        robot_embeddings: torch.Tensor,
        graph_embedding: torch.Tensor,
        density_embedding: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        flat_macro_features: list[torch.Tensor] = []
        macro_slices: list[tuple[int, int]] = []
        for candidates in observation.macro_candidates:
            start_index = len(flat_macro_features)
            flat_macro_features.extend(
                _macro_feature_tensor(observation, candidate, device=device) for candidate in candidates
            )
            macro_slices.append((start_index, len(flat_macro_features)))
        macro_base = self.macro_encoder(torch.stack(flat_macro_features))
        incidence_messages = torch.zeros_like(macro_base)
        if observation.robot_macro_incidence_edges:
            for edge_index, (robot_index, macro_index) in enumerate(observation.robot_macro_incidence_edges):
                incidence_feature = torch.tensor(
                    observation.robot_macro_incidence_features[edge_index],
                    dtype=torch.float32,
                    device=device,
                )
                fused = self.robot_macro_fusion(
                    torch.cat(
                        [
                            robot_embeddings[robot_index],
                            self.robot_macro_edge_encoder(incidence_feature),
                            graph_embedding + density_embedding,
                        ],
                        dim=-1,
                    )
                )
                incidence_messages[macro_index] = incidence_messages[macro_index] + fused
        macro_embeddings = macro_base + incidence_messages
        if observation.macro_conflict_edges:
            edge_index = torch.tensor(observation.macro_conflict_edges, dtype=torch.long, device=device).T.contiguous()
            edge_attr = self.macro_conflict_edge_encoder(
                torch.tensor(observation.macro_conflict_features, dtype=torch.float32, device=device)
            )
            for conv in self.macro_conflict_convs:
                macro_embeddings = torch.relu(conv(macro_embeddings, edge_index, edge_attr))
                macro_embeddings = self.dropout(macro_embeddings)
        return macro_embeddings, macro_slices

    def _parallel_score_matrix(
        self,
        *,
        graph_embedding: torch.Tensor,
        density_embedding: torch.Tensor,
        robot_embeddings: torch.Tensor,
        macro_embeddings: torch.Tensor,
        macro_slices: list[tuple[int, int]],
    ) -> torch.Tensor:
        device = graph_embedding.device
        num_robots = robot_embeddings.shape[0]
        num_macros = macro_embeddings.shape[0]
        if num_robots == 0 or num_macros == 0:
            return torch.empty((num_robots, num_macros), dtype=torch.float32, device=device)
        repeated_graph = graph_embedding.unsqueeze(0).unsqueeze(0).expand(num_robots, num_macros, -1)
        repeated_robot = robot_embeddings.unsqueeze(1).expand(-1, num_macros, -1)
        repeated_macro = macro_embeddings.unsqueeze(0).expand(num_robots, -1, -1)
        repeated_density = density_embedding.unsqueeze(0).unsqueeze(0).expand(num_robots, num_macros, -1)
        logits = self.candidate_head(
            torch.cat([repeated_graph, repeated_robot, repeated_macro, repeated_density], dim=-1)
        ).squeeze(-1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        feasible_mask = torch.zeros((num_robots, num_macros), dtype=torch.bool, device=device)
        for robot_index, (start_index, end_index) in enumerate(macro_slices):
            feasible_mask[robot_index, start_index:end_index] = True
        return logits.masked_fill(~feasible_mask, float("-inf"))

    def _decode_parallel_assignment(
        self,
        *,
        observation: IntegratedObservation,
        graph_embedding: torch.Tensor,
        robot_embeddings: torch.Tensor,
        macro_slices: list[tuple[int, int]],
        macro_embeddings: torch.Tensor,
        chosen_indices: tuple[int, ...] | None,
        greedy: bool,
    ) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor]:
        device = self._device()
        density_embedding = self._encode_density(observation, device=device)
        score_matrix = self._parallel_score_matrix(
            graph_embedding=graph_embedding,
            density_embedding=density_embedding,
            robot_embeddings=robot_embeddings,
            macro_embeddings=macro_embeddings,
            macro_slices=macro_slices,
        )
        if chosen_indices is None:
            chosen_indices = _parallel_match_global_candidates(
                observation=observation,
                score_matrix=score_matrix,
                greedy=greedy,
            )
        log_prob_total = torch.tensor(0.0, dtype=torch.float32, device=device)
        entropy_total = torch.tensor(0.0, dtype=torch.float32, device=device)
        for robot_index, (start_index, end_index) in enumerate(macro_slices):
            local_logits = score_matrix[robot_index, start_index:end_index]
            if local_logits.numel() == 0:
                local_logits = torch.zeros((1,), dtype=torch.float32, device=device)
            local_logits = torch.nan_to_num(local_logits, nan=0.0, posinf=50.0, neginf=-50.0)
            if not torch.isfinite(local_logits).any():
                local_logits = torch.zeros_like(local_logits)
            distribution = torch.distributions.Categorical(logits=local_logits)
            chosen_index = int(chosen_indices[robot_index]) if robot_index < len(chosen_indices) else 0
            chosen_index = max(0, min(chosen_index, end_index - start_index - 1))
            chosen_tensor = torch.tensor(chosen_index, dtype=torch.long, device=device)
            log_prob_total = log_prob_total + distribution.log_prob(chosen_tensor)
            entropy_total = entropy_total + distribution.entropy()
        return tuple(chosen_indices), log_prob_total, entropy_total

    def _device(self) -> torch.device:
        return next(self.parameters()).device


def load_conflict_graph_macro_model(
    path: Path,
    device: torch.device | str = "cpu",
) -> LoadedConflictGraphMacroModel:
    artifact = load_end_to_end_macro_artifact(path)
    if artifact.model_type != "conflict_graph_macro_ppo":
        raise ValueError(f"Expected conflict_graph_macro_ppo artifact, got {artifact.model_type}")
    parameters = artifact.parameters
    model = ConflictGraphMacroPolicyNetwork(
        node_dim=int(parameters["node_dim"]),
        edge_dim=int(parameters["edge_dim"]),
        robot_dim=int(parameters["robot_dim"]),
        task_dim=int(parameters["task_dim"]),
        macro_dim=int(parameters["macro_dim"]),
        density_dim=int(parameters["density_dim"]),
        robot_robot_edge_dim=int(parameters["robot_robot_edge_dim"]),
        robot_macro_edge_dim=int(parameters["robot_macro_edge_dim"]),
        macro_conflict_edge_dim=int(parameters["macro_conflict_edge_dim"]),
        hidden_dim=int(parameters.get("hidden_dim", 64)),
        warehouse_message_passing_layers=int(parameters.get("warehouse_message_passing_layers", 1)),
        conflict_message_passing_layers=int(parameters.get("conflict_message_passing_layers", 2)),
        dropout=float(parameters.get("dropout", 0.0)),
        top_k_conflicting_robots=int(parameters.get("top_k_conflicting_robots", 4)),
    )
    state_path = path.parent / str(parameters["state_dict_path"])
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.to(device)
    model.eval()
    return LoadedConflictGraphMacroModel(artifact=artifact, model=model)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, *, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _top_k_conflict_edges(
    edges: tuple[tuple[int, int], ...],
    features: tuple[tuple[float, ...], ...],
    *,
    top_k: int,
) -> tuple[list[tuple[int, int]], list[tuple[float, ...]]]:
    ranked_by_source: dict[int, list[tuple[float, tuple[int, int], tuple[float, ...]]]] = {}
    for edge, feature in zip(edges, features, strict=True):
        ranked_by_source.setdefault(edge[0], []).append((float(feature[-1]), edge, feature))
    kept_edges: list[tuple[int, int]] = []
    kept_features: list[tuple[float, ...]] = []
    for ranked in ranked_by_source.values():
        ranked.sort(key=lambda item: item[0], reverse=True)
        for _score, edge, feature in ranked[:top_k]:
            kept_edges.append(edge)
            kept_features.append(feature)
    return kept_edges, kept_features


def _ensure_batch_integrated_observation(
    observations: list[IntegratedObservation] | tuple[IntegratedObservation, ...] | BatchIntegratedObservation,
) -> BatchIntegratedObservation:
    if isinstance(observations, BatchIntegratedObservation):
        return observations
    items = tuple(observations)
    graph_batch_index = tuple(index for index, observation in enumerate(items) for _ in observation.node_features)
    robot_batch_index = tuple(index for index, observation in enumerate(items) for _ in observation.robot_features)
    macro_batch_index = tuple(
        index
        for index, observation in enumerate(items)
        for candidates in observation.macro_candidates
        for _ in candidates
    )
    return BatchIntegratedObservation(
        observations=items,
        graph_batch_index=graph_batch_index,
        robot_batch_index=robot_batch_index,
        macro_batch_index=macro_batch_index,
    )


def _parallel_match_global_candidates(
    *,
    observation: IntegratedObservation,
    score_matrix: torch.Tensor,
    greedy: bool,
) -> tuple[int, ...]:
    if not observation.macro_candidates:
        return ()
    assignments: list[int | None] = [None] * len(observation.macro_candidates)
    claimed_tasks: set[int] = set()
    no_task_edges: list[tuple[float, int, int]] = []
    task_edges: list[tuple[float, int, int, int]] = []
    for robot_index, candidates in enumerate(observation.macro_candidates):
        for local_index, _candidate in enumerate(candidates):
            global_index = observation.robot_candidate_slices[robot_index][0] + local_index
            score = float(score_matrix[robot_index, global_index].item())
            task_index = (
                observation.global_candidate_task_indices[global_index]
                if global_index < len(observation.global_candidate_task_indices)
                else -1
            )
            if task_index >= 0:
                task_edges.append((score, robot_index, local_index, task_index))
            else:
                no_task_edges.append((score, robot_index, local_index))
    task_edges.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    for _score, robot_index, local_index, task_index in task_edges:
        if assignments[robot_index] is not None or task_index in claimed_tasks:
            continue
        assignments[robot_index] = local_index
        claimed_tasks.add(task_index)
    no_task_edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _score, robot_index, local_index in no_task_edges:
        if assignments[robot_index] is None:
            assignments[robot_index] = local_index
    resolved: list[int] = []
    for robot_index, chosen_index in enumerate(assignments):
        if chosen_index is not None:
            resolved.append(chosen_index)
            continue
        candidates = observation.macro_candidates[robot_index]
        if not candidates:
            resolved.append(0)
            continue
        best_index = 0
        best_score = float("-inf")
        start_index, _end_index = observation.robot_candidate_slices[robot_index]
        for local_index, candidate in enumerate(candidates):
            global_index = start_index + local_index
            task_index = observation.global_candidate_task_indices[global_index]
            if task_index >= 0 and task_index in claimed_tasks:
                continue
            score = float(score_matrix[robot_index, global_index].item())
            if score > best_score:
                best_score = score
                best_index = local_index
        task_id = candidates[best_index].task_id
        if task_id is not None:
            global_index = start_index + best_index
            task_index = observation.global_candidate_task_indices[global_index]
            if task_index >= 0:
                claimed_tasks.add(task_index)
        resolved.append(best_index)
    return tuple(resolved)


class IntegratedCoordinationRLEnv:
    """Replanning-boundary RL environment for integrated coordination."""

    def __init__(
        self,
        config: ExperimentConfig,
        seed: int,
        *,
        policy_name: str = "trained_conflict_graph_macro_ppo",
    ) -> None:
        seeded_config = replace(config, demand=replace(config.demand, seed=seed))
        self._experiment_config = seeded_config
        self._policy_name = policy_name
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
        before_planner_wait = sum(plan.wait_insertion_time for plan in self._planner_plans)
        before_path_conflicts = sum(plan.pre_resolution_conflict_count for plan in self._planner_plans)
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
                    selected_by_policy=self._policy_name,
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
                        planner_name=self._policy_name,
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
                    planner_name=self._policy_name,
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
        planner_wait_delta = sum(plan.wait_insertion_time for plan in self._planner_plans) - before_planner_wait
        path_conflict_delta = sum(plan.pre_resolution_conflict_count for plan in self._planner_plans) - before_path_conflicts
        reward = (
            reward_config.task_completion * completed_delta
            + reward_config.waiting_time * wait_delta
            + reward_config.congestion_delay * delay_delta
            + reward_config.safety_violation * safety_delta
            + reward_config.planner_wait_time * planner_wait_delta
            + reward_config.path_conflict * path_conflict_delta
            + reward_config.wait_insertion_time * planner_wait_delta
        )
        return (None if done else self._observation()), float(reward), done, {
            "completed_delta": float(completed_delta),
            "wait_delta": float(wait_delta),
            "delay_delta": float(delay_delta),
            "safety_delta": float(safety_delta),
            "planner_wait_delta": float(planner_wait_delta),
            "path_conflict_delta": float(path_conflict_delta),
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
            policy_name=self._policy_name,
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


def _resolve_integrated_training_device(requested_device: str) -> torch.device:
    if requested_device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _conflict_graph_model_kwargs(sample_observation: IntegratedObservation, config: IntegratedRLTrainingConfig) -> dict[str, int | float]:
    return {
        "node_dim": len(sample_observation.node_features[0]),
        "edge_dim": len(sample_observation.edge_features[0]) if sample_observation.edge_features else 3,
        "robot_dim": len(sample_observation.robot_features[0]),
        "task_dim": len(sample_observation.task_features[0]) if sample_observation.task_features else 5,
        "macro_dim": len(_macro_feature_tensor(sample_observation, sample_observation.macro_candidates[0][0])),
        "density_dim": len(sample_observation.global_density_features) if sample_observation.global_density_features else 1,
        "robot_robot_edge_dim": len(sample_observation.robot_robot_conflict_features[0]) if sample_observation.robot_robot_conflict_features else 6,
        "robot_macro_edge_dim": len(sample_observation.robot_macro_incidence_features[0]) if sample_observation.robot_macro_incidence_features else 6,
        "macro_conflict_edge_dim": len(sample_observation.macro_conflict_features[0]) if sample_observation.macro_conflict_features else 6,
        "hidden_dim": config.model.hidden_dim,
        "warehouse_message_passing_layers": config.model.warehouse_message_passing_layers,
        "conflict_message_passing_layers": config.model.conflict_message_passing_layers,
        "dropout": config.model.dropout,
        "top_k_conflicting_robots": config.model.top_k_conflicting_robots,
    }


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().to("cpu") for name, tensor in model.state_dict().items()}


def _short_progress_name(name: str, *, max_length: int = 28) -> str:
    if len(name) <= max_length:
        return name
    return name[: max_length - 3] + "..."


def _effective_rollout_workers(config: IntegratedRLTrainingConfig, *, phase: str) -> int:
    return max(1, config.runtime.rollout_workers)


def _should_skip_optimal_teacher(env: IntegratedCoordinationRLEnv, observation: IntegratedObservation) -> bool:
    scenario_name = env._experiment_config.name
    total_candidates = sum(len(candidates) for candidates in observation.macro_candidates)
    robot_count = len(observation.robot_ids)
    if scenario_name in {"integrated_high_fleet_density_heavy", "integrated_dense_merge_heavy"}:
        return True
    if robot_count >= 6 and total_candidates >= 24:
        return True
    if total_candidates >= 40:
        return True
    return False


def _checkpoint_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "artifact": output_dir / "model_artifact.json",
        "checkpoint": output_dir / "checkpoint.pt",
        "training_metrics": output_dir / "training_metrics.csv",
        "warm_start_metrics": output_dir / "warm_start_metrics.csv",
        "evaluation_rollouts": output_dir / "evaluation_rollouts.json",
        "claim_gate": output_dir / "claim_gate.json",
    }


def _load_integrated_training_checkpoint(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("checkpoint_kind") != "integrated_training":
        return None
    return payload


def _save_integrated_training_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint_kind": "integrated_training", **payload}, path)


def _rollout_worker(
    *,
    scenario: ExperimentConfig,
    seed: int,
    reward_config,
    model_kwargs: dict[str, int | float],
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[IntegratedPolicyStep], dict[str, object]]:
    model = ConflictGraphMacroPolicyNetwork(**model_kwargs)
    model.load_state_dict(state_dict)
    model.eval()
    env = IntegratedCoordinationRLEnv(scenario, seed)
    observation = env.reset()
    done = False
    transitions: list[IntegratedPolicyStep] = []
    while not done:
        with torch.no_grad():
            output = model.act(observation, greedy=False)
        next_observation, reward, done, _info = env.step(output.chosen_indices, reward_config)
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
    row = {
        "tasks_completed": result.metrics.tasks_completed,
        "throughput_per_hour": result.metrics.throughput_per_hour,
        "safety_violations_total": result.metrics.safety_violations_total,
        "path_conflicts_before_resolution_total": result.metrics.path_conflicts_before_resolution_total,
        "planner_wait_time_total": result.metrics.planner_wait_time_total,
    }
    return transitions, row


def _warm_start_samples_worker(
    *,
    scenario: ExperimentConfig,
    seed: int,
    teacher_mixture: dict[str, float],
    epoch_seed: int,
    reward_config,
) -> list[tuple[IntegratedObservation, tuple[int, ...], str]]:
    env = IntegratedCoordinationRLEnv(scenario, seed)
    observation = env.reset()
    done = False
    rng = random.Random(epoch_seed + seed)
    rows: list[tuple[IntegratedObservation, tuple[int, ...], str]] = []
    while not done:
        teacher_name, teacher_output = _teacher_output(env, observation, teacher_mixture, rng=rng)
        rows.append((observation, teacher_output.chosen_indices, teacher_name))
        next_observation, _reward, done, _info = env.step(teacher_output.chosen_indices, reward_config)
        observation = next_observation or observation
    return rows


def run_integrated_rl_training_from_config(config: IntegratedRLTrainingConfig) -> dict[str, Path]:
    """Train a conflict-aware PPO macro controller on integrated scenarios."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _checkpoint_output_paths(output_dir)
    checkpoint = _load_integrated_training_checkpoint(output_paths["checkpoint"])
    if checkpoint is not None and checkpoint.get("phase") == "completed" and all(
        path.exists() for key, path in output_paths.items() if key != "checkpoint"
    ):
        return output_paths

    scenarios = [load_experiment_config(path) for path in config.curriculum.scenario_configs]
    env = IntegratedCoordinationRLEnv(scenarios[0], config.curriculum.train_seeds[0])
    sample_observation = env.reset()
    learner_device = _resolve_integrated_training_device(config.runtime.device)
    model_kwargs = _conflict_graph_model_kwargs(sample_observation, config)
    model = ConflictGraphMacroPolicyNetwork(**model_kwargs)
    model.to(learner_device)
    ppo_optimizer = optim.Adam(model.parameters(), lr=config.ppo.learning_rate)
    warm_optimizer = optim.Adam(model.parameters(), lr=config.warm_start.learning_rate)
    warm_start_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    teacher_mixture = _resolved_teacher_mixture(config)
    selected_state_dict = _cpu_state_dict(model)
    selected_validation_rows: list[dict[str, object]] = []
    selected_gate_payload: dict[str, object] = {
        "claim_ready": False,
        "observed_safety_violations": float("inf"),
        "observed_task_completion_rate": 0.0,
        "observed_throughput_ratio_vs_baseline": 0.0,
        "observed_policy_distinctness_vs_teacher": 0.0,
    }
    warm_start_gate_payload: dict[str, object] = dict(selected_gate_payload)
    selected_stage = "initial"
    rng = random.Random(0)
    warm_start_completed_epochs = 0
    ppo_completed_episodes = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        selected_state_dict = dict(checkpoint.get("selected_state_dict", selected_state_dict))
        selected_validation_rows = list(checkpoint.get("selected_validation_rows", []))
        selected_gate_payload = dict(checkpoint.get("selected_gate_payload", selected_gate_payload))
        warm_start_gate_payload = dict(checkpoint.get("warm_start_gate_payload", warm_start_gate_payload))
        selected_stage = str(checkpoint.get("selected_stage", selected_stage))
        warm_start_rows = list(checkpoint.get("warm_start_rows", []))
        training_rows = list(checkpoint.get("training_rows", []))
        warm_start_completed_epochs = int(checkpoint.get("warm_start_completed_epochs", 0))
        ppo_completed_episodes = int(checkpoint.get("ppo_completed_episodes", 0))
        if "rng_state" in checkpoint:
            rng.setstate(checkpoint["rng_state"])
        if checkpoint.get("phase") in {"warm_start", "post_warm_start"} and checkpoint.get("warm_optimizer_state_dict"):
            warm_optimizer.load_state_dict(checkpoint["warm_optimizer_state_dict"])
        if checkpoint.get("phase") in {"ppo", "completed"} and checkpoint.get("ppo_optimizer_state_dict"):
            ppo_optimizer.load_state_dict(checkpoint["ppo_optimizer_state_dict"])
    rollout_workers = _effective_rollout_workers(config, phase="ppo")
    episodes_per_sync = max(1, config.runtime.episodes_per_sync)
    cpu_state = _cpu_state_dict(model)
    overall_progress = ProgressTracker(
        label="integrated_rl",
        total=max(1, config.warm_start.epochs + config.ppo.total_episodes + len(scenarios)),
        unit="phase",
    )
    completed_phases = warm_start_completed_epochs + ppo_completed_episodes
    overall_progress.update(completed_phases, extra="starting integrated training", force=True)

    warm_progress = ProgressTracker(label="integrated_warm_start", total=max(1, config.warm_start.epochs), unit="epoch")
    warm_progress.update(warm_start_completed_epochs, extra="resuming warm start", force=True)
    for epoch in range(warm_start_completed_epochs, config.warm_start.epochs):
        samples = _collect_warm_start_samples(
            scenarios=scenarios,
            config=config,
            teacher_mixture=teacher_mixture,
            epoch=epoch,
            rollout_workers=_effective_rollout_workers(config, phase="warm_start"),
        )
        row = _run_warm_start_epoch(
            model=model,
            optimizer=warm_optimizer,
            samples=samples,
            config=config,
            epoch=epoch,
            teacher_mixture=teacher_mixture,
        )
        warm_start_rows.append(row)
        warm_start_completed_epochs = epoch + 1
        completed_phases = warm_start_completed_epochs + ppo_completed_episodes
        warm_progress.update(
            warm_start_completed_epochs,
            extra=f"bc_loss={row['mean_bc_loss']:.4f} match={row['teacher_action_match_rate']:.3f}",
            force=True,
        )
        overall_progress.update(completed_phases, extra="warm start complete", force=True)
        _save_integrated_training_checkpoint(
            output_paths["checkpoint"],
            {
                "phase": "warm_start",
                "model_state_dict": _cpu_state_dict(model),
                "warm_optimizer_state_dict": warm_optimizer.state_dict(),
                "ppo_optimizer_state_dict": ppo_optimizer.state_dict(),
                "warm_start_completed_epochs": warm_start_completed_epochs,
                "ppo_completed_episodes": ppo_completed_episodes,
                "warm_start_rows": warm_start_rows,
                "training_rows": training_rows,
                "selected_state_dict": selected_state_dict,
                "selected_validation_rows": selected_validation_rows,
                "selected_gate_payload": selected_gate_payload,
                "warm_start_gate_payload": warm_start_gate_payload,
                "selected_stage": selected_stage,
                "rng_state": rng.getstate(),
            },
        )
    warm_progress.close(extra="warm start complete")

    if config.warm_start.epochs > 0 and (not selected_validation_rows or selected_stage == "initial"):
        warm_start_validation_rows, warm_start_gate_payload = _evaluate_integrated_macro_model(model, scenarios, config)
        selected_state_dict = _cpu_state_dict(model)
        selected_validation_rows = warm_start_validation_rows
        selected_gate_payload = warm_start_gate_payload
        selected_stage = "warm_start"
        _save_integrated_training_checkpoint(
            output_paths["checkpoint"],
            {
                "phase": "post_warm_start",
                "model_state_dict": _cpu_state_dict(model),
                "warm_optimizer_state_dict": warm_optimizer.state_dict(),
                "ppo_optimizer_state_dict": ppo_optimizer.state_dict(),
                "warm_start_completed_epochs": warm_start_completed_epochs,
                "ppo_completed_episodes": ppo_completed_episodes,
                "warm_start_rows": warm_start_rows,
                "training_rows": training_rows,
                "selected_state_dict": selected_state_dict,
                "selected_validation_rows": selected_validation_rows,
                "selected_gate_payload": selected_gate_payload,
                "warm_start_gate_payload": warm_start_gate_payload,
                "selected_stage": selected_stage,
                "rng_state": rng.getstate(),
            },
        )

    episode = ppo_completed_episodes
    transitions: list[IntegratedPolicyStep] = []
    ppo_progress = ProgressTracker(label="integrated_ppo", total=config.ppo.total_episodes, unit="episode")
    ppo_progress.update(episode, extra="waiting for first rollout sync", force=True)
    while episode < config.ppo.total_episodes:
        batch_size = min(episodes_per_sync, config.ppo.total_episodes - episode)
        rollout_specs: list[tuple[int, ExperimentConfig, int]] = []
        for batch_offset in range(batch_size):
            scenario = _sample_curriculum_scenario(scenarios, config, rng=rng)
            seed = config.curriculum.train_seeds[(episode + batch_offset) % len(config.curriculum.train_seeds)]
            rollout_specs.append((episode + batch_offset, scenario, seed))
        rollout_progress = ProgressTracker(
            label="integrated_rollout_sync",
            total=batch_size,
            unit="rollout",
            unit_plural="rollouts",
            min_interval_seconds=0.0,
        )
        rollout_progress.update(0, extra=f"sync_start episode={episode}", force=True)
        if rollout_workers > 1 and batch_size > 1:
            with ProcessPoolExecutor(max_workers=min(rollout_workers, batch_size)) as executor:
                future_map = {
                    executor.submit(
                        _rollout_worker,
                        scenario=scenario,
                        seed=seed,
                        reward_config=config.reward,
                        model_kwargs=model_kwargs,
                        state_dict=cpu_state,
                    ): (episode_index, scenario, seed)
                    for episode_index, scenario, seed in rollout_specs
                }
                rollout_outputs_by_key: dict[tuple[int, str, int], tuple[list[IntegratedPolicyStep], dict[str, object]]] = {}
                completed_rollouts = 0
                for future in as_completed(future_map):
                    episode_index, scenario, seed = future_map[future]
                    rollout_outputs_by_key[(episode_index, scenario.name, seed)] = future.result()
                    completed_rollouts += 1
                    rollout_progress.update(
                        completed_rollouts,
                        extra=f"completed {_short_progress_name(scenario.name)} seed {seed}",
                        force=True,
                    )
                rollout_outputs = [
                    rollout_outputs_by_key[(episode_index, scenario.name, seed)]
                    for episode_index, scenario, seed in rollout_specs
                ]
        else:
            rollout_outputs = []
            for rollout_index, (_episode_index, scenario, seed) in enumerate(rollout_specs, start=1):
                rollout_outputs.append(
                    _rollout_worker(
                        scenario=scenario,
                        seed=seed,
                        reward_config=config.reward,
                        model_kwargs=model_kwargs,
                        state_dict=cpu_state,
                    )
                )
                rollout_progress.update(
                    rollout_index,
                    extra=f"completed {_short_progress_name(scenario.name)} seed {seed}",
                    force=True,
                )
        rollout_progress.close(extra=f"sync_done episode={episode + batch_size}")
        for (episode_index, scenario, seed), (episode_transitions, row) in zip(rollout_specs, rollout_outputs, strict=True):
            transitions.extend(episode_transitions)
            training_rows.append(
                {
                    "episode": episode_index,
                    "scenario_name": scenario.name,
                    "seed": seed,
                    "scenario_weight": _scenario_weight(config, scenario.name),
                    **row,
                }
            )
        _ppo_update(model, ppo_optimizer, transitions, config)
        transitions.clear()
        cpu_state = _cpu_state_dict(model)
        episode += batch_size
        ppo_completed_episodes = episode
        ppo_progress.update(
            episode,
            extra=f"last_scenario={_short_progress_name(rollout_specs[-1][1].name)} workers={min(rollout_workers, batch_size)}",
            force=True,
        )
        completed_phases = warm_start_completed_epochs + episode
        overall_progress.update(completed_phases, extra="ppo collecting/updating", force=True)
        _save_integrated_training_checkpoint(
            output_paths["checkpoint"],
            {
                "phase": "ppo",
                "model_state_dict": _cpu_state_dict(model),
                "warm_optimizer_state_dict": warm_optimizer.state_dict(),
                "ppo_optimizer_state_dict": ppo_optimizer.state_dict(),
                "warm_start_completed_epochs": warm_start_completed_epochs,
                "ppo_completed_episodes": ppo_completed_episodes,
                "warm_start_rows": warm_start_rows,
                "training_rows": training_rows,
                "selected_state_dict": selected_state_dict,
                "selected_validation_rows": selected_validation_rows,
                "selected_gate_payload": selected_gate_payload,
                "warm_start_gate_payload": warm_start_gate_payload,
                "selected_stage": selected_stage,
                "rng_state": rng.getstate(),
            },
        )
    ppo_progress.close(extra="ppo complete")

    final_validation_rows, final_gate_payload = _evaluate_integrated_macro_model(model, scenarios, config)
    overall_progress.close(extra="training and validation complete")
    if _is_better_gate_payload(final_gate_payload, selected_gate_payload):
        selected_state_dict = _cpu_state_dict(model)
        selected_validation_rows = final_validation_rows
        selected_gate_payload = final_gate_payload
        selected_stage = "ppo_final"
    model.load_state_dict(selected_state_dict)
    state_path = output_dir / "conflict_graph_macro_ppo.pt"
    torch.save(_cpu_state_dict(model), state_path)
    artifact = EndToEndMacroArtifact(
        artifact_version=1,
        model_type="conflict_graph_macro_ppo",
        parameters={
            "node_dim": len(sample_observation.node_features[0]),
            "edge_dim": len(sample_observation.edge_features[0]) if sample_observation.edge_features else 3,
            "robot_dim": len(sample_observation.robot_features[0]),
            "task_dim": len(sample_observation.task_features[0]) if sample_observation.task_features else 5,
            "macro_dim": len(_macro_feature_tensor(sample_observation, sample_observation.macro_candidates[0][0])),
            "density_dim": len(sample_observation.global_density_features) if sample_observation.global_density_features else 1,
            "robot_robot_edge_dim": len(sample_observation.robot_robot_conflict_features[0]) if sample_observation.robot_robot_conflict_features else 6,
            "robot_macro_edge_dim": len(sample_observation.robot_macro_incidence_features[0]) if sample_observation.robot_macro_incidence_features else 6,
            "macro_conflict_edge_dim": len(sample_observation.macro_conflict_features[0]) if sample_observation.macro_conflict_features else 6,
            "hidden_dim": config.model.hidden_dim,
            "warehouse_message_passing_layers": config.model.warehouse_message_passing_layers,
            "conflict_message_passing_layers": config.model.conflict_message_passing_layers,
            "dropout": config.model.dropout,
            "top_k_conflicting_robots": config.model.top_k_conflicting_robots,
            "state_dict_path": state_path.name,
        },
        metadata={
            "benchmark_gate": selected_gate_payload,
            "selected_checkpoint_stage": selected_stage,
            "decoder_type": "parallel_matching",
            "training_runtime": "mps_learner_cpu_actors",
            "candidate_gate_evaluations": {
                "warm_start": warm_start_gate_payload,
                "ppo_final": final_gate_payload,
            },
            "warm_start": {
                "epochs": config.warm_start.epochs,
                "teacher_mixture": _resolved_teacher_mixture(config),
            },
            "runtime": {
                "requested_device": config.runtime.device,
                "resolved_device": str(learner_device),
                "rollout_workers": rollout_workers,
                "episodes_per_sync": episodes_per_sync,
                "inference_batch_size": config.runtime.inference_batch_size,
                "learner_minibatch_size": config.ppo.learner_minibatch_size,
            },
        },
    )
    artifact_path = write_end_to_end_macro_artifact(artifact, output_dir / "model_artifact.json")
    _write_csv(output_paths["training_metrics"], training_rows)
    _write_csv(output_paths["warm_start_metrics"], warm_start_rows)
    output_paths["evaluation_rollouts"].write_text(json.dumps(selected_validation_rows, indent=2), encoding="utf-8")
    output_paths["claim_gate"].write_text(json.dumps(selected_gate_payload, indent=2), encoding="utf-8")
    _save_integrated_training_checkpoint(
        output_paths["checkpoint"],
        {
            "phase": "completed",
            "model_state_dict": _cpu_state_dict(model),
            "warm_optimizer_state_dict": warm_optimizer.state_dict(),
            "ppo_optimizer_state_dict": ppo_optimizer.state_dict(),
            "warm_start_completed_epochs": warm_start_completed_epochs,
            "ppo_completed_episodes": ppo_completed_episodes,
            "warm_start_rows": warm_start_rows,
            "training_rows": training_rows,
            "selected_state_dict": selected_state_dict,
            "selected_validation_rows": selected_validation_rows,
            "selected_gate_payload": selected_gate_payload,
            "warm_start_gate_payload": warm_start_gate_payload,
            "selected_stage": selected_stage,
            "rng_state": rng.getstate(),
        },
    )
    output_paths["artifact"] = artifact_path
    return output_paths


def _collect_warm_start_samples(
    *,
    scenarios: list[ExperimentConfig],
    config: IntegratedRLTrainingConfig,
    teacher_mixture: dict[str, float],
    epoch: int,
    rollout_workers: int,
) -> list[tuple[IntegratedObservation, tuple[int, ...], str]]:
    sample_specs = [
        (scenario, seed)
        for scenario in _weighted_scenario_iteration(scenarios, config)
        for seed in config.curriculum.train_seeds
    ]
    sample_progress = ProgressTracker(
        label="integrated_warm_start_samples",
        total=len(sample_specs),
        unit="trace_batch",
        unit_plural="trace_batches",
        min_interval_seconds=0.0,
    )
    sample_progress.update(0, extra=f"epoch={epoch}", force=True)
    collected: list[tuple[IntegratedObservation, tuple[int, ...], str]] = []
    if rollout_workers > 1 and len(sample_specs) > 1:
        with ProcessPoolExecutor(max_workers=min(rollout_workers, len(sample_specs))) as executor:
            future_map = {
                executor.submit(
                    _warm_start_samples_worker,
                    scenario=scenario,
                    seed=seed,
                    teacher_mixture=teacher_mixture,
                    epoch_seed=epoch,
                    reward_config=config.reward,
                ): (scenario.name, seed)
                for scenario, seed in sample_specs
            }
            completed_batches = 0
            for future in as_completed(future_map):
                scenario_name, seed = future_map[future]
                collected.extend(future.result())
                completed_batches += 1
                sample_progress.update(
                    completed_batches,
                    extra=f"epoch={epoch} collected {_short_progress_name(scenario_name)} seed {seed}",
                    force=True,
                )
    else:
        for sample_index, (scenario, seed) in enumerate(sample_specs, start=1):
            collected.extend(
                _warm_start_samples_worker(
                    scenario=scenario,
                    seed=seed,
                    teacher_mixture=teacher_mixture,
                    epoch_seed=epoch,
                    reward_config=config.reward,
                )
            )
            sample_progress.update(
                sample_index,
                extra=f"epoch={epoch} collected {_short_progress_name(scenario.name)} seed {seed}",
                force=True,
            )
    sample_progress.close(extra=f"epoch={epoch} teacher traces ready")
    return collected


def _run_warm_start_epoch(
    *,
    model: ConflictGraphMacroPolicyNetwork,
    optimizer: optim.Optimizer,
    samples: list[tuple[IntegratedObservation, tuple[int, ...], str]],
    config: IntegratedRLTrainingConfig,
    epoch: int,
    teacher_mixture: dict[str, float],
) -> dict[str, object]:
    batch_size = max(1, config.ppo.learner_minibatch_size)
    epoch_losses: list[float] = []
    matched = 0
    total = 0
    shuffled = list(samples)
    random.Random(epoch).shuffle(shuffled)
    total_updates = max(1, (len(shuffled) + batch_size - 1) // batch_size)
    bc_progress = ProgressTracker(
        label="integrated_warm_start_opt",
        total=total_updates,
        unit="update",
        unit_plural="updates",
        min_interval_seconds=0.0,
    )
    bc_progress.update(0, extra=f"epoch={epoch} applying BC updates", force=True)
    completed_updates = 0
    for start_index in range(0, len(shuffled), batch_size):
        batch = shuffled[start_index : start_index + batch_size]
        observations = [observation for observation, _chosen_indices, _teacher_name in batch]
        chosen_indices_batch = [chosen_indices for _observation, chosen_indices, _teacher_name in batch]
        log_probs, _values, _entropies = model.evaluate_batch(observations, chosen_indices_batch)
        loss = -log_probs.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.item()))
        with torch.no_grad():
            greedy_outputs = model.act_batch(observations, greedy=True)
        for greedy_output, (_observation, chosen_indices, _teacher_name) in zip(greedy_outputs, batch, strict=True):
            matched += sum(
                int(predicted == target)
                for predicted, target in zip(greedy_output.chosen_indices, chosen_indices, strict=True)
            )
            total += len(chosen_indices)
        completed_updates += 1
        bc_progress.update(
            completed_updates,
            extra=f"epoch={epoch} loss={float(loss.item()):.4f}",
            force=True,
        )
    bc_progress.close(extra=f"epoch={epoch} BC updates complete")
    return {
        "warm_start_epoch": epoch,
        "teacher_policy": ",".join(f"{name}:{weight:.3f}" for name, weight in teacher_mixture.items()),
        "mean_bc_loss": sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0,
        "teacher_action_match_rate": matched / total if total else 0.0,
        "teacher_sample_count": len(samples),
    }


def _ppo_update(
    model: ConflictGraphMacroPolicyNetwork,
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
    device = model._device()
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    old_log_probs = torch.tensor([transition.old_log_prob for transition in transitions], dtype=torch.float32, device=device)
    minibatch_size = max(1, min(config.ppo.learner_minibatch_size, len(transitions)))

    for _ in range(config.ppo.ppo_epochs):
        for start_index in range(0, len(transitions), minibatch_size):
            batch = transitions[start_index : start_index + minibatch_size]
            if not batch:
                continue
            observations = [transition.observation for transition in batch]
            chosen_indices_batch = [transition.chosen_indices for transition in batch]
            log_probs, values, entropies = model.evaluate_batch(observations, chosen_indices_batch)
            log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=50.0, neginf=-50.0)
            values = torch.nan_to_num(values, nan=0.0, posinf=1e3, neginf=-1e3)
            entropies = torch.nan_to_num(entropies, nan=0.0, posinf=10.0, neginf=0.0)
            index_tensor = torch.arange(start_index, start_index + len(batch), device=device)
            ratio = torch.exp(log_probs - old_log_probs[index_tensor])
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.0)
            unclipped = ratio * advantages_tensor[index_tensor]
            clipped = (
                torch.clamp(ratio, 1.0 - config.ppo.clip_epsilon, 1.0 + config.ppo.clip_epsilon)
                * advantages_tensor[index_tensor]
            )
            actor_loss = -torch.minimum(unclipped, clipped)
            value_loss = torch.nn.functional.mse_loss(values, returns_tensor[index_tensor])
            loss = actor_loss.mean() + 0.5 * value_loss - 0.01 * entropies.mean()
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e3, neginf=-1e3)
            optimizer.zero_grad()
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                optimizer.zero_grad()
                continue
            optimizer.step()


def _evaluate_integrated_macro_model(
    model: ConflictGraphMacroPolicyNetwork,
    scenarios: list[ExperimentConfig],
    config: IntegratedRLTrainingConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    validation_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    distinct_matches = 0
    distinct_total = 0
    teacher = PrioritizedSIPPCoordinatorPolicy()
    validation_seeds = config.curriculum.validation_seeds or config.curriculum.train_seeds[:1]
    progress = ProgressTracker(
        label="integrated_eval",
        total=max(1, len(scenarios) * len(validation_seeds)),
        unit="rollout",
    )
    completed_rollouts = 0
    progress.update(0, extra="starting validation rollouts", force=True)
    for scenario in scenarios:
        for seed in validation_seeds:
            env = IntegratedCoordinationRLEnv(scenario, seed)
            observation = env.reset()
            done = False
            while not done:
                teacher_indices = teacher.select_macros(observation).chosen_indices
                output = model.act(observation, greedy=True)
                distinct_matches += sum(
                    int(predicted != baseline)
                    for predicted, baseline in zip(output.chosen_indices, teacher_indices, strict=True)
                )
                distinct_total += len(output.chosen_indices)
                observation, _reward, done, _info = env.step(output.chosen_indices, config.reward)
                if observation is None:
                    break
            result = env.build_result()
            validation_rows.append(
                {
                    "scenario_name": scenario.name,
                    "seed": seed,
                    "policy": "trained_conflict_graph_macro_ppo",
                    "tasks_completed": result.metrics.tasks_completed,
                    "tasks_generated": result.metrics.tasks_generated,
                    "throughput_per_hour": result.metrics.throughput_per_hour,
                    "safety_violations_total": result.metrics.safety_violations_total,
                    "collision_event_count": result.metrics.safety_violations_total,
                    "path_conflicts_before_resolution_total": result.metrics.path_conflicts_before_resolution_total,
                    "planner_wait_time_total": result.metrics.planner_wait_time_total,
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
                    "collision_event_count": baseline_result.metrics.safety_violations_total,
                    "path_conflicts_before_resolution_total": baseline_result.metrics.path_conflicts_before_resolution_total,
                    "planner_wait_time_total": baseline_result.metrics.planner_wait_time_total,
                }
            )
            completed_rollouts += 1
            progress.update(completed_rollouts, extra=f"{scenario.name} seed {seed}", force=True)
    throughput_ratio = (
        sum(row["throughput_per_hour"] for row in validation_rows) / max(sum(row["throughput_per_hour"] for row in baseline_rows), 1e-6)
    )
    task_completion_rate = (
        sum(row["tasks_completed"] for row in validation_rows) / max(sum(row["tasks_generated"] for row in validation_rows), 1)
    )
    safety_violations = sum(row["safety_violations_total"] for row in validation_rows)
    policy_distinctness = distinct_matches / max(distinct_total, 1)
    gate_payload = {
        "max_safety_violations": config.benchmark_gate.max_safety_violations,
        "min_task_completion_rate": config.benchmark_gate.min_task_completion_rate,
        "min_throughput_ratio_vs_baseline": config.benchmark_gate.min_throughput_ratio_vs_baseline,
        "min_policy_distinctness_vs_teacher": config.benchmark_gate.min_policy_distinctness_vs_teacher,
        "observed_safety_violations": safety_violations,
        "observed_task_completion_rate": task_completion_rate,
        "observed_throughput_ratio_vs_baseline": throughput_ratio,
        "observed_policy_distinctness_vs_teacher": policy_distinctness,
        "claim_ready": (
            safety_violations <= config.benchmark_gate.max_safety_violations
            and task_completion_rate >= config.benchmark_gate.min_task_completion_rate
            and throughput_ratio >= config.benchmark_gate.min_throughput_ratio_vs_baseline
            and policy_distinctness >= config.benchmark_gate.min_policy_distinctness_vs_teacher
        ),
    }
    progress.close(extra=f"throughput_ratio={throughput_ratio:.3f} claim_ready={gate_payload['claim_ready']}")
    return validation_rows, gate_payload


def _teacher_policy(name: str):
    if name == "prioritized_sipp_coordinator":
        return PrioritizedSIPPCoordinatorPolicy()
    if name == "optimal_mapf_coordinator":
        return OptimalMAPFCoordinatorPolicy()
    raise ValueError(f"Unsupported warm-start teacher policy: {name}")


def _behavior_cloning_loss(
    model: ConflictGraphMacroPolicyNetwork,
    observation: IntegratedObservation,
    chosen_indices: tuple[int, ...],
) -> torch.Tensor:
    log_prob, _value, _entropy = model.evaluate(observation, chosen_indices)
    return -log_prob / max(len(chosen_indices), 1)


def _is_better_gate_payload(candidate: dict[str, object], incumbent: dict[str, object]) -> bool:
    return _gate_selection_key(candidate) > _gate_selection_key(incumbent)


def _gate_selection_key(payload: dict[str, object]) -> tuple[float, float, float, float, float]:
    return (
        1.0 if bool(payload.get("claim_ready", False)) else 0.0,
        -float(payload.get("observed_safety_violations", 0.0)),
        float(payload.get("observed_task_completion_rate", 0.0)),
        float(payload.get("observed_throughput_ratio_vs_baseline", 0.0)),
        float(payload.get("observed_policy_distinctness_vs_teacher", 0.0)),
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
            1.0 if macro_type == "charge_route" else 0.0,
            route_time,
            float(len(candidate.route_nodes)),
            float(len(candidate.route_edges)),
            1.0 if candidate.task_id is not None else 0.0,
            candidate.shared_route_segment_count,
            candidate.shared_chokepoint_count,
            candidate.predicted_overlap_time,
            candidate.competing_task_usage_count,
            candidate.planner_estimated_conflict_count,
        ],
        dtype=torch.float32,
        device=device,
    )


def _resolved_teacher_mixture(config: IntegratedRLTrainingConfig) -> dict[str, float]:
    if config.warm_start.teacher_mixture:
        return dict(config.warm_start.teacher_mixture)
    return {config.warm_start.teacher_policy: 1.0}


def _scenario_weight(config: IntegratedRLTrainingConfig, scenario_name: str) -> float:
    return float(config.curriculum.scenario_weights.get(scenario_name, 1.0))


def _sample_curriculum_scenario(
    scenarios: list[ExperimentConfig],
    config: IntegratedRLTrainingConfig,
    *,
    rng: random.Random,
) -> ExperimentConfig:
    weights = [_scenario_weight(config, scenario.name) for scenario in scenarios]
    return rng.choices(scenarios, weights=weights, k=1)[0]


def _weighted_scenario_iteration(
    scenarios: list[ExperimentConfig],
    config: IntegratedRLTrainingConfig,
) -> list[ExperimentConfig]:
    weighted: list[ExperimentConfig] = []
    for scenario in scenarios:
        repeats = max(1, int(round(_scenario_weight(config, scenario.name))))
        weighted.extend([scenario] * repeats)
    return weighted


def _teacher_output(
    env: IntegratedCoordinationRLEnv,
    observation: IntegratedObservation,
    teacher_mixture: dict[str, float],
    *,
    rng: random.Random,
) -> tuple[str, IntegratedPolicyOutput]:
    if _should_skip_optimal_teacher(env, observation):
        prioritized = _teacher_output_for_policy(env, observation, "prioritized_sipp_coordinator")
        return "prioritized_sipp_coordinator", prioritized
    teacher_outputs = {
        name: _teacher_output_for_policy(env, observation, name)
        for name in teacher_mixture
    }
    if len(teacher_outputs) == 1:
        teacher_name = next(iter(teacher_outputs))
        return teacher_name, teacher_outputs[teacher_name]

    prioritized = teacher_outputs.get("prioritized_sipp_coordinator")
    optimal = teacher_outputs.get("optimal_mapf_coordinator")
    if prioritized is None or optimal is None:
        teacher_name = rng.choices(list(teacher_outputs), weights=list(teacher_mixture.values()), k=1)[0]
        return teacher_name, teacher_outputs[teacher_name]
    prioritized_conflict = _teacher_conflict_score(env, observation, prioritized)
    optimal_conflict = _teacher_conflict_score(env, observation, optimal)
    materially_better_optimal = (
        optimal.chosen_indices != prioritized.chosen_indices
        and optimal_conflict < prioritized_conflict
    )
    if materially_better_optimal:
        chosen = rng.choices(
            ["prioritized_sipp_coordinator", "optimal_mapf_coordinator"],
            weights=[
                teacher_mixture.get("prioritized_sipp_coordinator", 1.0),
                teacher_mixture.get("optimal_mapf_coordinator", 1.0),
            ],
            k=1,
        )[0]
        return chosen, teacher_outputs[chosen]
    return "prioritized_sipp_coordinator", prioritized


def _teacher_output_for_policy(
    env: IntegratedCoordinationRLEnv,
    observation: IntegratedObservation,
    policy_name: str,
) -> IntegratedPolicyOutput:
    teacher = _teacher_policy(policy_name)
    if policy_name == "optimal_mapf_coordinator":
        planned = teacher.plan_joint_macros(
            observation,
            environment=env._environment,
            occupancy=env._occupancy,
            robot_states=env._robot_states,
            tasks=env._tasks,
            current_time=env._current_time,
            config=env._simulation_config,
        )
        if planned is not None:
            return planned
    return teacher.select_macros(observation)


def _teacher_conflict_score(
    env: IntegratedCoordinationRLEnv,
    observation: IntegratedObservation,
    output: IntegratedPolicyOutput,
) -> int:
    assert env._occupancy is not None
    return _count_pre_resolution_conflicts(
        environment=env._environment,
        observation=observation,
        output=output,
        robot_states=env._robot_states,
        occupancy=env._occupancy,
        current_time=env._current_time,
        config=env._simulation_config,
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
