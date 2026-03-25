"""PyG graph-conditioned dispatch scorer models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from warehouse_sim.learning.artifacts import DispatchModelArtifact, load_dispatch_model_artifact


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class GraphDispatchScorer(nn.Module):
    """Global-graph embedding scorer over dispatch candidates."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        candidate_dim: int,
        hidden_dim: int = 64,
        message_passing_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_encoder = _mlp(node_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.edge_encoder = _mlp(edge_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.convs = nn.ModuleList(
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
                for _ in range(message_passing_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.candidate_head = nn.Sequential(
            nn.Linear(hidden_dim + candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def encode_graph(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_encoder(node_features)
        edge_attr = self.edge_encoder(edge_features)
        for conv in self.convs:
            x = torch.relu(conv(x, edge_index, edge_attr))
            x = self.dropout(x)
        return x.mean(dim=0)

    def score_candidates(
        self,
        graph_embedding: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        repeated_graph = graph_embedding.unsqueeze(0).expand(candidate_features.shape[0], -1)
        logits = self.candidate_head(torch.cat([repeated_graph, candidate_features], dim=-1))
        return logits.squeeze(-1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_embedding = self.encode_graph(node_features, edge_index, edge_features)
        logits = self.score_candidates(graph_embedding, candidate_features)
        return logits, graph_embedding


class GraphDispatchActorCritic(nn.Module):
    """Masked PPO actor-critic using the graph dispatch scorer backbone."""

    def __init__(self, scorer: GraphDispatchScorer, hidden_dim: int = 64) -> None:
        super().__init__()
        self.scorer = scorer
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_actor(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, graph_embedding = self.scorer(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            candidate_features=candidate_features,
        )
        return logits, graph_embedding

    def forward_value(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        return self.value_head(graph_embedding).squeeze(-1)


@dataclass(frozen=True)
class LoadedGraphDispatchModel:
    artifact: DispatchModelArtifact
    model: GraphDispatchScorer


def load_graph_dispatch_model(artifact_path: Path, device: torch.device | str = "cpu") -> LoadedGraphDispatchModel:
    """Load a graph dispatch scorer and weights from an artifact manifest."""

    artifact = load_dispatch_model_artifact(artifact_path)
    if artifact.model_type != "pyg_graph_dispatch":
        raise ValueError(f"Expected pyg_graph_dispatch artifact, got {artifact.model_type}")

    parameters = artifact.parameters
    model = GraphDispatchScorer(
        node_dim=int(parameters["node_dim"]),
        edge_dim=int(parameters["edge_dim"]),
        candidate_dim=int(parameters["candidate_dim"]),
        hidden_dim=int(parameters["hidden_dim"]),
        message_passing_layers=int(parameters["message_passing_layers"]),
        dropout=float(parameters["dropout"]),
    )
    state_path = artifact_path.parent / str(parameters["state_dict_path"])
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedGraphDispatchModel(artifact=artifact, model=model)
