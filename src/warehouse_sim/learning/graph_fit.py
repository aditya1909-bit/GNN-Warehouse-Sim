"""Offline fitting for the PyG graph-conditioned dispatch scorer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import optim

from warehouse_sim.learning.artifacts import DispatchModelArtifact, write_dispatch_model_artifact
from warehouse_sim.learning.graph_data import GraphDispatchDataset
from warehouse_sim.learning.graph_model import GraphDispatchScorer
from warehouse_sim.learning.training import DispatchTrainingResult


@dataclass(frozen=True)
class GraphDispatchFitConfig:
    """Hyperparameters for offline graph-dispatch fitting."""

    node_feature_names: tuple[str, ...]
    edge_feature_names: tuple[str, ...]
    candidate_feature_names: tuple[str, ...]
    hidden_dim: int = 64
    message_passing_layers: int = 2
    dropout: float = 0.0
    batch_size: int = 8
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 10
    seed: int = 0
    benchmark_weighting: bool = False


def fit_graph_dispatch_model(
    train_dataset: GraphDispatchDataset,
    validation_dataset: GraphDispatchDataset,
    *,
    config: GraphDispatchFitConfig,
    output_dir: Path,
    artifact_name: str = "graph_dispatch_model",
) -> DispatchTrainingResult:
    """Fit the graph-conditioned dispatch scorer with grouped softmax loss."""

    if not train_dataset.examples:
        raise ValueError("train_dataset must contain at least one example.")
    torch.manual_seed(config.seed)
    model = GraphDispatchScorer(
        node_dim=len(config.node_feature_names),
        edge_dim=len(config.edge_feature_names),
        candidate_dim=len(config.candidate_feature_names),
        hidden_dim=config.hidden_dim,
        message_passing_layers=config.message_passing_layers,
        dropout=config.dropout,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss = _dataset_loss(
            model,
            train_dataset,
            optimizer=optimizer,
            example_weights=train_dataset.group_weights() if config.benchmark_weighting else None,
        )
        model.eval()
        with torch.no_grad():
            validation_loss = _dataset_loss(
                model,
                validation_dataset,
                optimizer=None,
                example_weights=validation_dataset.group_weights() if config.benchmark_weighting else None,
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
            }
        )
        if validation_loss + 1e-9 < best_loss:
            best_loss = float(validation_loss)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = output_dir / f"{artifact_name}.pt"
    torch.save(model.state_dict(), state_dict_path)
    artifact = DispatchModelArtifact(
        artifact_version=2,
        model_type="pyg_graph_dispatch",
        objective="dispatch_group_softmax_cross_entropy",
        feature_names=config.candidate_feature_names,
        parameters={
            "node_feature_names": list(config.node_feature_names),
            "edge_feature_names": list(config.edge_feature_names),
            "candidate_feature_names": list(config.candidate_feature_names),
            "node_dim": len(config.node_feature_names),
            "edge_dim": len(config.edge_feature_names),
            "candidate_dim": len(config.candidate_feature_names),
            "hidden_dim": config.hidden_dim,
            "message_passing_layers": config.message_passing_layers,
            "dropout": config.dropout,
            "state_dict_path": state_dict_path.name,
        },
        metadata={
            "training": {
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "benchmark_weighting": config.benchmark_weighting,
            }
        },
    )
    write_dispatch_model_artifact(artifact, output_dir / "model_artifact.json")
    return DispatchTrainingResult(
        artifact=artifact,
        training_history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        training_metadata={
            "node_feature_names": list(config.node_feature_names),
            "edge_feature_names": list(config.edge_feature_names),
            "candidate_feature_names": list(config.candidate_feature_names),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
    )


def _dataset_loss(
    model: GraphDispatchScorer,
    dataset: GraphDispatchDataset,
    *,
    optimizer: optim.Optimizer | None,
    example_weights = None,
) -> float:
    if not dataset.examples:
        return 0.0
    total = 0.0
    total_weight = 0.0
    for example_index, example in enumerate(dataset.examples):
        node_features = torch.tensor(example.node_features, dtype=torch.float32)
        edge_index = torch.tensor(example.edge_index, dtype=torch.long)
        edge_features = torch.tensor(example.edge_features, dtype=torch.float32)
        candidate_features = torch.tensor(example.candidate_features, dtype=torch.float32)
        labels = torch.tensor(example.labels, dtype=torch.float32)
        logits, _ = model(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            candidate_features=candidate_features,
        )
        target_index = int(torch.argmax(labels).item())
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([target_index]))
        weight = 1.0 if example_weights is None else float(example_weights[example_index].item())
        weighted_loss = loss * weight
        total += float(weighted_loss.item())
        total_weight += weight
        if optimizer is not None:
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()
    return total / max(total_weight, 1e-12)
