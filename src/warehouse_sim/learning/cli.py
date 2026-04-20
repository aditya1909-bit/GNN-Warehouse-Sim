"""CLI and orchestration for offline dispatch policy fitting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from warehouse_sim.config import (
    load_integrated_rl_training_config,
    OfflineTrainingConfig,
    load_offline_training_config,
    load_rl_fine_tuning_config,
)
from warehouse_sim.learning.artifacts import load_dispatch_model_artifact, write_dispatch_model_artifact
from warehouse_sim.learning.datasets import load_dispatch_observation_dataset
from warehouse_sim.learning.evaluation import evaluate_dispatch_model, write_offline_evaluation_report
from warehouse_sim.learning.graph_data import (
    DEFAULT_GRAPH_CANDIDATE_FEATURES,
    DEFAULT_GRAPH_EDGE_FEATURES,
    DEFAULT_GRAPH_NODE_FEATURES,
    load_graph_dispatch_dataset,
)
from warehouse_sim.learning.graph_evaluation import evaluate_graph_dispatch_artifact
from warehouse_sim.learning.graph_fit import GraphDispatchFitConfig, fit_graph_dispatch_model
from warehouse_sim.learning.linear_fit import GroupedLinearFitConfig, fit_grouped_linear_model
from warehouse_sim.learning.mlp_fit import GroupedMLPFitConfig, fit_grouped_mlp_model
from warehouse_sim.learning.splits import SplitConfig, split_dispatch_observation_dataset
from warehouse_sim.utils.dependencies import require_dependency


def run_offline_training_from_config(config: OfflineTrainingConfig) -> dict[str, Path]:
    """Fit an offline scorer and write reproducible training artifacts."""

    output_dir = config.reporting.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.model.type == "graph_dispatch":
        dataset = load_graph_dispatch_dataset(
            config.dataset.source,
            candidate_feature_names=config.dataset.feature_names or DEFAULT_GRAPH_CANDIDATE_FEATURES,
            node_feature_names=config.dataset.node_feature_names or DEFAULT_GRAPH_NODE_FEATURES,
            edge_feature_names=config.dataset.edge_feature_names or DEFAULT_GRAPH_EDGE_FEATURES,
        )
        splits = split_dispatch_observation_dataset(
            dataset,
            SplitConfig(
                train_fraction=config.split.train_fraction,
                validation_fraction=config.split.validation_fraction,
                test_fraction=config.split.test_fraction,
                split_unit=config.split.split_unit,
                seed=config.seed,
            ),
        )
        training_result = fit_graph_dispatch_model(
            train_dataset=splits.train.dataset,
            validation_dataset=splits.validation.dataset if splits.validation.row_count else splits.train.dataset,
            config=GraphDispatchFitConfig(
                node_feature_names=tuple(config.dataset.node_feature_names or DEFAULT_GRAPH_NODE_FEATURES),
                edge_feature_names=tuple(config.dataset.edge_feature_names or DEFAULT_GRAPH_EDGE_FEATURES),
                candidate_feature_names=tuple(config.dataset.feature_names or DEFAULT_GRAPH_CANDIDATE_FEATURES),
                hidden_dim=config.model.hidden_dim,
                message_passing_layers=config.model.message_passing_layers,
                dropout=config.model.dropout,
                batch_size=config.model.batch_size,
                learning_rate=config.model.learning_rate,
                max_epochs=config.model.max_epochs,
                patience=config.model.patience,
                seed=config.seed,
                benchmark_weighting=config.model.benchmark_weighting,
            ),
            output_dir=output_dir,
        )
        artifact_path = output_dir / "model_artifact.json"
        history_path = _write_training_history(output_dir / "training_history.csv", training_result.training_history)
        written_paths: dict[str, Path] = {
            "artifact": artifact_path,
            "state_dict": output_dir / "graph_dispatch_model.pt",
            "training_history": history_path,
        }
        evaluation_summaries: dict[str, dict[str, float | int | None]] = {}
        split_summaries = {}
        for split in (splits.train, splits.validation, splits.test):
            evaluation = evaluate_graph_dispatch_artifact(split.dataset, artifact_path)
            written_paths.update(write_offline_evaluation_report(output_dir, split.name, evaluation))
            evaluation_summaries[split.name] = evaluation.metrics
            split_summaries[split.name] = {
                "rows": split.row_count,
                "dispatch_groups": split.dataset.group_count,
                "split_units": list(split.split_units),
            }
        summary_path = output_dir / "training_summary.json"
        summary_payload = {
            "name": config.name,
            "seed": config.seed,
            "dataset_source": str(config.dataset.source),
            "graph_node_feature_names": list(config.dataset.node_feature_names or DEFAULT_GRAPH_NODE_FEATURES),
            "graph_edge_feature_names": list(config.dataset.edge_feature_names or DEFAULT_GRAPH_EDGE_FEATURES),
            "candidate_feature_names": list(config.dataset.feature_names or DEFAULT_GRAPH_CANDIDATE_FEATURES),
            "split": {
                "split_unit": config.split.split_unit,
                "train_fraction": config.split.train_fraction,
                "validation_fraction": config.split.validation_fraction,
                "test_fraction": config.split.test_fraction,
                "splits": split_summaries,
            },
            "model": {
                "type": config.model.type,
                "best_epoch": training_result.best_epoch,
                "best_validation_loss": training_result.best_validation_loss,
                "metadata": training_result.training_metadata,
                "benchmark_weighting": config.model.benchmark_weighting,
            },
            "evaluations": evaluation_summaries,
            "artifact_path": str(artifact_path),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        written_paths["training_summary"] = summary_path
        return written_paths

    dataset = load_dispatch_observation_dataset(
        config.dataset.source,
        feature_names=config.dataset.feature_names,
    )
    splits = split_dispatch_observation_dataset(
        dataset,
        SplitConfig(
            train_fraction=config.split.train_fraction,
            validation_fraction=config.split.validation_fraction,
            test_fraction=config.split.test_fraction,
            split_unit=config.split.split_unit,
            seed=config.seed,
        ),
    )

    if config.model.type == "linear":
        training_result = fit_grouped_linear_model(
            train_split=splits.train,
            validation_split=splits.validation if splits.validation.row_count else splits.train,
            config=GroupedLinearFitConfig(
                learning_rate=config.model.learning_rate,
                max_epochs=config.model.max_epochs,
                l2_regularization=config.model.l2_regularization,
                patience=config.model.patience,
                benchmark_weighting=config.model.benchmark_weighting,
            ),
        )
    else:
        training_result = fit_grouped_mlp_model(
            train_split=splits.train,
            validation_split=splits.validation if splits.validation.row_count else splits.train,
            config=GroupedMLPFitConfig(
                hidden_dim=config.model.hidden_dim,
                learning_rate=config.model.learning_rate,
                max_epochs=config.model.max_epochs,
                l2_regularization=config.model.l2_regularization,
                patience=config.model.patience,
                seed=config.seed,
                benchmark_weighting=config.model.benchmark_weighting,
            ),
        )

    artifact_path = write_dispatch_model_artifact(
        training_result.artifact,
        output_dir / "model_artifact.json",
    )
    history_path = _write_training_history(output_dir / "training_history.csv", training_result.training_history)

    written_paths: dict[str, Path] = {
        "artifact": artifact_path,
        "training_history": history_path,
    }
    evaluation_summaries: dict[str, dict[str, float | int | None]] = {}
    split_summaries = {}

    for split in (splits.train, splits.validation, splits.test):
        evaluation = evaluate_dispatch_model(split.dataset, training_result.artifact)
        written_paths.update(write_offline_evaluation_report(output_dir, split.name, evaluation))
        evaluation_summaries[split.name] = evaluation.metrics
        split_summaries[split.name] = {
            "rows": split.row_count,
            "dispatch_groups": split.dataset.group_count,
            "split_units": list(split.split_units),
        }

    summary_path = output_dir / "training_summary.json"
    summary_payload = {
        "name": config.name,
        "seed": config.seed,
        "dataset_source": str(config.dataset.source),
        "feature_names": list(dataset.feature_names),
        "split": {
            "split_unit": config.split.split_unit,
            "train_fraction": config.split.train_fraction,
            "validation_fraction": config.split.validation_fraction,
            "test_fraction": config.split.test_fraction,
            "splits": split_summaries,
        },
        "model": {
            "type": config.model.type,
            "best_epoch": training_result.best_epoch,
            "best_validation_loss": training_result.best_validation_loss,
            "metadata": training_result.training_metadata,
            "benchmark_weighting": config.model.benchmark_weighting,
        },
        "evaluations": evaluation_summaries,
        "artifact_path": str(artifact_path),
        "artifact_metadata": training_result.artifact.metadata,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    written_paths["training_summary"] = summary_path
    return written_paths


def run_offline_training_from_path(config_path: Path) -> dict[str, Path]:
    """Load an offline fitting config and execute it."""

    config = load_offline_training_config(config_path)
    return run_offline_training_from_config(config)


def run_offline_evaluation(
    artifact_path: Path,
    dataset_source: Path,
    output_dir: Path,
    split_name: str = "evaluation",
) -> dict[str, Path]:
    """Evaluate an existing artifact against a dataset source."""

    artifact = load_dispatch_model_artifact(artifact_path)
    if artifact.model_type == "pyg_graph_dispatch":
        dataset = load_graph_dispatch_dataset(
            dataset_source,
            candidate_feature_names=artifact.parameters["candidate_feature_names"],
            node_feature_names=artifact.parameters["node_feature_names"],
            edge_feature_names=artifact.parameters["edge_feature_names"],
        )
        evaluation = evaluate_graph_dispatch_artifact(dataset, artifact_path)
    else:
        dataset = load_dispatch_observation_dataset(dataset_source, feature_names=artifact.feature_names)
        evaluation = evaluate_dispatch_model(dataset, artifact)
    return write_offline_evaluation_report(output_dir, split_name, evaluation)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for offline policy fitting and evaluation."""

    parser = argparse.ArgumentParser(description="Offline dispatch-policy fitting and evaluation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fit a model from an offline training config.")
    train_parser.add_argument("--config", type=Path, required=True, help="Offline training TOML config.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate an existing artifact on a dataset.")
    evaluate_parser.add_argument("--artifact", type=Path, required=True)
    evaluate_parser.add_argument("--dataset", type=Path, required=True)
    evaluate_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser.add_argument("--split-name", default="evaluation")

    rl_parser = subparsers.add_parser("train-rl", help="Fine-tune a graph dispatch model with masked PPO.")
    rl_parser.add_argument("--config", type=Path, required=True, help="RL fine-tuning TOML config.")

    integrated_rl_parser = subparsers.add_parser(
        "train-integrated-rl",
        help="Train an integrated end-to-end macro PPO controller.",
    )
    integrated_rl_parser.add_argument("--config", type=Path, required=True, help="Integrated PPO TOML config.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the offline fitting CLI."""

    args = build_parser().parse_args(argv)
    if args.command == "train":
        written = run_offline_training_from_path(args.config)
        print(f"Offline training config: {args.config}")
    elif args.command == "evaluate":
        written = run_offline_evaluation(
            artifact_path=args.artifact,
            dataset_source=args.dataset,
            output_dir=args.output_dir,
            split_name=args.split_name,
        )
        print(f"Artifact: {args.artifact}")
    elif args.command == "train-rl":
        require_dependency("gymnasium", feature="Dispatch RL fine-tuning")
        require_dependency("torch", feature="Dispatch RL fine-tuning")
        require_dependency("torch_geometric", feature="Dispatch RL fine-tuning")
        from warehouse_sim.learning.rl import run_rl_fine_tuning_from_config

        written = run_rl_fine_tuning_from_config(load_rl_fine_tuning_config(args.config))
        print(f"RL fine-tuning config: {args.config}")
    else:
        require_dependency("torch", feature="Integrated RL training")
        require_dependency("torch_geometric", feature="Integrated RL training")
        from warehouse_sim.learning.integrated_rl import run_integrated_rl_training_from_config

        written = run_integrated_rl_training_from_config(load_integrated_rl_training_config(args.config))
        print(f"Integrated RL training config: {args.config}")

    for label, path in written.items():
        print(f"{label}: {path}")


def _write_training_history(path: Path, history: tuple[dict[str, float], ...]) -> Path:
    if not history:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    return path


if __name__ == "__main__":
    main()
