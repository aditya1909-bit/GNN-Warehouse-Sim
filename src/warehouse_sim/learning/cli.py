"""CLI and orchestration for offline dispatch policy fitting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from warehouse_sim.config import OfflineTrainingConfig, load_offline_training_config
from warehouse_sim.learning.artifacts import load_dispatch_model_artifact, write_dispatch_model_artifact
from warehouse_sim.learning.datasets import load_dispatch_observation_dataset
from warehouse_sim.learning.evaluation import evaluate_dispatch_model, write_offline_evaluation_report
from warehouse_sim.learning.linear_fit import GroupedLinearFitConfig, fit_grouped_linear_model
from warehouse_sim.learning.mlp_fit import GroupedMLPFitConfig, fit_grouped_mlp_model
from warehouse_sim.learning.splits import SplitConfig, split_dispatch_observation_dataset


def run_offline_training_from_config(config: OfflineTrainingConfig) -> dict[str, Path]:
    """Fit an offline scorer and write reproducible training artifacts."""

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
            ),
        )

    output_dir = config.reporting.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the offline fitting CLI."""

    args = build_parser().parse_args(argv)
    if args.command == "train":
        written = run_offline_training_from_path(args.config)
        print(f"Offline training config: {args.config}")
    else:
        written = run_offline_evaluation(
            artifact_path=args.artifact,
            dataset_source=args.dataset,
            output_dir=args.output_dir,
            split_name=args.split_name,
        )
        print(f"Artifact: {args.artifact}")

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
