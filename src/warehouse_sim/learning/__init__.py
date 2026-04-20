"""Offline and online learning helpers with lazy exports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DispatchModelArtifact": "warehouse_sim.learning.artifacts",
    "load_dispatch_model_artifact": "warehouse_sim.learning.artifacts",
    "write_dispatch_model_artifact": "warehouse_sim.learning.artifacts",
    "DispatchObservationDataset": "warehouse_sim.learning.datasets",
    "load_dispatch_observation_dataset": "warehouse_sim.learning.datasets",
    "OfflineEvaluationResult": "warehouse_sim.learning.evaluation",
    "evaluate_dispatch_model": "warehouse_sim.learning.evaluation",
    "write_offline_evaluation_report": "warehouse_sim.learning.evaluation",
    "DEFAULT_GRAPH_CANDIDATE_FEATURES": "warehouse_sim.learning.graph_data",
    "DEFAULT_GRAPH_EDGE_FEATURES": "warehouse_sim.learning.graph_data",
    "DEFAULT_GRAPH_NODE_FEATURES": "warehouse_sim.learning.graph_data",
    "GraphDispatchDataset": "warehouse_sim.learning.graph_data",
    "GraphDispatchExample": "warehouse_sim.learning.graph_data",
    "build_graph_dispatch_example_from_context": "warehouse_sim.learning.graph_data",
    "load_graph_dispatch_dataset": "warehouse_sim.learning.graph_data",
    "evaluate_graph_dispatch_artifact": "warehouse_sim.learning.graph_evaluation",
    "GraphDispatchFitConfig": "warehouse_sim.learning.graph_fit",
    "fit_graph_dispatch_model": "warehouse_sim.learning.graph_fit",
    "DEFAULT_CANDIDATE_FEATURES": "warehouse_sim.learning.features",
    "LABEL_COLUMN": "warehouse_sim.learning.features",
    "REQUIRED_DISPATCH_COLUMNS": "warehouse_sim.learning.features",
    "SUPPORTED_SPLIT_UNITS": "warehouse_sim.learning.features",
    "candidate_feature_names_from_columns": "warehouse_sim.learning.features",
    "validate_candidate_feature_names": "warehouse_sim.learning.features",
    "GroupedLinearFitConfig": "warehouse_sim.learning.linear_fit",
    "fit_grouped_linear_model": "warehouse_sim.learning.linear_fit",
    "GroupedMLPFitConfig": "warehouse_sim.learning.mlp_fit",
    "fit_grouped_mlp_model": "warehouse_sim.learning.mlp_fit",
    "DatasetSplit": "warehouse_sim.learning.splits",
    "DatasetSplits": "warehouse_sim.learning.splits",
    "SplitConfig": "warehouse_sim.learning.splits",
    "split_dispatch_observation_dataset": "warehouse_sim.learning.splits",
    "DispatchTrainingResult": "warehouse_sim.learning.training",
    "GraphDispatchActorCritic": "warehouse_sim.learning.graph_model",
    "GraphDispatchScorer": "warehouse_sim.learning.graph_model",
    "load_graph_dispatch_model": "warehouse_sim.learning.graph_model",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
