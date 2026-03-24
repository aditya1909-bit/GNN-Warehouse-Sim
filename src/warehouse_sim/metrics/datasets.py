"""Dataset export utilities for future learned-policy experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from warehouse_sim.environment import WarehouseEnvironment
from warehouse_sim.graph import build_graph_features

if TYPE_CHECKING:
    from warehouse_sim.simulation.models import SimulationResult


def write_observation_dataset(
    output_dir: Path,
    environment: WarehouseEnvironment,
    result: "SimulationResult",
    experiment_name: str,
) -> dict[str, Path]:
    """Write graph features and dispatch traces for policy-learning experiments."""

    output_dir.mkdir(parents=True, exist_ok=True)

    graph_features = build_graph_features(environment.graph, zone_lookup=environment.zone_for_node)
    nodes_path = output_dir / "graph_nodes.csv"
    arcs_path = output_dir / "graph_arcs.csv"
    dispatch_path = output_dir / "dispatch_observations.csv"
    manifest_path = output_dir / "dataset_manifest.json"

    _write_csv(nodes_path, [asdict(node) for node in graph_features.nodes])
    _write_csv(arcs_path, [asdict(arc) for arc in graph_features.arcs])
    _write_csv(dispatch_path, [asdict(record) for record in result.dispatch_traces])

    manifest = {
        "experiment_name": experiment_name,
        "policy_name": result.policy_name,
        "dispatch_events": len({record.dispatch_index for record in result.dispatch_traces}),
        "candidate_rows": len(result.dispatch_traces),
        "files": {
            "graph_nodes": nodes_path.name,
            "graph_arcs": arcs_path.name,
            "dispatch_observations": dispatch_path.name,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "dataset_manifest": manifest_path,
        "graph_nodes": nodes_path,
        "graph_arcs": arcs_path,
        "dispatch_observations": dispatch_path,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
