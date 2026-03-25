"""Tests for reproducible artifact manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path

from warehouse_sim.reporting.artifact_manifest import (
    write_artifact_manifest,
    write_config_snapshot,
    write_seed_bundle,
)


def test_write_manifest_and_snapshots(tmp_path: Path) -> None:
    config_snapshot = write_config_snapshot(
        tmp_path / "config_snapshot.toml",
        {"configs/example.toml": "[benchmark]\nname = 'example'\n"},
    )
    seed_bundle = write_seed_bundle(
        tmp_path / "seed_bundle.json",
        {"scenario_seeds": {"example": [7, 11]}},
    )
    manifest = write_artifact_manifest(
        tmp_path / "manifest.json",
        benchmark_name="example",
        generated_paths={"summary": tmp_path / "benchmark_summary.csv"},
        config_snapshot_path=config_snapshot,
        seed_bundle_path=seed_bundle,
        extra_metadata={"metric_schema_version": "1.0"},
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["benchmark_name"] == "example"
    assert payload["config_snapshot_path"] == str(config_snapshot)
    assert payload["seed_bundle_path"] == str(seed_bundle)
    assert payload["metadata"]["metric_schema_version"] == "1.0"
