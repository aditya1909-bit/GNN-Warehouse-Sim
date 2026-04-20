"""Artifact snapshot and manifest writers for reproducible experiments."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, UTC
from pathlib import Path
from typing import Mapping


def write_seed_bundle(output_path: Path, payload: Mapping[str, object]) -> Path:
    """Write a machine-readable bundle of benchmark seeds."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    return output_path


def write_config_snapshot(output_path: Path, config_sources: Mapping[str, str]) -> Path:
    """Write a concatenated config snapshot used to regenerate an artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for source_name, contents in config_sources.items():
        if lines:
            lines.append("\n")
        lines.append(f"# --- {source_name} ---\n")
        stripped = contents.rstrip()
        lines.append(f"{stripped}\n" if stripped else "\n")
    output_path.write_text("".join(lines), encoding="utf-8")
    return output_path


def write_artifact_manifest(
    output_path: Path,
    *,
    benchmark_name: str,
    generated_paths: Mapping[str, Path],
    config_snapshot_path: Path | None = None,
    seed_bundle_path: Path | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> Path:
    """Write a reproducibility manifest for a benchmark or report artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "benchmark_name": benchmark_name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit_hash": _git_commit_hash(output_path.parent),
        "generated_paths": {
            label: str(path)
            for label, path in sorted(generated_paths.items())
        },
    }
    if config_snapshot_path is not None:
        payload["config_snapshot_path"] = str(config_snapshot_path)
    if seed_bundle_path is not None:
        payload["seed_bundle_path"] = str(seed_bundle_path)
    if extra_metadata:
        payload["metadata"] = dict(extra_metadata)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_artifact_aliases(path: Path) -> dict[str, Path]:
    """Load named artifact aliases from a manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    aliases = metadata.get("artifact_aliases", {})
    if not isinstance(aliases, dict):
        raise ValueError(f"Artifact manifest {path} does not contain a valid artifact_aliases mapping.")
    return {
        str(alias): (path.parent / Path(str(relative_path))).resolve()
        for alias, relative_path in aliases.items()
    }


def _git_commit_hash(start_dir: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start_dir,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None
