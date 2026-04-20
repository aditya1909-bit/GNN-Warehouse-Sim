"""Tests for the documented install and dependency contract."""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_declares_rl_runtime_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = set(payload["project"]["dependencies"])

    assert "gymnasium>=1.2" in dependencies
    assert "torch>=2.11" in dependencies
    assert "torch_geometric>=2.7" in dependencies


def test_readme_documents_supported_install_path() -> None:
    readme_path = Path(__file__).resolve().parents[1] / "README.md"
    readme = readme_path.read_text(encoding="utf-8")

    assert "python3 -m pip install -e .[dev]" in readme
    assert "python3 -m pytest -q" in readme
