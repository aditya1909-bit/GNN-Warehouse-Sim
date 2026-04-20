"""Helpers for lazily importing optional heavy dependencies."""

from __future__ import annotations

from importlib import import_module


def require_dependency(module_name: str, *, feature: str):
    """Import an optional dependency or raise a feature-scoped error."""

    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = (exc.name or module_name).split(".", maxsplit=1)[0]
        requested = module_name.split(".", maxsplit=1)[0]
        if missing != requested:
            raise
        raise ImportError(
            f"{feature} requires optional dependency '{requested}'. "
            f"Install the learning/geometry dependencies and retry."
        ) from exc
