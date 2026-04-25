"""Lightweight console progress reporting with ETA."""

from __future__ import annotations

import sys
import time


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds == float("inf"):
        return "--:--"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ProgressTracker:
    """Render a single-line progress bar with elapsed time and ETA."""

    def __init__(
        self,
        *,
        label: str,
        total: int,
        unit: str = "step",
        unit_plural: str | None = None,
        width: int = 28,
        min_interval_seconds: float = 0.5,
    ) -> None:
        self.label = label
        self.total = max(1, total)
        self.unit = unit
        self.unit_plural = unit_plural or f"{unit}s"
        self.width = width
        self.min_interval_seconds = min_interval_seconds
        self._start = time.monotonic()
        self._last_render = 0.0
        self._stream = sys.stdout
        self._isatty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._last_line_length = 0

    def update(self, completed: int, *, extra: str | None = None, force: bool = False) -> None:
        now = time.monotonic()
        completed = min(max(0, completed), self.total)
        if not force and completed < self.total and now - self._last_render < self.min_interval_seconds:
            return
        self._last_render = now
        elapsed = now - self._start
        rate = completed / elapsed if elapsed > 0 and completed > 0 else 0.0
        eta_seconds = ((self.total - completed) / rate) if rate > 0 else None
        fraction = completed / self.total
        filled = min(self.width, int(round(fraction * self.width)))
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" | elapsed {_format_seconds(elapsed)} | eta {_format_seconds(eta_seconds)}"
        if extra:
            suffix += f" | {extra}"
        line = (
            f"{self.label}: [{bar}] {completed}/{self.total} "
            f"{self.unit if self.total == 1 else self.unit_plural} ({fraction * 100:5.1f}%)"
            f"{suffix}"
        )
        if self._isatty:
            padded = line
            if len(line) < self._last_line_length:
                padded = line + (" " * (self._last_line_length - len(line)))
            self._stream.write("\r" + padded)
            if completed >= self.total:
                self._stream.write("\n")
            self._stream.flush()
            self._last_line_length = len(line)
        else:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self, *, extra: str | None = None) -> None:
        self.update(self.total, extra=extra, force=True)
