"""CLI smoke tests for the legacy-compatible demand generator entrypoint."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def test_legacy_script_cli_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "task_demand.csv"
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "generate_task_demand.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--output",
            str(output_path),
            "--min-tasks",
            "0",
            "--seed",
            "7",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()
    with output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    assert rows[0] == ["Task_ID", "Timestamp", "Interarrival_Time", "Regime"]
    assert len(rows) > 1
    assert "Wrote" in completed.stdout

