"""Run the canonical dispatch and integrated benchmark suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from warehouse_sim.reporting.canonical_suite import run_canonical_suite_from_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the canonical warehouse benchmark suite.")
    parser.add_argument("--config", type=Path, required=True, help="Canonical suite TOML config.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = run_canonical_suite_from_path(args.config)
    print(f"Canonical suite: {args.config}")
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
