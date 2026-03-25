"""Analyze canonical benchmark claim tables into one headline results bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from warehouse_sim.reporting.canonical_suite import analyze_canonical_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Combine canonical benchmark outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite-name", type=str, default="canonical_full_matrix")
    parser.add_argument("--dispatch-claims", type=Path, required=True)
    parser.add_argument("--integrated-claims", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = analyze_canonical_suite(
        output_dir=args.output_dir,
        suite_name=args.suite_name,
        dispatch_claims_path=args.dispatch_claims,
        integrated_claims_path=args.integrated_claims,
        config_path=args.config,
    )
    print(f"Canonical analysis: {args.suite_name}")
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
