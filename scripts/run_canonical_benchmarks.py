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
    parser.add_argument("--parallel-workers", type=str, default=None, help='Worker count or "auto".')
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--use-mps-for-learned-policies", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    written = run_canonical_suite_from_path(
        args.config,
        parallel_workers_override=None if args.parallel_workers in {None, "auto"} else int(args.parallel_workers),
        resume_override=args.resume,
        fail_fast_override=args.fail_fast or None,
        use_mps_for_learned_policies_override=args.use_mps_for_learned_policies or None,
    )
    print(f"Canonical suite: {args.config}")
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
