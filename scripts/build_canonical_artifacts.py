"""Build the trained artifact bundle required by the canonical benchmark suite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from warehouse_sim.reporting.canonical_artifacts import build_canonical_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the canonical trained artifact bundle.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "canonical_artifacts",
        help="Root output directory for the canonical artifact bundle.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_root = REPO_ROOT / "configs" / "canonical_artifacts"
    written = build_canonical_artifacts(
        repo_root=REPO_ROOT,
        dispatch_corpus_config_path=config_root / "canonical_dispatch_corpus.toml",
        linear_config_path=config_root / "canonical_linear_fit.toml",
        mlp_config_path=config_root / "canonical_mlp_fit.toml",
        graph_config_path=config_root / "canonical_graph_dispatch_fit.toml",
        macro_config_path=config_root / "canonical_macro_ppo_training.toml",
        output_dir=args.output_dir.resolve(),
    )
    print(f"Canonical artifact bundle: {args.output_dir}")
    for label, path in written.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
