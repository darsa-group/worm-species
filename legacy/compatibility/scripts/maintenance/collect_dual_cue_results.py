"""Compatibility wrapper for dual-cue result collection."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.worm_species.experiments.result_collection import (
    add_equivalent_condition_aliases,
    build_comparison,
    collect_nested_csv,
    collect_results,
    main,
    matched_results_long,
    read_json,
)

__all__ = [
    "add_equivalent_condition_aliases",
    "build_comparison",
    "collect_nested_csv",
    "collect_results",
    "main",
    "matched_results_long",
    "read_json",
]


if __name__ == "__main__":
    main()
