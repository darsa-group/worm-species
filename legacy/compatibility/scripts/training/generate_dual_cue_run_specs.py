"""Compatibility wrapper for dual-cue condition and run-spec generation."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.worm_species.experiments.conditions import (
    condition_overrides,
    format_override,
    generate_conditions,
    inclusive_sequence,
    slug,
    sweep_combinations,
)
from src.worm_species.experiments.run_specs import main, write_run_specs

__all__ = [
    "condition_overrides",
    "format_override",
    "generate_conditions",
    "inclusive_sequence",
    "main",
    "slug",
    "sweep_combinations",
    "write_run_specs",
]


if __name__ == "__main__":
    main()
