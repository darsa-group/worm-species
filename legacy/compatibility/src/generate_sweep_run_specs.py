#!/usr/bin/env python3
"""Compatibility wrapper for the historical ordinary sweep generator."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from legacy.python.experiments.generate_sweep_run_specs import *  # noqa: F401,F403,E402
from legacy.python.experiments.generate_sweep_run_specs import main  # noqa: E402


if __name__ == "__main__":
    main()
