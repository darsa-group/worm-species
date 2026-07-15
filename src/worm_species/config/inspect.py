from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .loading import load_config
from .overrides import apply_overrides
from .validation import (
    ConfigValidationError,
    resolve_workflow,
    validate_config,
    validate_override_items,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a resolved experiment configuration without running it."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Apply existing dotted key=value overrides before inspection.",
    )
    parser.add_argument(
        "--workflow",
        choices=("auto", "training", "run_specs", "saved"),
        default="auto",
    )
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("--format", choices=("yaml", "json"), default="yaml")
    return parser


def identify_experiment_type(config: dict[str, Any]) -> str:
    input_condition = config.get("input_condition", {}) or {}
    if isinstance(input_condition, dict) and bool(input_condition.get("enabled", False)):
        transform = str(input_condition.get("transform", "original")).lower()
        return "matched_condition_execution" if transform != "original" else "rgb_training_execution"
    matched = config.get("matched_condition_training", {}) or {}
    if isinstance(matched, dict) and bool(matched.get("enabled", False)):
        return "matched_condition_plan"
    colour = config.get("colour_ablation", {}) or {}
    if isinstance(colour, dict) and bool(colour.get("enabled", False)):
        return "colour_ablation"
    cue = config.get("test_cue_suppression", {}) or {}
    if isinstance(cue, dict) and bool(cue.get("enabled", False)):
        return "fixed_rgb_stress_evaluation"
    return "ordinary_training"


def inspection_summary(config: dict[str, Any], workflow: str) -> dict[str, Any]:
    # Lazy imports keep config loading and --help lightweight.
    from ..experiments.conditions import generate_conditions, sweep_combinations

    combinations = sweep_combinations(config)
    matched = config.get("matched_condition_training", {}) or {}
    if isinstance(matched, dict) and bool(matched.get("enabled", False)):
        conditions = generate_conditions(config)
    else:
        conditions = [{"condition": "original", "transform": "original"}]

    base_model = str((config.get("model", {}) or {}).get("name", "model"))
    models = {
        str(combination.get("model.name", base_model))
        for combination in combinations
    }
    return {
        "experiment_type": identify_experiment_type(config),
        "workflow": workflow,
        "expected_model_count": len(models),
        "expected_sweep_combination_count": len(combinations),
        "expected_condition_count": len(conditions),
        "expected_total_run_count": len(combinations) * len(conditions),
        "models": sorted(models),
        "condition_names": [condition["condition"] for condition in conditions],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_override_items(args.override)
        config = apply_overrides(load_config(args.config), args.override)
        workflow = resolve_workflow(config, args.workflow)
        validate_config(
            config,
            workflow=args.workflow,
            check_paths=args.check_paths,
            check_model_registry=True,
        )
        summary = inspection_summary(config, workflow)
    except (ConfigValidationError, OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    payload = {
        "summary": summary,
        "applied_overrides": list(args.override),
        "resolved_config": config,
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(yaml.safe_dump(payload, sort_keys=False).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
