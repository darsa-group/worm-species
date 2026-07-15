from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config.loading import load_config
from ..config.validation import validate_config
from .conditions import condition_overrides, format_override, generate_conditions, sweep_combinations


def write_run_specs(config_path: Path, run_specs_dir: Path, sweep_plan_path: Path) -> int:
    config = load_config(config_path)
    # Validate before creating directories or removing stale specifications.
    # Run-spec generation intentionally does not require local data paths or a
    # torchvision import: it is a dry-run/cluster-submission workflow.
    validate_config(
        config,
        workflow="run_specs",
        check_paths=False,
        check_model_registry=False,
    )
    conditions = generate_conditions(config)
    combinations = sweep_combinations(config)
    matched_config = config.get("matched_condition_training", {}) or {}
    evaluate_rgb_all = bool(
        matched_config.get("evaluate_original_model_on_all_test_conditions", True)
    )

    run_specs_dir.mkdir(parents=True, exist_ok=True)
    for old in run_specs_dir.glob("run_*.args"):
        old.unlink()

    plan_lines = [
        "run_index\tarray_name\tmodel\ttrain_condition\ttrain_transform\toverrides"
    ]
    run_index = 0
    for combination in combinations:
        model_name = str(combination.get("model.name", config.get("model", {}).get("name", "model")))
        for condition in conditions:
            array_name = f"run_{run_index:03d}"
            override_lines = [
                f"{key}={format_override(value)}" for key, value in combination.items()
            ]
            override_lines.extend(condition_overrides(condition))
            cue_enabled = evaluate_rgb_all and condition["transform"] == "original"
            override_lines.append(
                f"test_cue_suppression.enabled={'true' if cue_enabled else 'false'}"
            )
            override_lines.append("matched_condition_training.enabled=false")
            (run_specs_dir / f"{array_name}.args").write_text(
                "\n".join(override_lines) + "\n"
            )
            plan_lines.append("\t".join([
                str(run_index), array_name, model_name, condition["condition"],
                condition["transform"], " ".join(override_lines),
            ]))
            run_index += 1

    sweep_plan_path.write_text("\n".join(plan_lines) + "\n")
    metadata = {
        "n_sweep_combinations": len(combinations),
        "n_unique_training_conditions": len(conditions),
        "n_total_runs": run_index,
        "conditions": conditions,
        "sweep_combinations": combinations,
    }
    (sweep_plan_path.parent / "dual_cue_experiment_plan.json").write_text(
        json.dumps(metadata, indent=2)
    )
    return run_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("run_specs_dir")
    parser.add_argument("sweep_plan")
    args = parser.parse_args()
    count = write_run_specs(
        Path(args.config), Path(args.run_specs_dir), Path(args.sweep_plan)
    )
    print(count)
