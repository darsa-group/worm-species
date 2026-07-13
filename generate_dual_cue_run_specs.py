from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml


def inclusive_sequence(start: float, stop: float, step: float) -> list[float]:
    start = float(start)
    stop = float(stop)
    step = abs(float(step))
    if step <= 0:
        raise ValueError("step must be greater than zero")

    direction = -1.0 if start > stop else 1.0
    tolerance = step * 1e-6
    values: list[float] = []
    current = start

    if direction < 0:
        while current >= stop - tolerance:
            values.append(round(current, 10))
            current -= step
    else:
        while current <= stop + tolerance:
            values.append(round(current, 10))
            current += step

    if not values or not math.isclose(values[-1], stop, abs_tol=tolerance):
        values.append(stop)
    return values


def slug(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "value"


def format_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return ",".join(str(x) for x in value)
    return str(value)


def generate_conditions(cfg: dict) -> list[dict]:
    matched_cfg = cfg.get("matched_condition_training", {}) or {}
    if not bool(matched_cfg.get("enabled", False)):
        return [
            {
                "condition": "original",
                "feature": "baseline",
                "transform": "original",
                "strength": 0.0,
            }
        ]

    cue_cfg = cfg.get("test_cue_suppression", {}) or {}
    include_original = bool(matched_cfg.get("include_original", True))
    deduplicate = bool(matched_cfg.get("deduplicate_equivalent_conditions", True))
    conditions: list[dict] = []

    if include_original:
        conditions.append(
            {
                "condition": "original",
                "feature": "baseline",
                "transform": "original",
                "strength": 0.0,
            }
        )

    saturation_cfg = cue_cfg.get("saturation", {}) or {}
    if bool(saturation_cfg.get("enabled", True)):
        values = saturation_cfg.get("values")
        if values is None:
            values = inclusive_sequence(
                saturation_cfg.get("start", 1.0),
                saturation_cfg.get("stop", 0.0),
                saturation_cfg.get("step", 0.01),
            )
        for raw_retention in values:
            retention = float(raw_retention)
            if not 0.0 <= retention <= 1.0:
                raise ValueError(f"Saturation retention must be in [0, 1], got {retention}")
            if deduplicate and include_original and math.isclose(retention, 1.0, abs_tol=1e-12):
                continue
            grayscale_enabled = bool((cue_cfg.get("grayscale", {}) or {}).get("enabled", True))
            if deduplicate and grayscale_enabled and math.isclose(retention, 0.0, abs_tol=1e-12):
                continue
            percentage = int(round(retention * 100))
            conditions.append(
                {
                    "condition": f"saturation_{percentage:03d}pct",
                    "feature": "colour",
                    "transform": "saturation",
                    "strength": round(float(1.0 - retention), 10),
                    "retention": retention,
                }
            )

    grayscale_cfg = cue_cfg.get("grayscale", {}) or {}
    if bool(grayscale_cfg.get("enabled", True)):
        conditions.append(
            {
                "condition": "grayscale",
                "feature": "colour",
                "transform": "grayscale",
                "strength": 1.0,
            }
        )

    channel_cfg = cue_cfg.get("channel_shuffle", {}) or {}
    if bool(channel_cfg.get("enabled", True)):
        for order in channel_cfg.get("orders", [[2, 0, 1]]):
            order = [int(i) for i in order]
            if sorted(order) != [0, 1, 2]:
                raise ValueError(f"Invalid RGB channel order: {order}")
            conditions.append(
                {
                    "condition": "channel_shuffle_" + "".join(str(i) for i in order),
                    "feature": "colour",
                    "transform": "channel_shuffle",
                    "strength": 1.0,
                    "order": order,
                }
            )

    bilateral_cfg = cue_cfg.get("bilateral_filter", {}) or {}
    if bool(bilateral_cfg.get("enabled", True)):
        settings = bilateral_cfg.get(
            "settings",
            [
                {"diameter": 5, "sigma_colour": 25, "sigma_space": 25},
                {"diameter": 7, "sigma_colour": 50, "sigma_space": 50},
                {"diameter": 9, "sigma_colour": 100, "sigma_space": 100},
            ],
        )
        for setting in settings:
            diameter = int(setting["diameter"])
            sigma_colour = float(setting["sigma_colour"])
            sigma_space = float(setting["sigma_space"])
            conditions.append(
                {
                    "condition": f"bilateral_d{diameter}_c{sigma_colour:g}_s{sigma_space:g}",
                    "feature": "texture",
                    "transform": "bilateral_filter",
                    "strength": sigma_colour,
                    "diameter": diameter,
                    "sigma_colour": sigma_colour,
                    "sigma_space": sigma_space,
                }
            )

    gaussian_cfg = cue_cfg.get("gaussian_blur", {}) or {}
    if bool(gaussian_cfg.get("enabled", True)):
        for raw_sigma in gaussian_cfg.get("sigmas", [0.5, 1.0, 2.0, 4.0]):
            sigma = float(raw_sigma)
            conditions.append(
                {
                    "condition": f"gaussian_sigma_{sigma:g}",
                    "feature": "texture",
                    "transform": "gaussian_blur",
                    "strength": sigma,
                    "sigma": sigma,
                }
            )

    patch_cfg = cue_cfg.get("patch_shuffle", {}) or {}
    if bool(patch_cfg.get("enabled", True)):
        seed = int(patch_cfg.get("seed", cfg.get("seed", 0)))
        for raw_grid_size in patch_cfg.get("grid_sizes", [2, 4, 8]):
            grid_size = int(raw_grid_size)
            conditions.append(
                {
                    "condition": f"patch_shuffle_grid_{grid_size}",
                    "feature": "shape",
                    "transform": "patch_shuffle",
                    "strength": grid_size,
                    "grid_size": grid_size,
                    "seed": seed,
                }
            )

    requested_names = matched_cfg.get("condition_names")
    if requested_names:
        requested = {str(name) for name in requested_names}
        conditions = [c for c in conditions if c["condition"] in requested]
        missing = requested - {c["condition"] for c in conditions}
        if missing:
            raise ValueError(f"Unknown matched training condition names: {sorted(missing)}")

    names = [c["condition"] for c in conditions]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate matched training conditions: {duplicates}")
    if not conditions:
        raise ValueError("No matched training conditions were generated")
    return conditions


def sweep_combinations(cfg: dict) -> list[dict[str, Any]]:
    sweep_cfg = cfg.get("sweep", {}) or {}
    if not bool(sweep_cfg.get("enabled", False)):
        return [{}]

    params = sweep_cfg.get("parameters", {}) or {}
    if not isinstance(params, dict):
        raise TypeError("sweep.parameters must be a dictionary")
    if not params:
        return [{}]

    keys = list(params)
    value_lists: list[list[Any]] = []
    for key in keys:
        values = params[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.parameters.{key} must be a non-empty list")
        value_lists.append(values)

    return [dict(zip(keys, values)) for values in itertools.product(*value_lists)]


def condition_overrides(condition: dict) -> list[str]:
    lines = [
        "input_condition.enabled=true",
        f"input_condition.condition={format_override(condition['condition'])}",
        f"input_condition.feature={format_override(condition['feature'])}",
        f"input_condition.transform={format_override(condition['transform'])}",
        f"input_condition.strength={format_override(condition.get('strength', 0.0))}",
    ]
    for key in (
        "retention",
        "order",
        "diameter",
        "sigma_colour",
        "sigma_space",
        "sigma",
        "grid_size",
        "seed",
    ):
        if key in condition:
            lines.append(f"input_condition.{key}={format_override(condition[key])}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("run_specs_dir")
    parser.add_argument("sweep_plan")
    args = parser.parse_args()

    config_path = Path(args.config)
    run_specs_dir = Path(args.run_specs_dir)
    sweep_plan_path = Path(args.sweep_plan)

    cfg = yaml.safe_load(config_path.read_text()) or {}
    conditions = generate_conditions(cfg)
    combinations = sweep_combinations(cfg)
    matched_cfg = cfg.get("matched_condition_training", {}) or {}
    evaluate_rgb_all = bool(
        matched_cfg.get("evaluate_original_model_on_all_test_conditions", True)
    )

    run_specs_dir.mkdir(parents=True, exist_ok=True)
    for old in run_specs_dir.glob("run_*.args"):
        old.unlink()

    plan_lines = [
        "run_index\tarray_name\tmodel\ttrain_condition\ttrain_transform\toverrides"
    ]
    run_index = 0

    for combo in combinations:
        model_name = str(combo.get("model.name", cfg.get("model", {}).get("name", "model")))
        for condition in conditions:
            array_name = f"run_{run_index:03d}"
            override_lines = [
                f"{key}={format_override(value)}" for key, value in combo.items()
            ]
            override_lines.extend(condition_overrides(condition))
            cue_enabled = evaluate_rgb_all and condition["transform"] == "original"
            override_lines.append(
                f"test_cue_suppression.enabled={'true' if cue_enabled else 'false'}"
            )
            override_lines.append("matched_condition_training.enabled=false")

            spec_path = run_specs_dir / f"{array_name}.args"
            spec_path.write_text("\n".join(override_lines) + "\n")
            plan_lines.append(
                "\t".join(
                    [
                        str(run_index),
                        array_name,
                        model_name,
                        condition["condition"],
                        condition["transform"],
                        " ".join(override_lines),
                    ]
                )
            )
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
    print(run_index)


if __name__ == "__main__":
    main()
