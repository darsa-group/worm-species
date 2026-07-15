"""Canonical and byte-compatible legacy command-line entry points."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ..config.loading import load_config
from ..config.overrides import apply_overrides
from ..config.sweeps import generate_sweep_configs
from .modes import PROFILES
from .modes import get_profile
from .modes import infer_experiment_type
from .modes import resolve_configured_profile
from .modes import resolved_run_name
from .modes import validate_training_semantics


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help=(
            "Override config values, e.g. model.name=vit_b_16 "
            "training.lr=1e-5"
        ),
    )
    parser.add_argument(
        "--sweep",
        nargs="*",
        default=[],
        help=(
            "Multi-run sweep, e.g. model.name=resnet18,vit_b_16 "
            "data.image_col=rel_path_seg,rel_path_raw"
        ),
    )
    return parser


def _canonical_parser() -> argparse.ArgumentParser:
    parser = _legacy_parser()
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    parser.add_argument("--single-run", action="store_true")
    return parser


def resolve_plan(
    config_path: str,
    overrides: list[str],
    sweep: list[str],
    explicit_profile: str | None,
):
    source = load_config(config_path)
    overridden = apply_overrides(source, overrides)
    chosen = explicit_profile or overridden.get("training", {}).get("profile")
    compatibility_profile = get_profile(str(chosen)) if chosen else None
    profile = compatibility_profile or resolve_configured_profile(overridden)
    expanded = generate_sweep_configs(
        overridden,
        sweep,
        include_colour_ablation=profile.colour_sweep,
    )
    resolved = []
    resolved_types = []

    for item in expanded:
        cfg = copy.deepcopy(item)
        if compatibility_profile is None:
            item_profile = resolve_configured_profile(cfg)
            if item_profile != profile:
                raise ValueError(
                    "One canonical invocation cannot sweep over training feature "
                    "switches that resolve to different loader or output contracts"
                )
            experiment_type = infer_experiment_type(cfg)
            validate_training_semantics(cfg, item_profile, experiment_type)
            resolved.append(cfg)
            resolved_types.append(experiment_type)
            continue

        condition = cfg.get("input_condition", {}) or {}
        transformed = bool(condition.get("enabled", False)) and str(
            condition.get("transform", "original")
        ) != "original"
        stress = bool(
            (cfg.get("test_cue_suppression", {}) or {}).get("enabled", False)
        )

        if profile.loader_mode == "standard":
            default_type = "standard"
        elif profile.loader_mode == "colour":
            default_type = "matched_condition"
        elif stress:
            default_type = "rgb_stress_test"
        elif condition.get("enabled", False):
            default_type = "matched_condition"
        else:
            default_type = "standard"

        experiment_type = str(
            (cfg.get("experiment", {}) or {}).get("type") or default_type
        )
        allowed = {
            "standard",
            "matched_condition",
            "rgb_stress_test",
            "matched_and_rgb_stress",
        }
        if experiment_type not in allowed:
            raise ValueError(f"Unknown experiment.type {experiment_type!r}")
        if (
            experiment_type in {"rgb_stress_test", "matched_and_rgb_stress"}
            and profile.name != "cue_suppression"
        ):
            raise ValueError(
                f"experiment.type={experiment_type} requires profile cue_suppression"
            )
        if stress and transformed:
            raise ValueError(
                "Fixed-RGB stress evaluation requires an original-trained "
                "input condition"
            )
        if (
            experiment_type in {"rgb_stress_test", "matched_and_rgb_stress"}
            and not stress
        ):
            raise ValueError(
                f"experiment.type={experiment_type} requires "
                "test_cue_suppression.enabled=true"
            )
        if profile.loader_mode == "standard" and experiment_type != "standard":
            raise ValueError(
                f"profile {profile.name} requires experiment.type=standard"
            )
        if (
            profile.loader_mode == "colour"
            and experiment_type != "matched_condition"
        ):
            raise ValueError(
                "profile colour_ablation requires experiment.type=matched_condition"
            )
        if (
            profile.loader_mode == "condition"
            and experiment_type == "standard"
            and (condition.get("enabled", False) or stress)
        ):
            raise ValueError(
                "cue_suppression standard experiment cannot enable "
                "input_condition or stress evaluation"
            )

        resolved.append(cfg)
        resolved_types.append(experiment_type)

    external = bool(
        (overridden.get("input_condition", {}) or {}).get("enabled", False)
    )
    if external:
        active = []
        if sweep:
            active.append("CLI --sweep")
        for key in ("sweep", "colour_ablation", "matched_condition_training"):
            if bool((overridden.get(key, {}) or {}).get("enabled", False)):
                active.append(f"{key}.enabled")
        if active or len(resolved) != 1:
            raise ValueError(
                "External input_condition requires exactly one run and all "
                "internal expanders disabled; active: " + ", ".join(active)
            )

    return profile, resolved, resolved_types


def _plan_summary(profile, configs, experiment_types):
    models = sorted({str(c.get("model", {}).get("name")) for c in configs})
    conditions = sorted(
        {
            str(
                (c.get("input_condition", {}) or {}).get("condition", "original")
            )
            for c in configs
        }
    )
    first = configs[0]
    tasks = first.get("data", {}).get("target_cols", {})
    hierarchy = (
        first.get("multi_task", {}).get("hierarchy_loss", {})
        if profile.hierarchy
        else {}
    )
    return {
        "selected_profile": profile.name,
        "configuration_driven": profile.name == "configured",
        "loader_mode": profile.loader_mode,
        "experiment_type": experiment_types[0],
        "expected_internal_training_runs": len(configs),
        "model_count": len(models),
        "models": models,
        "tasks": tasks,
        "loss_weights": first.get("multi_task", {}).get("loss_weights", {}),
        "normalize_loss_by_active_tasks": first.get("multi_task", {}).get(
            "normalize_loss_by_active_tasks", True
        ),
        "hierarchy_enabled": bool(hierarchy.get("enabled", False)),
        "hierarchy_weight": hierarchy.get("weight"),
        "wandb_enabled": bool(
            profile.wandb and first.get("wandb", {}).get("enabled", False)
        ),
        "masked_labels": profile.masked_labels,
        "condition_count": len(conditions),
        "resolved_training_conditions": conditions,
        "post_training_rgb_stress": bool(
            profile.stress_evaluation
            and first.get("test_cue_suppression", {}).get("enabled", False)
        ),
        "expected_output_paths": [
            str(
                Path(c.get("output", {}).get("out_dir", "outputs"))
                / resolved_run_name(c, profile)
            )
            for c in configs
        ],
    }


def execute(args, forced_profile: str | None = None):
    profile, configs, experiment_types = resolve_plan(
        args.config,
        args.override,
        args.sweep,
        forced_profile or getattr(args, "profile", None),
    )
    if getattr(args, "single_run", False) and len(configs) != 1:
        raise ValueError(
            f"--single-run requires exactly one resolved run, got {len(configs)}"
        )

    summary = _plan_summary(profile, configs, experiment_types)
    if getattr(args, "dry_run", False) or getattr(
        args, "print_resolved_config", False
    ):
        print(
            json.dumps(
                {"plan": summary, "resolved_configs": configs},
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return []

    import pandas as pd

    from .runner import run_one

    results = [run_one(cfg, profile) for cfg in configs]
    out = Path(configs[0]["output"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    if profile.sort_colour_results and "colour_percent" in frame:
        frame = frame.sort_values("colour_percent", ascending=False).reset_index(
            drop=True
        )
    frame.to_csv(out / "multi_run_results.csv", index=False)
    return results


def main(argv=None):
    return execute(_canonical_parser().parse_args(argv))


def legacy_main(profile: str, argv=None):
    return execute(_legacy_parser().parse_args(argv), forced_profile=profile)
