"""Explicit behavior profiles for the unified trainer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    loader_mode: str
    hierarchy: bool
    wandb: bool
    colour_sweep: bool = False
    run_summary: bool = False
    stress_evaluation: bool = False
    sort_colour_results: bool = False


PROFILES = {
    "masked": TrainingProfile("masked", "standard", False, False),
    "masked_hloss": TrainingProfile("masked_hloss", "standard", True, False),
    "masked_hloss_wandb": TrainingProfile(
        "masked_hloss_wandb", "standard", True, True
    ),
    "colour_ablation": TrainingProfile(
        "colour_ablation", "colour", True, True, True, True, False, True
    ),
    "cue_suppression": TrainingProfile(
        "cue_suppression", "condition", True, True, True, True, True, True
    ),
}
DEFAULT_PROFILE = "masked_hloss_wandb"


def get_profile(name: str) -> TrainingProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown training profile {name!r}; choose from {sorted(PROFILES)}"
        ) from exc


def resolved_run_name(cfg: dict, profile: TrainingProfile) -> str:
    from src.utils import make_run_name

    base = make_run_name(cfg)
    if profile.loader_mode == "standard":
        return base

    percent = int(
        round(float(cfg.get("data", {}).get("colour_retention", 1.0)) * 100)
    )
    if profile.loader_mode == "colour":
        if "colour_retention" not in cfg.get("data", {}):
            return base
        suffix = f"colour_{percent:03d}pct"
    else:
        raw = cfg.get("input_condition", {}) or {}
        condition = (
            str(
                raw.get("condition")
                or raw.get("name")
                or raw.get("transform", "original")
            )
            if raw.get("enabled", False)
            else "original"
        )
        parts = []
        if "colour_retention" in cfg.get("data", {}):
            parts.append(f"basecolour_{percent:03d}pct")
        parts.append(f"train_{condition.replace(' ', '_')}")
        suffix = "_".join(parts)

    return base if suffix in base else f"{base}_{suffix}"
