#!/usr/bin/env python3
"""Dry-run or submit exactly the five missing publication resolutions."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_ablation_pipeline import run_pipeline
from worm_species.slurm.config import load_submission_config
from worm_species.slurm.planning import plan_submission


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE = (
    PROJECT_ROOT
    / "dev"
    / "genome_publication_30seed_resolution_gapfill_pipeline.yaml"
)
DEFAULT_CONFIG = (
    PROJECT_ROOT / "dev" / "genome_publication_30seed_resolution_gapfill.yaml"
)
CLUSTER = PROJECT_ROOT / "configs" / "clusters" / "genome.yaml"
EXPECTED_SEEDS = set(range(40, 2941, 100))
EXPECTED_PIXELS = {
    "resolution_loss_090pct": 22,
    "resolution_loss_095pct": 11,
    "resolution_loss_097pct": 7,
    "resolution_loss_098pct": 4,
    "resolution_loss_099pct": 2,
}


def validate_gapfill_plan(config_path: Path = DEFAULT_CONFIG) -> None:
    config = load_submission_config(config_path, cluster_config=CLUSTER)
    plan = plan_submission(config)
    observed_conditions = {spec.training_condition for spec in plan.run_specs}
    observed_seeds = {int(spec.resolved_config["seed"]) for spec in plan.run_specs}
    if plan.array_size != 150:
        raise ValueError(
            f"Resolution gap-fill must contain exactly 150 fits, got {plan.array_size}"
        )
    if set(plan.models) != {"convnext_base"}:
        raise ValueError(f"Unexpected models in gap-fill: {plan.models}")
    if observed_conditions != set(EXPECTED_PIXELS):
        raise ValueError(
            "Gap-fill conditions changed: "
            f"expected {sorted(EXPECTED_PIXELS)}, got {sorted(observed_conditions)}"
        )
    if observed_seeds != EXPECTED_SEEDS:
        raise ValueError("Gap-fill seeds no longer match the 30-seed paper design")
    for spec in plan.run_specs:
        condition = spec.resolved_config["input_condition"]
        percent = float(condition["parameters"]["percent"])
        pixels = max(1, int(round(224 * (1.0 - percent / 100.0))))
        expected = EXPECTED_PIXELS[spec.training_condition]
        if pixels != expected:
            raise ValueError(
                f"{spec.training_condition} resolves to {pixels}px, expected {expected}px"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "submit"), default="dry-run")
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    args = parser.parse_args()
    validate_gapfill_plan()
    summary = run_pipeline(args.pipeline, args.mode)
    print(
        "Resolution gap-fill validated: 5 levels, 30 seeds, "
        f"150 fits; mode={args.mode}."
    )
    print(f"Artifacts: {summary['artifact_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
