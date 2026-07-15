"""Compatibility profile wrapper for matched training and RGB stress tests."""

from pathlib import Path

from src.worm_species.config.sweeps import generate_sweep_configs as _generate_sweep_configs
from src.worm_species.data.labels import build_label_maps
from src.worm_species.data.labels import read_csvs_from_dir
from src.worm_species.evaluation.cue_suppression import *
from src.worm_species.evaluation.cue_suppression import _inclusive_float_sequence
from src.worm_species.evaluation.cue_suppression import _test_condition_signature
from src.worm_species.training.cli import legacy_main
from src.worm_species.training.epochs import run_hierarchy_epoch as run_epoch
from src.worm_species.training.loaders import get_input_condition
from src.worm_species.training.loaders import legacy_cue_loader_tuple
from src.worm_species.training.loaders import make_profile_loaders
from src.worm_species.training.losses import *
from src.worm_species.training.metrics import safe_metric as _safe_metric
from src.worm_species.training.metrics import score_for_selection as _score_for_selection
from src.worm_species.training.modes import get_profile
from src.worm_species.training.runner import _flatten_wandb_config
from src.worm_species.training.runner import _wandb_metrics
from src.worm_species.training.runner import get_colour_metadata
from src.worm_species.training.runner import (
    initialise_wandb_run as _initialise_wandb_run,
)
from src.worm_species.training.runner import make_experiment_run_name as _name
from src.worm_species.training.runner import run_one

PROFILE = get_profile("cue_suppression")


def generate_sweep_configs(
    base_cfg: dict,
    cli_sweep_items: list[str] | None = None,
) -> list[dict]:
    return _generate_sweep_configs(
        base_cfg,
        cli_sweep_items,
        include_colour_ablation=True,
    )


def make_loaders(cfg: dict):
    return legacy_cue_loader_tuple(make_profile_loaders(cfg, PROFILE))


def initialise_wandb_run(cfg: dict, run_name: str, out_dir: Path):
    return _initialise_wandb_run(cfg, run_name, out_dir, PROFILE)


def make_experiment_run_name(cfg: dict) -> str:
    return _name(cfg, PROFILE)


def train_one_run(cfg: dict) -> dict:
    return run_one(cfg, PROFILE)


def main():
    return legacy_main(PROFILE.name)


if __name__ == "__main__":
    main()
