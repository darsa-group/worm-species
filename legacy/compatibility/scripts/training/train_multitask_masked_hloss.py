"""Compatibility profile wrapper for the historical hierarchy trainer."""

from src.worm_species.config.sweeps import generate_sweep_configs
from src.worm_species.data.labels import build_label_maps
from src.worm_species.data.labels import read_csvs_from_dir
from src.worm_species.training.cli import legacy_main
from src.worm_species.training.epochs import run_hierarchy_epoch as run_epoch
from src.worm_species.training.loaders import make_standard_loaders as make_loaders
from src.worm_species.training.losses import *
from src.worm_species.training.metrics import safe_metric as _safe_metric
from src.worm_species.training.metrics import score_for_selection as _score_for_selection
from src.worm_species.training.modes import get_profile
from src.worm_species.training.runner import run_one

PROFILE = get_profile("masked_hloss")


def train_one_run(cfg: dict) -> dict:
    return run_one(cfg, PROFILE)


def main():
    return legacy_main(PROFILE.name)


if __name__ == "__main__":
    main()
