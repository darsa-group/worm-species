"""Compatibility profile wrapper for the historical masked trainer."""

from src.worm_species.config.sweeps import generate_sweep_configs
from src.worm_species.data.labels import build_label_maps
from src.worm_species.data.labels import read_csvs_from_dir
from src.worm_species.training.cli import legacy_main
from src.worm_species.training.epochs import run_hierarchy_epoch as _run_epoch
from src.worm_species.training.loaders import make_standard_loaders as make_loaders
from src.worm_species.training.losses import build_criteria
from src.worm_species.training.losses import compute_individual_class_weights
from src.worm_species.training.metrics import safe_metric as _safe_metric
from src.worm_species.training.metrics import score_for_selection as _score_for_selection
from src.worm_species.training.modes import get_profile
from src.worm_species.training.runner import run_one

PROFILE = get_profile("masked")


def run_epoch(
    model,
    loader,
    criteria,
    optimizer,
    device,
    train,
    scaler=None,
    use_amp: bool = True,
    task_loss_weights: dict[str, float] | None = None,
    normalize_loss_by_active_tasks: bool = True,
):
    return _run_epoch(
        model,
        loader,
        criteria,
        optimizer,
        device,
        train,
        scaler,
        use_amp,
        task_loss_weights,
        normalize_loss_by_active_tasks,
        {},
        None,
    )


def train_one_run(cfg: dict) -> dict:
    return run_one(cfg, PROFILE)


def main():
    return legacy_main(PROFILE.name)


if __name__ == "__main__":
    main()
