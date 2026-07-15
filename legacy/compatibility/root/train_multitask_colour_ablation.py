"""Root compatibility wrapper for the colour-ablation trainer."""

from scripts.training.train_multitask_colour_ablation import *  # noqa: F401,F403
from scripts.training.train_multitask_colour_ablation import (
    _flatten_wandb_config,
    _safe_metric,
    _score_for_selection,
    _wandb_metrics,
    main,
)


if __name__ == "__main__":
    main()
