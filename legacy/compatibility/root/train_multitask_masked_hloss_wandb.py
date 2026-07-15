"""Root compatibility wrapper for the hierarchy-loss W&B trainer."""

from scripts.training.train_multitask_masked_hloss_wandb import *  # noqa: F401,F403
from scripts.training.train_multitask_masked_hloss_wandb import (
    _flatten_wandb_config,
    _safe_metric,
    _score_for_selection,
    _wandb_metrics,
    main,
)


if __name__ == "__main__":
    main()
