"""Root compatibility wrapper for matched training and fixed-RGB evaluation."""

from scripts.training.train_multitask_cue_suppression import *  # noqa: F401,F403
from scripts.training.train_multitask_cue_suppression import (
    _flatten_wandb_config,
    _inclusive_float_sequence,
    _safe_metric,
    _score_for_selection,
    _test_condition_signature,
    _wandb_metrics,
    main,
)


if __name__ == "__main__":
    main()
