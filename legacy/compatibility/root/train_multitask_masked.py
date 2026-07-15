"""Root compatibility wrapper for the standard multi-task trainer."""

from scripts.training.train_multitask_masked import *  # noqa: F401,F403
from scripts.training.train_multitask_masked import _safe_metric, _score_for_selection, main


if __name__ == "__main__":
    main()
