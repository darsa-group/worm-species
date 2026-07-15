"""Root compatibility wrapper for the hierarchy-loss trainer."""

from scripts.training.train_multitask_masked_hloss import *  # noqa: F401,F403
from scripts.training.train_multitask_masked_hloss import _safe_metric, _score_for_selection, main


if __name__ == "__main__":
    main()
