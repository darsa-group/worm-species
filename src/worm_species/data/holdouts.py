"""Plain, auditable removal of biological cohorts from model development."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataHoldoutResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    development_cohort: pd.DataFrame | None
    test_cohort: pd.DataFrame | None
    audit: dict | None

    @property
    def evaluation_cohort(self) -> pd.DataFrame | None:
        """Compatibility alias for the independent test cohort."""
        return self.test_cohort


def _filter_mask(
    frame: pd.DataFrame,
    where: dict[str, str],
    target_cols: dict[str, str],
) -> pd.Series:
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for task, expected in where.items():
        column = target_cols[task]
        if column not in frame.columns:
            raise ValueError(
                f"Data holdout task {task!r} uses missing column {column!r}."
            )
        mask &= frame[column].astype("string").eq(str(expected)).fillna(False)
    return mask


def _column_filter_mask(
    frame: pd.DataFrame,
    where: dict[str, str],
) -> pd.Series:
    """Select an auditable raw-data cohort without inventing task labels."""
    mask = pd.Series(True, index=frame.index, dtype=bool)
    for column, expected in where.items():
        if column not in frame.columns:
            raise ValueError(
                f"Data holdout cohort selector uses missing column {column!r}."
            )
        mask &= frame[column].astype("string").eq(str(expected)).fillna(False)
    return mask


def select_holdout_frame(
    frame: pd.DataFrame,
    where: dict[str, str],
    target_cols: dict[str, str],
) -> pd.DataFrame:
    """Return one cohort without mutating its source split."""
    return frame.loc[
        _filter_mask(frame, where, target_cols)
    ].copy().reset_index(drop=True)


def _counts(frame: pd.DataFrame, mask: pd.Series, group_col: str) -> dict[str, int]:
    selected = frame.loc[mask]
    return {
        "rows": int(len(selected)),
        "individuals": int(selected[group_col].nunique()),
    }


def apply_data_holdout(
    *,
    config: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    target_cols: dict[str, str],
    group_col: str,
) -> DataHoldoutResult:
    """Remove one configured cohort without ever modifying the test split."""
    holdout = copy.deepcopy(config.get("data_holdout", {}) or {})
    if not bool(holdout.get("enabled", False)):
        return DataHoldoutResult(train, validation, test, None, None, None)

    where = dict(holdout.get("where") or {})
    cohort_where = dict(holdout.get("cohort_where") or {})
    if not where and not cohort_where:
        raise ValueError(
            "Enabled data holdout requires data_holdout.where or "
            "data_holdout.cohort_where."
        )
    evaluation_where = dict(holdout.get("evaluation_where") or where)
    remove_from = list(holdout.get("remove_from", ["train", "validation"]))
    frames = {
        "train": train.copy(),
        "validation": validation.copy(),
    }
    removed: dict[str, dict[str, int]] = {}
    removed_frames = []
    for split_name in remove_from:
        frame = frames[split_name]
        mask = (
            _column_filter_mask(frame, cohort_where)
            if cohort_where
            else _filter_mask(frame, where, target_cols)
        )
        removed[split_name] = _counts(frame, mask, group_col)
        selected = frame.loc[mask].copy()
        selected["_holdout_source_split"] = split_name
        removed_frames.append(selected)
        frames[split_name] = frame.loc[~mask].reset_index(drop=True)

    if removed.get("train", {}).get("rows", 0) == 0:
        raise ValueError(
            f"Data holdout {holdout['name']!r} removed no training rows. "
            "Check data_holdout.where/cohort_where="
            f"{(cohort_where or where)!r}."
        )

    evaluation_mask = _filter_mask(test, evaluation_where, target_cols)
    test_cohort = test.loc[evaluation_mask].copy()
    test_cohort["_holdout_source_split"] = "test"
    test_cohort = test_cohort.reset_index(drop=True)
    if test_cohort.empty:
        raise ValueError(
            f"Data holdout {holdout['name']!r} has no matching test examples. "
            f"Check data_holdout.evaluation_where={evaluation_where!r}."
        )
    development_cohort = pd.concat(
        removed_frames, ignore_index=True
    )
    if development_cohort.empty:
        raise ValueError(
            f"Data holdout {holdout['name']!r} has no development cohort."
        )

    audit = {
        "name": holdout["name"],
        "question": holdout.get("question", ""),
        "remove_from": remove_from,
        "where": where,
        "cohort_where": cohort_where,
        "evaluation_where": evaluation_where,
        "primary_tasks": list(holdout["primary_tasks"]),
        "removed": removed,
        "development_cohort": {
            "rows": int(len(development_cohort)),
            "individuals": int(
                development_cohort[group_col].nunique()
            ),
        },
        "test_cohort": _counts(test, evaluation_mask, group_col),
        # Historical name retained for old result readers.
        "evaluation_cohort": _counts(test, evaluation_mask, group_col),
        "remaining": {
            "train": {
                "rows": int(len(frames["train"])),
                "individuals": int(frames["train"][group_col].nunique()),
            },
            "validation": {
                "rows": int(len(frames["validation"])),
                "individuals": int(
                    frames["validation"][group_col].nunique()
                ),
            },
        },
        "test_unchanged": True,
    }
    return DataHoldoutResult(
        frames["train"],
        frames["validation"],
        test,
        development_cohort,
        test_cohort,
        audit,
    )


__all__ = [
    "DataHoldoutResult",
    "apply_data_holdout",
    "select_holdout_frame",
]
