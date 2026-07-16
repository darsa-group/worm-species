from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def make_individual_level_splits(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    root_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits by individual/group, not by image row.

    This prevents images of the same worm from appearing in both train and test sets.
    Assumes each individual belongs to one target class.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1")

    missing_columns = {target_col, group_col}.difference(df.columns)
    if missing_columns:
        raise KeyError(f"Missing split columns: {sorted(missing_columns)}")

    group_df = (
        df[[group_col, target_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    counts_per_group = group_df.groupby(group_col)[target_col].nunique()
    bad_groups = counts_per_group[counts_per_group > 1]

    if len(bad_groups) > 0:
        raise ValueError(
            "Some groups have more than one target label. "
            "Check barcode/group parsing."
        )

    # One row per individual
    group_df = group_df.drop_duplicates(group_col).reset_index(drop=True)

    splitter_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )

    trainval_idx, test_idx = next(
        splitter_test.split(group_df[group_col], group_df[target_col])
    )

    group_trainval = group_df.iloc[trainval_idx].reset_index(drop=True)
    group_test = group_df.iloc[test_idx].reset_index(drop=True)

    relative_val_size = val_size / (1.0 - test_size)

    splitter_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=relative_val_size,
        random_state=seed + 1,
    )

    train_idx, val_idx = next(
        splitter_val.split(group_trainval[group_col], group_trainval[target_col])
    )

    group_train = group_trainval.iloc[train_idx]
    group_val = group_trainval.iloc[val_idx]

    train_groups = set(group_train[group_col])
    val_groups = set(group_val[group_col])
    test_groups = set(group_test[group_col])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)

    if root_dir is not None:
        split_dir = Path(root_dir) / "split_csv"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(split_dir / "train_split.csv", index=False)
        val_df.to_csv(split_dir / "val_split.csv", index=False)
        test_df.to_csv(split_dir / "test_split.csv", index=False)
        print(f"Saved train/val/test splits to {split_dir}")

    return train_df, val_df, test_df
