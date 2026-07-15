from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


MISSING_LABEL = -1
MISSING_LABEL_INDEX = -1
DEFAULT_MISSING_VALUES = {
    "", "na", "n/a", "nan", "none", "null", "unknown", "unidentified",
    "missing", "not_available",
}


def is_missing_label(value, missing_values: list[str] | None = None) -> bool:
    """Return whether a task label must be masked without dropping its row."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    text = str(value).strip()
    if text == "":
        return True
    configured = {
        "na", "n/a", "nan", "none", "null", "missing", "unknown",
        "unidentified",
    }
    if missing_values is not None:
        configured.update(str(item).strip().lower() for item in missing_values)
    return text.lower() in configured


def missing_values_from_config(config: dict) -> set[str]:
    values = config.get("data", {}).get("missing_label_values", [])
    return DEFAULT_MISSING_VALUES | {str(value).strip().lower() for value in values}


def normalise_missing_series(
    series: pd.Series,
    missing_values: set[str],
) -> pd.Series:
    """Convert configured missing strings to ``pd.NA`` without changing labels."""
    result = series.copy()
    result = result.replace(r"^\s*$", pd.NA, regex=True)
    mask = result.astype("string").str.strip().str.lower().isin(missing_values)
    return result.mask(mask, pd.NA)


def clean_label_value(value: Any, missing_values: set[str]) -> str | None:
    if pd.isna(value):
        return None
    cleaned = str(value).strip()
    if cleaned.lower() in missing_values:
        return None
    return cleaned


def get_target_cols(config: dict) -> dict[str, str]:
    """Return the configured task-to-metadata-column mapping."""
    target_cols = config.get("data", {}).get("target_cols")
    if target_cols is None:
        target_cols = {
            "genus": "genus",
            "species": "species_label",
            "age": "life_stage",
        }
    if not isinstance(target_cols, dict) or len(target_cols) == 0:
        raise ValueError(
            "data.target_cols must be a non-empty mapping, for example: "
            "{genus: genus, species: species_label, age: life_stage}"
        )
    return target_cols


def build_label_maps(
    train_df: pd.DataFrame,
    target_cols: dict[str, str],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[int, str]]]:
    """Build sorted, training-only label maps for each task."""
    label_to_index_by_task = {}
    index_to_label_by_task = {}
    for task, column in target_cols.items():
        labels = sorted(
            train_df.loc[train_df[column].notna(), column].astype(str).unique()
        )
        if len(labels) == 0:
            raise ValueError(
                f"Task '{task}' has no labelled examples in the training split. "
                f"Check column '{column}' or reduce min_individuals_per_class for this task."
            )
        if len(labels) == 1:
            print(
                f"Warning: task '{task}' has only one class in training: {labels}. "
                "The head can train, but evaluation is not biologically informative."
            )
        label_to_index = {label: index for index, label in enumerate(labels)}
        index_to_label = {index: label for label, index in label_to_index.items()}
        label_to_index_by_task[task] = label_to_index
        index_to_label_by_task[task] = index_to_label
    return label_to_index_by_task, index_to_label_by_task


def read_csvs_from_dir(
    dir_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the legacy ``<dir>/split_csv/{train,val,test}_split.csv`` paths."""
    train_csv_path = Path(os.path.join(dir_path, "split_csv", "train_split.csv"))
    val_csv_path = Path(os.path.join(dir_path, "split_csv", "val_split.csv"))
    test_csv_path = Path(os.path.join(dir_path, "split_csv", "test_split.csv"))
    print(
        "Reading train/val/test splits from "
        f"{train_csv_path}, {val_csv_path}, {test_csv_path}"
    )
    if not train_csv_path.exists() or not val_csv_path.exists() or not test_csv_path.exists():
        raise FileNotFoundError(f"One or more split CSV files not found in {dir_path}")
    return (
        pd.read_csv(train_csv_path),
        pd.read_csv(val_csv_path),
        pd.read_csv(test_csv_path),
    )
