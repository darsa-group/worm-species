"""Bounded helpers for the read-only condition-matrix dashboard view."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from .data_loader import load_csv_rows, load_json
from src.worm_species.results.normalization import canonical_condition_relation
from src.worm_species.results.normalization import matrix_evaluation_relation


RELATIONS = ("matched", "rgb_stress", "cross_condition")
CONDITION_RELATIONS = ("original", "matched", "rgb_stress", "cross_condition")
REQUIRED_COLUMNS = (
    "run_name",
    "model",
    "train_condition",
    "test_condition",
    "task",
    "macro_f1",
)


def normalise_matrix_rows(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate lightweight task rows without mutating caller-owned values."""
    valid: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, source in enumerate(rows):
        row = dict(source)
        missing = [
            column
            for column in REQUIRED_COLUMNS
            if row.get(column) is None or str(row.get(column)).strip() == ""
        ]
        if missing:
            warnings.append(f"row {index}: missing {missing}")
            continue
        relation_value = row.get("evaluation_relation")
        if relation_value is None or str(relation_value).strip() == "":
            relation_value = matrix_evaluation_relation(
                row["train_condition"], row["test_condition"]
            )
            row["evaluation_relation"] = relation_value
        relation = str(relation_value)
        if relation not in RELATIONS:
            warnings.append(f"row {index}: unknown relation {relation!r}")
            continue
        condition_relation = row.get("condition_relation")
        if condition_relation is None or str(condition_relation).strip() == "":
            condition_relation = canonical_condition_relation(
                row["train_condition"], row["test_condition"]
            )
            row["condition_relation"] = condition_relation
        if str(condition_relation) not in CONDITION_RELATIONS:
            warnings.append(
                f"row {index}: unknown condition relation {condition_relation!r}"
            )
            continue
        try:
            macro_f1 = float(row["macro_f1"])
        except (TypeError, ValueError):
            warnings.append(f"row {index}: macro_f1 is not numeric")
            continue
        if not math.isfinite(macro_f1) or not 0.0 <= macro_f1 <= 1.0:
            warnings.append(f"row {index}: macro_f1 is outside [0, 1]")
            continue
        row["macro_f1"] = macro_f1
        valid.append(row)
    return valid, warnings


def filter_matrix_rows(
    rows: Iterable[dict[str, Any]],
    *,
    models: set[str] | None = None,
    train_conditions: set[str] | None = None,
    test_conditions: set[str] | None = None,
    relations: set[str] | None = None,
    tasks: set[str] | None = None,
) -> list[dict[str, Any]]:
    filters = {
        "model": models,
        "train_condition": train_conditions,
        "test_condition": test_conditions,
        "evaluation_relation": relations,
        "task": tasks,
    }
    return [
        dict(row)
        for row in rows
        if all(
            selected is None or str(row.get(column)) in selected
            for column, selected in filters.items()
        )
    ]


def matrix_relation_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    cells = {
        (
            str(row.get("run_name")),
            str(row.get("model")),
            str(row.get("train_condition")),
            str(row.get("test_condition")),
        ): str(row.get("evaluation_relation"))
        for row in rows
    }
    counts = Counter(cells.values())
    return {relation: int(counts.get(relation, 0)) for relation in RELATIONS}


def macro_f1_pivot(
    rows: Iterable[dict[str, Any]], *, model: str, task: str
) -> Any:
    """Return a stable train-by-test DataFrame for one model/task selection."""
    import pandas as pd

    selected = [
        row
        for row in rows
        if str(row.get("model")) == model and str(row.get("task")) == task
    ]
    if not selected:
        return pd.DataFrame()
    frame = pd.DataFrame(selected)
    frame["macro_f1"] = pd.to_numeric(frame["macro_f1"], errors="coerce")
    pivot = frame.pivot_table(
        index="train_condition",
        columns="test_condition",
        values="macro_f1",
        aggfunc="mean",
    )
    return pivot.sort_index().sort_index(axis=1)


def _artifact(
    artifacts: Iterable[dict[str, Any]], kind: str
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in artifacts
            if item.get("kind") == kind and item.get("available")
        ),
        None,
    )


def select_matrix_task_artifacts(
    experiment: dict[str, Any] | None,
    runs: Iterable[dict[str, Any]],
    *,
    max_run_artifacts: int = 1_000,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Prefer one aggregate artifact, otherwise use already indexed run files."""
    aggregate = _artifact(
        (experiment or {}).get("artifacts", []),
        "condition_matrix_task_metrics.csv",
    )
    if aggregate is not None:
        return "aggregate", [aggregate], []
    artifacts = sorted(
        (
            artifact
            for run in runs
            for artifact in run.get("artifacts", [])
            if artifact.get("kind") == "condition_matrix/task_metrics.csv"
            and artifact.get("available")
        ),
        key=lambda item: str(item.get("path", "")),
    )
    warnings = []
    if len(artifacts) > max_run_artifacts:
        warnings.append(
            f"bounded fallback uses the first {max_run_artifacts} of "
            f"{len(artifacts)} indexed run artifacts"
        )
        artifacts = artifacts[:max_run_artifacts]
    return "per_run", artifacts, warnings


def load_indexed_matrix_rows(
    experiment: dict[str, Any] | None,
    runs: Iterable[dict[str, Any]],
    *,
    max_rows: int = 200_000,
    max_run_artifacts: int = 1_000,
) -> tuple[list[dict[str, Any]], list[str], list[str], str]:
    """Read only selected indexed CSV paths, with a global row bound."""
    mode, artifacts, warnings = select_matrix_task_artifacts(
        experiment,
        runs,
        max_run_artifacts=max_run_artifacts,
    )
    rows: list[dict[str, Any]] = []
    paths: list[str] = []
    for artifact in artifacts:
        remaining = max_rows - len(rows)
        if remaining <= 0:
            warnings.append(f"condition-matrix rows were capped at {max_rows}")
            break
        path = str(artifact["path"])
        paths.append(path)
        try:
            loaded = load_csv_rows(path, max_rows=remaining)
            rows.extend(loaded)
            if len(loaded) >= remaining:
                warnings.append(
                    f"condition-matrix rows reached the {max_rows} row bound; "
                    "additional rows may be omitted"
                )
        except Exception as exc:
            warnings.append(f"could not read {path}: {exc}")
    valid, row_warnings = normalise_matrix_rows(rows)
    warnings.extend(row_warnings)
    return valid, warnings, paths, mode


def load_matrix_completion_summary(
    experiment: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    artifact = _artifact(
        (experiment or {}).get("artifacts", []),
        "condition_matrix_collection_summary.json",
    )
    if artifact is None:
        return None, None
    try:
        return load_json(artifact["path"]), None
    except Exception as exc:
        return None, f"could not read {artifact['path']}: {exc}"


__all__ = [
    "CONDITION_RELATIONS",
    "RELATIONS",
    "filter_matrix_rows",
    "load_indexed_matrix_rows",
    "load_matrix_completion_summary",
    "macro_f1_pivot",
    "matrix_relation_counts",
    "normalise_matrix_rows",
    "select_matrix_task_artifacts",
]
