"""Evaluate original-image baselines on the exact biological control cohorts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ..data.datasets import MultiTaskWormImageDataset
from ..data.holdouts import select_holdout_frame
from ..data.transforms import build_split_transform
from ..results.writing import save_json
from ..training.epochs import run_hierarchy_epoch


def _cohort_loaders(
    context: dict[str, Any],
    definition: dict[str, Any],
) -> dict[str, DataLoader]:
    split_frames = context["split_frames"]
    target_cols = context["target_cols"]
    where = dict(definition["where"])
    evaluation_where = dict(definition.get("evaluation_where") or where)
    development_parts = []
    for split_name in ("train", "validation"):
        selected = select_holdout_frame(
            split_frames[split_name], where, target_cols
        )
        selected["_holdout_source_split"] = split_name
        development_parts.append(selected)
    cohorts = {
        "development_withheld": pd.concat(
            development_parts, ignore_index=True
        ),
        "independent_test": select_holdout_frame(
            split_frames["test"], evaluation_where, target_cols
        ),
    }
    if any(frame.empty for frame in cohorts.values()):
        empty = [name for name, frame in cohorts.items() if frame.empty]
        raise ValueError(
            f"Baseline holdout control {definition['name']!r} has empty "
            f"cohorts: {empty}"
        )
    transform = build_split_transform(
        split="validation",
        preprocessing=context["preprocessing"],
        augmentation=context["augmentation"],
        condition={
            "condition": "original",
            "feature": "baseline",
            "transform": "original",
            "strength": 0.0,
        },
        original_colour_retention=context["original_colour_retention"],
    )
    return {
        name: DataLoader(
            MultiTaskWormImageDataset(
                frame,
                transform=transform,
                **context["dataset_kwargs"],
            ),
            batch_size=context["batch_size"],
            shuffle=False,
            **context["loader_kwargs"],
        )
        for name, frame in cohorts.items()
    }


def evaluate_holdout_controls(
    *,
    cfg: dict,
    out_dir: Path,
    model: Any,
    bundle: Any,
    criteria: dict,
    device: Any,
    use_amp: bool,
    task_loss_weights: dict[str, float],
    normalize_loss_by_active_tasks: bool,
    hierarchy_cfg: dict,
    child_to_parent_matrix: Any,
    use_masked_labels: bool,
) -> dict[str, Any]:
    controls = (
        (cfg.get("evaluation", {}) or {}).get(
            "data_holdout_controls", {}
        )
        or {}
    )
    if not bool(controls.get("enabled", False)):
        return {"enabled": False, "tasks": []}
    definitions = controls.get("definitions", [])
    if not isinstance(definitions, list) or not definitions:
        raise ValueError(
            "evaluation.data_holdout_controls.definitions must be non-empty"
        )
    if bundle.test_loader_context is None:
        raise ValueError("Baseline holdout controls require loader context")

    rows = []
    metrics_by_cohort = {}
    for definition in definitions:
        loaders = _cohort_loaders(bundle.test_loader_context, definition)
        evaluation_where = dict(
            definition.get("evaluation_where")
            or definition.get("where")
            or {}
        )
        for cohort_name, loader in loaders.items():
            metrics, true, pred = run_hierarchy_epoch(
                model=model,
                loader=loader,
                criteria=criteria,
                optimizer=None,
                device=device,
                train=False,
                scaler=None,
                use_amp=use_amp,
                task_loss_weights=task_loss_weights,
                normalize_loss_by_active_tasks=normalize_loss_by_active_tasks,
                hierarchy_cfg=hierarchy_cfg,
                child_to_parent_matrix=child_to_parent_matrix,
                use_masked_labels=use_masked_labels,
            )
            metrics_by_cohort[
                f"{definition['name']}::{cohort_name}"
            ] = metrics
            for task in definition["primary_tasks"]:
                label = evaluation_where.get(task)
                label_map = bundle.label_to_index_by_task[task]
                supported = label is None or label in label_map
                y_true = np.asarray(true.get(task, []), dtype=int)
                y_pred = np.asarray(pred.get(task, []), dtype=int)
                target_n = 0
                target_recall = float("nan")
                if supported and label is not None:
                    target_index = label_map[label]
                    target_mask = y_true == target_index
                    target_n = int(target_mask.sum())
                    if target_n:
                        target_recall = float(
                            (y_pred[target_mask] == target_index).mean()
                        )
                rows.append({
                    "holdout": definition["name"],
                    "question": definition.get("question", ""),
                    "cohort": cohort_name,
                    "task": task,
                    "target_label": label,
                    "class_supported_by_training_head": bool(supported),
                    "n": int(metrics.get(f"{task}_n", 0)),
                    "accuracy": metrics.get(f"{task}_accuracy"),
                    "balanced_accuracy": metrics.get(
                        f"{task}_balanced_accuracy"
                    ),
                    "macro_f1": metrics.get(f"{task}_macro_f1"),
                    "target_n": target_n,
                    "target_recall": target_recall,
                })

    root = Path(out_dir) / "data_holdout_control_evaluation"
    root.mkdir(parents=True, exist_ok=True)
    task_path = root / "task_metrics.csv"
    pd.DataFrame(rows).to_csv(task_path, index=False)
    result = {
        "enabled": True,
        "metrics_by_cohort": metrics_by_cohort,
        "tasks": rows,
        "task_metrics_path": str(task_path),
    }
    save_json(result, root / "summary.json")
    return result


__all__ = ["evaluate_holdout_controls"]
