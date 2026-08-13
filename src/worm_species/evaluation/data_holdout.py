"""Evaluation focused on the biological cohort removed from development data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..results.writing import save_json
from ..training.epochs import run_hierarchy_epoch
from ..training.metrics import classification_metric_summary


def evaluate_data_holdout(
    *,
    cfg: dict,
    out_dir: Path,
    model,
    bundle,
    criteria,
    device,
    use_amp: bool,
    task_loss_weights: dict[str, float],
    normalize_loss_by_active_tasks: bool,
    hierarchy_cfg: dict,
    child_to_parent_matrix,
    use_masked_labels: bool,
    full_test_true: dict[str, list[int]] | None = None,
    full_test_pred: dict[str, list[int]] | None = None,
    full_test_probabilities: dict[str, np.ndarray] | None = None,
) -> dict:
    """Write one obvious answer for each task named by the holdout question."""
    holdout = cfg.get("data_holdout", {}) or {}
    if not bool(holdout.get("enabled", False)):
        return {"enabled": False}
    loaders = bundle.data_holdout_loaders or (
        {"independent_test": bundle.data_holdout_loader}
        if bundle.data_holdout_loader is not None
        else {}
    )
    if not loaders or bundle.data_holdout_audit is None:
        raise ValueError("Enabled data holdout has no evaluation cohort loader.")

    evaluation_where = dict(
        holdout.get("evaluation_where") or holdout.get("where") or {}
    )
    task_rows = []
    metrics_by_cohort = {}
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
        metrics_by_cohort[cohort_name] = metrics
        for task in holdout["primary_tasks"]:
            label = evaluation_where.get(task)
            label_to_index = bundle.label_to_index_by_task[task]
            supported = label is None or label in label_to_index
            y_true = np.asarray(true.get(task, []), dtype=int)
            y_pred = np.asarray(pred.get(task, []), dtype=int)
            target_n = 0
            target_recall = float("nan")
            expanded_metrics: dict[str, float | int] = {}
            if supported and label is not None:
                target_index = label_to_index[label]
                target_mask = y_true == target_index
                target_n = int(target_mask.sum())
                if target_n:
                    target_recall = float(
                        (y_pred[target_mask] == target_index).mean()
                    )
                if full_test_true is not None and full_test_pred is not None:
                    probability_matrix = (
                        (full_test_probabilities or {}).get(task)
                    )
                    target_probabilities = (
                        probability_matrix[:, target_index]
                        if probability_matrix is not None
                        and probability_matrix.ndim == 2
                        and probability_matrix.shape[1] > target_index
                        else None
                    )
                    calculated = classification_metric_summary(
                        full_test_true.get(task, []),
                        full_test_pred.get(task, []),
                        target_index=target_index,
                        target_probabilities=target_probabilities,
                    )
                    expanded_metrics = {
                        key if key.startswith("target_") else f"full_test_{key}": value
                        for key, value in calculated.items()
                    }
                    target_recall = float(
                        expanded_metrics.get("target_recall", target_recall)
                    )
            cohort_metrics = classification_metric_summary(y_true, y_pred)
            task_rows.append({
                "holdout": holdout["name"],
                "question": holdout["question"],
                "cohort": cohort_name,
                "task": task,
                "target_label": label,
                "class_supported_by_training_head": bool(supported),
                "n": int(metrics.get(f"{task}_n", 0)),
                "accuracy": metrics.get(f"{task}_accuracy"),
                "balanced_accuracy": metrics.get(f"{task}_balanced_accuracy"),
                "macro_f1": metrics.get(f"{task}_macro_f1"),
                "target_n": target_n,
                "target_recall": target_recall,
                **{
                    f"cohort_{key}": value
                    for key, value in cohort_metrics.items()
                },
                **expanded_metrics,
            })

    root = Path(out_dir) / "data_holdout_evaluation"
    root.mkdir(parents=True, exist_ok=True)
    task_path = root / "task_metrics.csv"
    pd.DataFrame(task_rows).to_csv(task_path, index=False)
    result = {
        "enabled": True,
        "name": holdout["name"],
        "question": holdout["question"],
        "audit": bundle.data_holdout_audit,
        "metrics_by_cohort": metrics_by_cohort,
        "tasks": task_rows,
        "task_metrics_path": str(task_path),
    }
    save_json(result, root / "summary.json")
    return result


__all__ = ["evaluate_data_holdout"]
