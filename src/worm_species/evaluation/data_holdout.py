"""Evaluation focused on the biological cohort removed from development data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..results.writing import save_json
from ..training.epochs import run_hierarchy_epoch
from .predictions import collect_probability_predictions
from .predictions import public_prediction_frame
from .predictions import structured_target_metrics


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
    image_prediction_frames = []
    individual_prediction_frames = []
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
        image_predictions, individual_predictions, probability_metrics = (
            collect_probability_predictions(
                models=[model],
                loader=loader,
                tasks=bundle.target_cols,
                index_to_label_by_task=bundle.index_to_label_by_task,
                device=device,
                use_amp=use_amp,
                run_id=Path(out_dir).name,
                checkpoint="best",
                split="structured_holdout",
                holdout=cohort_name,
                maximum_images_per_individual=(
                    bundle.multiview_evaluation_max_images
                ),
            )
        )
        metrics.update(probability_metrics)
        image_prediction_frames.append(image_predictions)
        individual_prediction_frames.append(individual_predictions)
        for task in holdout["primary_tasks"]:
            if task not in criteria or task not in bundle.label_to_index_by_task:
                continue
            label = evaluation_where.get(task)
            label_to_index = bundle.label_to_index_by_task[task]
            supported = label is None or label in label_to_index
            y_true = np.asarray(true.get(task, []), dtype=int)
            y_pred = np.asarray(pred.get(task, []), dtype=int)
            target_n = 0
            target_recall = float("nan")
            if supported and label is not None:
                target_index = label_to_index[label]
                target_mask = y_true == target_index
                target_n = int(target_mask.sum())
                if target_n:
                    target_recall = float(
                        (y_pred[target_mask] == target_index).mean()
                    )
            task_image = image_predictions[image_predictions["task"] == task]
            task_individual = individual_predictions[
                individual_predictions["task"] == task
            ]
            biological = structured_target_metrics(
                task_image, task_individual, target_label=label if supported else None
            )
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
                **biological,
            })

    root = Path(out_dir) / "data_holdout_evaluation"
    root.mkdir(parents=True, exist_ok=True)
    task_path = root / "task_metrics.csv"
    if image_prediction_frames:
        public_prediction_frame(pd.concat(image_prediction_frames, ignore_index=True)).to_csv(
            root / "predictions_best.csv", index=False
        )
        public_prediction_frame(pd.concat(individual_prediction_frames, ignore_index=True)).to_csv(
            root / "individual_predictions_best.csv", index=False
        )
    pd.DataFrame(
        task_rows,
        columns=[
            "holdout",
            "question",
            "cohort",
            "task",
            "target_label",
            "class_supported_by_training_head",
            "n",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "target_n",
            "target_recall",
            "target_n_images",
            "target_n_individuals",
            "target_recall_image",
            "target_recall_individual",
        ],
    ).to_csv(task_path, index=False)
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
