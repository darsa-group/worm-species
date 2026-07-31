"""Auditable image- and individual-level probability evaluation."""

from __future__ import annotations

import json
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from ..data.labels import MISSING_LABEL
from ..models.multitask import task_logits
from ..training.losses import infer_parent_label_from_child_label


PREDICTION_COLUMNS = [
    "run_id",
    "checkpoint",
    "split",
    "holdout",
    "image_path",
    "barcode",
    "task",
    "true_label",
    "predicted_label",
    "probabilities",
]


def _scores(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    if frame.empty:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_accuracy": float("nan"),
            f"{prefix}_balanced_accuracy": float("nan"),
            f"{prefix}_macro_f1": float("nan"),
        }
    true = frame["true_label"].astype(str)
    predicted = frame["predicted_label"].astype(str)
    return {
        f"{prefix}_n": int(len(frame)),
        f"{prefix}_accuracy": float(accuracy_score(true, predicted)),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(true, predicted)
        ),
        f"{prefix}_macro_f1": float(
            f1_score(true, predicted, average="macro", zero_division=0)
        ),
    }


def aggregate_individual_probabilities(
    predictions: pd.DataFrame,
    maximum_images: int | None = None,
) -> pd.DataFrame:
    """Average all image probabilities within barcode/task groups."""
    if predictions.empty:
        return pd.DataFrame(columns=[*PREDICTION_COLUMNS, "n_images"])
    rows = []
    keys = ["run_id", "checkpoint", "split", "holdout", "barcode", "task"]
    for values, group in predictions.groupby(keys, dropna=False, sort=True):
        if maximum_images is not None:
            if maximum_images <= 0:
                raise ValueError("maximum_images must be positive")
            group = group.sort_values("image_path").head(int(maximum_images))
        true_labels = group["true_label"].dropna().astype(str).unique()
        if len(true_labels) != 1:
            raise ValueError(
                "One barcode has inconsistent true labels for task "
                f"{values[-1]!r}: {true_labels.tolist()}"
            )
        probabilities = np.stack([
            np.asarray(json.loads(value), dtype=float)
            for value in group["probabilities"]
        ])
        averaged = probabilities.mean(axis=0)
        class_names = group.iloc[0]["_class_names"]
        predicted = class_names[int(np.argmax(averaged))]
        rows.append({
            **dict(zip(keys, values, strict=True)),
            "image_path": "|".join(group["image_path"].astype(str)),
            "true_label": true_labels[0],
            "predicted_label": predicted,
            "probabilities": json.dumps(averaged.tolist()),
            "n_images": int(len(group)),
            "_class_names": class_names,
        })
    return pd.DataFrame(rows)


def prediction_metrics(
    image_predictions: pd.DataFrame,
    individual_predictions: pd.DataFrame,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}
    tasks = sorted(set(image_predictions.get("task", [])))
    for task in tasks:
        image_task = image_predictions[image_predictions["task"] == task]
        individual_task = individual_predictions[
            individual_predictions["task"] == task
        ]
        metrics.update(_scores(image_task, f"{task}_image"))
        metrics.update(_scores(individual_task, f"{task}_individual"))
        # Preserve the historical metric API: unqualified metrics are image-level.
        for metric in ("n", "accuracy", "balanced_accuracy", "macro_f1"):
            metrics[f"{task}_{metric}"] = metrics[f"{task}_image_{metric}"]
    macro = [
        metrics[f"{task}_individual_macro_f1"]
        for task in tasks
        if np.isfinite(metrics[f"{task}_individual_macro_f1"])
    ]
    metrics["individual_mean_macro_f1"] = (
        float(np.mean(macro)) if macro else float("nan")
    )
    for level, frame, keys in (
        ("image", image_predictions, ["image_path", "barcode"]),
        ("individual", individual_predictions, ["barcode"]),
    ):
        if {"genus", "species"}.issubset(set(frame.get("task", []))):
            pivot = frame.pivot_table(
                index=keys,
                columns="task",
                values="predicted_label",
                aggfunc="first",
            ).dropna(subset=["genus", "species"])
            implied = pivot["species"].map(infer_parent_label_from_child_label)
            metrics[f"{level}_genus_species_agreement_rate"] = (
                float(pivot["genus"].eq(implied).mean())
                if len(pivot) else float("nan")
            )
    if "image_genus_species_agreement_rate" in metrics:
        metrics["genus_species_agreement_rate"] = metrics[
            "image_genus_species_agreement_rate"
        ]
    return metrics


def collect_probability_predictions(
    *,
    models: Iterable[torch.nn.Module],
    loader,
    tasks: Iterable[str],
    index_to_label_by_task: dict[str, dict[int, str]],
    device: torch.device,
    use_amp: bool,
    run_id: str,
    checkpoint: str,
    split: str,
    holdout: str = "",
    maximum_images_per_individual: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    """Collect probabilities, averaging them across models for an ensemble."""
    model_list = list(models)
    if not model_list:
        raise ValueError("At least one model is required for prediction")
    for model in model_list:
        model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            probability_sets: list[dict[str, torch.Tensor]] = []
            for model in model_list:
                with torch.amp.autocast(
                    enabled=use_amp and device.type == "cuda",
                    device_type=device.type,
                ):
                    outputs = model(images)
                probability_sets.append({
                    task: torch.softmax(task_logits(outputs, task), dim=1)
                    for task in tasks
                })
            paths = list(batch["path"])
            barcodes = [str(value) for value in batch["barcode"]]
            for task in tasks:
                labels = batch["labels"][task]
                class_map = index_to_label_by_task[task]
                class_names = [class_map[index] for index in range(len(class_map))]
                probabilities = torch.stack([
                    values[task] for values in probability_sets
                ]).mean(dim=0).detach().float().cpu().numpy()
                for index, encoded in enumerate(labels.tolist()):
                    if encoded == MISSING_LABEL:
                        continue
                    probability = probabilities[index]
                    rows.append({
                        "run_id": run_id,
                        "checkpoint": checkpoint,
                        "split": split,
                        "holdout": holdout,
                        "image_path": str(paths[index]),
                        "barcode": barcodes[index],
                        "task": task,
                        "true_label": class_map[int(encoded)],
                        "predicted_label": class_names[int(np.argmax(probability))],
                        "probabilities": json.dumps(probability.tolist()),
                        "_class_names": class_names,
                    })
    image = pd.DataFrame(rows)
    individual = aggregate_individual_probabilities(
        image, maximum_images=maximum_images_per_individual
    )
    return image, individual, prediction_metrics(image, individual)


def public_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in PREDICTION_COLUMNS if column in frame.columns]
    if "n_images" in frame.columns:
        columns.append("n_images")
    return frame.loc[:, columns]


def ensemble_prediction_frames(
    frames: list[pd.DataFrame],
    *,
    checkpoint: str = "ensemble",
) -> pd.DataFrame:
    """Average aligned row-level probabilities from distinct checkpoints."""
    if not frames:
        raise ValueError("At least one prediction frame is required")
    keys = ["run_id", "split", "holdout", "image_path", "barcode", "task", "true_label"]
    indexed = [frame.set_index(keys).sort_index() for frame in frames]
    reference = indexed[0].index
    if any(not frame.index.equals(reference) for frame in indexed[1:]):
        raise ValueError("Ensemble checkpoints produced non-identical prediction rows")
    rows = []
    for position, values in enumerate(reference):
        probabilities = np.stack([
            np.asarray(json.loads(frame.iloc[position]["probabilities"]), dtype=float)
            for frame in indexed
        ]).mean(axis=0)
        class_names = indexed[0].iloc[position]["_class_names"]
        rows.append({
            **dict(zip(keys, values, strict=True)),
            "checkpoint": checkpoint,
            "predicted_label": class_names[int(np.argmax(probabilities))],
            "probabilities": json.dumps(probabilities.tolist()),
            "_class_names": class_names,
        })
    return pd.DataFrame(rows)


def structured_target_metrics(
    image: pd.DataFrame,
    individual: pd.DataFrame,
    *,
    target_label: str | None,
) -> dict[str, float | int]:
    if target_label is None:
        return {
            "target_n_images": 0,
            "target_n_individuals": 0,
            "target_recall_image": float("nan"),
            "target_recall_individual": float("nan"),
        }
    image_target = image[image["true_label"].astype(str) == str(target_label)]
    individual_target = individual[
        individual["true_label"].astype(str) == str(target_label)
    ]
    return {
        "target_n_images": int(len(image_target)),
        "target_n_individuals": int(len(individual_target)),
        "target_recall_image": (
            float((image_target["predicted_label"] == target_label).mean())
            if len(image_target) else float("nan")
        ),
        "target_recall_individual": (
            float((individual_target["predicted_label"] == target_label).mean())
            if len(individual_target) else float("nan")
        ),
    }


__all__ = [
    "PREDICTION_COLUMNS",
    "aggregate_individual_probabilities",
    "collect_probability_predictions",
    "ensemble_prediction_frames",
    "prediction_metrics",
    "public_prediction_frame",
    "structured_target_metrics",
]
