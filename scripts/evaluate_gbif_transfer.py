#!/usr/bin/env python3
"""Evaluate one completed transfer checkpoint on the immutable PETI/GBIF tests.

This command is intentionally inference-only.  It writes prediction-level and
GBIF occurrence-level outputs and never starts training or submits a job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from worm_species.data.datasets import MultiTaskWormImageDataset
from worm_species.data.transforms import build_split_transform
from worm_species.gbif.domain_data import DOMAINS, TASK_COLUMNS, file_sha256, load_domain_config
from worm_species.models.multitask import build_multitask_model


def _ece(correct: np.ndarray, confidence: np.ndarray, bins: int = 15) -> float | None:
    if not len(correct):
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (confidence <= high if high == 1 else confidence < high)
        if not selected.any():
            continue
        total += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return total


def _task_metrics(labels: list[str], predictions: list[str], probabilities: np.ndarray | None, classes: list[str]) -> dict:
    valid_indices = [index for index, (label, pred) in enumerate(zip(labels, predictions)) if label and pred]
    valid = [(labels[index], predictions[index]) for index in valid_indices]
    if not valid:
        return {"n": 0, "accuracy": None, "balanced_accuracy": None, "macro_f1": None}
    y_true, y_pred = zip(*valid)
    observed = sorted(set(y_true))
    result = {
        "n": len(valid),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=observed, average="macro", zero_division=0)),
        "confusion_matrix_labels": observed,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=observed).tolist(),
    }
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=observed, zero_division=0
    )
    result["per_class"] = [
        {"label": label, "precision": float(p), "recall": float(r), "f1": float(score), "support": int(n)}
        for label, p, r, score, n in zip(observed, precision, recall, f1, support)
    ]
    prevalence = pd.Series(y_true).value_counts()
    majority = str(prevalence.index[0])
    result["majority_label"] = majority
    result["majority_baseline"] = float((np.asarray(y_true) == majority).mean())
    result["balanced_chance_1_over_k"] = float(1.0 / len(observed))
    if probabilities is not None and len(probabilities):
        confidence = probabilities[np.asarray(valid_indices)].max(axis=1)
        correct = np.asarray(y_true) == np.asarray(y_pred)
        result["mean_confidence"] = float(confidence.mean())
        result["ece"] = _ece(correct, confidence)
        try:
            result["confidence_correctness_auroc"] = float(roc_auc_score(correct.astype(int), confidence))
        except ValueError:
            result["confidence_correctness_auroc"] = None
    else:
        result["mean_confidence"] = None
        result["ece"] = None
        result["confidence_correctness_auroc"] = None
    return result


def _load_model(checkpoint: Path, label_maps: dict, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_maps = payload.get("label_to_index_by_task")
    if checkpoint_maps != label_maps:
        raise ValueError("Checkpoint label maps do not match immutable prepared label maps")
    cfg = payload.get("cfg", {})
    model_name = cfg.get("model", {}).get("name") or payload.get("experiment_spec", {}).get("model")
    if not model_name:
        raise ValueError("Checkpoint does not record its architecture")
    model = build_multitask_model(
        {"model": {"name": model_name, "pretrained": False}},
        {task: len(mapping) for task, mapping in label_maps.items()},
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload


def evaluate_checkpoint(config: dict, checkpoint: Path, output: Path, datasets: tuple[str, ...], device_name: str) -> dict:
    prepared = Path(config["paths"]["output_root"]) / "prepared"
    label_maps = json.loads((prepared / "label_maps.json").read_text())
    taxonomy = pd.concat(
        [pd.read_csv(prepared / f"petri_{split}.csv", dtype=str, keep_default_na=False) for split in ("train", "validation", "test")],
        ignore_index=True,
    )
    species_to_genus = (
        taxonomy.loc[taxonomy["species"].ne(""), ["species", "genus"]]
        .drop_duplicates("species").set_index("species")["genus"].to_dict()
    )
    device = torch.device(device_name)
    model, payload = _load_model(checkpoint, label_maps, device)
    preprocessing = payload.get("cfg", {}).get("preprocessing", {
        "image_size": int(config["data"]["image_size"]),
        "normalisation": {"enabled": True, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    })
    transform = build_split_transform(split="test", preprocessing=preprocessing, apply_augmentation=False)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    probability_rows: dict[str, list[np.ndarray]] = {task: [] for task in TASK_COLUMNS}
    for domain in datasets:
        frame = pd.read_csv(prepared / f"{domain}_test.csv", dtype=str, keep_default_na=False)
        dataset = MultiTaskWormImageDataset(
            frame, root_dir="/", image_col="image_path", target_cols=TASK_COLUMNS,
            label_to_index_by_task=label_maps, transform=transform, crop_to_foreground=False,
        )
        loader = DataLoader(dataset, batch_size=int(config["inference"]["batch_size"]), shuffle=False, num_workers=0)
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                logits = model(batch["image"].to(device))
                count = len(batch["image"])
                for index in range(count):
                    source = frame.iloc[offset + index]
                    record = {
                        "image_id": source.get("sample_id", ""), "sample_id": source.get("sample_id", ""),
                        "occurrence_id": source.get("gbif_id", "") or source.get("group_id", ""),
                        "dataset": domain, "split": "test", "country": source.get("country", ""),
                        "true_genus": source.get("true_genus", source.get("genus", "")),
                        "true_species": source.get("true_species", source.get("species", "")),
                        "true_age": source.get("age", ""),
                        "age_label_available": bool(source.get("age", "")),
                        "genus_known_to_model": bool(source.get("genus", "")),
                        "species_known_to_model": bool(source.get("species", "")),
                    }
                    for task in TASK_COLUMNS:
                        probabilities = torch.softmax(logits[task][index], dim=0).cpu().numpy()
                        probability_rows[task].append(probabilities)
                        best = int(probabilities.argmax())
                        names = {int(value): str(key) for key, value in label_maps[task].items()}
                        record[f"pred_{task}"] = names[best]
                        record[f"{task}_confidence"] = float(probabilities[best])
                        true = source.get(task, "")
                        record[f"_mapped_{task}"] = true
                        record[f"{task}_correct"] = bool(true and true == names[best])
                        if task == "species":
                            top3 = np.argsort(probabilities)[-min(3, len(probabilities)):][::-1]
                            record["species_top3_correct"] = bool(true and true in {names[int(value)] for value in top3})
                    record.update({
                        "training_regime": payload.get("experiment_spec", {}).get("strategy", ""),
                        "training_stage": payload.get("experiment_spec", {}).get("stage", ""),
                        "hierarchy_loss_weight": payload.get("experiment_spec", {}).get("hierarchy_loss_weight", 0.0),
                        "seed": payload.get("experiment_spec", {}).get("seed", ""),
                        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
                    })
                    record["genus_species_prediction_consistent"] = bool(
                        record.get("pred_species", "")
                        and species_to_genus.get(record["pred_species"], "") == record.get("pred_genus", "")
                    )
                    rows.append(record)
                offset += count
    predictions = pd.DataFrame(rows)
    predictions.to_csv(output / "predictions.csv", index=False)
    metrics: dict = {"checkpoint": str(checkpoint.resolve()), "image_level": {}, "occurrence_level": {}}
    for domain in datasets:
        subset = predictions[predictions["dataset"].eq(domain)]
        for task in TASK_COLUMNS:
            true_col = f"_mapped_{task}"
            labels = subset[true_col].astype(str).tolist() if true_col in subset else []
            preds = subset[f"pred_{task}"].astype(str).tolist()
            probs = (
                subset[[f"{task}_confidence"]].to_numpy(dtype=float)
                if f"{task}_confidence" in subset else None
            )
            metrics["image_level"].setdefault(domain, {})[task] = _task_metrics(labels, preds, probs, list(label_maps[task]))
        if domain in predictions["dataset"].unique() and "species_top3_correct" in subset:
            species_rows = subset[subset["_mapped_species"].ne("")]
            metrics["image_level"].setdefault(domain, {})["species_top3_accuracy"] = (
                float(species_rows["species_top3_correct"].mean()) if not species_rows.empty else None
            )
            metrics["image_level"].setdefault(domain, {})["genus_species_consistency"] = float(
                subset["genus_species_prediction_consistent"].mean()
            )
        if domain == "gbif" and not subset.empty:
            grouped = []
            for occurrence, group in subset.groupby("occurrence_id", sort=True):
                item = group.iloc[0].to_dict()
                item["occurrence_id"] = occurrence
                for task in TASK_COLUMNS:
                    mean_probabilities = np.vstack(probability_rows[task])[group.index].mean(axis=0)
                    names = {int(value): str(key) for key, value in label_maps[task].items()}
                    best = int(mean_probabilities.argmax())
                    item[f"pred_{task}"] = names[best]
                    item[f"{task}_confidence"] = float(mean_probabilities[best])
                grouped.append(item)
            occurrence = pd.DataFrame(grouped)
            occurrence.to_csv(output / "gbif_occurrence_predictions.csv", index=False)
            for task in TASK_COLUMNS:
                metrics["occurrence_level"].setdefault("gbif", {})[task] = _task_metrics(
                    occurrence.get(f"_mapped_{task}", pd.Series(dtype=str)).astype(str).tolist(),
                    occurrence[f"pred_{task}"].astype(str).tolist(),
                    occurrence[[f"{task}_confidence"]].to_numpy(dtype=float),
                    list(label_maps[task]),
                )
    metrics["checkpoint_sha256"] = file_sha256(checkpoint)
    metrics["interpretation"] = "GBIF agreement is not independently verified taxonomic accuracy."
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gbif_training.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate_checkpoint(load_domain_config(args.config), Path(args.checkpoint), Path(args.output), tuple(args.datasets), args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
