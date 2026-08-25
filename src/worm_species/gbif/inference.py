from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data.transforms import build_split_transform
from ..models.multitask import build_multitask_model
from ..training.checkpoints import load_checkpoint
from .pipeline import resolve_manifest_image_path


class InferenceImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["_resolved_local_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(row["image_id"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inverse_label_map(label_to_index: dict[str, int]) -> dict[int, str]:
    inverse = {int(index): str(label) for label, index in label_to_index.items()}
    if len(inverse) != len(label_to_index):
        raise ValueError("Checkpoint label indices must be unique")
    return inverse


def infer_existing_checkpoint(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 4,
    device_name: str = "auto",
    curation_labels: tuple[str, ...] = ("keep",),
) -> dict:
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(checkpoint_file)
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if "download_status" in frame.columns:
        frame = frame.loc[frame["download_status"] == "downloaded"].copy()
    if "curation_label" in frame.columns and curation_labels:
        frame = frame.loc[frame["curation_label"].isin(curation_labels)].copy()
    frame["_resolved_local_path"] = frame["local_path"].map(
        lambda value: str(resolve_manifest_image_path(manifest_path, value))
    )
    frame = frame.loc[
        frame["_resolved_local_path"].map(lambda path: Path(path).is_file())
    ].copy()
    if frame.empty:
        raise ValueError("No eligible downloaded images are available for inference")

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )
    checkpoint = load_checkpoint(checkpoint_file, map_location="cpu")
    config = copy.deepcopy(checkpoint["cfg"])
    # The checkpoint supplies every model weight. Avoid a redundant ImageNet or
    # DINOv3 download while reconstructing the architecture for strict loading.
    config.setdefault("model", {})["pretrained"] = False
    label_maps = checkpoint["label_to_index_by_task"]
    class_counts = {task: len(mapping) for task, mapping in label_maps.items()}
    model = build_multitask_model(config, class_counts)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval().to(device)

    transform = build_split_transform(
        split="test",
        preprocessing=config.get("preprocessing", {}),
        augmentation=config.get("augmentation", {}),
        condition={"transform": "original"},
        apply_augmentation=False,
    )
    loader = DataLoader(
        InferenceImageDataset(frame, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    predictions: dict[str, list] = {"image_id": []}
    for task in label_maps:
        predictions.update(
            {
                f"predicted_{task}": [],
                f"predicted_{task}_confidence": [],
                f"predicted_{task}_top3": [],
            }
        )
    inverse = {task: _inverse_label_map(mapping) for task, mapping in label_maps.items()}
    with torch.inference_mode():
        for images, image_ids in loader:
            outputs = model(images.to(device, non_blocking=True))
            predictions["image_id"].extend(image_ids)
            for task, logits in outputs.items():
                probabilities = torch.softmax(logits.float(), dim=1)
                count = min(3, probabilities.shape[1])
                top_probabilities, top_indices = probabilities.topk(count, dim=1)
                for row_indices, row_probabilities in zip(
                    top_indices.cpu().tolist(), top_probabilities.cpu().tolist()
                ):
                    predictions[f"predicted_{task}"].append(inverse[task][row_indices[0]])
                    predictions[f"predicted_{task}_confidence"].append(row_probabilities[0])
                    predictions[f"predicted_{task}_top3"].append(
                        json.dumps([
                            {"label": inverse[task][index], "probability": probability}
                            for index, probability in zip(row_indices, row_probabilities)
                        ])
                    )

    prediction_frame = pd.DataFrame(predictions)
    result = frame.drop(columns=["_resolved_local_path"]).merge(
        prediction_frame, on="image_id", how="left", validate="one_to_one"
    )
    known_genera = set(label_maps.get("genus", {}))
    known_species = set(label_maps.get("species", {}))
    result["checkpoint_genus_scope"] = np.where(
        result["genus"].isin(known_genera), "known", "unknown"
    )
    result["checkpoint_species_scope"] = np.where(
        result["species_label"].isin(known_species), "known", "unknown"
    )
    if "predicted_genus" in result:
        result["genus_label_agreement"] = np.where(
            result["checkpoint_genus_scope"] == "known",
            result["predicted_genus"] == result["genus"],
            pd.NA,
        )
    if "predicted_species" in result:
        result["species_label_agreement"] = np.where(
            result["checkpoint_species_scope"] == "known",
            result["predicted_species"] == result["species_label"],
            pd.NA,
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    summary = {
        "rows": int(len(result)),
        "device": str(device),
        "checkpoint": str(checkpoint_file.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_file),
        "checkpoint_model": config["model"]["name"],
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_selection_metric": checkpoint.get("selection_metric"),
        "known_genus_rows": int((result["checkpoint_genus_scope"] == "known").sum()),
        "known_species_rows": int((result["checkpoint_species_scope"] == "known").sum()),
        "genus_label_agreement": (
            float(result.loc[result["checkpoint_genus_scope"] == "known", "genus_label_agreement"].mean())
            if (result["checkpoint_genus_scope"] == "known").any()
            else None
        ),
        "species_label_agreement": (
            float(result.loc[result["checkpoint_species_scope"] == "known", "species_label_agreement"].mean())
            if (result["checkpoint_species_scope"] == "known").any()
            else None
        ),
        "interpretation": (
            "Agreement compares predictions with GBIF occurrence labels and is not "
            "an independently verified accuracy estimate."
        ),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary
