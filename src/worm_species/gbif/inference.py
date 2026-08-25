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
    prefetch_factor: int = 4,
    device_name: str = "auto",
    curation_labels: tuple[str, ...] = ("keep",),
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"Invalid inference shard {shard_index}/{shard_count}; "
            "require 0 <= shard_index < shard_count"
        )
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
    eligible_rows = int(len(frame))
    frame = frame.loc[
        frame["image_id"].map(
            lambda value: int(
                hashlib.sha256(str(value).encode("utf-8")).hexdigest(), 16
            ) % shard_count == shard_index
        )
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
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers:
        loader_kwargs.update(
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )
    loader = DataLoader(InferenceImageDataset(frame, transform), **loader_kwargs)
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
        "eligible_rows_all_shards": eligible_rows,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
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


def merge_inference_shards(
    manifest_path: str | Path,
    shard_dir: str | Path,
    output_path: str | Path,
    *,
    shard_count: int,
    curation_labels: tuple[str, ...] = ("keep",),
) -> dict:
    """Validate exact shard coverage before publishing merged inference."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if "download_status" in manifest:
        manifest = manifest.loc[manifest["download_status"].eq("downloaded")].copy()
    if "curation_label" in manifest and curation_labels:
        manifest = manifest.loc[manifest["curation_label"].isin(curation_labels)].copy()
    manifest["_resolved_local_path"] = manifest["local_path"].map(
        lambda value: str(resolve_manifest_image_path(manifest_path, value))
    )
    manifest = manifest.loc[
        manifest["_resolved_local_path"].map(lambda path: Path(path).is_file())
    ].copy()
    expected_ids = manifest["image_id"].astype(str).tolist()
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Eligible inference manifest contains duplicate image_id values")

    root = Path(shard_dir)
    frames = []
    summaries = []
    for index in range(shard_count):
        shard = root / f"shard-{index:03d}.csv"
        summary_path = shard.with_suffix(".summary.json")
        if not shard.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing inference shard artifacts for index {index}")
        frame = pd.read_csv(shard, dtype=str, keep_default_na=False)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("shard_index", -1)) != index:
            raise ValueError(f"Shard {index} summary has the wrong shard_index")
        if int(summary.get("shard_count", -1)) != shard_count:
            raise ValueError(f"Shard {index} summary has the wrong shard_count")
        frames.append(frame)
        summaries.append(summary)
    checkpoint_hashes = {item.get("checkpoint_sha256") for item in summaries}
    if None in checkpoint_hashes or len(checkpoint_hashes) != 1:
        raise ValueError("Inference shards used different checkpoints")
    combined = pd.concat(frames, ignore_index=True)
    if combined["image_id"].duplicated().any():
        duplicates = combined.loc[combined["image_id"].duplicated(), "image_id"].head().tolist()
        raise ValueError(f"Inference shard outputs overlap: {duplicates}")
    actual_ids = set(combined["image_id"].astype(str))
    missing = set(expected_ids).difference(actual_ids)
    unexpected = actual_ids.difference(expected_ids)
    if missing or unexpected:
        raise ValueError(
            f"Inference shard coverage mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )
    order = {image_id: index for index, image_id in enumerate(expected_ids)}
    combined["_manifest_order"] = combined["image_id"].map(order)
    combined = combined.sort_values("_manifest_order").drop(columns="_manifest_order")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    summary = {
        "rows": int(len(combined)),
        "shard_count": shard_count,
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "checkpoint": summaries[0].get("checkpoint"),
        "checkpoint_model": summaries[0].get("checkpoint_model"),
        "known_genus_rows": int(
            combined.get("checkpoint_genus_scope", pd.Series(dtype=str)).eq("known").sum()
        ),
        "known_species_rows": int(
            combined.get("checkpoint_species_scope", pd.Series(dtype=str)).eq("known").sum()
        ),
        "coverage_validated": True,
        "interpretation": (
            "Agreement compares predictions with GBIF occurrence labels and is not "
            "an independently verified accuracy estimate."
        ),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
