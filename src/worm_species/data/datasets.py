from __future__ import annotations

from pathlib import Path
import random

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from .cropping import (
    foreground_bbox_from_image,
    foreground_bbox_from_mask,
    pad_square_bbox,
)
from .image_validation import resolve_path
from .labels import (
    MISSING_LABEL,
    MISSING_LABEL_INDEX,
    clean_label_value,
    is_missing_label,
)


class MultiTaskWormImageDataset(Dataset):
    """Image dataset with task-by-task missing-label masking."""

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str | Path,
        image_col: str,
        target_col: str | None = None,
        label_to_index: dict[str, int] | None = None,
        transform=None,
        mask_col: str | None = None,
        crop_to_foreground: bool = True,
        crop_pad: float = 0.15,
        target_cols: dict[str, str] | None = None,
        label_to_index_by_task: dict[str, dict[str, int]] | None = None,
        missing_label_index: int = MISSING_LABEL_INDEX,
        image_is_tensor: bool = False,
        barcode_col: str = "barcode",
        metadata_cols: dict[str, str] | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.image_col = image_col
        self.mask_col = mask_col
        self.target_col = target_col
        self.label_to_index = label_to_index
        self.target_cols = target_cols
        self.label_to_index_by_task = label_to_index_by_task
        self.missing_label_index = missing_label_index
        self.image_is_tensor = image_is_tensor
        self.barcode_col = barcode_col
        self.metadata_cols = metadata_cols or {}
        self.transform = transform
        self.crop_to_foreground = crop_to_foreground
        self.crop_pad = crop_pad
        self.multi_task = target_cols is not None
        if self.multi_task:
            if label_to_index_by_task is None:
                raise ValueError(
                    "label_to_index_by_task must be provided for multi-task training."
                )
        elif target_col is None or label_to_index is None:
            raise ValueError(
                "target_col and label_to_index must be provided for single-task training."
            )

    def __len__(self) -> int:
        return len(self.df)

    def _encode_label(self, task: str, value) -> int:
        # ``missing_values`` is intentionally not initialised here. This private,
        # unused legacy method is retained verbatim as a compatibility surface.
        cleaned = clean_label_value(value, self.missing_values)
        if cleaned is None:
            return MISSING_LABEL
        return self.label_to_index_by_task[task].get(cleaned, MISSING_LABEL)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        barcode = (
            str(row[self.barcode_col])
            if self.barcode_col in self.df.columns
            else str(index)
        )
        image_path = resolve_path(self.root_dir, row[self.image_col])
        if self.image_is_tensor:
            image = torch.load(
                image_path,
                map_location="cpu",
                weights_only=True,
            )
            if not torch.is_tensor(image) or image.ndim != 3:
                raise ValueError(
                    f"Cached condition image must be a [C, H, W] tensor: "
                    f"{image_path}"
                )
            image = image.to(dtype=torch.float32)
        else:
            image = Image.open(image_path).convert("RGB")

        if self.crop_to_foreground and not self.image_is_tensor:
            bbox = None
            if (
                self.mask_col is not None
                and self.mask_col in row
                and pd.notna(row[self.mask_col])
            ):
                mask_path = resolve_path(self.root_dir, row[self.mask_col])
                bbox = foreground_bbox_from_mask(mask_path)
            if bbox is None:
                bbox = foreground_bbox_from_image(image)
            if bbox is not None:
                width, height = image.size
                bbox = pad_square_bbox(
                    bbox, width, height, self.crop_pad
                )
                image = image.crop(bbox)

        if self.transform is not None:
            image = self.transform(image)

        if self.multi_task:
            labels = {}
            label_names = {}
            for task, column in self.target_cols.items():
                label_name = row[column]
                if is_missing_label(label_name):
                    encoded = self.missing_label_index
                    label_names[task] = "<MISSING>"
                else:
                    label_name = str(label_name)
                    encoded = self.label_to_index_by_task[task].get(
                        label_name, self.missing_label_index
                    )
                    label_names[task] = label_name
                labels[task] = torch.tensor(encoded, dtype=torch.long)
            metadata_label_names = {
                task: (
                    "<MISSING>"
                    if column not in row or is_missing_label(row[column])
                    else str(row[column])
                )
                for task, column in self.metadata_cols.items()
            }
            return {
                "image": image,
                "labels": labels,
                "label_names": label_names,
                "metadata_label_names": metadata_label_names,
                "path": str(image_path),
                "barcode": barcode,
            }

        label_name = row[self.target_col]
        encoded = self.label_to_index[label_name]
        return {
            "image": image,
            "label": torch.tensor(encoded, dtype=torch.long),
            "label_name": label_name,
            "path": str(image_path),
            "barcode": barcode,
        }


class MultiViewWormImageDataset(Dataset):
    """One training item per biological individual.

    Images are sampled again whenever the epoch changes.  Sampling never mixes
    barcodes and never duplicates an image merely because an individual has
    fewer images than ``images_per_individual``.
    """

    def __init__(
        self,
        image_dataset: MultiTaskWormImageDataset,
        *,
        images_per_individual: int = 3,
        image_sampling: str = "random",
        seed: int = 0,
    ) -> None:
        if images_per_individual <= 0:
            raise ValueError("images_per_individual must be positive")
        if image_sampling not in {"random", "first"}:
            raise ValueError("image_sampling must be random or first")
        self.image_dataset = image_dataset
        self.images_per_individual = int(images_per_individual)
        self.image_sampling = image_sampling
        self.seed = int(seed)
        self.epoch = 0
        barcode_col = image_dataset.barcode_col
        if barcode_col not in image_dataset.df.columns:
            raise ValueError(
                f"Multi-view training requires barcode column {barcode_col!r}"
            )
        groups = image_dataset.df.groupby(barcode_col, sort=True).indices
        if image_dataset.target_cols:
            for task, column in image_dataset.target_cols.items():
                inconsistent = image_dataset.df.groupby(barcode_col)[column].nunique(
                    dropna=True
                )
                bad = inconsistent[inconsistent > 1]
                if len(bad):
                    raise ValueError(
                        f"Barcode-level multi-view item has inconsistent {task} "
                        f"labels for {bad.index.astype(str).tolist()[:5]}"
                    )
        self.barcodes = tuple(str(value) for value in groups)
        self.indices_by_barcode = {
            str(value): tuple(int(index) for index in indices)
            for value, indices in groups.items()
        }

    @property
    def df(self):
        return self.image_dataset.df

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.barcodes)

    def __getitem__(self, index: int):
        barcode = self.barcodes[index]
        available = list(self.indices_by_barcode[barcode])
        count = min(self.images_per_individual, len(available))
        if self.image_sampling == "random":
            rng = random.Random(self.seed + self.epoch * len(self) + index)
            selected = rng.sample(available, count)
        else:
            selected = available[:count]
        views = [self.image_dataset[item_index] for item_index in selected]
        first = views[0]
        return {
            "image": torch.stack([view["image"] for view in views]),
            "view_mask": torch.ones(len(views), dtype=torch.bool),
            "labels": first["labels"],
            "label_names": first["label_names"],
            "metadata_label_names": first.get(
                "metadata_label_names", first["label_names"]
            ),
            "path": [view["path"] for view in views],
            "barcode": barcode,
        }


def multiview_collate(batch: list[dict]) -> dict:
    """Pad variable image counts and expose a mask for mean pooling."""
    if not batch:
        raise ValueError("Cannot collate an empty multi-view batch")
    max_views = max(int(item["image"].shape[0]) for item in batch)
    prototype = batch[0]["image"]
    images = prototype.new_zeros((len(batch), max_views, *prototype.shape[1:]))
    view_mask = torch.zeros((len(batch), max_views), dtype=torch.bool)
    for row, item in enumerate(batch):
        count = int(item["image"].shape[0])
        images[row, :count] = item["image"]
        view_mask[row, :count] = True
    tasks = batch[0]["labels"]
    return {
        "image": images,
        "view_mask": view_mask,
        "labels": {
            task: torch.stack([item["labels"][task] for item in batch])
            for task in tasks
        },
        "label_names": {
            task: [item["label_names"][task] for item in batch]
            for task in tasks
        },
        "metadata_label_names": {
            task: [
                item.get("metadata_label_names", item["label_names"])[task]
                for item in batch
            ]
            for task in batch[0].get(
                "metadata_label_names", batch[0]["label_names"]
            )
        },
        "path": [item["path"] for item in batch],
        "barcode": [item["barcode"] for item in batch],
    }


__all__ = [
    "MultiTaskWormImageDataset",
    "MultiViewWormImageDataset",
    "multiview_collate",
]
