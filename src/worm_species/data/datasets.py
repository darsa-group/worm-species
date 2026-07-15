from __future__ import annotations

from pathlib import Path

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
        image_path = resolve_path(self.root_dir, row[self.image_col])
        image = Image.open(image_path).convert("RGB")

        if self.crop_to_foreground:
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
            return {
                "image": image,
                "labels": labels,
                "label_names": label_names,
                "path": str(image_path),
            }

        label_name = row[self.target_col]
        encoded = self.label_to_index[label_name]
        return {
            "image": image,
            "label": torch.tensor(encoded, dtype=torch.long),
            "label_name": label_name,
            "path": str(image_path),
        }
