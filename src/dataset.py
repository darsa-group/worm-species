from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

def resolve_path(root_dir: str | Path, path_value) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return Path(root_dir) / path

def prepare_metadata(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data"]["metadata_csv"])

    # Safer parsing of labels such as:
    # Aporrectodea_rosea_Adult
    # Aporrectodea_caliginosa_tuberculata_Juvenile
    df['species'] = df['barcode'].str.replace(r'_\d+$', '', regex=True)
    if "species_label" not in df.columns or "life_stage" not in df.columns:
        extracted = df["species"].str.extract(r"^(.+)_(Adult|Juvenile)$")
        df["species_label"] = extracted[0]
        df["life_stage"] = extracted[1]

    if "genus" not in df.columns:
        df["genus"] = df["species_label"].str.split("_").str[0]

    group_col = cfg["data"]["group_col"]

    df[group_col] = df[group_col].astype(str)

    if cfg["data"].get("strip_final_number_from_group", False):
        df[group_col] = df[group_col].str.replace(r"_(\d+)$", "", regex=True)

    target_col = cfg["data"]["target_col"]

    df = df.dropna(subset=[
        cfg["data"]["image_col"],
        target_col,
        group_col,
    ]).reset_index(drop=True)
    
    #remove invalid images
    def is_valid_image(path: str) -> bool:
        try:
            img = Image.open(path)
            img.verify()  # Verify that the image can be opened
            return True
        except (IOError, SyntaxError):
            return False
    print(f"Initial dataset size: {len(df)}")
    df = df[df[cfg["data"]["image_col"]].apply(lambda x: is_valid_image(Path(cfg["data"]["root_dir"]) / x))].reset_index(drop=True)
    print(f"After removing invalid images, dataset size: {len(df)}")
    # Remove rare classes using number of unique individuals, not image rows.
    min_n = cfg["data"].get("min_individuals_per_class", 1)

    class_counts = (
        df.groupby(target_col)[group_col]
        .nunique()
        .sort_values(ascending=False)
    )

    keep_classes = class_counts[class_counts >= min_n].index
    df = df[df[target_col].isin(keep_classes)].reset_index(drop=True)

    print("\nClasses retained:")
    print(
        df.groupby(target_col)[group_col]
        .nunique()
        .sort_values(ascending=False)
    )

    return df


def foreground_bbox_from_image(img: Image.Image) -> tuple[int, int, int, int] | None:
    arr = np.asarray(img.convert("RGB"))
    gray = arr.mean(axis=2)

    mask = gray > 5

    if mask.sum() < 20:
        return None

    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def foreground_bbox_from_mask(mask_path: Path) -> tuple[int, int, int, int] | None:
    if not mask_path.exists():
        return None

    mask = Image.open(mask_path).convert("L")
    arr = np.asarray(mask)

    fg = arr > 0

    if fg.sum() < 20:
        return None

    ys, xs = np.where(fg)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def pad_square_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    side = max(bw, bh)
    side = int(side * (1.0 + pad))

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    nx1 = max(0, cx - side // 2)
    ny1 = max(0, cy - side // 2)
    nx2 = min(img_w, nx1 + side)
    ny2 = min(img_h, ny1 + side)

    # Correct if clipping reduced the box on one side
    nx1 = max(0, nx2 - side)
    ny1 = max(0, ny2 - side)

    return nx1, ny1, nx2, ny2


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            # transforms.ColorJitter(
            #     brightness=0.15,
            #     contrast=0.15,
            #     saturation=0.10,
            #     hue=0.03,
            # ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class WormImageDataset(Dataset):
    """
    Dataset for raw, segmented, or mask images.

    Required metadata columns:
    - image path column, e.g. rel_path_seg
    - target column, e.g. species_label
    - group column, e.g. barcode
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root_dir: str | Path,
        image_col: str,
        target_col: str,
        label_to_index: dict[str, int],
        transform=None,
        mask_col: str | None = None,
        crop_to_foreground: bool = True,
        crop_pad: float = 0.15,
    ):
        self.df = df.reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.image_col = image_col
        self.mask_col = mask_col
        self.target_col = target_col
        self.label_to_index = label_to_index
        self.transform = transform
        self.crop_to_foreground = crop_to_foreground
        self.crop_pad = crop_pad

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = resolve_path(self.root_dir, row[self.image_col])
        img = Image.open(img_path).convert("RGB")

        if self.crop_to_foreground:
            bbox = None

            if self.mask_col is not None and self.mask_col in row and pd.notna(row[self.mask_col]):
                mask_path = self.root_dir / row[self.mask_col]
                bbox = foreground_bbox_from_mask(mask_path)

            if bbox is None:
                bbox = foreground_bbox_from_image(img)

            if bbox is not None:
                w, h = img.size
                bbox = pad_square_bbox(bbox, w, h, self.crop_pad)
                img = img.crop(bbox)

        if self.transform is not None:
            img = self.transform(img)

        label_name = row[self.target_col]
        y = self.label_to_index[label_name]

        return {
            "image": img,
            "label": torch.tensor(y, dtype=torch.long),
            "label_name": label_name,
            "path": str(img_path),
        }