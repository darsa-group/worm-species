from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


MISSING_LABEL = -1
DEFAULT_MISSING_VALUES = {
    "", "na", "n/a", "nan", "none", "null", "unknown", "unidentified", "missing", "not_available",
}


def resolve_path(root_dir: str | Path, path_value) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return Path(root_dir) / path

MISSING_LABEL_INDEX = -1


def is_missing_label(value, missing_values: list[str] | None = None) -> bool:
    """
    Return True if a label should be treated as missing.

    Missing labels are ignored for the relevant task only.
    For example, a missing species label still allows the image
    to train the genus and life-stage heads.
    """
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass

    text = str(value).strip()

    if text == "":
        return True

    default_missing = {
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "missing",
        "unknown",
        "unidentified",
    }

    if missing_values is not None:
        default_missing.update(str(v).strip().lower() for v in missing_values)

    return text.lower() in default_missing


def strip_final_number(value: str) -> str:
    """
    Remove final barcode replicate number.

    Example:
    Aporrectodea_rosea_Adult_12
    -> Aporrectodea_rosea_Adult
    """
    return re.sub(r"_\d+$", "", str(value))


def parse_taxonomy_from_barcode(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Create genus, species_label, life_stage, and taxon_label columns.

    This handles labels such as:
    - Aporrectodea_rosea_Adult
    - Aporrectodea_caliginosa_tuberculata_Juvenile
    - Lumbricus_sp
    - Lumbricus_sp_terrestris_herculeus

    Important:
    - taxon_label keeps the original practical ID category.
    - species_label is the strict species target and may be masked.
    """
    data_cfg = cfg["data"]
    barcode_col = data_cfg.get("barcode_col", "barcode")

    if barcode_col not in df.columns:
        raise ValueError(f"barcode_col='{barcode_col}' not found in metadata.")

    barcode_base = df[barcode_col].astype(str).map(strip_final_number)

    # Extract life stage if it is encoded at the end of the barcode label.
    extracted = barcode_base.str.extract(r"^(.+)_(Adult|Juvenile)$")

    parsed_taxon = extracted[0].where(extracted[0].notna(), barcode_base)
    parsed_life_stage = extracted[1]

    # Preserve the original practical identification category.
    # This may include uncertain categories such as Lumbricus_sp.
    if "taxon_label" not in df.columns:
        df["taxon_label"] = parsed_taxon

    # Create or fill species_label.
    if "species_label" not in df.columns:
        df["species_label"] = parsed_taxon
    else:
        df["species_label"] = df["species_label"].where(
            df["species_label"].notna(),
            parsed_taxon,
        )

    # Create or fill life_stage.
    if "life_stage" not in df.columns:
        df["life_stage"] = parsed_life_stage
    else:
        df["life_stage"] = df["life_stage"].where(
            df["life_stage"].notna(),
            parsed_life_stage,
        )

    # Genus should be inferred from the practical taxon label,
    # not from strict species_label, because species_label may be masked.
    if "genus" not in df.columns:
        df["genus"] = df["taxon_label"].astype(str).str.split("_").str[0]

    return df


def apply_taxonomic_uncertainty_rules(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Mask uncertain labels for the strict species task.

    Example behaviour:
    Lumbricus_sp
    -> genus = Lumbricus
    -> species_label = missing
    -> life_stage = kept

    Aporrectodea_caliginosa_tuberculata
    -> genus = Aporrectodea
    -> species_label = missing
    -> life_stage = kept

    Lumbricus_sp_terrestris_herculeus
    -> genus = Lumbricus
    -> species_label = Lumbricus_terrestris_herculeus
    -> life_stage = Adult
    """
    data_cfg = cfg["data"]
    uncertainty_cfg = data_cfg.get("taxonomic_uncertainty", {})

    missing_values = data_cfg.get("missing_label_values", [])

    uncertain_labels = set(
        str(x).strip()
        for x in uncertainty_cfg.get("uncertain_species_labels", [])
    )

    uncertain_patterns = [
        re.compile(pattern)
        for pattern in uncertainty_cfg.get("uncertain_species_patterns", [])
    ]

    resolved_overrides = {
        str(k).strip(): str(v).strip()
        for k, v in uncertainty_cfg.get("resolved_species_label_overrides", {}).items()
    }

    life_stage_overrides = {
        str(k).strip(): str(v).strip()
        for k, v in uncertainty_cfg.get("life_stage_overrides", {}).items()
    }

    raw_taxon = df["taxon_label"].astype(str).str.strip()

    # Apply explicit resolved species overrides first.
    # This prevents Lumbricus_sp_terrestris_herculeus from being masked
    # by the general "_sp_" uncertainty rule.
    for raw_label, resolved_label in resolved_overrides.items():
        mask = raw_taxon == raw_label
        df.loc[mask, "species_label"] = resolved_label

    # Apply life-stage overrides.
    for raw_label, stage in life_stage_overrides.items():
        mask = raw_taxon == raw_label
        df.loc[mask, "life_stage"] = stage

    # Recompute after overrides.
    species_text = df["species_label"].astype(str).str.strip()
    raw_taxon = df["taxon_label"].astype(str).str.strip()

    has_resolved_override = raw_taxon.isin(resolved_overrides.keys())

    uncertain_mask = raw_taxon.isin(uncertain_labels)

    for pattern in uncertain_patterns:
        uncertain_mask = uncertain_mask | species_text.apply(
            lambda x: bool(pattern.search(x))
        )

    # Do not mask explicitly resolved labels.
    uncertain_mask = uncertain_mask & ~has_resolved_override

    if uncertain_mask.any():
        print("\nStrict species labels masked because of taxonomic uncertainty:")
        print(df.loc[uncertain_mask, "taxon_label"].value_counts())

    df.loc[uncertain_mask, "species_label"] = pd.NA

    # Also mask generic missing values.
    for col in ["genus", "species_label", "life_stage"]:
        if col in df.columns:
            missing_mask = df[col].apply(
                lambda x: is_missing_label(x, missing_values)
            )
            df.loc[missing_mask, col] = pd.NA

    # Split label: use strict species if available, otherwise genus.
    # This avoids discarding genus-only specimens.
    df["__taxon_for_split__"] = df["species_label"].where(
        df["species_label"].notna(),
        df["genus"],
    )

    return df


def mask_rare_classes_per_task(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    For multi-task training, rare labels should be masked per task,
    not removed as whole rows.

    Example:
    If a species label is too rare, the image can still train genus and age.
    """
    data_cfg = cfg["data"]
    target_cols = data_cfg["target_cols"]
    group_col = data_cfg["group_col"]
    min_n = data_cfg.get("min_individuals_per_class", 1)
    print(f"\nMasking rare classes per task (min_individuals_per_class={min_n})")
    for task, col in target_cols.items():
        if col not in df.columns:
            raise ValueError(f"Target column for task '{task}' not found: {col}")

        valid = df[col].notna()

        class_counts = (
            df.loc[valid]
            .groupby(col)[group_col]
            .nunique()
            .sort_values(ascending=False)
        )

        keep_classes = class_counts[class_counts >= min_n].index
        rare_classes = class_counts[class_counts < min_n].index

        if len(rare_classes) > 0:
            df.loc[df[col].isin(rare_classes), col] = pd.NA

        print(f"\nClasses retained for task '{task}' ({col}):")
        print(class_counts.loc[class_counts.index.isin(keep_classes)])

        if len(rare_classes) > 0:
            print(f"\nClasses masked for task '{task}' because they are rare:")
            print(class_counts.loc[class_counts.index.isin(rare_classes)])

    # Keep rows that have at least one usable target.
    target_col_names = list(target_cols.values())
    has_any_label = df[target_col_names].notna().any(axis=1)
    df = df.loc[has_any_label].reset_index(drop=True)

    return df

def _missing_values_from_cfg(cfg: dict) -> set[str]:
    values = cfg.get("data", {}).get("missing_label_values", [])
    return DEFAULT_MISSING_VALUES | {str(v).strip().lower() for v in values}


def _normalise_missing_series(s: pd.Series, missing_values: set[str]) -> pd.Series:
    """Convert common missing-label strings to pd.NA while preserving valid labels."""
    out = s.copy()
    out = out.replace(r"^\s*$", pd.NA, regex=True)
    mask = out.astype("string").str.strip().str.lower().isin(missing_values)
    out = out.mask(mask, pd.NA)
    return out


def _clean_label_value(value: Any, missing_values: set[str]) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value.lower() in missing_values:
        return None
    return value


def get_target_cols(cfg: dict) -> dict[str, str]:
    """
    Return task -> dataframe column mapping.

    Defaults are matched to the metadata produced by prepare_metadata_multitask():
        genus   -> genus
        species -> species_label
        age     -> life_stage
    """
    target_cols = cfg.get("data", {}).get("target_cols")

    if target_cols is None:
        target_cols = {
            "genus": "genus",
            "species": "species_label",
            "age": "life_stage",
        }

    if not isinstance(target_cols, dict) or len(target_cols) == 0:
        raise ValueError(
            "data.target_cols must be a non-empty mapping, for example: "
            "{genus: genus, species: species_label, age: life_stage}"
        )

    return target_cols


def _derive_taxonomy_and_stage(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Derive genus, species_label and life_stage from barcode-like strings.

    Expected examples:
        Aporrectodea_rosea_Adult
        Aporrectodea_caliginosa_tuberculata_Juvenile
        Lumbricus_Adult

    The last example is interpreted as genus-only. It gives:
        genus = Lumbricus
        species_label = missing
        life_stage = Adult
    """
    df = df.copy()
    barcode_col = cfg.get("data", {}).get("barcode_col", "barcode")

    if barcode_col not in df.columns:
        # If your metadata has explicit genus/species/life_stage columns, this is fine.
        return df

    barcode_base = df[barcode_col].astype("string").str.replace(r"_\d+$", "", regex=True)

    extracted = barcode_base.str.extract(r"^(.+)_(Adult|Juvenile)$", expand=True)
    taxon_from_barcode = extracted[0]
    stage_from_barcode = extracted[1]

    if "life_stage" not in df.columns:
        df["life_stage"] = stage_from_barcode
    else:
        df["life_stage"] = df["life_stage"].where(df["life_stage"].notna(), stage_from_barcode)

    if "species_label" not in df.columns:
        df["species_label"] = taxon_from_barcode
    else:
        df["species_label"] = df["species_label"].where(df["species_label"].notna(), taxon_from_barcode)

    # If there is an old column called species, keep it if useful, but do not rely on it.
    if "species" not in df.columns:
        df["species"] = barcode_base

    # Derive genus from the most specific available taxon string.
    taxon_for_genus = df["species_label"].where(df["species_label"].notna(), taxon_from_barcode)
    if "genus" not in df.columns:
        df["genus"] = taxon_for_genus.astype("string").str.split("_").str[0]
    else:
        inferred_genus = taxon_for_genus.astype("string").str.split("_").str[0]
        df["genus"] = df["genus"].where(df["genus"].notna(), inferred_genus)

    # Important: a taxon string with no underscore is genus-only, not species-level ID.
    # Example: Lumbricus_Adult -> species_label should be missing.
    species_requires_binomial = cfg.get("data", {}).get("species_requires_binomial", True)
    if species_requires_binomial and "species_label" in df.columns:
        species_as_str = df["species_label"].astype("string")
        genus_only = df["species_label"].notna() & ~species_as_str.str.contains("_", regex=False, na=False)
        df.loc[genus_only, "species_label"] = pd.NA

    # Convenience alias if you prefer to call the task 'age' in the config.
    if "age" not in df.columns and "life_stage" in df.columns:
        df["age"] = df["life_stage"]

    return df


# def prepare_metadata_multitask(cfg: dict) -> pd.DataFrame:
    """
    Prepare metadata for masked multi-task classification.

    Difference from the single-task prepare_metadata:
    - only image path and group id are required for keeping a row;
    - genus/species/age labels may be missing;
    - missing task labels are retained and later encoded as -1;
    - rare classes are masked per task rather than removing the full image row.
    """
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["metadata_csv"])
    df = _derive_taxonomy_and_stage(df, cfg)
    target_col = data_cfg.get("target_col", "species_label")
    target_cols = get_target_cols(cfg)
    missing_values = _missing_values_from_cfg(cfg)

    image_col = data_cfg["image_col"]
    group_col = data_cfg["group_col"]

    required_cols = [image_col, group_col]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required metadata columns: {missing_required}")

    missing_target_cols = [col for col in target_cols.values() if col not in df.columns]
    if missing_target_cols:
        raise ValueError(
            f"Missing target columns for multi-task training: {missing_target_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    df[group_col] = df[group_col].astype(str)

    if data_cfg.get("strip_final_number_from_group", False):
        df[group_col] = df[group_col].str.replace(r"_(\d+)$", "", regex=True)

    # Keep rows with image path and group id. Do not require every label.
    df = df.dropna(subset=[image_col, group_col]).reset_index(drop=True)

    # Normalise missing values in each target column.
    for col in set(target_cols.values()):
        df[col] = _normalise_missing_series(df[col], missing_values)
        valid = df[col].notna()
        df.loc[valid, col] = df.loc[valid, col].astype(str).str.strip()

    # Remove invalid images.
    def is_valid_image(path_value) -> bool:
        try:
            img_path = resolve_path(data_cfg["root_dir"], path_value)
            with Image.open(img_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    print(f"Initial dataset size: {len(df)}")
    df = df[df[image_col].apply(is_valid_image)].reset_index(drop=True)
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
        .sort_values(ascending=False))
    # Mask rare classes per task using number of unique individuals, not image rows.
    default_min_n = data_cfg.get("min_individuals_per_class", 1)
    min_n_by_task = data_cfg.get("min_individuals_per_class_by_task", {})

    for task, col in target_cols.items():
        min_n = min_n_by_task.get(task, default_min_n)
        labelled = df[df[col].notna()]

        if labelled.empty:
            print(f"\nTask '{task}' has no labels in column '{col}'.")
            continue

        class_counts = labelled.groupby(col)[group_col].nunique().sort_values(ascending=False)
        keep_classes = class_counts[class_counts >= min_n].index
        rare_classes = class_counts[class_counts < min_n].index

        # Do not remove the image row; just remove this task label.
        if len(rare_classes) > 0:
            df.loc[df[col].isin(rare_classes), col] = pd.NA

        print(f"\nClasses retained for task '{task}' ({col}):")
        retained_counts = (
            df[df[col].notna()]
            .groupby(col)[group_col]
            .nunique()
            .sort_values(ascending=False)
        )
        print(retained_counts)

    # Keep rows that have at least one usable task label.
    any_label = np.zeros(len(df), dtype=bool)
    for col in target_cols.values():
        any_label |= df[col].notna().to_numpy()
    df = df.loc[any_label].reset_index(drop=True)

    # Internal split column: use species when present, otherwise genus.
    species_col = target_cols.get("species", "species_label")
    genus_col = target_cols.get("genus", "genus")

    split_col = data_cfg.get("split_target_col", "__taxon_for_split__")
    if split_col == "__taxon_for_split__":
        df["__taxon_for_split__"] = df[species_col].where(df[species_col].notna(), df[genus_col])
        df["__taxon_for_split__"] = df["__taxon_for_split__"].fillna("unknown_taxon")

    print("\nLabel availability:")
    for task, col in target_cols.items():
        print(f"  {task:>8s}: {int(df[col].notna().sum())} labelled / {len(df)} rows")
    
    df.to_csv(data_cfg.get("prepared_metadata_csv", "prepared_metadata_multitask.csv"), index=False)
    return df

def prepare_metadata(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data"]["metadata_csv"])

    data_cfg = cfg["data"]
    image_col = data_cfg["image_col"]
    group_col = data_cfg["group_col"]
    target_col = data_cfg.get("target_col", "species_label")

    # ------------------------------------------------------------
    # Shared preprocessing
    # ------------------------------------------------------------
    df = parse_taxonomy_from_barcode(df, cfg)
    df = apply_taxonomic_uncertainty_rules(df, cfg)

    df[group_col] = df[group_col].astype(str)
    min_n = cfg["data"].get("min_individuals_per_class", 1)

    class_counts = (
        df.groupby(target_col)[group_col]
        .nunique()
        .sort_values(ascending=False)
    )

    keep_classes = class_counts[class_counts >= min_n].index
    df = df[df[target_col].isin(keep_classes)].reset_index(drop=True)
    if data_cfg.get("strip_final_number_from_group", False):
        df[group_col] = df[group_col].str.replace(r"_(\d+)$", "", regex=True)

    # ------------------------------------------------------------
    # Drop only rows missing image path or group.
    # Do not drop rows just because species is missing.
    # ------------------------------------------------------------
    df = df.dropna(subset=[
        image_col,
        group_col,
    ]).reset_index(drop=True)

    # ------------------------------------------------------------
    # Remove invalid images
    # ------------------------------------------------------------
    def is_valid_image(path: Path) -> bool:
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except (IOError, SyntaxError, FileNotFoundError):
            return False

    print(f"Initial dataset size: {len(df)}")

    df = df[
        df[image_col].apply(
            lambda x: is_valid_image(resolve_path(data_cfg["root_dir"], x))
        )
    ].reset_index(drop=True)

    print(f"After removing invalid images, dataset size: {len(df)}")

    # ------------------------------------------------------------
    # Multi-task case: genus + species + age
    # ------------------------------------------------------------
    if "target_cols" in data_cfg:
        df = mask_rare_classes_per_task(df, cfg)

        print("\nFinal usable labels per task:")
        for task, col in data_cfg["target_cols"].items():
            print(f"{task}: {df[col].notna().sum()} labelled rows")

        return df

    # ------------------------------------------------------------
    # Backward-compatible single-task case
    # ------------------------------------------------------------
    target_col = data_cfg["target_col"]

    df = df.dropna(subset=[
        target_col,
    ]).reset_index(drop=True)

    min_n = data_cfg.get("min_individuals_per_class", 1)

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

    # Correct if clipping reduced the box on one side.
    nx1 = max(0, nx2 - side)
    ny1 = max(0, ny2 - side)

    return nx1, ny1, nx2, ny2


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=270),
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


class MultiTaskWormImageDataset(Dataset):
    """
    Dataset for raw, segmented, or mask images with partially missing labels.

    Required metadata columns:
    - image path column, e.g. rel_path_seg
    - group column, e.g. barcode

    Label behaviour:
    - valid labels are mapped to class indices;
    - missing labels or labels unseen in training are returned as -1;
    - the training loop masks -1 labels for the corresponding task.
    """

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
        else:
            if target_col is None or label_to_index is None:
                raise ValueError(
                    "target_col and label_to_index must be provided for single-task training."
                )
        

    def __len__(self) -> int:
        return len(self.df)

    def _encode_label(self, task: str, value) -> int:
        cleaned = _clean_label_value(value, self.missing_values)
        if cleaned is None:
            return MISSING_LABEL
        return self.label_to_index_by_task[task].get(cleaned, MISSING_LABEL)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = resolve_path(self.root_dir, row[self.image_col])
        img = Image.open(img_path).convert("RGB")

        if self.crop_to_foreground:
            bbox = None

            if self.mask_col is not None and self.mask_col in row and pd.notna(row[self.mask_col]):
                mask_path = resolve_path(self.root_dir, row[self.mask_col])
                bbox = foreground_bbox_from_mask(mask_path)

            if bbox is None:
                bbox = foreground_bbox_from_image(img)

            if bbox is not None:
                w, h = img.size
                bbox = pad_square_bbox(bbox, w, h, self.crop_pad)
                img = img.crop(bbox)

        if self.transform is not None:
            img = self.transform(img)

          # ------------------------------------------------------------
        # Multi-task output
        # ------------------------------------------------------------
        if self.multi_task:
            labels = {}
            label_names = {}

            for task, col in self.target_cols.items():
                label_name = row[col]

                if is_missing_label(label_name):
                    y = self.missing_label_index
                    label_names[task] = "<MISSING>"
                else:
                    label_name = str(label_name)
                    y = self.label_to_index_by_task[task].get(
                        label_name,
                        self.missing_label_index,
                    )
                    label_names[task] = label_name

                labels[task] = torch.tensor(y, dtype=torch.long)

            return {
                "image": img,
                "labels": labels,
                "label_names": label_names,
                "path": str(img_path),
            }

        # ------------------------------------------------------------
        # Backward-compatible single-task output
        # ------------------------------------------------------------
        label_name = row[self.target_col]
        y = self.label_to_index[label_name]

        return {
            "image": img,
            "label": torch.tensor(y, dtype=torch.long),
            "label_name": label_name,
            "path": str(img_path),
        }
