from __future__ import annotations

import pandas as pd

from .image_validation import is_valid_image, resolve_path
from .taxonomy import apply_taxonomic_uncertainty_rules, parse_taxonomy_from_barcode


def mask_rare_classes_per_task(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Mask rare task labels without discarding rows useful to other tasks."""
    data_cfg = cfg["data"]
    target_cols = data_cfg["target_cols"]
    group_col = data_cfg["group_col"]
    min_n = data_cfg.get("min_individuals_per_class", 1)
    print(f"\nMasking rare classes per task (min_individuals_per_class={min_n})")
    for task, column in target_cols.items():
        if column not in df.columns:
            raise ValueError(f"Target column for task '{task}' not found: {column}")
        valid = df[column].notna()
        class_counts = (
            df.loc[valid].groupby(column)[group_col].nunique().sort_values(
                ascending=False
            )
        )
        keep_classes = class_counts[class_counts >= min_n].index
        rare_classes = class_counts[class_counts < min_n].index
        if len(rare_classes) > 0:
            df.loc[df[column].isin(rare_classes), column] = pd.NA
        print(f"\nClasses retained for task '{task}' ({column}):")
        print(class_counts.loc[class_counts.index.isin(keep_classes)])
        if len(rare_classes) > 0:
            print(f"\nClasses masked for task '{task}' because they are rare:")
            print(class_counts.loc[class_counts.index.isin(rare_classes)])

    target_col_names = list(target_cols.values())
    has_any_label = df[target_col_names].notna().any(axis=1)
    return df.loc[has_any_label].reset_index(drop=True)


def prepare_metadata(cfg: dict) -> pd.DataFrame:
    """Prepare metadata using the existing multi-task scientific rules."""
    df = pd.read_csv(cfg["data"]["metadata_csv"])
    data_cfg = cfg["data"]
    image_col = data_cfg["image_col"]
    group_col = data_cfg["group_col"]
    target_col = data_cfg.get("target_col", "species_label")

    df = parse_taxonomy_from_barcode(df, cfg)
    df = apply_taxonomic_uncertainty_rules(df, cfg)
    df[group_col] = df[group_col].astype(str)
    min_n = data_cfg.get("min_individuals_per_class", 1)
    class_counts = (
        df.groupby(target_col)[group_col].nunique().sort_values(ascending=False)
    )
    keep_classes = class_counts[class_counts >= min_n].index
    df = df[df[target_col].isin(keep_classes)].reset_index(drop=True)
    if data_cfg.get("strip_final_number_from_group", False):
        df[group_col] = df[group_col].str.replace(r"_(\d+)$", "", regex=True)

    df = df.dropna(subset=[image_col, group_col]).reset_index(drop=True)
    print(f"Initial dataset size: {len(df)}")
    df = df[
        df[image_col].apply(
            lambda value: is_valid_image(resolve_path(data_cfg["root_dir"], value))
        )
    ].reset_index(drop=True)
    print(f"After removing invalid images, dataset size: {len(df)}")

    if "target_cols" in data_cfg:
        df = mask_rare_classes_per_task(df, cfg)
        print("\nFinal usable labels per task:")
        for task, column in data_cfg["target_cols"].items():
            print(f"{task}: {df[column].notna().sum()} labelled rows")
        return df

    df = df.dropna(subset=[data_cfg["target_col"]]).reset_index(drop=True)
    target_col = data_cfg["target_col"]
    min_n = data_cfg.get("min_individuals_per_class", 1)
    class_counts = (
        df.groupby(target_col)[group_col].nunique().sort_values(ascending=False)
    )
    keep_classes = class_counts[class_counts >= min_n].index
    df = df[df[target_col].isin(keep_classes)].reset_index(drop=True)
    print("\nClasses retained:")
    print(df.groupby(target_col)[group_col].nunique().sort_values(ascending=False))
    return df
