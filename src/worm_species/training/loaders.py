"""Profile-aware loader assembly for the canonical trainer."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from torch.utils.data import DataLoader

from src.cache import build_image_cache
from src.splits import make_individual_level_splits

from ..data.datasets import MultiTaskWormImageDataset
from ..data.labels import build_label_maps
from ..data.labels import get_target_cols
from ..data.labels import read_csvs_from_dir
from ..data.metadata import prepare_metadata
from ..data.transforms import build_split_transform
from .modes import TrainingProfile
from .modes import get_profile


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    label_to_index_by_task: dict
    index_to_label_by_task: dict
    split_summary: dict
    train_df: object
    target_cols: dict
    test_loader_context: dict | None = None


def require_complete_task_labels(
    split_frames: dict[str, object],
    target_cols: dict[str, str],
) -> None:
    """Fail rather than changing samples when label masking is disabled."""
    missing = []
    for split_name, frame in split_frames.items():
        for task, column in target_cols.items():
            if column not in frame.columns:
                missing.append(f"{split_name}.{task}: missing column {column!r}")
                continue
            count = int(frame[column].isna().sum())
            if count:
                missing.append(f"{split_name}.{task}: {count} missing labels")
    if missing:
        raise ValueError(
            "training.use_masked_labels=false requires every selected task "
            "label to be present; no rows were dropped. " + "; ".join(missing)
        )


def get_input_condition(cfg: dict) -> dict:
    raw = copy.deepcopy(cfg.get("input_condition", {}) or {})
    if not bool(raw.get("enabled", False)):
        return {
            "condition": "original",
            "feature": "baseline",
            "transform": "original",
            "strength": 0.0,
        }

    transform_name = str(raw.get("transform", "original")).lower()
    condition = {
        "condition": str(
            raw.get("condition") or raw.get("name") or transform_name
        ),
        "feature": str(raw.get("feature", "baseline")),
        "transform": transform_name,
        "strength": float(raw.get("strength", 0.0)),
    }
    nested_parameters = raw.get("parameters", {}) or {}
    if not isinstance(nested_parameters, dict):
        raise TypeError("input_condition.parameters must be a mapping")
    parameter_keys = {
        "retention",
        "order",
        "diameter",
        "sigma_colour",
        "sigma_space",
        "sigma",
        "grid_size",
        "seed",
    }
    for key in parameter_keys:
        value = raw.get(key, nested_parameters.get(key))
        if value is not None:
            condition[key] = value

    if transform_name == "saturation":
        condition["retention"] = float(condition.get("retention", 1.0))
        if not 0.0 <= condition["retention"] <= 1.0:
            raise ValueError(
                "input_condition.retention must be in [0, 1], got "
                f"{condition['retention']}."
            )
    elif transform_name == "channel_shuffle":
        order = condition.get("order", [2, 0, 1])
        condition["order"] = (
            [int(value.strip()) for value in order.split(",")]
            if isinstance(order, str)
            else [int(value) for value in order]
        )
    elif transform_name == "bilateral_filter":
        condition.update(
            diameter=int(condition["diameter"]),
            sigma_colour=float(condition["sigma_colour"]),
            sigma_space=float(condition["sigma_space"]),
        )
    elif transform_name == "gaussian_blur":
        condition["sigma"] = float(condition["sigma"])
    elif transform_name == "patch_shuffle":
        condition.update(
            grid_size=int(condition["grid_size"]),
            seed=int(condition.get("seed", cfg.get("seed", 0))),
        )
    elif transform_name not in {"original", "grayscale"}:
        raise ValueError(
            f"Unsupported input condition transform: {transform_name!r}."
        )

    return condition


def make_profile_loaders(cfg: dict, profile: TrainingProfile) -> LoaderBundle:
    df = prepare_metadata(cfg)
    target_cols = get_target_cols(cfg)
    group_col = cfg["data"]["group_col"]
    split_target_col = cfg["data"].get(
        "split_target_col", "__taxon_for_split__"
    )
    if split_target_col not in df.columns:
        raise ValueError(
            f"data.split_target_col={split_target_col!r} is not in the metadata "
            "dataframe. Use '__taxon_for_split__' or an existing column."
        )

    cache_enabled = cfg.get("cache", {}).get("enabled", False)
    if cache_enabled:
        df = build_image_cache(cfg, df)
        df = df[df["_cached_image_path"].notna()].reset_index(drop=True)
        image_col_for_dataset = "_cached_image_path"
        crop_to_foreground_for_dataset = False
    else:
        image_col_for_dataset = cfg["data"]["image_col"]
        crop_to_foreground_for_dataset = cfg["data"].get(
            "crop_to_foreground", True
        )

    if cfg["split"].get("use_predefined_splits", False):
        train_df, val_df, test_df = read_csvs_from_dir(
            cfg["split"]["predefined_split_dir"]
        )
    else:
        train_df, val_df, test_df = make_individual_level_splits(
            df=df,
            group_col=group_col,
            target_col=split_target_col,
            test_size=cfg["split"]["test_size"],
            val_size=cfg["split"]["val_size"],
            seed=cfg["seed"],
            root_dir=(
                cfg["split"]["predefined_split_dir"]
                if cfg["split"].get("save_splits", False)
                else None
            ),
        )

    if cfg["split"].get("use_predefined_splits", False) and cfg.get(
        "cache", {}
    ).get("enabled", False):
        print(f"Using predefined splits from {cfg['split']['predefined_split_dir']}")
        train_df = build_image_cache(cfg, train_df)
        val_df = build_image_cache(cfg, val_df)
        test_df = build_image_cache(cfg, test_df)
        train_df = train_df[
            train_df["_cached_image_path"].notna()
        ].reset_index(drop=True)
        val_df = val_df[val_df["_cached_image_path"].notna()].reset_index(
            drop=True
        )
        test_df = test_df[test_df["_cached_image_path"].notna()].reset_index(
            drop=True
        )

    if not profile.masked_labels:
        require_complete_task_labels(
            {"train": train_df, "val": val_df, "test": test_df},
            target_cols,
        )

    label_to_index_by_task, index_to_label_by_task = build_label_maps(
        train_df, target_cols
    )
    preprocessing = copy.deepcopy(cfg.get("preprocessing", {}) or {})
    if not isinstance(preprocessing, dict):
        raise TypeError("preprocessing must be a mapping")
    if "image_size" not in preprocessing:
        preprocessing["image_size"] = cfg["data"]["image_size"]
    augmentation = copy.deepcopy(cfg.get("augmentation", {}) or {})
    if not isinstance(augmentation, dict):
        raise TypeError("augmentation must be a mapping")
    image_size = preprocessing["image_size"]
    colour_retention = (
        1.0
        if profile.loader_mode == "standard"
        else float(cfg.get("data", {}).get("colour_retention", 1.0))
    )
    input_condition = (
        get_input_condition(cfg)
        if profile.loader_mode == "condition"
        else {
            "condition": "original",
            "feature": "baseline",
            "transform": "original",
            "strength": 0.0,
        }
    )

    if profile.loader_mode == "colour":
        print(
            f"Using colour_retention={colour_retention} for data augmentation."
        )
    train_tf = build_split_transform(
        split="train",
        preprocessing=preprocessing,
        augmentation=augmentation,
        condition=input_condition,
        original_colour_retention=colour_retention,
    )
    eval_tf = build_split_transform(
        split="validation",
        preprocessing=preprocessing,
        augmentation=augmentation,
        condition=input_condition,
        original_colour_retention=colour_retention,
    )

    common_kwargs = {
        "root_dir": cfg["data"]["root_dir"],
        "image_col": image_col_for_dataset,
        "target_cols": target_cols,
        "label_to_index_by_task": label_to_index_by_task,
        "mask_col": cfg["data"].get("mask_col"),
        "crop_to_foreground": crop_to_foreground_for_dataset,
        "crop_pad": cfg["data"].get("crop_pad", 0.15),
    }
    train_ds = MultiTaskWormImageDataset(
        train_df, transform=train_tf, **common_kwargs
    )
    val_ds = MultiTaskWormImageDataset(val_df, transform=eval_tf, **common_kwargs)
    test_ds = MultiTaskWormImageDataset(
        test_df, transform=eval_tf, **common_kwargs
    )

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 4)
    train_loader_kwargs = {"num_workers": num_workers, "pin_memory": True}
    eval_loader_kwargs = {"num_workers": num_workers, "pin_memory": True}
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 4
        eval_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )

    split_summary = {}
    if profile.loader_mode in {"colour", "condition"}:
        split_summary["colour_retention"] = colour_retention
    if profile.loader_mode == "condition":
        split_summary["training_condition"] = input_condition
    if profile.loader_mode in {"colour", "condition"}:
        split_summary["colour_percent"] = int(round(colour_retention * 100))
    split_summary.update(
        {
            "target_cols": target_cols,
            "split_target_col": split_target_col,
            "num_classes_by_task": {
                task: len(label_to_index)
                for task, label_to_index in label_to_index_by_task.items()
            },
            "classes_by_task": {
                task: list(label_to_index.keys())
                for task, label_to_index in label_to_index_by_task.items()
            },
            "labelled_rows_by_task": {
                task: {
                    "train": int(train_df[col].notna().sum()),
                    "val": int(val_df[col].notna().sum()),
                    "test": int(test_df[col].notna().sum()),
                }
                for task, col in target_cols.items()
            },
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_individuals": train_df[group_col].nunique(),
            "val_individuals": val_df[group_col].nunique(),
            "test_individuals": test_df[group_col].nunique(),
        }
    )

    context = None
    if profile.loader_mode == "condition":
        context = {
            "test_df": test_df,
            "dataset_kwargs": common_kwargs,
            "batch_size": batch_size,
            "loader_kwargs": eval_loader_kwargs,
            "image_size": image_size,
            "preprocessing": preprocessing,
            "augmentation": augmentation,
            "original_colour_retention": colour_retention,
            "training_condition": input_condition,
        }

    return LoaderBundle(
        train_loader,
        val_loader,
        test_loader,
        label_to_index_by_task,
        index_to_label_by_task,
        split_summary,
        train_df,
        target_cols,
        context,
    )


def make_standard_loaders(cfg: dict):
    bundle = make_profile_loaders(cfg, get_profile("masked"))
    return legacy_loader_tuple(bundle)


def legacy_loader_tuple(bundle: LoaderBundle) -> tuple:
    return (
        bundle.train_loader,
        bundle.val_loader,
        bundle.test_loader,
        bundle.label_to_index_by_task,
        bundle.index_to_label_by_task,
        bundle.split_summary,
        bundle.train_df,
        bundle.target_cols,
    )


def legacy_cue_loader_tuple(bundle: LoaderBundle) -> tuple:
    return (*legacy_loader_tuple(bundle), bundle.test_loader_context)
