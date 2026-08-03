"""Profile-aware loader assembly for the canonical trainer."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from torch.utils.data import DataLoader

from src.cache import build_image_cache
from src.splits import make_individual_level_splits

from ..cache.condition_variants import DEFAULT_TRANSFORMS
from ..cache.condition_variants import TENSOR_COLUMN
from ..cache.condition_variants import attach_condition_cache
from ..cache.condition_variants import condition_cache_settings
from ..data.datasets import MultiTaskWormImageDataset
from ..data.datasets import MultiViewWormImageDataset
from ..data.datasets import multiview_collate
from ..data.holdouts import apply_data_holdout
from ..data.labels import build_label_maps
from ..data.labels import get_target_cols
from ..data.labels import read_csvs_from_dir
from ..data.metadata import prepare_metadata
from ..data.samplers import CrossSpeciesStageContrastiveBatchSampler
from ..data.samplers import JointSpeciesStageSampler
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
    val_df: object
    test_df: object
    target_cols: dict
    test_loader_context: dict | None = None
    data_holdout_loader: DataLoader | None = None
    data_holdout_loaders: dict[str, DataLoader] | None = None
    data_holdout_audit: dict | None = None
    sampler_summary: object | None = None
    holdout_frames: dict[str, object] | None = None
    multiview_evaluation_max_images: int | None = None


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
        "percent",
        "max_sigma",
        "operations",
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
    elif transform_name == "gaussian_blur_percent":
        condition["percent"] = float(condition["percent"])
        condition["max_sigma"] = float(condition.get("max_sigma", 16.0))
    elif transform_name == "resolution_loss":
        condition["percent"] = float(condition["percent"])
    elif transform_name == "patch_shuffle":
        condition.update(
            grid_size=int(condition["grid_size"]),
            seed=int(condition.get("seed", cfg.get("seed", 0))),
        )
    elif transform_name == "composed":
        operations = condition.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ValueError(
                "input_condition.operations must be a non-empty list for composed"
            )
        condition["operations"] = copy.deepcopy(operations)
    elif transform_name not in {"original", "grayscale"}:
        raise ValueError(
            f"Unsupported input condition transform: {transform_name!r}."
        )

    return condition


def make_profile_loaders(cfg: dict, profile: TrainingProfile) -> LoaderBundle:
    df = prepare_metadata(cfg)
    all_target_cols = get_target_cols(cfg)
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

    original_split_frames = {
        "train": train_df.copy(),
        "validation": val_df.copy(),
        "test": test_df.copy(),
    }
    holdout = apply_data_holdout(
        config=cfg,
        train=train_df,
        validation=val_df,
        test=test_df,
        target_cols=all_target_cols,
        group_col=group_col,
    )
    train_df = holdout.train
    val_df = holdout.validation
    test_df = holdout.test
    development_cohort = holdout.development_cohort
    test_cohort = holdout.test_cohort

    architecture = str(
        (cfg.get("model", {}) or {}).get(
            "multitask_architecture", "shared_heads"
        )
    )
    if architecture == "single_task":
        target_task = str(
            (cfg.get("model", {}) or {}).get("target_task", "")
        )
        if target_task not in all_target_cols:
            raise ValueError(
                f"model.target_task={target_task!r} is not in "
                f"data.target_cols={list(all_target_cols)}"
            )
        target_cols = {target_task: all_target_cols[target_task]}
    else:
        target_cols = all_target_cols

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
    condition_settings = condition_cache_settings(cfg)
    condition_cache_active = (
        profile.loader_mode == "condition"
        and condition_settings["enabled"]
        and input_condition["transform"] in DEFAULT_TRANSFORMS
    )
    if condition_cache_active:
        condition_root = (
            (cfg.get("cache", {}) or {})
            .get("condition_variants", {})
            .get("root")
        )
        if not isinstance(condition_root, str) or not condition_root:
            raise ValueError(
                "cache.condition_variants.root is required when condition "
                "caching is enabled"
            )
        attach_kwargs = {
            "cache_root": condition_root,
            "condition": input_condition,
            "protocol_version": int(
                condition_settings["protocol_version"]
            ),
        }
        train_df = attach_condition_cache(train_df, **attach_kwargs)
        val_df = attach_condition_cache(val_df, **attach_kwargs)
        test_df = attach_condition_cache(test_df, **attach_kwargs)
        if development_cohort is not None:
            development_cohort = attach_condition_cache(
                development_cohort, **attach_kwargs
            )
        if test_cohort is not None:
            test_cohort = attach_condition_cache(
                test_cohort, **attach_kwargs
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
        condition_precomputed=condition_cache_active,
    )
    eval_tf = build_split_transform(
        split="validation",
        preprocessing=preprocessing,
        augmentation=augmentation,
        condition=input_condition,
        original_colour_retention=colour_retention,
        condition_precomputed=condition_cache_active,
    )

    base_common_kwargs = {
        "root_dir": cfg["data"]["root_dir"],
        "image_col": (
            "_cached_image_path"
            if cache_enabled
            else cfg["data"]["image_col"]
        ),
        "target_cols": target_cols,
        "metadata_cols": all_target_cols,
        "label_to_index_by_task": label_to_index_by_task,
        "mask_col": cfg["data"].get("mask_col"),
        "crop_to_foreground": (
            False
            if cache_enabled
            else cfg["data"].get("crop_to_foreground", True)
        ),
        "crop_pad": cfg["data"].get("crop_pad", 0.15),
        "image_is_tensor": False,
    }
    common_kwargs = {
        "root_dir": cfg["data"]["root_dir"],
        "image_col": (
            TENSOR_COLUMN
            if condition_cache_active
            else image_col_for_dataset
        ),
        "target_cols": target_cols,
        "metadata_cols": all_target_cols,
        "label_to_index_by_task": label_to_index_by_task,
        "mask_col": cfg["data"].get("mask_col"),
        "crop_to_foreground": (
            False
            if condition_cache_active
            else crop_to_foreground_for_dataset
        ),
        "crop_pad": cfg["data"].get("crop_pad", 0.15),
        "image_is_tensor": condition_cache_active,
        "barcode_col": group_col,
    }
    train_image_ds = MultiTaskWormImageDataset(
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

    multiview_cfg = (cfg.get("data", {}) or {}).get("multiview", {}) or {}
    multiview_enabled = bool(multiview_cfg.get("enabled", False))
    train_ds = (
        MultiViewWormImageDataset(
            train_image_ds,
            images_per_individual=int(
                multiview_cfg.get("images_per_individual", 3)
            ),
            image_sampling=str(multiview_cfg.get("image_sampling", "random")),
            seed=int(cfg.get("seed", 0)),
        )
        if multiview_enabled
        else train_image_ds
    )
    sampler_cfg = (cfg.get("data", {}) or {}).get("sampler", {}) or {}
    sampler_type = str(sampler_cfg.get("type", "default"))
    train_sampler = None
    train_batch_sampler = None
    sampler_summary = None
    if sampler_type == "joint_species_stage":
        if multiview_enabled:
            raise ValueError(
                "joint_species_stage is image-indexed and cannot be combined "
                "with data.multiview.enabled; use "
                "cross_species_stage_contrastive"
            )
        required = {"species", "age"}
        if not required.issubset(all_target_cols):
            raise ValueError(
                "joint_species_stage sampling requires species and age in "
                "data.target_cols"
            )
        train_sampler = JointSpeciesStageSampler(
            train_df,
            species_col=all_target_cols["species"],
            stage_col=all_target_cols["age"],
            group_col=group_col,
            replacement=bool(sampler_cfg.get("replacement", True)),
            samples_per_epoch=sampler_cfg.get("samples_per_epoch"),
            seed=int(cfg.get("seed", 0)),
        )
        sampler_summary = train_sampler.summary
        print("Joint species-stage training sampler:")
        print(sampler_summary.to_string(index=False))
    elif sampler_type == "cross_species_stage_contrastive":
        train_batch_sampler = CrossSpeciesStageContrastiveBatchSampler(
            train_df,
            species_col=all_target_cols["species"],
            stage_col=all_target_cols["age"],
            group_col=group_col,
            species_per_stage=int(sampler_cfg.get("species_per_stage", 3)),
            individuals_per_species_stage=int(
                sampler_cfg.get("individuals_per_species_stage", 2)
            ),
            images_per_individual=int(
                sampler_cfg.get("images_per_individual", 1)
            ),
            replacement=bool(sampler_cfg.get("replacement", True)),
            samples_per_epoch=sampler_cfg.get("samples_per_epoch"),
            seed=int(cfg.get("seed", 0)),
            individual_dataset=multiview_enabled,
        )
        sampler_summary = train_batch_sampler.summary
        print("Cross-species stage contrastive batch sampler:")
        print(sampler_summary.to_string(index=False))
    elif sampler_type != "default":
        raise ValueError(
            "data.sampler.type must be default, joint_species_stage, or "
            "cross_species_stage_contrastive"
        )

    collate_fn = multiview_collate if multiview_enabled else None
    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            **train_loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_fn,
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
    data_holdout_loader = None
    data_holdout_loaders = {}
    for cohort_name, cohort_frame in (
        ("development_withheld", development_cohort),
        ("independent_test", test_cohort),
    ):
        if cohort_frame is None:
            continue
        holdout_ds = MultiTaskWormImageDataset(
            cohort_frame, transform=eval_tf, **common_kwargs
        )
        data_holdout_loaders[cohort_name] = DataLoader(
            holdout_ds,
            batch_size=batch_size,
            shuffle=False,
            **eval_loader_kwargs,
        )
    if data_holdout_loaders:
        data_holdout_loader = data_holdout_loaders.get("independent_test")

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
    if holdout.audit is not None:
        split_summary["data_holdout"] = holdout.audit

    context = None
    if profile.loader_mode == "condition":
        context = {
            "test_df": test_df,
            "dataset_kwargs": base_common_kwargs,
            "batch_size": batch_size,
            "loader_kwargs": eval_loader_kwargs,
            "image_size": image_size,
            "preprocessing": preprocessing,
            "augmentation": augmentation,
            "original_colour_retention": colour_retention,
            "training_condition": input_condition,
            "condition_cache_active": condition_cache_active,
            "split_frames": original_split_frames,
            "target_cols": target_cols,
            "group_col": group_col,
        }

    return LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_to_index_by_task=label_to_index_by_task,
        index_to_label_by_task=index_to_label_by_task,
        split_summary=split_summary,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_cols=target_cols,
        test_loader_context=context,
        data_holdout_loader=data_holdout_loader,
        data_holdout_loaders=data_holdout_loaders or None,
        data_holdout_audit=holdout.audit,
        sampler_summary=sampler_summary,
        holdout_frames={
            name: frame
            for name, frame in (
                ("development_withheld", development_cohort),
                ("independent_test", test_cohort),
            )
            if frame is not None
        },
        multiview_evaluation_max_images=multiview_cfg.get(
            "evaluation_max_images"
        ),
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
