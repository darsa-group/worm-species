from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import copy
import json
import itertools
import math

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.dataset_multitask import (
    MISSING_LABEL,
    MultiTaskWormImageDataset,
    build_transforms,
    build_condition_transform,
    build_test_condition_transform,
    get_target_cols,
    prepare_metadata,
)
from src.cache import build_image_cache
from src.worm_species.models.multitask import MultiTaskClassifier, build_multitask_model
from src.worm_species.data.labels import build_label_maps, read_csvs_from_dir
from src.splits import make_individual_level_splits
from src.utils import (
    load_config,
    apply_overrides,
    set_nested,
    parse_scalar,
    set_seed,
    save_json,
    make_run_name,
)
from src.worm_species.config.sweeps import (
    generate_colour_retention_values,
    generate_sweep_configs as _generate_sweep_configs,
    get_colour_sweep_parameters_from_config as get_sweep_parameters_from_config,
    get_sweep_parameters_from_cli,
    parse_sweep_item,
)


# ---------------------------------------------------------------------
# Sweep utilities: same style as your current single-task script
# ---------------------------------------------------------------------


def generate_sweep_configs(
    base_cfg: dict,
    cli_sweep_items: list[str] | None = None,
) -> list[dict]:
    return _generate_sweep_configs(
        base_cfg,
        cli_sweep_items,
        include_colour_ablation=True,
    )


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def make_loaders(cfg: dict):
    df = prepare_metadata(cfg)

    target_cols = get_target_cols(cfg)
    group_col = cfg["data"]["group_col"]

    split_target_col = cfg["data"].get("split_target_col", "__taxon_for_split__")
    if split_target_col not in df.columns:
        raise ValueError(
            f"data.split_target_col={split_target_col!r} is not in the metadata dataframe. "
            "Use '__taxon_for_split__' or an existing column."
        )
    cache_enabled = cfg.get("cache", {}).get("enabled", False)
    if cache_enabled:
        df = build_image_cache(cfg, df)
        df = df[df["_cached_image_path"].notna()].reset_index(drop=True)
        image_col_for_dataset = "_cached_image_path"
        crop_to_foreground_for_dataset = False
    else:
        image_col_for_dataset = cfg["data"]["image_col"]
        crop_to_foreground_for_dataset = cfg["data"].get("crop_to_foreground", True)

    train_df, val_df, test_df = read_csvs_from_dir(cfg["split"]["predefined_split_dir"]) if cfg["split"].get("use_predefined_splits", False) else make_individual_level_splits(
        df=df,
        group_col=group_col,
        target_col=split_target_col,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        seed=cfg["seed"],
        root_dir=cfg["split"]["predefined_split_dir"] if cfg["split"].get("save_splits", False) else None,
    )

    if cfg['split'].get('use_predefined_splits', False) and cfg.get("cache", {}).get("enabled", False):
        print(f"Using predefined splits from {cfg['split']['predefined_split_dir']}")
        train_df = build_image_cache(cfg, train_df)
        val_df = build_image_cache(cfg, val_df)
        test_df = build_image_cache(cfg, test_df)
        train_df = train_df[train_df["_cached_image_path"].notna()].reset_index(drop=True)
        val_df = val_df[val_df["_cached_image_path"].notna()].reset_index(drop=True)
        test_df = test_df[test_df["_cached_image_path"].notna()].reset_index(drop=True)








    label_to_index_by_task, index_to_label_by_task = build_label_maps(train_df, target_cols)

    image_size = cfg["data"]["image_size"]
    colour_retention = float(cfg.get("data", {}).get("colour_retention", 1.0))
    if not 0.0 <= colour_retention <= 1.0:
        raise ValueError(
            f"data.colour_retention must be between 0 and 1, got {colour_retention}."
        )

    input_condition = get_input_condition(cfg)
    train_tf = build_condition_transform(
        image_size=image_size,
        train=True,
        condition=input_condition,
        original_colour_retention=colour_retention,
    )
    eval_tf = build_condition_transform(
        image_size=image_size,
        train=False,
        condition=input_condition,
        original_colour_retention=colour_retention,
    )

    common_kwargs = dict(
        root_dir=cfg["data"]["root_dir"],
        image_col=image_col_for_dataset,
        target_cols=target_cols,
        label_to_index_by_task=label_to_index_by_task,
        mask_col=cfg["data"].get("mask_col"),
        crop_to_foreground=crop_to_foreground_for_dataset,
        crop_pad=cfg["data"].get("crop_pad", 0.15),
    )

    train_ds = MultiTaskWormImageDataset(train_df, transform=train_tf, **common_kwargs)
    val_ds = MultiTaskWormImageDataset(val_df, transform=eval_tf, **common_kwargs)
    test_ds = MultiTaskWormImageDataset(test_df, transform=eval_tf, **common_kwargs)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 4)

    train_loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    eval_loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
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

    test_loader_context = {
        "test_df": test_df,
        "dataset_kwargs": common_kwargs,
        "batch_size": batch_size,
        "loader_kwargs": eval_loader_kwargs,
        "image_size": image_size,
        "original_colour_retention": colour_retention,
        "training_condition": input_condition,
    }

    split_summary = {
        "colour_retention": colour_retention,
        "training_condition": input_condition,
        "colour_percent": int(round(colour_retention * 100)),
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

    return (
        train_loader,
        val_loader,
        test_loader,
        label_to_index_by_task,
        index_to_label_by_task,
        split_summary,
        train_df,
        target_cols,
        test_loader_context,
    )


# ---------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------


def compute_individual_class_weights(
    train_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    label_to_index: dict[str, int],
) -> torch.Tensor:
    labelled = train_df[train_df[target_col].notna()]
    counts = labelled.groupby(target_col)[group_col].nunique().to_dict()

    weights = []
    for label in label_to_index:
        weights.append(1.0 / max(float(counts.get(label, 1.0)), 1.0))

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def build_criteria(
    train_df: pd.DataFrame,
    target_cols: dict[str, str],
    group_col: str,
    label_to_index_by_task: dict[str, dict[str, int]],
    device: torch.device,
) -> dict[str, nn.Module]:
    criteria = {}

    for task, col in target_cols.items():
        class_weights = compute_individual_class_weights(
            train_df=train_df,
            target_col=col,
            group_col=group_col,
            label_to_index=label_to_index_by_task[task],
        ).to(device)

        criteria[task] = nn.CrossEntropyLoss(weight=class_weights)

    return criteria


def _safe_metric(metric_fn, y_true: np.ndarray, y_pred: np.ndarray, default: float = float("nan")) -> float:
    if len(y_true) == 0:
        return default
    return float(metric_fn(y_true, y_pred))


def infer_parent_label_from_child_label(child_label: str) -> str:
    """
    Infer the parent taxon from a child taxon label.

    For species labels such as "Lumbricus terrestris" or
    "Lumbricus_terrestris", this returns "Lumbricus".
    If your labels do not contain the genus name, provide an explicit
    child_to_parent mapping in the config.
    """
    child_label = str(child_label).strip()

    if " " in child_label:
        return child_label.split()[0]
    if "_" in child_label:
        return child_label.split("_")[0]

    return child_label


def build_child_to_parent_matrix(
    label_to_index_by_task: dict[str, dict[str, int]],
    parent_task: str,
    child_task: str,
    device: torch.device,
    child_to_parent: dict[str, str] | None = None,
) -> torch.Tensor:
    """
    Build a matrix that maps child-task probabilities to parent-task probabilities.

    Example:
        parent_task = "genus"
        child_task = "species"

    The returned matrix has shape [n_child_classes, n_parent_classes].
    Multiplying species probabilities by this matrix gives the genus
    distribution implied by the species head.
    """
    if parent_task not in label_to_index_by_task:
        raise ValueError(f"Parent task {parent_task!r} is not in label_to_index_by_task.")
    if child_task not in label_to_index_by_task:
        raise ValueError(f"Child task {child_task!r} is not in label_to_index_by_task.")

    parent_to_index = label_to_index_by_task[parent_task]
    child_to_index = label_to_index_by_task[child_task]
    child_to_parent = child_to_parent or {}

    matrix = torch.zeros(
        len(child_to_index),
        len(parent_to_index),
        dtype=torch.float32,
        device=device,
    )

    missing_parent_labels = []

    for child_label, child_index in child_to_index.items():
        parent_label = child_to_parent.get(
            child_label,
            infer_parent_label_from_child_label(child_label),
        )

        if parent_label not in parent_to_index:
            missing_parent_labels.append((child_label, parent_label))
            continue

        parent_index = parent_to_index[parent_label]
        matrix[child_index, parent_index] = 1.0

    if missing_parent_labels:
        examples = ", ".join(
            f"{child!r}->{parent!r}" for child, parent in missing_parent_labels[:10]
        )
        raise ValueError(
            f"Could not map {len(missing_parent_labels)} {child_task!r} labels "
            f"to valid {parent_task!r} labels. Examples: {examples}. "
            "Either make sure species labels start with the genus name, "
            "or provide multi_task.hierarchy_loss.child_to_parent in the config."
        )

    if not torch.all(matrix.sum(dim=1) == 1):
        raise ValueError(
            f"Each {child_task!r} class must map to exactly one {parent_task!r} class."
        )

    return matrix


def hierarchy_consistency_loss(
    parent_logits: torch.Tensor,
    child_logits: torch.Tensor,
    child_to_parent_matrix: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor | None:
    """
    Penalise disagreement between a parent-task head and the parent
    distribution implied by a child-task head.

    Example:
        parent = genus
        child = species

    The loss is applied only to samples selected by valid_mask. In your case,
    this means samples where both genus and species labels are available.
    """
    if not valid_mask.any():
        return None

    parent_logits = parent_logits[valid_mask]
    child_logits = child_logits[valid_mask]

    parent_probs = F.softmax(parent_logits, dim=1)
    child_probs = F.softmax(child_logits, dim=1)

    child_to_parent_matrix = child_to_parent_matrix.to(
        device=child_probs.device,
        dtype=child_probs.dtype,
    )

    implied_parent_probs = child_probs @ child_to_parent_matrix

    # Update the parent head to agree with the child-implied parent distribution.
    parent_loss = F.kl_div(
        (parent_probs + eps).log(),
        implied_parent_probs.detach(),
        reduction="batchmean",
    )

    # Update the child head so its implied parent distribution agrees with the parent head.
    child_loss = F.kl_div(
        (implied_parent_probs + eps).log(),
        parent_probs.detach(),
        reduction="batchmean",
    )

    return 0.5 * (parent_loss + child_loss)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criteria: dict[str, nn.Module],
    optimizer,
    device: torch.device,
    train: bool,
    scaler=None,
    use_amp: bool = True,
    task_loss_weights: dict[str, float] | None = None,
    normalize_loss_by_active_tasks: bool = True,
    hierarchy_cfg: dict | None = None,
    child_to_parent_matrix: torch.Tensor | None = None,
):
    if train:
        model.train()
    else:
        model.eval()

    tasks = list(criteria.keys())
    task_loss_weights = task_loss_weights or {task: 1.0 for task in tasks}
    hierarchy_cfg = hierarchy_cfg or {}
    hierarchy_enabled = bool(hierarchy_cfg.get("enabled", False))
    hierarchy_parent_task = hierarchy_cfg.get("parent_task", "genus")
    hierarchy_child_task = hierarchy_cfg.get("child_task", "species")
    hierarchy_weight = float(
        hierarchy_cfg.get(
            "weight",
            task_loss_weights.get("hierarchy", 0.1),
        )
    )
    use_hierarchy_loss = (
        hierarchy_enabled
        and hierarchy_weight > 0.0
        and child_to_parent_matrix is not None
        and hierarchy_parent_task in tasks
        and hierarchy_child_task in tasks
    )

    losses = []
    task_losses = {task: [] for task in tasks}
    hierarchy_losses = []
    all_true = {task: [] for task in tasks}
    all_pred = {task: [] for task in tasks}

    complete_exact_correct = 0
    complete_exact_total = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = {
            task: batch["labels"][task].to(device, non_blocking=True)
            for task in tasks
        }

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(
                enabled=(use_amp and device.type == "cuda"),
                device_type=device.type,
            ):
                logits_by_task = model(x)

                total_loss = torch.zeros((), device=device)
                active_weight_sum = 0.0
                loss_by_task: dict[str, torch.Tensor | None] = {}

                for task in tasks:
                    valid = y[task] != MISSING_LABEL

                    if valid.any():
                        task_loss = criteria[task](logits_by_task[task][valid], y[task][valid])
                        weight = float(task_loss_weights.get(task, 1.0))
                        total_loss = total_loss + weight * task_loss
                        active_weight_sum += weight
                        loss_by_task[task] = task_loss
                    else:
                        loss_by_task[task] = None

                if use_hierarchy_loss:
                    hierarchy_valid = (
                        (y[hierarchy_parent_task] != MISSING_LABEL)
                        & (y[hierarchy_child_task] != MISSING_LABEL)
                    )

                    hierarchy_loss = hierarchy_consistency_loss(
                        parent_logits=logits_by_task[hierarchy_parent_task],
                        child_logits=logits_by_task[hierarchy_child_task],
                        child_to_parent_matrix=child_to_parent_matrix,
                        valid_mask=hierarchy_valid,
                    )

                    if hierarchy_loss is not None:
                        total_loss = total_loss + hierarchy_weight * hierarchy_loss
                        active_weight_sum += hierarchy_weight
                        loss_by_task["hierarchy"] = hierarchy_loss
                    else:
                        loss_by_task["hierarchy"] = None

                if active_weight_sum == 0:
                    # Extremely unlikely because prepare_metadata_multitask removes fully unlabelled rows.
                    continue

                if normalize_loss_by_active_tasks:
                    total_loss = total_loss / active_weight_sum

            if train:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

        losses.append(float(total_loss.item()))
        if use_hierarchy_loss and loss_by_task.get("hierarchy") is not None:
            hierarchy_losses.append(float(loss_by_task["hierarchy"].item()))

        complete_mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)
        complete_correct = torch.ones(x.shape[0], dtype=torch.bool, device=device)

        for task in tasks:
            pred = logits_by_task[task].argmax(dim=1)
            valid = y[task] != MISSING_LABEL

            complete_mask &= valid
            complete_correct &= pred.eq(y[task])

            if valid.any():
                if loss_by_task[task] is not None:
                    task_losses[task].append(float(loss_by_task[task].item()))
                all_true[task].extend(y[task][valid].detach().cpu().numpy().tolist())
                all_pred[task].extend(pred[valid].detach().cpu().numpy().tolist())

        if complete_mask.any():
            complete_exact_total += int(complete_mask.sum().item())
            complete_exact_correct += int((complete_correct & complete_mask).sum().item())

    metrics = {"loss": float(np.mean(losses)) if losses else float("nan")}
    if use_hierarchy_loss:
        metrics["hierarchy_loss"] = (
            float(np.mean(hierarchy_losses)) if hierarchy_losses else float("nan")
        )

    if train:
        return metrics, all_true, all_pred
    else:
        macro_f1_values = []

        for task in tasks:
            y_true = np.array(all_true[task], dtype=int)
            y_pred = np.array(all_pred[task], dtype=int)

            if len(y_true) == 0:
                metrics[f"{task}_loss"] = float("nan")
                metrics[f"{task}_n"] = 0
                metrics[f"{task}_accuracy"] = float("nan")
                metrics[f"{task}_balanced_accuracy"] = float("nan")
                metrics[f"{task}_macro_f1"] = float("nan")
                continue

            task_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            macro_f1_values.append(task_macro_f1)

            metrics[f"{task}_loss"] = float(np.mean(task_losses[task])) if task_losses[task] else float("nan")
            metrics[f"{task}_n"] = int(len(y_true))
            metrics[f"{task}_accuracy"] = _safe_metric(accuracy_score, y_true, y_pred)
            metrics[f"{task}_balanced_accuracy"] = _safe_metric(balanced_accuracy_score, y_true, y_pred)
            metrics[f"{task}_macro_f1"] = float(task_macro_f1)

        metrics["mean_macro_f1"] = float(np.mean(macro_f1_values)) if macro_f1_values else float("nan")
        metrics["complete_exact_match_accuracy"] = (
            float(complete_exact_correct / complete_exact_total)
            if complete_exact_total > 0
            else float("nan")
        )
        metrics["complete_exact_match_n"] = int(complete_exact_total)

    return metrics, all_true, all_pred



# ---------------------------------------------------------------------
# Test-time cue suppression
# ---------------------------------------------------------------------


def _inclusive_float_sequence(start: float, stop: float, step: float) -> list[float]:
    """Return an inclusive, rounded sequence in either direction."""
    start = float(start)
    stop = float(stop)
    step = abs(float(step))
    if step == 0:
        raise ValueError("Sequence step must be greater than zero.")

    direction = -1.0 if start > stop else 1.0
    values = []
    current = start
    tolerance = step * 1e-6
    if direction < 0:
        while current >= stop - tolerance:
            values.append(round(current, 10))
            current -= step
    else:
        while current <= stop + tolerance:
            values.append(round(current, 10))
            current += step

    if not values or not math.isclose(values[-1], stop, abs_tol=tolerance):
        values.append(stop)
    return values


def generate_test_cue_conditions(cfg: dict) -> list[dict]:
    """Create deterministic test conditions from ``test_cue_suppression``."""
    cue_cfg = cfg.get("test_cue_suppression", {}) or {}
    if not bool(cue_cfg.get("enabled", False)):
        return []

    conditions: list[dict] = []

    saturation_cfg = cue_cfg.get("saturation", {}) or {}
    if bool(saturation_cfg.get("enabled", True)):
        values = saturation_cfg.get("values")
        if values is None:
            values = _inclusive_float_sequence(
                saturation_cfg.get("start", 1.0),
                saturation_cfg.get("stop", 0.0),
                saturation_cfg.get("step", 0.01),
            )
        for retention in values:
            retention = float(retention)
            if not 0.0 <= retention <= 1.0:
                raise ValueError(
                    f"Saturation retention values must be in [0, 1], got {retention}."
                )
            percentage = int(round(retention * 100))
            conditions.append({
                "condition": f"saturation_{percentage:03d}pct",
                "feature": "colour",
                "transform": "saturation",
                "strength": round(float(1.0 - retention), 10),
                "retention": retention,
            })

    grayscale_cfg = cue_cfg.get("grayscale", {}) or {}
    if bool(grayscale_cfg.get("enabled", True)):
        conditions.append({
            "condition": "grayscale",
            "feature": "colour",
            "transform": "grayscale",
            "strength": 1.0,
            "retention": 0.0,
        })

    channel_cfg = cue_cfg.get("channel_shuffle", {}) or {}
    if bool(channel_cfg.get("enabled", True)):
        orders = channel_cfg.get("orders", [[2, 0, 1]])
        for order in orders:
            order = [int(i) for i in order]
            conditions.append({
                "condition": "channel_shuffle_" + "".join(str(i) for i in order),
                "feature": "colour",
                "transform": "channel_shuffle",
                "strength": 1.0,
                "order": order,
            })

    bilateral_cfg = cue_cfg.get("bilateral_filter", {}) or {}
    if bool(bilateral_cfg.get("enabled", True)):
        settings = bilateral_cfg.get("settings", [
            {"diameter": 5, "sigma_colour": 25, "sigma_space": 25},
            {"diameter": 7, "sigma_colour": 50, "sigma_space": 50},
            {"diameter": 9, "sigma_colour": 100, "sigma_space": 100},
        ])
        for setting in settings:
            diameter = int(setting["diameter"])
            sigma_colour = float(setting["sigma_colour"])
            sigma_space = float(setting["sigma_space"])
            conditions.append({
                "condition": (
                    f"bilateral_d{diameter}_c{sigma_colour:g}_s{sigma_space:g}"
                ),
                "feature": "texture",
                "transform": "bilateral_filter",
                "strength": sigma_colour,
                "diameter": diameter,
                "sigma_colour": sigma_colour,
                "sigma_space": sigma_space,
            })

    gaussian_cfg = cue_cfg.get("gaussian_blur", {}) or {}
    if bool(gaussian_cfg.get("enabled", True)):
        for sigma in gaussian_cfg.get("sigmas", [0.5, 1.0, 2.0, 4.0]):
            sigma = float(sigma)
            conditions.append({
                "condition": f"gaussian_sigma_{sigma:g}",
                "feature": "texture",
                "transform": "gaussian_blur",
                "strength": sigma,
                "sigma": sigma,
            })

    patch_cfg = cue_cfg.get("patch_shuffle", {}) or {}
    if bool(patch_cfg.get("enabled", True)):
        seed = int(patch_cfg.get("seed", cfg.get("seed", 0)))
        for grid_size in patch_cfg.get("grid_sizes", [2, 4, 8]):
            grid_size = int(grid_size)
            conditions.append({
                "condition": f"patch_shuffle_grid_{grid_size}",
                "feature": "shape",
                "transform": "patch_shuffle",
                "strength": grid_size,
                "grid_size": grid_size,
                "seed": seed,
            })

    names = [condition["condition"] for condition in conditions]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate test cue condition names: {duplicates}")
    return conditions


def _test_condition_signature(condition: dict, original_colour_retention: float) -> str:
    """Identify conditions that produce exactly the same transformed input."""
    transform_name = condition["transform"]
    if transform_name == "original":
        return f"colour_retention:{original_colour_retention:.10f}"
    if transform_name == "saturation":
        return f"colour_retention:{float(condition['retention']):.10f}"
    if transform_name == "grayscale":
        return "colour_retention:0.0000000000"
    return json.dumps(condition, sort_keys=True)


def make_test_condition_loader(test_loader_context: dict, condition: dict) -> DataLoader:
    transform = build_test_condition_transform(
        image_size=int(test_loader_context["image_size"]),
        condition=condition,
        original_colour_retention=float(
            test_loader_context["original_colour_retention"]
        ),
    )
    dataset = MultiTaskWormImageDataset(
        test_loader_context["test_df"],
        transform=transform,
        **test_loader_context["dataset_kwargs"],
    )
    return DataLoader(
        dataset,
        batch_size=int(test_loader_context["batch_size"]),
        shuffle=False,
        **test_loader_context["loader_kwargs"],
    )


def evaluate_test_cue_suppression(
    *,
    cfg: dict,
    run_name: str,
    out_dir: Path,
    model: nn.Module,
    baseline_metrics: dict,
    test_loader_context: dict,
    criteria: dict[str, nn.Module],
    target_cols: dict[str, str],
    device: torch.device,
    use_amp: bool,
    task_loss_weights: dict[str, float],
    normalize_loss_by_active_tasks: bool,
    hierarchy_cfg: dict,
    child_to_parent_matrix: torch.Tensor | None,
    wandb_run=None,
) -> dict:
    """Evaluate one fixed checkpoint under all configured test manipulations."""
    conditions = generate_test_cue_conditions(cfg)
    if not conditions:
        return {
            "enabled": False,
            "n_conditions": 0,
        }

    cue_dir = out_dir / "cue_suppression"
    cue_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        cfg.get("test_cue_suppression", {}) or {},
        cue_dir / "cue_suppression_config.json",
    )

    original_colour_retention = float(
        test_loader_context["original_colour_retention"]
    )
    original_condition = {
        "condition": "original",
        "feature": "baseline",
        "transform": "original",
        "strength": 0.0,
        "retention": original_colour_retention,
    }

    metric_cache = {
        _test_condition_signature(
            original_condition,
            original_colour_retention,
        ): baseline_metrics
    }

    condition_metric_rows = []
    ratio_rows = []

    def record_condition(condition: dict, metrics: dict, reused: bool) -> None:
        condition_metric_rows.append({
            "run_name": run_name,
            "model": cfg.get("model", {}).get("name"),
            "condition": condition["condition"],
            "feature": condition["feature"],
            "transform": condition["transform"],
            "strength": condition.get("strength"),
            "parameters": json.dumps(
                {
                    key: value
                    for key, value in condition.items()
                    if key not in {"condition", "feature", "transform", "strength"}
                },
                sort_keys=True,
            ),
            "reused_identical_evaluation": bool(reused),
            **metrics,
        })

        for task in target_cols:
            metric_key = f"{task}_macro_f1"
            transformed_score = float(metrics.get(metric_key, float("nan")))
            original_score = float(baseline_metrics.get(metric_key, float("nan")))
            if (
                math.isnan(transformed_score)
                or math.isnan(original_score)
                or original_score == 0.0
            ):
                ratio = float("nan")
            else:
                ratio = transformed_score / original_score

            ratio_rows.append({
                "run_name": run_name,
                "model": cfg.get("model", {}).get("name"),
                "task": task,
                "condition": condition["condition"],
                "feature": condition["feature"],
                "transform": condition["transform"],
                "strength": condition.get("strength"),
                "parameters": json.dumps(
                    {
                        key: value
                        for key, value in condition.items()
                        if key not in {"condition", "feature", "transform", "strength"}
                    },
                    sort_keys=True,
                ),
                "n": metrics.get(f"{task}_n"),
                "macro_f1": transformed_score,
                "original_macro_f1": original_score,
                "ratio_to_original": ratio,
                "relative_drop": 1.0 - ratio if not math.isnan(ratio) else float("nan"),
            })

    record_condition(original_condition, baseline_metrics, reused=True)

    for condition_index, condition in enumerate(conditions, start=1):
        signature = _test_condition_signature(
            condition,
            original_colour_retention,
        )
        reused = signature in metric_cache
        if reused:
            metrics = metric_cache[signature]
            print(
                f"Cue test {condition_index}/{len(conditions)}: "
                f"{condition['condition']} reuses an identical evaluation."
            )
        else:
            print(
                f"Cue test {condition_index}/{len(conditions)}: "
                f"{condition['condition']}"
            )
            condition_loader = make_test_condition_loader(
                test_loader_context,
                condition,
            )
            metrics, _, _ = run_epoch(
                model=model,
                loader=condition_loader,
                criteria=criteria,
                optimizer=None,
                device=device,
                train=False,
                scaler=None,
                use_amp=use_amp,
                task_loss_weights=task_loss_weights,
                normalize_loss_by_active_tasks=normalize_loss_by_active_tasks,
                hierarchy_cfg=hierarchy_cfg,
                child_to_parent_matrix=child_to_parent_matrix,
            )
            metric_cache[signature] = metrics

        record_condition(condition, metrics, reused=reused)

    condition_metrics_df = pd.DataFrame(condition_metric_rows)
    ratios_df = pd.DataFrame(ratio_rows)
    condition_metrics_path = cue_dir / "test_condition_metrics.csv"
    ratios_path = cue_dir / "macro_f1_ratios.csv"
    condition_metrics_df.to_csv(condition_metrics_path, index=False)
    ratios_df.to_csv(ratios_path, index=False)

    feature_summary = (
        ratios_df[ratios_df["condition"] != "original"]
        .groupby(["model", "task", "feature", "transform"], dropna=False)
        .agg(
            mean_ratio_to_original=("ratio_to_original", "mean"),
            minimum_ratio_to_original=("ratio_to_original", "min"),
            mean_relative_drop=("relative_drop", "mean"),
            n_conditions=("condition", "count"),
        )
        .reset_index()
    )
    feature_summary_path = cue_dir / "transform_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)

    if wandb_run is not None and wandb is not None:
        try:
            wandb_run.log({
                "cue_suppression/macro_f1_ratios": wandb.Table(dataframe=ratios_df),
                "cue_suppression/transform_summary": wandb.Table(dataframe=feature_summary),
            })
        except Exception as exc:
            print(f"Warning: could not log cue-suppression tables to W&B: {exc}")

    print(f"Saved cue-suppression metrics to {cue_dir}")
    return {
        "enabled": True,
        "n_conditions": int(len(condition_metric_rows)),
        "n_unique_evaluations": int(len(metric_cache)),
        "condition_metrics_path": str(condition_metrics_path),
        "macro_f1_ratios_path": str(ratios_path),
        "transform_summary_path": str(feature_summary_path),
    }


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
# W&B CHANGE 2: small helpers keep all W&B-specific behaviour in one place.
def _wandb_metrics(prefix: str, metrics: dict) -> dict[str, int | float]:
    """Convert a metric dictionary to W&B's grouped naming convention."""
    output = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            output[f"{prefix}/{key}"] = int(value) if isinstance(value, (int, np.integer)) else float(value)
    return output


def _flatten_wandb_config(value, prefix: str = "") -> dict:
    """Flatten nested config keys and remove periods, which W&B reserves."""
    flattened = {}
    if isinstance(value, dict):
        for key, child in value.items():
            safe_key = str(key).replace(".", "_")
            child_prefix = f"{prefix}__{safe_key}" if prefix else safe_key
            flattened.update(_flatten_wandb_config(child, child_prefix))
    else:
        flattened[prefix] = value
    return flattened


def initialise_wandb_run(cfg: dict, run_name: str, out_dir: Path):
    wandb_cfg = cfg.get("wandb", {}) or {}
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    if wandb is None:
        raise ImportError(
            "W&B tracking is enabled, but the 'wandb' package is not installed. "
            "Install it with: python -m pip install wandb"
        )

    tags = list(wandb_cfg.get("tags", []) or [])
    if os.getenv("SLURM_JOB_ID"):
        tags.append("slurm")

    tracking_config = copy.deepcopy(cfg)
    tracking_config["runtime"] = {
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "hostname": os.getenv("HOSTNAME"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }

    run = wandb.init(
        project=wandb_cfg.get("project") or os.getenv("WANDB_PROJECT") or "worm-species",
        entity=wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY") or None,
        name=wandb_cfg.get("name") or os.getenv("WANDB_NAME") or run_name,
        group=wandb_cfg.get("group") or os.getenv("WANDB_RUN_GROUP") or None,
        job_type=wandb_cfg.get("job_type", "train"),
        tags=tags,
        config=_flatten_wandb_config(tracking_config),
        dir=str(out_dir),
        mode=wandb_cfg.get("mode") or os.getenv("WANDB_MODE") or "online",
        save_code=bool(wandb_cfg.get("save_code", True)),
    )

    # Use the actual epoch as the x-axis, including when validation is only
    # performed every few epochs.
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/*", step_metric="epoch")
    run.define_metric("learning_rate", step_metric="epoch")

    return run

def _score_for_selection(metrics: dict, selection_metric: str) -> float:
    value = float(metrics.get(selection_metric, float("nan")))
    if math.isnan(value):
        return -float("inf")
    return value


def get_input_condition(cfg: dict) -> dict:
    """Return the deterministic condition applied to train/val/test in this run."""
    raw = copy.deepcopy(cfg.get("input_condition", {}) or {})
    if not bool(raw.get("enabled", False)):
        return {
            "condition": "original",
            "feature": "baseline",
            "transform": "original",
            "strength": 0.0,
        }

    transform_name = str(raw.get("transform", "original")).lower()
    condition_name = str(raw.get("condition") or raw.get("name") or transform_name)
    condition = {
        "condition": condition_name,
        "feature": str(raw.get("feature", "baseline")),
        "transform": transform_name,
        "strength": float(raw.get("strength", 0.0)),
    }

    parameter_keys = {
        "retention", "order", "diameter", "sigma_colour", "sigma_space",
        "sigma", "grid_size", "seed",
    }
    for key in parameter_keys:
        if key in raw and raw[key] is not None:
            condition[key] = raw[key]

    if transform_name == "saturation":
        retention = float(condition.get("retention", 1.0))
        if not 0.0 <= retention <= 1.0:
            raise ValueError(
                f"input_condition.retention must be in [0, 1], got {retention}."
            )
        condition["retention"] = retention
    elif transform_name == "channel_shuffle":
        order = condition.get("order", [2, 0, 1])
        if isinstance(order, str):
            order = [int(x.strip()) for x in order.split(",")]
        condition["order"] = [int(x) for x in order]
    elif transform_name == "bilateral_filter":
        condition["diameter"] = int(condition["diameter"])
        condition["sigma_colour"] = float(condition["sigma_colour"])
        condition["sigma_space"] = float(condition["sigma_space"])
    elif transform_name == "gaussian_blur":
        condition["sigma"] = float(condition["sigma"])
    elif transform_name == "patch_shuffle":
        condition["grid_size"] = int(condition["grid_size"])
        condition["seed"] = int(condition.get("seed", cfg.get("seed", 0)))
    elif transform_name not in {"original", "grayscale"}:
        raise ValueError(f"Unsupported input condition transform: {transform_name!r}.")

    return condition


def get_colour_metadata(cfg: dict) -> tuple[float, int]:
    retention = float(cfg.get("data", {}).get("colour_retention", 1.0))
    if not 0.0 <= retention <= 1.0:
        raise ValueError(
            f"data.colour_retention must be between 0 and 1, got {retention}."
        )
    return retention, int(round(retention * 100))


def make_experiment_run_name(cfg: dict) -> str:
    base_name = make_run_name(cfg)
    input_condition = get_input_condition(cfg)
    condition_suffix = str(input_condition["condition"]).replace(" ", "_")
    suffixes = []

    if "colour_retention" in cfg.get("data", {}):
        _, colour_percent = get_colour_metadata(cfg)
        suffixes.append(f"basecolour_{colour_percent:03d}pct")

    suffixes.append(f"train_{condition_suffix}")
    suffix = "_".join(suffixes)
    if suffix in base_name:
        return base_name
    return f"{base_name}_{suffix}"


def train_one_run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    colour_retention, colour_percent = get_colour_metadata(cfg)
    input_condition = get_input_condition(cfg)
    cue_eval_enabled = bool(
        (cfg.get("test_cue_suppression", {}) or {}).get("enabled", False)
    )
    if cue_eval_enabled and input_condition["transform"] != "original":
        print(
            "Warning: the full test-condition battery is normally reserved for "
            "the original RGB-trained baseline, but this run is trained under "
            f"{input_condition['condition']!r}."
        )
    run_name = make_experiment_run_name(cfg)
    print(
        f"Base colour retention: {colour_retention:.2f} "
        f"({colour_percent}% chromatic information retained)"
    )
    print(
        "Matched train/validation/test condition: "
        f"{input_condition['condition']} ({input_condition['transform']})"
    )
    out_dir = Path(cfg["output"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(cfg, out_dir / "config.json")

    wandb_run = initialise_wandb_run(cfg, run_name, out_dir)


    (
        train_loader,
        val_loader,
        test_loader,
        label_to_index_by_task,
        index_to_label_by_task,
        split_summary,
        train_df,
        target_cols,
        test_loader_context,
    ) = make_loaders(cfg)

    save_json(split_summary, out_dir / "split_summary.json")
    save_json(label_to_index_by_task, out_dir / "label_to_index_by_task.json")
    print(f"Split summary and label maps saved to {out_dir}")
    num_classes_by_task = {
        task: len(label_to_index)
        for task, label_to_index in label_to_index_by_task.items()
    }

    model = build_multitask_model(
        cfg=cfg,
        num_classes_by_task=num_classes_by_task,
    ).to(device)
    print("Model built and moved to device.")
    criteria = build_criteria(
        train_df=train_df,
        target_cols=target_cols,
        group_col=cfg["data"]["group_col"],
        label_to_index_by_task=label_to_index_by_task,
        device=device,
    )

    task_loss_weights = cfg.get("multi_task", {}).get(
        "loss_weights",
        {task: 1.0 for task in target_cols},
    )
    normalize_loss_by_active_tasks = cfg.get("multi_task", {}).get(
        "normalize_loss_by_active_tasks",
        True,
    )

    hierarchy_cfg = cfg.get("multi_task", {}).get("hierarchy_loss", {})
    child_to_parent_matrix = None
    if hierarchy_cfg.get("enabled", False):
        hierarchy_parent_task = hierarchy_cfg.get("parent_task", "genus")
        hierarchy_child_task = hierarchy_cfg.get("child_task", "species")
        child_to_parent_matrix = build_child_to_parent_matrix(
            label_to_index_by_task=label_to_index_by_task,
            parent_task=hierarchy_parent_task,
            child_task=hierarchy_child_task,
            device=device,
            child_to_parent=hierarchy_cfg.get("child_to_parent"),
        )
        print(
            f"Using hierarchy loss: {hierarchy_child_task} -> {hierarchy_parent_task} "
            f"with weight {hierarchy_cfg.get('weight', task_loss_weights.get('hierarchy', 0.1))}"
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )

    use_amp = cfg["training"].get("use_amp", True)
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    history = []

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", True)
    patience = early_cfg.get("patience", 3)
    min_delta = early_cfg.get("min_delta", 0.001)

    selection_metric = cfg.get("multi_task", {}).get("selection_metric", "mean_macro_f1")
    val_interval = cfg["training"].get("val_interval", 3)

    best_val_score = -float("inf")
    epochs_without_improvement = 0
    best_epoch = 0
    print(f"Training for {cfg['training']['epochs']} epochs with early stopping: {early_enabled}, patience: {patience}, min_delta: {min_delta}")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criteria=criteria,
            optimizer=optimizer,
            device=device,
            train=True,
            scaler=scaler,
            use_amp=use_amp,
            task_loss_weights=task_loss_weights,
            normalize_loss_by_active_tasks=normalize_loss_by_active_tasks,
            hierarchy_cfg=hierarchy_cfg,
            child_to_parent_matrix=child_to_parent_matrix,
        )

        do_validation = (
            epoch == 1
            or epoch % val_interval == 0
            or epoch == cfg["training"]["epochs"]
        )

        if do_validation:
            val_metrics, _, _ = run_epoch(
                model=model,
                loader=val_loader,
                criteria=criteria,
                optimizer=None,
                device=device,
                train=False,
                scaler=None,
                use_amp=use_amp,
                task_loss_weights=task_loss_weights,
                normalize_loss_by_active_tasks=normalize_loss_by_active_tasks,
                hierarchy_cfg=hierarchy_cfg,
                child_to_parent_matrix=child_to_parent_matrix,
            )
        else:
            val_metrics = {}
        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "learning_rate": current_lr,
                **_wandb_metrics("train", train_metrics),
                **_wandb_metrics("val", val_metrics),
            })

        if do_validation:
            if selection_metric not in val_metrics:
                raise ValueError(
                    f"multi_task.selection_metric={selection_metric!r} is not available. "
                    f"Available validation metrics: {list(val_metrics.keys())}"
                )

            current_val_score = _score_for_selection(val_metrics, selection_metric)

            print(
                f"[{run_name}] Epoch {epoch:03d}/{cfg['training']['epochs']} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"val {selection_metric} {val_metrics[selection_metric]:.4f} | "
                f"complete exact-match {val_metrics['complete_exact_match_accuracy']:.4f} "
                f"n={val_metrics['complete_exact_match_n']}"
            )

            for task in target_cols:
                print(
                    f"    {task}: val macro-F1 {val_metrics[f'{task}_macro_f1']:.4f} | "
                    f"val bal-acc {val_metrics[f'{task}_balanced_accuracy']:.4f} | "
                    f"n={val_metrics[f'{task}_n']}"
                )

            improved = current_val_score > best_val_score + min_delta

            if improved or epoch == 1:
                best_val_score = current_val_score
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "cfg": cfg,
                        "label_to_index_by_task": label_to_index_by_task,
                        "index_to_label_by_task": index_to_label_by_task,
                        "best_val_score": best_val_score,
                        "selection_metric": selection_metric,
                        "best_epoch": best_epoch,
                        "colour_retention": colour_retention,
                        "colour_percent": colour_percent,
                        "training_condition": input_condition,
                    },
                    out_dir / "best_model.pt",
                )

                if wandb_run is not None:
                    wandb_run.summary["best_epoch"] = best_epoch
                    wandb_run.summary["best_val_score"] = best_val_score
                    wandb_run.summary["selection_metric"] = selection_metric

                print(
                    f"[{run_name}] New best model saved | "
                    f"best val {selection_metric} {best_val_score:.4f} at epoch {best_epoch}"
                )
            else:
                epochs_without_improvement += 1
                print(f"[{run_name}] No improvement for {epochs_without_improvement}/{patience} validation checks")

            if early_enabled and epochs_without_improvement >= patience:
                print(
                    f"[{run_name}] Early stopping at epoch {epoch}. "
                    f"Best val {selection_metric} {best_val_score:.4f} at epoch {best_epoch}."
                )
                break

        else:
            print(
                f"[{run_name}] Epoch {epoch:03d}/{cfg['training']['epochs']} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"validation skipped"
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics, y_true_by_task, y_pred_by_task = run_epoch(
        model=model,
        loader=test_loader,
        criteria=criteria,
        optimizer=None,
        device=device,
        train=False,
        scaler=None,
        use_amp=use_amp,
        task_loss_weights=task_loss_weights,
        normalize_loss_by_active_tasks=normalize_loss_by_active_tasks,
        hierarchy_cfg=hierarchy_cfg,
        child_to_parent_matrix=child_to_parent_matrix,
    )

    for task in target_cols:
        index_to_label = index_to_label_by_task[task]
        labels = list(range(len(index_to_label)))
        label_names = [index_to_label[i] for i in labels]

        y_true = np.array(y_true_by_task[task], dtype=int)
        y_pred = np.array(y_pred_by_task[task], dtype=int)

        if len(y_true) == 0:
            pd.DataFrame([{"note": "No labelled test examples for this task."}]).to_csv(
                out_dir / f"classification_report_{task}.csv",
                index=False,
            )
            pd.DataFrame().to_csv(out_dir / f"confusion_matrix_{task}.csv")
            continue

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        pd.DataFrame(report).transpose().to_csv(out_dir / f"classification_report_{task}.csv")
        pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
            out_dir / f"confusion_matrix_{task}.csv"
        )

    save_json(test_metrics, out_dir / "test_metrics.json")

    cue_suppression_result = evaluate_test_cue_suppression(
        cfg=cfg,
        run_name=run_name,
        out_dir=out_dir,
        model=model,
        baseline_metrics=test_metrics,
        test_loader_context=test_loader_context,
        criteria=criteria,
        target_cols=target_cols,
        device=device,
        use_amp=use_amp,
        task_loss_weights=task_loss_weights,
        normalize_loss_by_active_tasks=normalize_loss_by_active_tasks,
        hierarchy_cfg=hierarchy_cfg,
        child_to_parent_matrix=child_to_parent_matrix,
        wandb_run=wandb_run,
    )

    run_result = {
        "run_name": run_name,
        "model": cfg.get("model", {}).get("name"),
        "out_dir": str(out_dir),
        "colour_retention": colour_retention,
        "colour_percent": colour_percent,
        "train_condition": input_condition["condition"],
        "train_feature": input_condition["feature"],
        "train_transform": input_condition["transform"],
        "train_strength": input_condition.get("strength"),
        "train_condition_parameters": json.dumps(
            {
                key: value
                for key, value in input_condition.items()
                if key not in {"condition", "feature", "transform", "strength"}
            },
            sort_keys=True,
        ),
        "best_epoch": best_epoch,
        "best_val_score": best_val_score,
        "selection_metric": selection_metric,
        "cue_suppression_enabled": cue_suppression_result["enabled"],
        "cue_suppression_n_conditions": cue_suppression_result["n_conditions"],
        "cue_suppression_n_unique_evaluations": cue_suppression_result.get(
            "n_unique_evaluations", 0
        ),
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    # Save locally before optional external logging, so the SLURM collector can
    # still recover the completed result if W&B logging fails afterwards.
    save_json(run_result, out_dir / "run_summary.json")

    if wandb_run is not None:
        wandb_run.log(_wandb_metrics("test", test_metrics))
        for key, value in _wandb_metrics("test", test_metrics).items():
            wandb_run.summary[key] = value
        wandb_run.summary["colour_retention"] = colour_retention
        wandb_run.summary["colour_percent"] = colour_percent
        wandb_run.summary["train_condition"] = input_condition["condition"]
        wandb_run.summary["train_feature"] = input_condition["feature"]
        wandb_run.summary["train_transform"] = input_condition["transform"]
        wandb_run.summary["train_strength"] = input_condition.get("strength")

        if bool((cfg.get("wandb", {}) or {}).get("log_model", False)):
            model_artifact = wandb.Artifact(
                name=f"{run_name}-best-model",
                type="model",
                metadata={
                    "best_epoch": best_epoch,
                    "best_val_score": best_val_score,
                    "selection_metric": selection_metric,
                    "colour_retention": colour_retention,
                    "colour_percent": colour_percent,
                    "training_condition": input_condition,
                },
            )
            model_artifact.add_file(str(out_dir / "best_model.pt"))
            wandb_run.log_artifact(model_artifact)

        wandb_run.finish()
    print("\nTest metrics:")
    print(test_metrics)

    return run_result


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
    )

    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. model.name=vit_b_16 training.lr=1e-5",
    )

    parser.add_argument(
        "--sweep",
        nargs="*",
        default=[],
        help="Multi-run sweep, e.g. model.name=resnet18,vit_b_16 data.image_col=rel_path_seg,rel_path_raw",
    )

    args = parser.parse_args()

    base_cfg = load_config(args.config)
    base_cfg = apply_overrides(base_cfg, args.override)

    configs = generate_sweep_configs(
        base_cfg=base_cfg,
        cli_sweep_items=args.sweep,
    )

    all_results = []

    for i, cfg in enumerate(configs, start=1):
        print(f"\nRunning {i}/{len(configs)}")
        result = train_one_run(cfg)
        all_results.append(result)

    out_dir = Path(base_cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    if "colour_percent" in results_df.columns:
        results_df = results_df.sort_values(
            "colour_percent", ascending=False
        ).reset_index(drop=True)
    results_df.to_csv(out_dir / "multi_run_results.csv", index=False)

    print("\nAll runs finished.")
    print(results_df)


if __name__ == "__main__":
    main()
