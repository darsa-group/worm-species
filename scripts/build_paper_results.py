#!/usr/bin/env python3
"""Build clearly named paper tables and figures from the full ablation suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from cycler import cycler
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.transforms import functional as tv_functional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.worm_species.data.conditions import (
    build_condition_operations,
    ColourRetention,
    GaussianBlurPercent,
    PatchShuffle,
    ResolutionLoss,
)

DEFAULT_STYLE_PATH = (
    PROJECT_ROOT / "dev" / "paper_report_style.yaml"
)
REPORT_STYLE: dict[str, Any] = {}
FIGURE_SOURCE_ROOT: Path | None = None
RESOLUTION_INPUT_SIZE = 224
RESOLUTION_LOSS_LEVELS = (
    0.0,
    25.0,
    50.0,
    75.0,
    87.5,
    93.75,
    100.0,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_report_style(style_path: Path | None = None) -> dict[str, Any]:
    """Load editable report colours without touching completed experiments."""
    global REPORT_STYLE
    path = style_path or DEFAULT_STYLE_PATH
    style: dict[str, Any] = {}
    if path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Report style must be a YAML mapping: {path}")
        style = loaded
    palette = style.get(
        "palette",
        ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"],
    )
    if not isinstance(palette, list) or not palette:
        raise ValueError("paper report style.palette must be a non-empty list")
    if any(not isinstance(colour, str) or not colour for colour in palette):
        raise ValueError("paper report style.palette entries must be colours")
    dpi = int(style.get("dpi", 300))
    plt.rcParams.update({
        "axes.prop_cycle": cycler(color=palette),
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "axes.grid": bool(style.get("axes_grid", False)),
        "font.size": float(style.get("font_size", 10)),
    })
    REPORT_STYLE = style
    return style


def _style(name: str, default: Any) -> Any:
    return REPORT_STYLE.get(name, default)


def _save_figure_formats(fig: Any, path: Path) -> None:
    """Save every manuscript figure in raster and editable vector formats."""
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            path.with_suffix(f".{suffix}"),
            dpi=int(_style("dpi", 300)),
            bbox_inches="tight",
        )


def _figure_source_dir(path: Path) -> Path:
    root = FIGURE_SOURCE_ROOT or path.parent.parent / "figure_sources"
    directory = root / path.stem
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_figure_sources(
    path: Path,
    frame: pd.DataFrame,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    directory = _figure_source_dir(path)
    frame.to_csv(directory / "plot_data.csv", index=False)
    run_dirs = (
        sorted(frame["run_dir"].dropna().astype(str).unique().tolist())
        if "run_dir" in frame
        else []
    )
    manifest = {
        "figure": path.stem,
        "plot_data": "plot_data.csv",
        "rows": int(len(frame)),
        "run_directories": run_dirs,
        "style": REPORT_STYLE,
        **(extra or {}),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return directory


def _decorate_hloss_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Make hierarchy weight an explicit plotting series without mixing seeds."""
    result = frame.copy()
    if result.empty:
        return result
    result["model"] = (
        result["model"].astype(str)
        + " / h="
        + result["hierarchy_loss_weight"].astype(float).map(
            lambda value: f"{value:g}"
        )
    )
    return result


def _loss_name(weights: dict) -> str:
    return "_".join(
        f"{task}-{float(weights.get(task, 0)):g}"
        for task in ("genus", "species", "age")
    )


def _condition_parameter(
    condition: dict[str, Any], key: str, default: Any = None
) -> Any:
    """Read both saved canonical configs and already-resolved run configs."""
    parameters = condition.get("parameters", {}) or {}
    if isinstance(parameters, dict) and key in parameters:
        return parameters[key]
    return condition.get(key, default)


def _format_number(value: Any) -> str:
    return f"{float(value):g}"


def resolution_loss_schedule(
    image_size: int = RESOLUTION_INPUT_SIZE,
) -> pd.DataFrame:
    """Describe every configured spatial-resolution control."""
    rows = []
    for percent in RESOLUTION_LOSS_LEVELS:
        retained = 100.0 - percent
        intermediate = max(
            1, int(round(image_size * retained / 100.0))
        )
        rows.append(
            {
                "resolution_loss_percent": percent,
                "retained_linear_dimension_percent": retained,
                "input_size": f"{image_size}x{image_size}",
                "intermediate_size": f"{intermediate}x{intermediate}",
                "intermediate_pixels_per_side": intermediate,
                "interpretation": (
                    "Extreme spatial-information control: mean colour "
                    "without spatial structure"
                    if percent == 100.0
                    else "Controlled spatial-resolution reduction"
                ),
            }
        )
    return pd.DataFrame(rows)


def _resolution_tick_labels() -> tuple[list[float], list[str]]:
    schedule = resolution_loss_schedule()
    ticks = schedule["resolution_loss_percent"].tolist()
    labels = [
        (
            f"{_format_number(row.resolution_loss_percent)}\n"
            f"{_format_number(row.retained_linear_dimension_percent)}% / "
            f"{int(row.intermediate_pixels_per_side)}px"
        )
        for row in schedule.itertuples()
    ]
    return ticks, labels


def collect_runs(paper_root: Path) -> pd.DataFrame:
    rows = []
    runs_root = paper_root / "runs"
    for config_path in sorted(runs_root.rglob("config.json")):
        run_dir = config_path.parent
        summary_path = run_dir / "run_summary.json"
        if not summary_path.is_file():
            continue
        config = _read_json(config_path)
        summary = _read_json(summary_path)
        parameter_path = run_dir / "model_parameters.json"
        parameter_counts = (
            _read_json(parameter_path) if parameter_path.is_file() else {}
        )
        relative = config_path.relative_to(runs_root)
        stage = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        model = str((config.get("model", {}) or {}).get("name", "unknown"))
        weights = dict(
            (config.get("multi_task", {}) or {}).get("loss_weights", {}) or {}
        )
        hierarchy = dict(
            (config.get("multi_task", {}) or {}).get(
                "hierarchy_loss", {}
            )
            or {}
        )
        hierarchy_enabled = bool(hierarchy.get("enabled", False))
        hierarchy_weight = (
            float(hierarchy.get("weight", 0.0))
            if hierarchy_enabled else 0.0
        )
        condition = dict(config.get("input_condition", {}) or {})
        holdout = dict(config.get("data_holdout", {}) or {})
        label_path = run_dir / "label_to_index_by_task.json"
        label_maps = _read_json(label_path) if label_path.is_file() else {}
        class_counts = {
            task: len(labels)
            for task, labels in label_maps.items()
            if isinstance(labels, dict)
        }
        chances = {
            task: (
                1.0 / class_counts[task]
                if class_counts.get(task, 0) > 0
                else np.nan
            )
            for task in ("genus", "species", "age")
        }
        finite_chances = [
            value for value in chances.values() if pd.notna(value)
        ]
        mean_chance = (
            float(np.mean(finite_chances)) if finite_chances else np.nan
        )
        operations = _condition_parameter(condition, "operations", [])
        gaussian_percent = _condition_parameter(condition, "percent")
        paired_transform = None
        paired_level = None
        if isinstance(operations, list) and operations:
            gaussian_operation = next(
                (
                    item for item in operations
                    if isinstance(item, dict)
                    and item.get("transform") == "gaussian_blur_percent"
                ),
                {},
            )
            gaussian_percent = _condition_parameter(
                gaussian_operation, "percent"
            )
            paired = next(
                (
                    item for item in operations
                    if isinstance(item, dict)
                    and item.get("transform") != "gaussian_blur_percent"
                ),
                {},
            )
            paired_transform = paired.get("transform")
            paired_parameters = paired.get("parameters", {}) or {}
            paired_level = next(
                (
                    paired_parameters[key]
                    for key in ("percent", "grid_size", "retention")
                    if key in paired_parameters
                ),
                None,
            )
        max_sigma = _condition_parameter(condition, "max_sigma")
        rows.append({
            "stage": stage,
            "model": model,
            "seed": config.get("seed"),
            "total_parameters": (
                parameter_counts.get("total_parameters")
                or summary.get("total_parameters")
            ),
            "trainable_parameters": (
                parameter_counts.get("trainable_parameters")
                or summary.get("trainable_parameters")
            ),
            "pretrained": (config.get("model", {}) or {}).get("pretrained"),
            "learning_rate": (config.get("training", {}) or {}).get("lr"),
            "batch_size": (config.get("training", {}) or {}).get("batch_size"),
            "selection_metric": (
                (config.get("multi_task", {}) or {}).get("selection_metric")
                or summary.get("selection_metric")
            ),
            "loss_recipe": _loss_name(weights),
            "genus_weight": weights.get("genus"),
            "species_weight": weights.get("species"),
            "age_weight": weights.get("age"),
            "hierarchy_loss_enabled": hierarchy_enabled,
            "hierarchy_loss_weight": hierarchy_weight,
            "hierarchy_loss_label": f"h={hierarchy_weight:g}",
            "condition": (
                condition.get("condition")
                or condition.get("name")
                or "original"
            ),
            "transform": condition.get("transform", "original"),
            "strength": condition.get("strength", 0.0),
            "percent": _condition_parameter(condition, "percent"),
            "grid_size": _condition_parameter(condition, "grid_size"),
            "colour_retention": _condition_parameter(condition, "retention"),
            "gaussian_percent": gaussian_percent,
            "gaussian_max_sigma": max_sigma,
            "gaussian_sigma": (
                float(gaussian_percent) * float(max_sigma) / 100.0
                if gaussian_percent is not None and max_sigma is not None
                else (
                    float(gaussian_percent) * 64.0 / 100.0
                    if isinstance(operations, list) and operations
                    and gaussian_percent is not None
                    else np.nan
                )
            ),
            "paired_transform": paired_transform,
            "paired_level": paired_level,
            "operations": json.dumps(operations, sort_keys=True),
            "holdout": holdout.get("name"),
            "holdout_question": holdout.get("question"),
            "best_epoch": summary.get("best_epoch"),
            "best_val_score": summary.get("best_val_score"),
            "test_mean_macro_f1": summary.get("test_mean_macro_f1"),
            "test_genus_macro_f1": summary.get("test_genus_macro_f1"),
            "test_species_macro_f1": summary.get("test_species_macro_f1"),
            "test_age_macro_f1": summary.get("test_age_macro_f1"),
            "genus_class_count": class_counts.get("genus"),
            "species_class_count": class_counts.get("species"),
            "age_class_count": class_counts.get("age"),
            "chance_genus_macro_f1": chances["genus"],
            "chance_species_macro_f1": chances["species"],
            "chance_age_macro_f1": chances["age"],
            "chance_mean_macro_f1": mean_chance,
            "run_dir": str(run_dir),
        })
    return pd.DataFrame(rows)


def collect_holdouts(paper_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(
        (paper_root / "runs" / "data_holdouts").rglob(
            "data_holdout_evaluation/task_metrics.csv"
        )
    ):
        frame = pd.read_csv(path)
        config = _read_json(path.parents[1] / "config.json")
        label_path = path.parents[1] / "label_to_index_by_task.json"
        label_maps = _read_json(label_path) if label_path.is_file() else {}
        frame.insert(0, "model", config["model"]["name"])
        frame.insert(1, "seed", config.get("seed"))
        hierarchy = (
            (config.get("multi_task", {}) or {}).get(
                "hierarchy_loss", {}
            )
            or {}
        )
        frame["hierarchy_loss_weight"] = (
            float(hierarchy.get("weight", 0.0))
            if bool(hierarchy.get("enabled", False)) else 0.0
        )
        if "cohort" not in frame:
            frame["cohort"] = "independent_test"
        frame["chance"] = frame["task"].map({
            task: 1.0 / len(labels)
            for task, labels in label_maps.items()
            if isinstance(labels, dict) and labels
        })
        frame["run_dir"] = str(path.parents[1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_holdout_controls(paper_root: Path) -> pd.DataFrame:
    """Collect baseline predictions on the exact biological cohorts."""
    frames = []
    for path in sorted(
        (paper_root / "runs" / "baseline").rglob(
            "data_holdout_control_evaluation/task_metrics.csv"
        )
    ):
        frame = pd.read_csv(path)
        config = _read_json(path.parents[1] / "config.json")
        label_path = path.parents[1] / "label_to_index_by_task.json"
        label_maps = _read_json(label_path) if label_path.is_file() else {}
        frame.insert(0, "model", config["model"]["name"])
        frame.insert(1, "seed", config.get("seed"))
        hierarchy = (
            (config.get("multi_task", {}) or {}).get(
                "hierarchy_loss", {}
            )
            or {}
        )
        frame["hierarchy_loss_weight"] = (
            float(hierarchy.get("weight", 0.0))
            if bool(hierarchy.get("enabled", False)) else 0.0
        )
        frame["loss_recipe"] = _loss_name(
            dict(
                (config.get("multi_task", {}) or {}).get(
                    "loss_weights", {}
                )
                or {}
            )
        )
        frame["chance"] = frame["task"].map({
            task: 1.0 / len(labels)
            for task, labels in label_maps.items()
            if isinstance(labels, dict) and labels
        })
        frame["run_dir"] = str(path.parents[1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_baseline_condition_controls(paper_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(
        (paper_root / "runs" / "baseline").rglob(
            "condition_matrix_evaluation/task_metrics.csv"
        )
    ):
        frame = pd.read_csv(path)
        config = _read_json(path.parents[1] / "config.json")
        frame["seed"] = config.get("seed")
        hierarchy = (
            (config.get("multi_task", {}) or {}).get(
                "hierarchy_loss", {}
            )
            or {}
        )
        frame["hierarchy_loss_weight"] = (
            float(hierarchy.get("weight", 0.0))
            if bool(hierarchy.get("enabled", False)) else 0.0
        )
        frame["loss_recipe"] = _loss_name(
            dict(
                (config.get("multi_task", {}) or {}).get(
                    "loss_weights", {}
                )
                or {}
            )
        )
        frame["run_dir"] = str(path.parents[1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_original_cross_conditions(paper_root: Path) -> pd.DataFrame:
    frames = []
    for stage in ("visual_ablation", "visual_interactions"):
        for path in sorted(
            (paper_root / "runs" / stage).rglob(
                "condition_matrix_evaluation/task_metrics.csv"
            )
        ):
            frame = pd.read_csv(path)
            config = _read_json(path.parents[1] / "config.json")
            label_path = path.parents[1] / "label_to_index_by_task.json"
            label_maps = (
                _read_json(label_path) if label_path.is_file() else {}
            )
            frame["stage"] = stage
            frame["seed"] = config.get("seed")
            hierarchy = (
                (config.get("multi_task", {}) or {}).get(
                    "hierarchy_loss", {}
                )
                or {}
            )
            frame["hierarchy_loss_weight"] = (
                float(hierarchy.get("weight", 0.0))
                if bool(hierarchy.get("enabled", False)) else 0.0
            )
            frame["loss_recipe"] = _loss_name(
                dict(
                    (config.get("multi_task", {}) or {}).get(
                        "loss_weights", {}
                    )
                    or {}
                )
            )
            frame["chance"] = frame["task"].map({
                task: 1.0 / len(labels)
                for task, labels in label_maps.items()
                if isinstance(labels, dict) and labels
            })
            frame["run_dir"] = str(path.parents[1])
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_parameter_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


def _original_test_condition_summary(cross: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-task original-image evaluation into one row per run."""
    if cross.empty:
        return cross
    selected = cross[cross["test_condition"].eq("original")].copy()
    if selected.empty:
        return selected
    selected["percent"] = selected["train_parameters"].apply(
        lambda raw: _parse_parameter_json(raw).get("percent")
    )
    selected["grid_size"] = selected["train_parameters"].apply(
        lambda raw: _parse_parameter_json(raw).get("grid_size")
    )
    if "chance" not in selected:
        selected["chance"] = np.nan
    return (
        selected.groupby(
            [
                "run_name",
                "model",
                "seed",
                "loss_recipe",
                "hierarchy_loss_weight",
                "train_condition",
                "train_transform",
                "percent",
                "grid_size",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            chance_mean_macro_f1=("chance", "mean"),
        )
    )


def _baseline_condition_summary(controls: pd.DataFrame) -> pd.DataFrame:
    if controls.empty:
        return controls
    frame = controls.copy()
    frame["percent"] = frame["test_parameters"].apply(
        lambda raw: _parse_parameter_json(raw).get("percent")
    )
    frame["grid_size"] = frame["test_parameters"].apply(
        lambda raw: _parse_parameter_json(raw).get("grid_size")
    )
    return (
        frame.groupby(
            [
                "run_name",
                "model",
                "seed",
                "loss_recipe",
                "hierarchy_loss_weight",
                "test_condition",
                "test_transform",
                "percent",
                "grid_size",
                "run_dir",
            ],
            dropna=False,
            as_index=False,
        )["macro_f1"]
        .mean()
        .rename(columns={"macro_f1": "mean_macro_f1"})
    )


def _with_original_anchor(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    x: str,
    anchor: float,
) -> pd.DataFrame:
    """Reuse the matching original-image fit as severity zero."""
    if frame.empty or baseline.empty:
        return frame
    recipes = frame["loss_recipe"].dropna().unique().tolist()
    source = baseline[
        baseline["loss_recipe"].isin(recipes)
    ][
        [
            "model",
            "loss_recipe",
            "seed",
            "test_mean_macro_f1",
            "chance_mean_macro_f1",
            "run_dir",
        ]
    ].copy()
    if source.empty:
        return frame
    source[x] = anchor
    return pd.concat([source, frame], ignore_index=True, sort=False)


def _ci_summary(
    frame: pd.DataFrame,
    *,
    groups: list[str],
    value: str,
) -> pd.DataFrame:
    """Return mean and seed-level 95% t intervals."""
    if frame.empty:
        return pd.DataFrame(
            columns=[*groups, "mean", "std", "n_seeds", "ci95"]
        )
    clean = frame.dropna(subset=[value]).copy()
    if clean.empty:
        return pd.DataFrame(
            columns=[*groups, "mean", "std", "n_seeds", "ci95"]
        )
    summary = (
        clean.groupby(groups, dropna=False)[value]
        .agg(mean="mean", std="std", n_seeds="count")
        .reset_index()
    )
    critical_by_df = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }
    summary["ci95"] = summary.apply(
        lambda row: (
            critical_by_df.get(
                max(1, int(row["n_seeds"]) - 1), 1.96
            )
            * float(row["std"])
            / np.sqrt(float(row["n_seeds"]))
            if int(row["n_seeds"]) > 1 and pd.notna(row["std"])
            else np.nan
        ),
        axis=1,
    )
    return summary


def _split_paths(split_root: Path | None) -> dict[str, Path]:
    if split_root is None:
        return {}
    candidates = [split_root, split_root / "split_csv"]
    filenames = {
        "training": "train_split.csv",
        "validation": "val_split.csv",
        "test": "test_split.csv",
    }
    for candidate in candidates:
        paths = {
            split: candidate / filename
            for split, filename in filenames.items()
        }
        if all(path.is_file() for path in paths.values()):
            return paths
    return {}


def load_split_frames(
    split_root: Path | None,
) -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_csv(path)
        for split, path in _split_paths(split_root).items()
    }


def dataset_composition_table(
    split_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    columns = [
        "genus",
        "species",
        "stage",
        "individuals",
        "images",
        "training_individuals",
        "training_images",
        "validation_individuals",
        "validation_images",
        "test_individuals",
        "test_images",
    ]
    if not split_frames:
        return pd.DataFrame(columns=columns)
    keys = ["genus", "species_label", "life_stage"]
    parts = []
    complete = []
    for split, frame in split_frames.items():
        available = frame.copy()
        for key in keys:
            if key not in available:
                available[key] = pd.NA
        grouped = (
            available.groupby(keys, dropna=False)
            .agg(
                **{
                    f"{split}_individuals": ("barcode", "nunique"),
                    f"{split}_images": ("barcode", "size"),
                }
            )
            .reset_index()
        )
        parts.append(grouped)
        tagged = available[keys + ["barcode"]].copy()
        complete.append(tagged)
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on=keys, how="outer")
    totals = (
        pd.concat(complete, ignore_index=True)
        .groupby(keys, dropna=False)
        .agg(individuals=("barcode", "nunique"), images=("barcode", "size"))
        .reset_index()
    )
    result = totals.merge(result, on=keys, how="left")
    count_columns = [
        column
        for column in result
        if column.endswith("_individuals")
        or column.endswith("_images")
        or column in {"individuals", "images"}
    ]
    result[count_columns] = result[count_columns].fillna(0).astype(int)
    result = result.rename(
        columns={"species_label": "species", "life_stage": "stage"}
    )
    return result[columns].sort_values(
        ["genus", "species", "stage"], na_position="last"
    )


def model_training_table(runs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "backbone",
        "total_parameters",
        "trainable_parameters",
        "pretraining",
        "loss_recipe",
        "hierarchy_loss_enabled",
        "hierarchy_loss_weight",
        "learning_rate",
        "batch_size",
        "seed_count",
        "seeds",
        "model_selection_criterion",
    ]
    if runs.empty:
        return pd.DataFrame(columns=columns)
    source = runs[runs["stage"].eq("baseline")].copy()
    rows = []
    group_columns = [
        "model",
        "loss_recipe",
        "hierarchy_loss_enabled",
        "hierarchy_loss_weight",
        "learning_rate",
        "batch_size",
        "selection_metric",
        "pretrained",
    ]
    for values, group in source.groupby(group_columns, dropna=False):
        (
            model,
            loss_recipe,
            hierarchy_loss_enabled,
            hierarchy_loss_weight,
            learning_rate,
            batch_size,
            selection_metric,
            pretrained,
        ) = values
        seeds = sorted(
            int(seed) for seed in group["seed"].dropna().unique()
        )
        rows.append({
            "backbone": model,
            "total_parameters": group["total_parameters"].dropna().max(),
            "trainable_parameters": (
                group["trainable_parameters"].dropna().max()
            ),
            "pretraining": bool(pretrained),
            "loss_recipe": loss_recipe,
            "hierarchy_loss_enabled": bool(hierarchy_loss_enabled),
            "hierarchy_loss_weight": float(hierarchy_loss_weight),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
            "model_selection_criterion": selection_metric,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["backbone", "loss_recipe", "hierarchy_loss_weight"]
    )


def experimental_ablation_table(
    runs: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "ablation",
        "training_condition",
        "test_condition",
        "removed_information",
        "scientific_question",
        "retained_linear_dimension_percent",
        "intermediate_size_at_224px",
    ]
    if runs.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    visual = runs[runs["stage"].eq("visual_ablation")].drop_duplicates(
        ["condition", "transform", "percent", "grid_size"]
    )
    questions = {
        "saturation": "Can the model identify worms without chromatic information?",
        "patch_shuffle": "How much does recognition depend on global spatial arrangement?",
        "gaussian_blur_percent": "How much fine texture is required for recognition?",
        "resolution_loss": "How much spatial detail is required for recognition?",
    }
    for _, row in visual.iterrows():
        transform = row["transform"]
        retained = np.nan
        intermediate_size = None
        if transform == "saturation":
            removed = "All colour information"
        elif transform == "patch_shuffle":
            removed = f"Global layout using a {int(row['grid_size'])}x{int(row['grid_size'])} shuffled grid"
        elif transform == "gaussian_blur_percent":
            removed = (
                "Fine texture; blur severity "
                f"{_format_number(row['percent'])}%"
            )
        else:
            percent = float(row["percent"])
            retained = 100.0 - percent
            intermediate = max(
                1,
                int(round(RESOLUTION_INPUT_SIZE * retained / 100.0)),
            )
            intermediate_size = f"{intermediate}x{intermediate}"
            removed = (
                "Spatial resolution; information loss "
                f"{_format_number(percent)}%, retaining "
                f"{_format_number(retained)}% linear dimension "
                f"({intermediate_size} from 224x224)"
            )
            if percent == 100.0:
                removed += (
                    "; extreme mean-colour control with no spatial structure"
                )
        rows.append({
            "ablation": transform,
            "training_condition": row["condition"],
            "test_condition": "matched condition and original images",
            "removed_information": removed,
            "scientific_question": questions[transform],
            "retained_linear_dimension_percent": retained,
            "intermediate_size_at_224px": intermediate_size,
        })
    holdout_runs = runs[
        runs["stage"].eq("data_holdouts")
    ].drop_duplicates(["holdout"])
    for _, row in holdout_runs.iterrows():
        rows.append({
            "ablation": "cohort_holdout",
            "training_condition": f"development data excluding {row['holdout']}",
            "test_condition": f"preserved test cohort {row['holdout']}",
            "removed_information": f"Biological cohort {row['holdout']}",
            "scientific_question": row["holdout_question"],
            "retained_linear_dimension_percent": np.nan,
            "intermediate_size_at_224px": None,
        })
    return pd.DataFrame(rows, columns=columns)


def collect_holdout_definitions(paper_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(
        (paper_root / "runs" / "data_holdouts").rglob("split_summary.json")
    ):
        payload = _read_json(path)
        audit = payload.get("data_holdout")
        if not isinstance(audit, dict):
            continue
        config = _read_json(path.parent / "config.json")
        removed = audit.get("removed", {}) or {}
        rows.append({
            "holdout": audit.get("name"),
            "question": audit.get("question"),
            "model": (config.get("model", {}) or {}).get("name"),
            "seed": config.get("seed"),
            "where": json.dumps(audit.get("where", {}), sort_keys=True),
            "removed_training_images": (
                removed.get("train", {}) or {}
            ).get("rows", 0),
            "removed_training_individuals": (
                removed.get("train", {}) or {}
            ).get("individuals", 0),
            "removed_validation_images": (
                removed.get("validation", {}) or {}
            ).get("rows", 0),
            "removed_validation_individuals": (
                removed.get("validation", {}) or {}
            ).get("individuals", 0),
            "test_cohort_images": (
                audit.get("evaluation_cohort", {}) or {}
            ).get("rows", 0),
            "test_cohort_individuals": (
                audit.get("evaluation_cohort", {}) or {}
            ).get("individuals", 0),
        })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    invariant = [
        "holdout",
        "question",
        "where",
        "removed_training_images",
        "removed_training_individuals",
        "removed_validation_images",
        "removed_validation_individuals",
        "test_cohort_images",
        "test_cohort_individuals",
    ]
    return frame.drop_duplicates(invariant)[invariant].sort_values("holdout")


def baseline_task_table(baseline: pd.DataFrame) -> pd.DataFrame:
    if baseline.empty:
        return pd.DataFrame()
    task_frames = []
    for task in ("genus", "species", "age", "mean"):
        source = baseline[
            [
                "model",
                "loss_recipe",
                "seed",
                f"test_{task}_macro_f1",
                f"chance_{task}_macro_f1",
            ]
        ].rename(columns={
            f"test_{task}_macro_f1": "macro_f1",
            f"chance_{task}_macro_f1": "chance",
        })
        source["task"] = task
        task_frames.append(source)
    long = pd.concat(task_frames, ignore_index=True)
    summary = _ci_summary(
        long,
        groups=["model", "loss_recipe", "task"],
        value="macro_f1",
    )
    chance = (
        long.groupby(
            ["model", "loss_recipe", "task"], as_index=False
        )["chance"].mean()
    )
    return summary.merge(
        chance, on=["model", "loss_recipe", "task"], how="left"
    )


def joined_holdout_metrics(
    runs: pd.DataFrame,
    holdouts: pd.DataFrame,
) -> pd.DataFrame:
    if runs.empty or holdouts.empty:
        return pd.DataFrame()
    full = runs[runs["stage"].eq("data_holdouts")][
        [
            "model",
            "seed",
            "hierarchy_loss_weight",
            "holdout",
            "test_mean_macro_f1",
            "test_genus_macro_f1",
            "test_species_macro_f1",
            "test_age_macro_f1",
        ]
    ].copy()
    cohort = holdouts.copy()
    full["seed"] = full["seed"].astype("string")
    cohort["seed"] = cohort["seed"].astype("string")
    return cohort.merge(
        full,
        on=["model", "seed", "hierarchy_loss_weight", "holdout"],
        how="left",
    )


def _save_grouped_bar(
    frame: pd.DataFrame,
    *,
    x: str,
    series: str,
    value: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    summary = _ci_summary(frame, groups=[x, series], value=value)
    if summary.empty:
        return
    x_values = summary[x].drop_duplicates().tolist()
    series_values = summary[series].drop_duplicates().tolist()
    positions = np.arange(len(x_values), dtype=float)
    width = 0.8 / max(1, len(series_values))
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, series_value in enumerate(series_values):
        group = summary[summary[series].eq(series_value)].set_index(x)
        ax.bar(
            positions + (index - (len(series_values) - 1) / 2) * width,
            [group["mean"].get(item, np.nan) for item in x_values],
            width=width,
            yerr=[group["ci95"].get(item, np.nan) for item in x_values],
            capsize=3,
            label=series_value,
        )
    ax.set_xticks(positions, x_values, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    chance = (
        float(frame["chance_mean_macro_f1"].dropna().mean())
        if "chance_mean_macro_f1" in frame
        and frame["chance_mean_macro_f1"].notna().any()
        else np.nan
    )
    if pd.notna(chance):
        ax.axhline(
            chance,
            color=_style("chance_colour", "#777777"),
            linestyle=":",
            linewidth=1.4,
            label=f"Chance ({chance:.3f})",
        )
    elif "chance" in frame and frame["chance"].notna().any():
        chance_by_x = frame.groupby(x, dropna=False)["chance"].mean()
        ax.plot(
            positions,
            [chance_by_x.get(item, np.nan) for item in x_values],
            color=_style("chance_colour", "#777777"),
            linestyle=":",
            marker="_",
            markersize=12,
            linewidth=1.4,
            label="Chance",
        )
    ax.legend(title=series.replace("_", " "), bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _write_figure_sources(path, frame, extra={"summary_rows": len(summary)})
    summary.to_csv(
        _figure_source_dir(path) / "seed_summary.csv", index=False
    )
    _save_figure_formats(fig, path)
    plt.close(fig)


def _save_lines(
    frame: pd.DataFrame,
    *,
    x: str,
    value: str,
    title: str,
    xlabel: str,
    ylabel: str = "Matched-test mean macro-F1",
    path: Path,
    xticks: list[float] | None = None,
    xticklabels: list[str] | None = None,
    thumbnails: dict[str, Path] | None = None,
) -> None:
    if frame.empty:
        return
    summary = _ci_summary(frame, groups=["model", x], value=value)
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, group in summary.groupby("model"):
        ordered = group.sort_values(x)
        ax.errorbar(
            ordered[x],
            ordered["mean"],
            yerr=ordered["ci95"],
            marker="o",
            capsize=3,
            label=model,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_ylim(0, 1)
    chance = (
        float(frame["chance_mean_macro_f1"].dropna().mean())
        if "chance_mean_macro_f1" in frame
        and frame["chance_mean_macro_f1"].notna().any()
        else np.nan
    )
    if pd.notna(chance):
        ax.axhline(
            chance,
            color=_style("chance_colour", "#777777"),
            linestyle=":",
            linewidth=1.4,
            label=f"Chance ({chance:.3f})",
        )
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    thumbnails = thumbnails or {}
    if thumbnails and "condition" in frame:
        _add_condition_thumbnails(
            ax,
            frame,
            x_column=x,
            thumbnails=thumbnails,
        )
    fig.tight_layout(rect=(0, 0.13 if thumbnails else 0, 1, 1))
    _write_figure_sources(
        path,
        frame,
        extra={
            "summary_rows": len(summary),
            "thumbnails": {
                name: str(image_path)
                for name, image_path in thumbnails.items()
            },
        },
    )
    summary.to_csv(
        _figure_source_dir(path) / "seed_summary.csv", index=False
    )
    _save_figure_formats(fig, path)
    plt.close(fig)


def best_baseline_configuration(
    baseline: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if baseline.empty:
        return {}, baseline
    grouped = (
        baseline.groupby(["model", "loss_recipe"], as_index=False)
        .agg(
            mean_validation_score=("best_val_score", "mean"),
            mean_test_macro_f1=("test_mean_macro_f1", "mean"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(
            ["mean_validation_score", "mean_test_macro_f1"],
            ascending=False,
        )
    )
    winner = grouped.iloc[0].to_dict()
    selected = baseline[
        baseline["model"].eq(winner["model"])
        & baseline["loss_recipe"].eq(winner["loss_recipe"])
    ].copy()
    return winner, selected


def _save_workflow(path: Path) -> None:
    labels = [
        "Specimen\ncollection",
        "Image\nacquisition",
        "Segmentation",
        "Individual-level\nsplit",
        "Model\ntraining",
        "Visual\nablations",
        "Cohort\nablations",
        "Evaluation and\npaper outputs",
    ]
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for index, label in enumerate(labels):
        x = index + 0.08
        colour = (
            _style("workflow_data_colour", "#D9EDF7")
            if index < 4
            else _style("workflow_model_colour", "#DFF0D8")
            if index == 4
            else _style("workflow_experiment_colour", "#FCE8C3")
        )
        box = FancyBboxPatch(
            (x, 0.3),
            0.84,
            0.4,
            boxstyle="round,pad=0.03",
            facecolor=colour,
            edgecolor="#333333",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + 0.42, 0.5, label, ha="center", va="center", fontsize=10)
        if index < len(labels) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.86, 0.5),
                    (x + 1.06, 0.5),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color="#555555",
                )
            )
    ax.set_title("Earthworm image-classification study workflow", fontsize=14)
    fig.tight_layout()
    _write_figure_sources(
        path,
        pd.DataFrame({
            "order": np.arange(1, len(labels) + 1),
            "stage": [label.replace("\n", " ") for label in labels],
        }),
    )
    _save_figure_formats(fig, path)
    plt.close(fig)


def _resolve_image(data_root: Path | None, raw: Any) -> Path | None:
    if data_root is None or not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = data_root / path
    return path if path.is_file() else None


def _image_array(path: Path, mode: str = "RGB") -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert(mode))


def _candidate_image_rows(
    split_frames: dict[str, pd.DataFrame],
    data_root: Path | None,
) -> list[tuple[pd.Series, float]]:
    if not split_frames or data_root is None:
        return []
    frame = pd.concat(split_frames.values(), ignore_index=True)
    if "rel_path_seg" not in frame:
        return []
    candidates = []
    sampled = frame.drop_duplicates("barcode").head(400)
    for _, row in sampled.iterrows():
        segmented = _resolve_image(data_root, row.get("rel_path_seg"))
        if segmented is None:
            continue
        mask_path = _resolve_image(data_root, row.get("rel_path_segmask"))
        fraction = 0.0
        if mask_path is not None:
            mask = _image_array(mask_path, "L")
            fraction = float((mask > 0).mean())
        candidates.append((row, fraction))
    return candidates


def _select_representatives(
    split_frames: dict[str, pd.DataFrame],
    data_root: Path | None,
) -> list[tuple[pd.Series, str]]:
    candidates = _candidate_image_rows(split_frames, data_root)
    if not candidates:
        return []
    fractions = np.asarray([fraction for _, fraction in candidates])
    median = float(np.median(fractions))
    ranked_typical = sorted(
        candidates, key=lambda item: abs(item[1] - median)
    )
    ranked_difficult = sorted(
        candidates, key=lambda item: abs(item[1] - median), reverse=True
    )
    selected: list[tuple[pd.Series, str]] = []
    used_barcodes: set[str] = set()
    used_labels: set[tuple[str, str]] = set()
    for label, pool in (
        ("typical segmentation", ranked_typical),
        ("typical segmentation", ranked_typical),
        ("difficult mask geometry", ranked_difficult),
        ("difficult mask geometry", ranked_difficult),
    ):
        choice = None
        for row, _ in pool:
            barcode = str(row.get("barcode"))
            taxon_stage = (
                str(row.get("genus")),
                str(row.get("life_stage")),
            )
            if barcode in used_barcodes:
                continue
            if taxon_stage in used_labels and len(used_labels) < 4:
                continue
            choice = row
            used_barcodes.add(barcode)
            used_labels.add(taxon_stage)
            break
        if choice is not None:
            selected.append((choice, label))
    return selected


def _save_representative_images(
    *,
    split_frames: dict[str, pd.DataFrame],
    data_root: Path | None,
    path: Path,
) -> bool:
    selected = _select_representatives(split_frames, data_root)
    if not selected:
        return False
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.2 * len(selected)))
    axes = np.atleast_2d(axes)
    records = []
    for row_index, (row, quality) in enumerate(selected):
        paths = [
            _resolve_image(data_root, row.get("rel_path_raw")),
            _resolve_image(data_root, row.get("rel_path_seg")),
            _resolve_image(data_root, row.get("rel_path_segmask")),
        ]
        for column, (label, image_path) in enumerate(
            zip(("Raw", "Segmented", "Mask"), paths)
        ):
            ax = axes[row_index, column]
            ax.axis("off")
            if image_path is not None:
                ax.imshow(
                    _image_array(
                        image_path, "L" if column == 2 else "RGB"
                    ),
                    cmap="gray" if column == 2 else None,
                )
                records.append({
                    "barcode": row.get("barcode"),
                    "quality": quality,
                    "kind": label,
                    "source_path": str(image_path),
                    "sha256": hashlib.sha256(
                        image_path.read_bytes()
                    ).hexdigest(),
                })
            ax.set_title(label)
        taxon = row.get("species_label")
        if pd.isna(taxon):
            taxon = row.get("taxon_label") or row.get("genus")
        axes[row_index, 0].set_ylabel(
            f"{taxon}\n{row.get('life_stage')}\n{quality}",
            fontsize=9,
        )
    fig.suptitle(
        "Representative taxa and segmentation outcomes\n"
        "(difficulty is defined by atypical foreground-mask area)",
        fontsize=13,
    )
    fig.tight_layout()
    _write_figure_sources(path, pd.DataFrame(records))
    _save_figure_formats(fig, path)
    plt.close(fig)
    return True


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def _runtime_condition_from_run(row: pd.Series) -> dict[str, Any]:
    transform = str(row.get("transform", "original"))
    parameters: dict[str, Any] = {}
    if transform == "saturation":
        parameters["retention"] = float(row["colour_retention"])
    elif transform == "patch_shuffle":
        parameters.update(
            grid_size=int(row["grid_size"]),
            seed=2026,
        )
    elif transform == "gaussian_blur_percent":
        raw_max_sigma = row.get("gaussian_max_sigma")
        parameters.update(
            percent=float(row["percent"]),
            max_sigma=(
                float(raw_max_sigma)
                if pd.notna(raw_max_sigma)
                else 64.0
            ),
        )
    elif transform == "resolution_loss":
        parameters["percent"] = float(row["percent"])
    elif transform == "composed":
        operations = _parse_parameter_json(row.get("operations"))
        if not operations and isinstance(row.get("operations"), str):
            try:
                operations = json.loads(row["operations"])
            except json.JSONDecodeError:
                operations = []
        parameters["operations"] = operations
    return {
        "condition": str(row["condition"]),
        "transform": transform,
        "parameters": parameters,
    }


def _save_visual_level_images(
    *,
    frame: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    data_root: Path | None,
    figure_path: Path,
) -> dict[str, Path]:
    """Save the exact pre-normalisation image shown for every plotted level."""
    selected = _select_representatives(split_frames, data_root)
    if frame.empty or not selected:
        return {}
    row = selected[0][0]
    image_path = _resolve_image(data_root, row.get("rel_path_seg"))
    if image_path is None:
        return {}
    with Image.open(image_path) as image:
        tensor = tv_functional.to_tensor(image.convert("RGB"))
    tensor = tv_functional.resize(tensor, [224, 224], antialias=True)
    directory = _figure_source_dir(figure_path)
    transformed_dir = directory / "transformed_images"
    transformed_dir.mkdir(parents=True, exist_ok=True)
    original_path = directory / "source_image.png"
    Image.fromarray(
        np.uint8(np.rint(_tensor_to_image(tensor) * 255.0))
    ).save(original_path)
    outputs: dict[str, Path] = {}
    records = []
    unique = frame.drop_duplicates("condition")
    for _, condition_row in unique.iterrows():
        condition = _runtime_condition_from_run(condition_row)
        transformed = tensor
        for operation in build_condition_operations(condition):
            transformed = operation(transformed)
        destination = transformed_dir / f"{condition['condition']}.png"
        Image.fromarray(
            np.uint8(np.rint(_tensor_to_image(transformed) * 255.0))
        ).save(destination)
        outputs[condition["condition"]] = destination
        records.append({
            "condition": condition["condition"],
            "transform": condition["transform"],
            "parameters": json.dumps(
                condition["parameters"], sort_keys=True
            ),
            "image": str(destination.relative_to(directory)),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        })
    pd.DataFrame(records).to_csv(
        directory / "transformed_images.csv", index=False
    )
    metadata = {
        "selection_rule": (
            "first deterministic typical-segmentation representative"
        ),
        "barcode": str(row.get("barcode")),
        "source_path": str(image_path),
        "source_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "render_stage": "resized tensor before ImageNet normalisation",
        "operation_order": "configured list order",
    }
    (directory / "image_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def _add_condition_thumbnails(
    ax: Any,
    frame: pd.DataFrame,
    *,
    x_column: str,
    thumbnails: dict[str, Path],
) -> None:
    unique = frame.dropna(subset=[x_column]).drop_duplicates(
        [x_column, "condition"]
    )
    zoom = 0.11 if len(unique) > 7 else 0.15
    for _, row in unique.iterrows():
        path = thumbnails.get(str(row["condition"]))
        if path is None:
            continue
        image = plt.imread(path)
        box = AnnotationBbox(
            OffsetImage(image, zoom=zoom),
            (float(row[x_column]), -0.17),
            xycoords=("data", "axes fraction"),
            frameon=True,
            pad=0.05,
            annotation_clip=False,
        )
        ax.add_artist(box)


def _save_transformation_examples(
    *,
    split_frames: dict[str, pd.DataFrame],
    data_root: Path | None,
    path: Path,
) -> bool:
    selected = _select_representatives(split_frames, data_root)
    if not selected:
        return False
    row = selected[0][0]
    image_path = _resolve_image(data_root, row.get("rel_path_seg"))
    if image_path is None:
        return False
    with Image.open(image_path) as image:
        tensor = tv_functional.to_tensor(image.convert("RGB"))
    tensor = tv_functional.resize(tensor, [224, 224], antialias=True)
    examples = [
        ("Original", tensor),
        ("0% colour", ColourRetention(0)(tensor)),
        ("Patch 4x4", PatchShuffle(4, seed=2026)(tensor)),
        ("Patch 16x16", PatchShuffle(16, seed=2026)(tensor)),
        ("Blur 25% (sigma 16)", GaussianBlurPercent(25, 64)(tensor)),
        ("Blur 50% (sigma 32)", GaussianBlurPercent(50, 64)(tensor)),
        ("Blur 100% (sigma 64)", GaussianBlurPercent(100, 64)(tensor)),
        ("Resolution 50% (112px)", ResolutionLoss(50)(tensor)),
        ("Resolution 87.5% (28px)", ResolutionLoss(87.5)(tensor)),
        ("Resolution 100% (1px)", ResolutionLoss(100)(tensor)),
    ]
    directory = _figure_source_dir(path)
    image_dir = directory / "transformed_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for index, (ax, (title, transformed)) in enumerate(
        zip(axes.flat, examples)
    ):
        ax.imshow(_tensor_to_image(transformed))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        destination = image_dir / f"{index:02d}_{title.lower().replace(' ', '_').replace('%', 'pct')}.png"
        Image.fromarray(
            np.uint8(np.rint(_tensor_to_image(transformed) * 255.0))
        ).save(destination)
        records.append({
            "order": index,
            "label": title,
            "image": str(destination.relative_to(directory)),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "source_path": str(image_path),
        })
    fig.suptitle("Visual ablations applied to the same individual", fontsize=14)
    fig.tight_layout()
    _write_figure_sources(path, pd.DataFrame(records))
    _save_figure_formats(fig, path)
    plt.close(fig)
    return True


def _save_baseline_ci(
    baseline_summary: pd.DataFrame,
    path: Path,
) -> None:
    selected = baseline_summary[baseline_summary["task"].eq("mean")]
    if selected.empty:
        return
    models = selected["model"].drop_duplicates().tolist()
    recipes = selected["loss_recipe"].drop_duplicates().tolist()
    x = np.arange(len(models), dtype=float)
    width = 0.8 / max(1, len(recipes))
    fig, ax = plt.subplots(figsize=(13, 6))
    for index, recipe in enumerate(recipes):
        group = selected[selected["loss_recipe"].eq(recipe)].set_index("model")
        means = [group["mean"].get(model, np.nan) for model in models]
        errors = [group["ci95"].get(model, np.nan) for model in models]
        ax.bar(
            x + (index - (len(recipes) - 1) / 2) * width,
            means,
            width=width,
            yerr=errors,
            capsize=4,
            label=recipe,
        )
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel("Test mean macro-F1")
    ax.set_ylim(0, 1)
    chance = (
        float(selected["chance"].dropna().mean())
        if "chance" in selected and selected["chance"].notna().any()
        else np.nan
    )
    if pd.notna(chance):
        ax.axhline(
            chance,
            color=_style("chance_colour", "#777777"),
            linestyle=":",
            linewidth=1.4,
            label=f"Chance ({chance:.3f})",
        )
    ax.set_title("Baseline comparison with seed-level 95% t confidence intervals")
    ax.legend(title="Loss recipe", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _write_figure_sources(path, selected)
    _save_figure_formats(fig, path)
    plt.close(fig)


def _save_visual_overview(
    visual: pd.DataFrame,
    best_score: float,
    path: Path,
    thumbnails: dict[str, Path] | None = None,
) -> None:
    if visual.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    panels = [
        ("patch_shuffle", "grid_size", "Patch grid per side", "Patch shuffle"),
        (
            "gaussian_blur_percent",
            "percent",
            "Blur severity (%)",
            "Gaussian blur",
        ),
        (
            "resolution_loss",
            "percent",
            "Resolution discarded (%)",
            "Resolution loss",
        ),
    ]
    plotted_summaries = []
    thumbnails = thumbnails or {}
    for ax, (transform, x_column, xlabel, title) in zip(axes.flat, panels):
        frame = visual[visual["transform"].eq(transform)]
        summary = _ci_summary(
            frame,
            groups=["model", x_column],
            value="test_mean_macro_f1",
        )
        summary["transform"] = transform
        plotted_summaries.append(summary)
        for model, group in summary.groupby("model"):
            ordered = group.sort_values(x_column)
            ax.errorbar(
                ordered[x_column],
                ordered["mean"],
                yerr=ordered["ci95"],
                marker="o",
                capsize=3,
                label=model,
            )
        ax.axhline(
            best_score,
            color=_style("best_baseline_colour", "#111111"),
            linestyle="--",
            linewidth=1.2,
            label="Best baseline",
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Matched-test mean macro-F1")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        if transform == "resolution_loss":
            ticks, labels = _resolution_tick_labels()
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, fontsize=7)
        _add_condition_thumbnails(
            ax,
            frame,
            x_column=x_column,
            thumbnails=thumbnails,
        )
    colour_ax = axes.flat[3]
    colour = visual[visual["transform"].eq("saturation")]
    colour_summary = _ci_summary(
        colour, groups=["model"], value="test_mean_macro_f1"
    )
    colour_ax.bar(
        colour_summary["model"],
        colour_summary["mean"],
        yerr=colour_summary["ci95"],
        capsize=3,
        color=_style("palette", ["#4C78A8"])[0],
    )
    colour_ax.axhline(
        best_score,
        color=_style("best_baseline_colour", "#111111"),
        linestyle="--",
        linewidth=1.2,
    )
    colour_ax.set_title("0% colour")
    colour_ax.set_ylabel("Matched-test mean macro-F1")
    colour_ax.set_ylim(0, 1)
    colour_ax.tick_params(axis="x", rotation=20)
    colour_image = next(
        (
            thumbnails.get(str(condition))
            for condition in colour["condition"].dropna().unique()
            if thumbnails.get(str(condition)) is not None
        ),
        None,
    )
    if colour_image is not None:
        colour_ax.add_artist(AnnotationBbox(
            OffsetImage(plt.imread(colour_image), zoom=0.22),
            (0.5, -0.18),
            xycoords=("axes fraction", "axes fraction"),
            frameon=True,
            annotation_clip=False,
        ))
    chance = (
        float(visual["chance_mean_macro_f1"].dropna().mean())
        if "chance_mean_macro_f1" in visual
        and visual["chance_mean_macro_f1"].notna().any()
        else np.nan
    )
    if pd.notna(chance):
        for ax in axes.flat:
            ax.axhline(
                chance,
                color=_style("chance_colour", "#777777"),
                linestyle=":",
                linewidth=1.2,
            )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Visual-information ablations compared with the best baseline")
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    _write_figure_sources(
        path,
        visual,
        extra={
            "thumbnails": {
                name: str(image_path)
                for name, image_path in thumbnails.items()
            }
        },
    )
    if plotted_summaries:
        pd.concat(plotted_summaries, ignore_index=True).to_csv(
            _figure_source_dir(path) / "seed_summary.csv", index=False
        )
    _save_figure_formats(fig, path)
    plt.close(fig)


def _save_notebook_baseline(
    baseline: pd.DataFrame,
    path: Path,
) -> None:
    """Recreate the notebook's combined task plot from completed runs."""
    if baseline.empty:
        return
    downstream_recipe = _loss_name(
        {"genus": 1.0, "species": 0.5, "age": 2.0}
    )
    selected = baseline[baseline["loss_recipe"].eq(downstream_recipe)].copy()
    if selected.empty:
        selected = baseline.copy()
    metrics = [
        ("test_mean_macro_f1", "Mean", "chance_mean_macro_f1"),
        ("test_genus_macro_f1", "Genus", "chance_genus_macro_f1"),
        ("test_species_macro_f1", "Species", "chance_species_macro_f1"),
        ("test_age_macro_f1", "Developmental stage", "chance_age_macro_f1"),
    ]
    models = selected["model"].drop_duplicates().tolist()
    x = np.arange(len(models), dtype=float)
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(12, 6))
    summaries = []
    for index, (metric, label, chance_column) in enumerate(metrics):
        summary = _ci_summary(
            selected, groups=["model"], value=metric
        )
        summary["task"] = label
        summaries.append(summary)
        group = summary.set_index("model")
        ax.bar(
            x + (index - (len(metrics) - 1) / 2) * width,
            [group["mean"].get(model, np.nan) for model in models],
            width=width,
            yerr=[group["ci95"].get(model, np.nan) for model in models],
            capsize=3,
            label=label,
        )
        chance = selected[chance_column].dropna()
        if not chance.empty:
            ax.axhline(
                float(chance.mean()),
                color=_style("chance_colour", "#777777"),
                linestyle=":",
                linewidth=0.8,
                alpha=0.55,
            )
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Baseline performance across prediction tasks "
        "(mean and seed-level 95% CI)"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    _write_figure_sources(
        path,
        selected,
        extra={"loss_recipe": downstream_recipe},
    )
    pd.concat(summaries, ignore_index=True).to_csv(
        _figure_source_dir(path) / "seed_summary.csv", index=False
    )
    _save_figure_formats(fig, path)
    plt.close(fig)


def _save_interaction_figures(
    interactions: pd.DataFrame,
    figures: Path,
    *,
    suffix: str = "",
) -> list[str]:
    if interactions.empty:
        return []
    created = []
    labels = {
        "saturation": ("Colour removal", "Gaussian blur severity (%)"),
        "patch_shuffle": ("Patch shuffle", "Patch grid per side"),
    }
    for transform, (title, xlabel) in labels.items():
        frame = interactions[
            interactions["paired_transform"].eq(transform)
        ].copy()
        if frame.empty:
            continue
        models = frame["model"].drop_duplicates().tolist()
        fig, axes = plt.subplots(
            len(models), 1, figsize=(11, 3.2 * len(models)), sharex=True
        )
        axes = np.atleast_1d(axes)
        summaries = []
        for ax, model in zip(axes, models):
            model_frame = frame[frame["model"].eq(model)]
            if transform == "saturation":
                summary = _ci_summary(
                    model_frame,
                    groups=["gaussian_percent"],
                    value="test_mean_macro_f1",
                )
                summaries.append(summary.assign(model=model))
                ordered = summary.sort_values("gaussian_percent")
                ax.errorbar(
                    ordered["gaussian_percent"],
                    ordered["mean"],
                    yerr=ordered["ci95"],
                    marker="o",
                    capsize=3,
                    label="0% colour",
                )
            else:
                summary = _ci_summary(
                    model_frame,
                    groups=["gaussian_percent", "paired_level"],
                    value="test_mean_macro_f1",
                )
                summaries.append(summary.assign(model=model))
                for blur_level, group in summary.groupby(
                    "gaussian_percent"
                ):
                    ordered = group.sort_values("paired_level")
                    ax.errorbar(
                        ordered["paired_level"],
                        ordered["mean"],
                        yerr=ordered["ci95"],
                        marker="o",
                        capsize=3,
                        label=f"Blur {float(blur_level):g}%",
                    )
            chance = model_frame["chance_mean_macro_f1"].dropna()
            if not chance.empty:
                ax.axhline(
                    float(chance.mean()),
                    color=_style("chance_colour", "#777777"),
                    linestyle=":",
                    label="Chance",
                )
            ax.set_ylim(0, 1)
            ax.set_ylabel("Macro-F1")
            ax.set_title(model)
            ax.grid(alpha=0.25)
            ax.legend(ncol=5, fontsize=8)
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(
            f"Gaussian blur × {title}: mean and seed-level 95% CI"
        )
        fig.tight_layout()
        path = figures / f"visual_interaction_{transform}{suffix}.png"
        _write_figure_sources(path, frame)
        pd.concat(summaries, ignore_index=True).to_csv(
            _figure_source_dir(path) / "seed_summary.csv", index=False
        )
        _save_figure_formats(fig, path)
        plt.close(fig)
        created.append(path.name)
    return created


def _save_interaction_contact_sheet(
    thumbnails: dict[str, Path],
    path: Path,
) -> bool:
    if not thumbnails:
        return False
    items = sorted(thumbnails.items())
    columns = 12
    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(24, 3.0 * rows))
    axes = np.atleast_2d(axes)
    for ax, item in zip(axes.flat, items):
        name, image_path = item
        ax.imshow(plt.imread(image_path))
        ax.set_title(name.replace("__", "\n"), fontsize=7)
        ax.axis("off")
    for ax in axes.flat[len(items):]:
        ax.axis("off")
    fig.suptitle("All Gaussian-blur pairwise interaction conditions")
    fig.tight_layout()
    _save_figure_formats(fig, path)
    plt.close(fig)
    return True


def _save_resolution_three_way(
    visual: pd.DataFrame,
    original_test: pd.DataFrame,
    baseline_controls: pd.DataFrame,
    path: Path,
) -> None:
    if (
        baseline_controls.empty
        or "test_transform" not in baseline_controls
    ):
        return
    matched = visual[visual["transform"].eq("resolution_loss")][
        [
            "model", "seed", "loss_recipe", "percent",
            "test_mean_macro_f1", "chance_mean_macro_f1", "run_dir",
        ]
    ].rename(columns={"test_mean_macro_f1": "score"})
    matched["evaluation"] = "Resolution-trained / matched test"
    original = original_test[
        original_test["train_transform"].eq("resolution_loss")
    ][
        [
            "model", "seed", "loss_recipe", "percent",
            "mean_macro_f1", "run_dir",
        ]
    ].rename(columns={"mean_macro_f1": "score"})
    original["evaluation"] = "Resolution-trained / original test"
    baseline = baseline_controls[
        baseline_controls["test_transform"].eq("resolution_loss")
    ][
        [
            "model", "seed", "loss_recipe", "percent",
            "mean_macro_f1", "run_dir",
        ]
    ].rename(columns={"mean_macro_f1": "score"})
    baseline["evaluation"] = "Baseline-trained / resolution test"
    fixed_recipe = _loss_name(
        {"genus": 1.0, "species": 0.5, "age": 2.0}
    )
    combined = pd.concat(
        [matched, original, baseline], ignore_index=True, sort=False
    )
    combined = combined[combined["loss_recipe"].eq(fixed_recipe)]
    if combined.empty:
        return
    summary = _ci_summary(
        combined,
        groups=["model", "percent", "evaluation"],
        value="score",
    )
    models = summary["model"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(models), 1, figsize=(11, 3.2 * len(models)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        model_summary = summary[summary["model"].eq(model)]
        for evaluation, group in model_summary.groupby("evaluation"):
            ordered = group.sort_values("percent")
            ax.errorbar(
                ordered["percent"],
                ordered["mean"],
                yerr=ordered["ci95"],
                marker="o",
                capsize=3,
                label=evaluation,
            )
        chance = combined[
            combined["model"].eq(model)
        ].get("chance_mean_macro_f1")
        if chance is not None and chance.notna().any():
            ax.axhline(
                float(chance.dropna().mean()),
                color=_style("chance_colour", "#777777"),
                linestyle=":",
                label="Chance",
            )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Mean macro-F1")
        ax.set_title(model)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Resolution loss (%)")
    fig.suptitle("Three-way spatial-resolution control comparison")
    fig.tight_layout()
    _write_figure_sources(path, combined)
    summary.to_csv(
        _figure_source_dir(path) / "seed_summary.csv", index=False
    )
    _save_figure_formats(fig, path)
    plt.close(fig)


def matched_vs_original_table(
    visual: pd.DataFrame,
    original_test: pd.DataFrame,
) -> pd.DataFrame:
    if visual.empty or original_test.empty:
        return pd.DataFrame()
    matched = visual[
        [
            "model", "seed", "hierarchy_loss_weight",
            "condition", "test_mean_macro_f1",
        ]
    ].rename(columns={"test_mean_macro_f1": "matched_mean_macro_f1"})
    original = original_test[
        [
            "model", "seed", "hierarchy_loss_weight",
            "train_condition", "mean_macro_f1",
        ]
    ].rename(
        columns={
            "train_condition": "condition",
            "mean_macro_f1": "original_mean_macro_f1",
        }
    )
    matched["seed"] = matched["seed"].astype("string")
    original["seed"] = original["seed"].astype("string")
    result = matched.merge(
        original,
        on=["model", "seed", "hierarchy_loss_weight", "condition"],
        how="inner",
    )
    result["matched_minus_original"] = (
        result["matched_mean_macro_f1"]
        - result["original_mean_macro_f1"]
    )
    return result


def _save_matched_vs_original(
    comparison: pd.DataFrame,
    path: Path,
) -> None:
    if comparison.empty:
        return
    matched = comparison.pivot_table(
        index="condition",
        columns="model",
        values="matched_mean_macro_f1",
        aggfunc="mean",
    )
    original = comparison.pivot_table(
        index="condition",
        columns="model",
        values="original_mean_macro_f1",
        aggfunc="mean",
    )
    columns = sorted(set(matched.columns) | set(original.columns))
    matched = matched.reindex(columns=columns)
    original = original.reindex(index=matched.index, columns=columns)
    matrix = np.concatenate([matched.to_numpy(), original.to_numpy()], axis=1)
    fig, ax = plt.subplots(
        figsize=(max(12, len(columns) * 2.0), max(8, len(matched) * 0.35))
    )
    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap=_style("heatmap_colormap", "viridis"),
    )
    ax.set_yticks(np.arange(len(matched.index)), matched.index, fontsize=8)
    ax.set_xticks(
        np.arange(2 * len(columns)),
        [f"{model}\nmatched" for model in columns]
        + [f"{model}\noriginal" for model in columns],
        rotation=35,
        ha="right",
    )
    ax.axvline(len(columns) - 0.5, color="white", linewidth=2)
    ax.set_title("Matched-condition versus original-image test performance")
    fig.colorbar(image, ax=ax, label="Mean macro-F1")
    fig.tight_layout()
    _write_figure_sources(path, comparison)
    _save_figure_formats(fig, path)
    plt.close(fig)
    sd_matched = comparison.pivot_table(
        index="condition",
        columns="model",
        values="matched_mean_macro_f1",
        aggfunc="std",
    ).reindex(index=matched.index, columns=columns)
    sd_original = comparison.pivot_table(
        index="condition",
        columns="model",
        values="original_mean_macro_f1",
        aggfunc="std",
    ).reindex(index=matched.index, columns=columns)
    sd_matrix = np.concatenate(
        [sd_matched.to_numpy(), sd_original.to_numpy()], axis=1
    )
    sd_fig, sd_ax = plt.subplots(
        figsize=(max(12, len(columns) * 2.0), max(8, len(matched) * 0.35))
    )
    sd_image = sd_ax.imshow(
        sd_matrix,
        aspect="auto",
        vmin=0,
        cmap="magma",
    )
    sd_ax.set_yticks(
        np.arange(len(matched.index)), matched.index, fontsize=8
    )
    sd_ax.set_xticks(
        np.arange(2 * len(columns)),
        [f"{model}\nmatched" for model in columns]
        + [f"{model}\noriginal" for model in columns],
        rotation=35,
        ha="right",
    )
    sd_ax.set_title("Seed SD for matched versus original-image performance")
    sd_fig.colorbar(sd_image, ax=sd_ax, label="Seed SD of mean macro-F1")
    sd_fig.tight_layout()
    _save_figure_formats(
        sd_fig, path.with_name(path.stem + "_seed_sd.png")
    )
    plt.close(sd_fig)


def _save_holdout_change(
    joined: pd.DataFrame,
    *,
    best_score: float,
    path: Path,
) -> None:
    if joined.empty:
        return
    cohort = (
        joined[joined["cohort"].eq("independent_test")]
        if "cohort" in joined
        else joined
    )
    aggregation = {
        "full_test_mean_macro_f1": ("test_mean_macro_f1", "first"),
        "cohort_target_recall": ("target_recall", "mean"),
    }
    if "chance" in cohort:
        aggregation["cohort_chance"] = ("chance", "mean")
    per_seed = cohort.groupby(
        ["holdout", "model", "seed"],
        as_index=False,
        dropna=False,
    ).agg(**aggregation)
    per_seed["change_from_best_baseline"] = (
        per_seed["full_test_mean_macro_f1"] - best_score
    )
    delta = _ci_summary(
        per_seed,
        groups=["holdout", "model"],
        value="change_from_best_baseline",
    )
    recall = _ci_summary(
        per_seed,
        groups=["holdout", "model"],
        value="cohort_target_recall",
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    holdout_names = delta["holdout"].drop_duplicates().tolist()
    models = delta["model"].drop_duplicates().tolist()
    x = np.arange(len(holdout_names), dtype=float)
    width = 0.8 / max(1, len(models))
    for model_index, model in enumerate(models):
        delta_group = delta[delta["model"].eq(model)].set_index("holdout")
        recall_group = recall[recall["model"].eq(model)].set_index("holdout")
        positions = x + (model_index - (len(models) - 1) / 2) * width
        axes[0].bar(
            positions,
            [delta_group["mean"].get(name, np.nan) for name in holdout_names],
            width=width,
            yerr=[
                delta_group["ci95"].get(name, np.nan)
                for name in holdout_names
            ],
            capsize=3,
            label=model,
        )
        axes[1].bar(
            positions,
            [recall_group["mean"].get(name, np.nan) for name in holdout_names],
            width=width,
            yerr=[
                recall_group["ci95"].get(name, np.nan)
                for name in holdout_names
            ],
            capsize=3,
            label=model,
        )
    axes[0].axhline(
        0,
        color=_style("best_baseline_colour", "#111111"),
        linewidth=1,
    )
    axes[0].set_ylabel("Full-test macro-F1 change")
    axes[0].set_title("Change relative to the best baseline")
    axes[1].set_ylabel("Held-out target recall")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Recognition within the preserved held-out cohort")
    if (
        "cohort_chance" in per_seed
        and per_seed["cohort_chance"].notna().any()
    ):
        chance_by_holdout = per_seed.groupby(
            "holdout", dropna=False
        )["cohort_chance"].mean()
        axes[1].plot(
            x,
            [
                chance_by_holdout.get(name, np.nan)
                for name in holdout_names
            ],
            color=_style("chance_colour", "#777777"),
            linestyle=":",
            marker="_",
            markersize=14,
            label="Chance",
        )
    for ax in axes:
        ax.set_xlabel("")
        ax.set_xticks(x, holdout_names, rotation=25, ha="right")
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    _write_figure_sources(path, per_seed)
    delta.assign(metric="full_test_change").to_csv(
        _figure_source_dir(path) / "full_test_seed_summary.csv", index=False
    )
    recall.assign(metric="target_recall").to_csv(
        _figure_source_dir(path) / "target_recall_seed_summary.csv",
        index=False,
    )
    _save_figure_formats(fig, path)
    plt.close(fig)


def _save_notebook_holdouts(
    holdouts: pd.DataFrame,
    controls: pd.DataFrame,
    path: Path,
) -> None:
    """Four-panel dual-cohort comparison against matched baselines."""
    if holdouts.empty:
        return
    fixed_recipe = _loss_name(
        {"genus": 1.0, "species": 0.5, "age": 2.0}
    )
    control = (
        controls[controls["loss_recipe"].eq(fixed_recipe)].copy()
        if not controls.empty and "loss_recipe" in controls
        else pd.DataFrame(columns=holdouts.columns)
    )
    trained = holdouts.copy()
    trained["system"] = "Holdout-trained"
    control["system"] = "Baseline"
    combined = (
        pd.concat([trained, control], ignore_index=True, sort=False)
        if not control.empty
        else trained
    )
    names = combined["holdout"].dropna().drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=True)
    axes = np.asarray(axes).ravel()
    summaries = []
    palette = _style(
        "palette",
        ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"],
    )
    for ax, holdout_name in zip(axes, names):
        frame = combined[combined["holdout"].eq(holdout_name)].copy()
        frame["category"] = (
            frame["cohort"].astype(str)
            + "\n"
            + frame["task"].astype(str)
        )
        summary = _ci_summary(
            frame,
            groups=["category", "model", "system"],
            value="target_recall",
        )
        summaries.append(summary.assign(holdout=holdout_name))
        categories = frame["category"].drop_duplicates().tolist()
        x = np.arange(len(categories), dtype=float)
        models = frame["model"].drop_duplicates().tolist()
        offsets = np.linspace(-0.32, 0.32, max(1, len(models)))
        for model_index, model in enumerate(models):
            for system, marker, fill in (
                ("Holdout-trained", "o", True),
                ("Baseline", "s", False),
            ):
                group = summary[
                    summary["model"].eq(model)
                    & summary["system"].eq(system)
                ].set_index("category")
                ax.errorbar(
                    x + offsets[model_index],
                    [group["mean"].get(item, np.nan) for item in categories],
                    yerr=[
                        group["ci95"].get(item, np.nan)
                        for item in categories
                    ],
                    fmt=marker,
                    markerfacecolor=(
                        palette[model_index % len(palette)]
                        if fill else "white"
                    ),
                    color=palette[model_index % len(palette)],
                    capsize=3,
                    label=f"{model} / {system}",
                )
        chance_by_category = (
            frame.groupby("category", as_index=False)["chance"].mean()
            if "chance" in frame
            else pd.DataFrame()
        )
        if not chance_by_category.empty:
            chance_map = chance_by_category.set_index("category")["chance"]
            ax.scatter(
                x,
                [chance_map.get(item, np.nan) for item in categories],
                marker="_",
                s=180,
                linewidths=2,
                color=_style("chance_colour", "#777777"),
                label="Chance",
                zorder=5,
            )
        ax.set_xticks(x, categories)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Target-class recall")
        ax.set_title(str(holdout_name).replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(names):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(
        "Structured biological holdouts on development-withheld and "
        "independent-test cohorts"
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    _write_figure_sources(
        path,
        combined,
        extra={"baseline_loss_recipe": fixed_recipe},
    )
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            _figure_source_dir(path) / "seed_summary.csv", index=False
        )
    _save_figure_formats(fig, path)
    plt.close(fig)


def _read_confusion_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame.astype(float)


def _save_validation_selected_confusions(
    baseline_runs: pd.DataFrame,
    path: Path,
) -> bool:
    if baseline_runs.empty:
        return False
    tasks = ("genus", "species", "age")
    aggregated: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    source_records = []
    directory = _figure_source_dir(path)
    matrix_dir = directory / "matrix_data"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        frames = []
        for row in baseline_runs.itertuples():
            matrix_path = (
                Path(row.run_dir)
                / f"confusion_matrix_best_{task}.csv"
            )
            if not matrix_path.is_file():
                continue
            frame = _read_confusion_frame(matrix_path)
            frames.append(frame)
            source_records.append({
                "task": task,
                "model": row.model,
                "seed": row.seed,
                "path": str(matrix_path),
            })
        if not frames:
            continue
        labels = frames[0].index.tolist()
        if any(
            frame.index.tolist() != labels
            or frame.columns.tolist() != labels
            for frame in frames
        ):
            raise ValueError(
                f"Confusion-matrix labels differ across seeds for {task}"
            )
        normalized = []
        for frame in frames:
            matrix = frame.to_numpy()
            row_sums = matrix.sum(axis=1, keepdims=True)
            normalized.append(np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix, dtype=float),
                where=row_sums != 0,
            ))
        stack = np.stack(normalized)
        mean = stack.mean(axis=0)
        sd = stack.std(axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(mean)
        aggregated[task] = (mean, sd, labels)
        pd.DataFrame(mean, index=labels, columns=labels).to_csv(
            matrix_dir / f"{task}_row_normalised_mean.csv"
        )
        pd.DataFrame(sd, index=labels, columns=labels).to_csv(
            matrix_dir / f"{task}_row_normalised_sd.csv"
        )
    if not aggregated:
        return False
    for statistic, index in (("mean", 0), ("sd", 1)):
        fig, axes = plt.subplots(
            1, len(aggregated), figsize=(19, 7), constrained_layout=True
        )
        axes = np.atleast_1d(axes)
        image = None
        for ax, (task, values) in zip(axes, aggregated.items()):
            matrix = values[index]
            labels = values[2]
            image = ax.imshow(
                matrix,
                cmap=(
                    _style("heatmap_colormap", "viridis")
                    if statistic == "mean" else "magma"
                ),
                vmin=0,
                vmax=1 if statistic == "mean" else None,
                aspect="equal",
            )
            ax.set_xticks(
                np.arange(len(labels)), labels, rotation=55, ha="right"
            )
            ax.set_yticks(np.arange(len(labels)), labels)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title(f"{task}: seed {statistic}")
        if image is not None:
            fig.colorbar(
                image,
                ax=axes.tolist(),
                label=(
                    "Mean proportion within true class"
                    if statistic == "mean" else "Seed SD"
                ),
            )
        destination = (
            path if statistic == "mean"
            else path.with_name(path.stem + "_seed_sd.png")
        )
        _save_figure_formats(fig, destination)
        plt.close(fig)
    pd.DataFrame(source_records).to_csv(
        directory / "matrix_sources.csv", index=False
    )
    _write_figure_sources(path, baseline_runs)
    return True


def _save_hloss_confusion_comparison(
    baseline_runs: pd.DataFrame,
    path: Path,
) -> bool:
    """Compare seed-mean confusion matrices for h=0 and h=0.2."""
    if baseline_runs.empty:
        return False
    tasks = ("genus", "species", "age")
    weights = sorted(
        float(value)
        for value in baseline_runs["hierarchy_loss_weight"]
        .dropna()
        .unique()
    )
    if not weights:
        return False
    aggregated: dict[tuple[float, str], tuple[np.ndarray, list[str]]] = {}
    source_records = []
    directory = _figure_source_dir(path)
    matrix_dir = directory / "matrix_data"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for weight in weights:
        weight_runs = baseline_runs[
            baseline_runs["hierarchy_loss_weight"].eq(weight)
        ]
        for task in tasks:
            frames = []
            for row in weight_runs.itertuples():
                matrix_path = (
                    Path(row.run_dir)
                    / f"confusion_matrix_best_{task}.csv"
                )
                if not matrix_path.is_file():
                    continue
                frame = _read_confusion_frame(matrix_path)
                frames.append(frame)
                source_records.append({
                    "task": task,
                    "model": row.model,
                    "seed": row.seed,
                    "hierarchy_loss_weight": weight,
                    "path": str(matrix_path),
                })
            if not frames:
                continue
            labels = frames[0].index.tolist()
            if any(
                frame.index.tolist() != labels
                or frame.columns.tolist() != labels
                for frame in frames
            ):
                raise ValueError(
                    "Confusion-matrix labels differ across hierarchy-loss "
                    f"runs for {task}"
                )
            normalized = []
            for frame in frames:
                matrix = frame.to_numpy()
                row_sums = matrix.sum(axis=1, keepdims=True)
                normalized.append(np.divide(
                    matrix,
                    row_sums,
                    out=np.zeros_like(matrix, dtype=float),
                    where=row_sums != 0,
                ))
            stack = np.stack(normalized)
            mean = stack.mean(axis=0)
            sd = (
                stack.std(axis=0, ddof=1)
                if len(stack) > 1
                else np.zeros_like(mean)
            )
            aggregated[(weight, task)] = (mean, labels)
            prefix = f"h{weight:g}_{task}".replace(".", "p")
            pd.DataFrame(mean, index=labels, columns=labels).to_csv(
                matrix_dir / f"{prefix}_row_normalised_mean.csv"
            )
            pd.DataFrame(sd, index=labels, columns=labels).to_csv(
                matrix_dir / f"{prefix}_row_normalised_sd.csv"
            )
    if not aggregated:
        return False
    fig, axes = plt.subplots(
        len(weights),
        len(tasks),
        figsize=(6.2 * len(tasks), 5.8 * len(weights)),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row_index, weight in enumerate(weights):
        for column_index, task in enumerate(tasks):
            ax = axes[row_index, column_index]
            values = aggregated.get((weight, task))
            if values is None:
                ax.axis("off")
                continue
            matrix, labels = values
            image = ax.imshow(
                matrix,
                cmap=_style("heatmap_colormap", "viridis"),
                vmin=0,
                vmax=1,
                aspect="equal",
            )
            ax.set_xticks(
                np.arange(len(labels)), labels, rotation=55, ha="right"
            )
            ax.set_yticks(np.arange(len(labels)), labels)
            ax.set_xlabel("Predicted class")
            ax.set_ylabel("True class")
            ax.set_title(f"{task}; hierarchy loss h={weight:g}")
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Mean proportion within true class",
        )
    _save_figure_formats(fig, path)
    plt.close(fig)
    pd.DataFrame(source_records).to_csv(
        directory / "matrix_sources.csv", index=False
    )
    _write_figure_sources(path, baseline_runs)
    return True


def build_report(
    paper_root: Path,
    *,
    split_root: Path | None = None,
    data_root: Path | None = None,
    style_path: Path | None = None,
) -> dict:
    global FIGURE_SOURCE_ROOT
    configured_style = configure_report_style(style_path)
    paper_root.mkdir(parents=True, exist_ok=True)
    if split_root is None:
        split_root = Path(__file__).resolve().parents[1]
    tables = paper_root / "tables"
    figures = paper_root / "figures"
    summary_dir = paper_root / "summary"
    FIGURE_SOURCE_ROOT = paper_root / "figure_sources"
    for directory in (tables, figures, summary_dir, FIGURE_SOURCE_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(paper_root)
    holdouts = collect_holdouts(paper_root)
    holdout_controls = collect_holdout_controls(paper_root)
    baseline_condition_rows = collect_baseline_condition_controls(paper_root)
    cross = collect_original_cross_conditions(paper_root)
    split_frames = load_split_frames(split_root)
    if data_root is None:
        for config_path in sorted((paper_root / "runs").rglob("config.json")):
            candidate = (
                _read_json(config_path).get("data", {}) or {}
            ).get("root_dir")
            if isinstance(candidate, str):
                resolved = Path(candidate).expanduser()
                if resolved.is_dir():
                    data_root = resolved
                    break
    runs.to_csv(tables / "all_completed_runs.csv", index=False)
    holdouts.to_csv(tables / "data_holdout_task_metrics.csv", index=False)
    holdout_controls.to_csv(
        tables / "baseline_data_holdout_control_metrics.csv", index=False
    )
    baseline_condition_rows.to_csv(
        tables / "baseline_visual_condition_control_metrics.csv",
        index=False,
    )
    cross.to_csv(tables / "visual_original_image_metrics.csv", index=False)

    baseline_all = (
        runs[runs["stage"].eq("baseline")] if not runs.empty else runs
    )
    visual_all = (
        runs[runs["stage"].eq("visual_ablation")]
        if not runs.empty else runs
    )
    interactions_all = (
        runs[runs["stage"].eq("visual_interactions")]
        if not runs.empty else runs
    )
    baseline = baseline_all[
        baseline_all["hierarchy_loss_weight"].eq(0.0)
    ]
    visual = visual_all[
        visual_all["hierarchy_loss_weight"].eq(0.0)
    ]
    interactions = interactions_all[
        interactions_all["hierarchy_loss_weight"].eq(0.0)
    ]
    holdouts_all = holdouts
    holdouts = (
        holdouts_all[holdouts_all["hierarchy_loss_weight"].eq(0.0)]
        if not holdouts_all.empty else holdouts_all
    )
    holdout_controls_all = holdout_controls
    holdout_controls = (
        holdout_controls_all[
            holdout_controls_all["hierarchy_loss_weight"].eq(0.0)
        ]
        if not holdout_controls_all.empty else holdout_controls_all
    )
    cross_all = cross
    cross = (
        cross_all[cross_all["hierarchy_loss_weight"].eq(0.0)]
        if not cross_all.empty else cross_all
    )
    baseline_condition_rows_all = baseline_condition_rows
    baseline_condition_rows = (
        baseline_condition_rows_all[
            baseline_condition_rows_all["hierarchy_loss_weight"].eq(0.0)
        ]
        if not baseline_condition_rows_all.empty
        else baseline_condition_rows_all
    )
    dataset_composition = dataset_composition_table(split_frames)
    model_training = model_training_table(runs)
    experimental_ablations = experimental_ablation_table(runs)
    resolution_schedule = resolution_loss_schedule()
    resolution_ticks, resolution_labels = _resolution_tick_labels()
    holdout_definitions = collect_holdout_definitions(paper_root)
    holdout_joined = joined_holdout_metrics(
        runs[runs["hierarchy_loss_weight"].eq(0.0)]
        if not runs.empty else runs,
        holdouts,
    )
    holdout_joined_all = joined_holdout_metrics(runs, holdouts_all)
    baseline_by_task = baseline_task_table(baseline)
    best_baseline, best_baseline_runs = best_baseline_configuration(baseline)
    baseline.to_csv(tables / "baseline_metrics.csv", index=False)
    baseline_all.to_csv(
        tables / "baseline_metrics_hloss_comparison.csv", index=False
    )
    visual.to_csv(tables / "visual_ablation_metrics.csv", index=False)
    visual_all.to_csv(
        tables / "visual_ablation_metrics_hloss_comparison.csv",
        index=False,
    )
    interactions.to_csv(
        tables / "visual_interaction_metrics.csv", index=False
    )
    interactions_all.to_csv(
        tables / "visual_interaction_metrics_hloss_comparison.csv",
        index=False,
    )
    original_test = _original_test_condition_summary(cross)
    original_test_all = _original_test_condition_summary(cross_all)
    original_test.to_csv(
        tables / "visual_trained_ablation_tested_on_original_metrics.csv",
        index=False,
    )
    original_test_all.to_csv(
        tables
        / "visual_trained_ablation_tested_on_original_hloss_comparison.csv",
        index=False,
    )
    matched_original = matched_vs_original_table(visual, original_test)
    matched_original_all = matched_vs_original_table(
        visual_all, original_test_all
    )
    baseline_condition_summary = _baseline_condition_summary(
        baseline_condition_rows
    )
    baseline_condition_summary_all = _baseline_condition_summary(
        baseline_condition_rows_all
    )
    baseline_condition_summary.to_csv(
        tables / "baseline_visual_condition_summary.csv", index=False
    )
    baseline_condition_summary_all.to_csv(
        tables / "baseline_visual_condition_summary_hloss_comparison.csv",
        index=False,
    )
    dataset_composition.to_csv(
        tables / "dataset_composition_by_taxon_stage_and_split.csv",
        index=False,
    )
    model_training.to_csv(
        tables / "model_and_training_configurations.csv", index=False
    )
    experimental_ablations.to_csv(
        tables / "experimental_ablation_definitions.csv", index=False
    )
    resolution_schedule.to_csv(
        tables / "resolution_loss_schedule.csv", index=False
    )
    baseline_by_task.to_csv(
        tables / "baseline_performance_by_task_with_ci.csv", index=False
    )
    holdout_definitions.to_csv(
        tables / "holdout_definitions_and_sample_counts.csv", index=False
    )
    holdout_joined.to_csv(
        tables / "full_test_and_heldout_cohort_metrics.csv", index=False
    )
    matched_original.to_csv(
        tables / "matched_vs_original_test_metrics.csv", index=False
    )
    matched_original_all.to_csv(
        tables / "matched_vs_original_test_metrics_hloss_comparison.csv",
        index=False,
    )
    (summary_dir / "best_baseline_configuration.json").write_text(
        json.dumps(best_baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not baseline.empty:
        _save_notebook_baseline(
            baseline,
            figures / "figure_01_combined_baseline.png",
        )
        _save_grouped_bar(
            baseline,
            x="model",
            series="loss_recipe",
            value="test_mean_macro_f1",
            title="Original-image baseline by backbone and loss recipe",
            ylabel="Test mean macro-F1",
            path=figures / "baseline_mean_macro_f1_by_model_and_loss.png",
        )
        _save_baseline_ci(
            baseline_by_task,
            figures / "figure_4_baseline_comparison_with_95ci.png",
        )
    if not baseline_all.empty:
        baseline_hloss = _decorate_hloss_models(baseline_all)
        _save_notebook_baseline(
            baseline_hloss,
            figures / "figure_01_combined_baseline_hloss_comparison.png",
        )
        _save_grouped_bar(
            baseline_hloss,
            x="model",
            series="loss_recipe",
            value="test_mean_macro_f1",
            title="Original-image baselines: hierarchy loss 0 versus 0.2",
            ylabel="Test mean macro-F1",
            path=(
                figures
                / "baseline_mean_macro_f1_by_model_and_loss_hloss_comparison.png"
            ),
        )
        _save_baseline_ci(
            baseline_task_table(baseline_hloss),
            figures
            / "figure_4_baseline_comparison_with_95ci_hloss_comparison.png",
        )
    if not visual.empty:
        visual_thumbnails = _save_visual_level_images(
            frame=visual,
            split_frames=split_frames,
            data_root=data_root,
            figure_path=(
                figures / "figure_02_combined_visual_ablation.png"
            ),
        )
        zero_colour = visual[visual["transform"].eq("saturation")]
        _save_grouped_bar(
            zero_colour,
            x="model",
            series="condition",
            value="test_mean_macro_f1",
            title="Training and testing with 0% colour",
            ylabel="Matched-test mean macro-F1",
            path=figures / "visual_zero_colour_by_model.png",
        )
        patches = visual[visual["transform"].eq("patch_shuffle")]
        patches = _with_original_anchor(
            patches, baseline, x="grid_size", anchor=1
        )
        _save_lines(
            patches,
            x="grid_size",
            value="test_mean_macro_f1",
            title="Patch-shuffle robustness",
            xlabel="Patch grid per side",
            path=figures / "visual_patch_shuffle_by_model.png",
            thumbnails=visual_thumbnails,
        )
        gaussian = visual[visual["transform"].eq("gaussian_blur_percent")]
        gaussian = _with_original_anchor(
            gaussian, baseline, x="percent", anchor=0
        )
        _save_lines(
            gaussian,
            x="percent",
            value="test_mean_macro_f1",
            title="Gaussian-blur severity",
            xlabel="Gaussian blur severity (%)",
            path=figures / "visual_gaussian_blur_by_model.png",
            thumbnails=visual_thumbnails,
        )
        resolution = visual[visual["transform"].eq("resolution_loss")]
        _save_lines(
            resolution,
            x="percent",
            value="test_mean_macro_f1",
            title="Spatial-resolution information loss",
            xlabel=(
                "Resolution loss (%)\n"
                "tick labels: retained linear dimension / intermediate side"
            ),
            path=figures / "visual_resolution_loss_by_model.png",
            xticks=resolution_ticks,
            xticklabels=resolution_labels,
            thumbnails=visual_thumbnails,
        )
        _save_visual_overview(
            visual,
            float(best_baseline.get("mean_test_macro_f1", np.nan)),
            figures / "figure_02_combined_visual_ablation.png",
            thumbnails=visual_thumbnails,
        )
        visual_hloss = _decorate_hloss_models(visual_all)
        baseline_hloss = _decorate_hloss_models(baseline_all)
        zero_colour_hloss = visual_hloss[
            visual_hloss["transform"].eq("saturation")
        ]
        _save_grouped_bar(
            zero_colour_hloss,
            x="model",
            series="condition",
            value="test_mean_macro_f1",
            title="0% colour: hierarchy loss 0 versus 0.2",
            ylabel="Matched-test mean macro-F1",
            path=figures / "visual_zero_colour_by_model_hloss_comparison.png",
        )
        for transform, x_column, title, output_name in (
            (
                "patch_shuffle",
                "grid_size",
                "Patch shuffle: hierarchy loss 0 versus 0.2",
                "visual_patch_shuffle_by_model_hloss_comparison.png",
            ),
            (
                "gaussian_blur_percent",
                "percent",
                "Gaussian blur: hierarchy loss 0 versus 0.2",
                "visual_gaussian_blur_by_model_hloss_comparison.png",
            ),
            (
                "resolution_loss",
                "percent",
                "Resolution loss: hierarchy loss 0 versus 0.2",
                "visual_resolution_loss_by_model_hloss_comparison.png",
            ),
        ):
            comparison_frame = visual_hloss[
                visual_hloss["transform"].eq(transform)
            ]
            if transform in {"patch_shuffle", "gaussian_blur_percent"}:
                comparison_frame = _with_original_anchor(
                    comparison_frame,
                    baseline_hloss,
                    x=x_column,
                    anchor=1 if transform == "patch_shuffle" else 0,
                )
            _save_lines(
                comparison_frame,
                x=x_column,
                value="test_mean_macro_f1",
                title=title,
                xlabel=x_column.replace("_", " "),
                path=figures / output_name,
                thumbnails=visual_thumbnails,
            )
        _save_visual_overview(
            visual_hloss,
            float(best_baseline.get("mean_test_macro_f1", np.nan)),
            figures
            / "figure_02_combined_visual_ablation_hloss_comparison.png",
            thumbnails=visual_thumbnails,
        )
        _save_visual_overview(
            visual_hloss,
            float(best_baseline.get("mean_test_macro_f1", np.nan)),
            figures
            / "figure_5_visual_ablation_performance_hloss_comparison.png",
            thumbnails=visual_thumbnails,
        )
        _save_visual_overview(
            visual,
            float(best_baseline.get("mean_test_macro_f1", np.nan)),
            figures / "figure_5_visual_ablation_performance.png",
            thumbnails=visual_thumbnails,
        )
    if not interactions.empty:
        interaction_path = (
            figures / "figure_08_visual_interaction_contact_sheet.png"
        )
        interaction_thumbnails = _save_visual_level_images(
            frame=interactions,
            split_frames=split_frames,
            data_root=data_root,
            figure_path=interaction_path,
        )
        _write_figure_sources(
            interaction_path,
            interactions,
            extra={
                "thumbnails": {
                    name: str(image_path)
                    for name, image_path in interaction_thumbnails.items()
                }
            },
        )
        _save_interaction_contact_sheet(
            interaction_thumbnails, interaction_path
        )
        _save_interaction_figures(interactions, figures)
        _save_interaction_figures(
            _decorate_hloss_models(interactions_all),
            figures,
            suffix="_hloss_comparison",
        )
    if not original_test.empty:
        original_zero_colour = original_test[
            original_test["train_transform"].eq("saturation")
        ]
        _save_grouped_bar(
            original_zero_colour,
            x="model",
            series="train_condition",
            value="mean_macro_f1",
            title="0% colour training, tested on original images",
            ylabel="Original-test mean macro-F1",
            path=figures / "visual_zero_colour_original_test_by_model.png",
        )
        _save_lines(
            original_test[
                original_test["train_transform"].eq("patch_shuffle")
            ],
            x="grid_size",
            value="mean_macro_f1",
            title="Patch-shuffle training, tested on original images",
            xlabel="Training patch grid per side",
            ylabel="Original-test mean macro-F1",
            path=figures / "visual_patch_shuffle_original_test_by_model.png",
        )
        _save_lines(
            original_test[
                original_test["train_transform"].eq(
                    "gaussian_blur_percent"
                )
            ],
            x="percent",
            value="mean_macro_f1",
            title="Gaussian-blur training, tested on original images",
            xlabel="Training Gaussian blur severity (%)",
            ylabel="Original-test mean macro-F1",
            path=figures / "visual_gaussian_blur_original_test_by_model.png",
        )
        _save_lines(
            original_test[
                original_test["train_transform"].eq("resolution_loss")
            ],
            x="percent",
            value="mean_macro_f1",
            title="Resolution-loss training, tested on original images",
            xlabel=(
                "Training resolution loss (%)\n"
                "tick labels: retained linear dimension / intermediate side"
            ),
            ylabel="Original-test mean macro-F1",
            path=figures / "visual_resolution_loss_original_test_by_model.png",
            xticks=resolution_ticks,
            xticklabels=resolution_labels,
        )
        _save_matched_vs_original(
            matched_original,
            figures / "figure_6_matched_vs_original_performance.png",
        )
        _save_resolution_three_way(
            visual,
            original_test,
            baseline_condition_summary,
            figures / "figure_09_resolution_three_way_comparison.png",
        )
        original_test_hloss = _decorate_hloss_models(original_test_all)
        for transform, x_column, title, output_name in (
            (
                "patch_shuffle",
                "grid_size",
                "Original-test patch training: hierarchy loss comparison",
                "visual_patch_shuffle_original_test_by_model_hloss_comparison.png",
            ),
            (
                "gaussian_blur_percent",
                "percent",
                "Original-test blur training: hierarchy loss comparison",
                "visual_gaussian_blur_original_test_by_model_hloss_comparison.png",
            ),
            (
                "resolution_loss",
                "percent",
                "Original-test resolution training: hierarchy loss comparison",
                "visual_resolution_loss_original_test_by_model_hloss_comparison.png",
            ),
        ):
            _save_lines(
                original_test_hloss[
                    original_test_hloss["train_transform"].eq(transform)
                ],
                x=x_column,
                value="mean_macro_f1",
                title=title,
                xlabel=x_column.replace("_", " "),
                ylabel="Original-test mean macro-F1",
                path=figures / output_name,
            )
        original_colour_hloss = original_test_hloss[
            original_test_hloss["train_transform"].eq("saturation")
        ]
        _save_grouped_bar(
            original_colour_hloss,
            x="model",
            series="train_condition",
            value="mean_macro_f1",
            title="Original-test colour removal: hierarchy loss comparison",
            ylabel="Original-test mean macro-F1",
            path=(
                figures
                / "visual_zero_colour_original_test_by_model_hloss_comparison.png"
            ),
        )
        _save_matched_vs_original(
            _decorate_hloss_models(matched_original_all),
            figures
            / "figure_6_matched_vs_original_performance_hloss_comparison.png",
        )
        _save_resolution_three_way(
            _decorate_hloss_models(visual_all),
            _decorate_hloss_models(original_test_all),
            _decorate_hloss_models(baseline_condition_summary_all),
            figures
            / "figure_09_resolution_three_way_comparison_hloss_comparison.png",
        )
    if not holdouts.empty:
        supported = holdouts[
            holdouts["class_supported_by_training_head"].astype(bool)
        ]
        supported = supported.assign(
            holdout_task=supported["holdout"] + " / " + supported["task"]
        )
        _save_grouped_bar(
            supported,
            x="holdout_task",
            series="model",
            value="target_recall",
            title="Detection on biologically held-out cohorts",
            ylabel="Target-class recall",
            path=figures / "data_holdout_target_recall_by_model.png",
        )
        _save_holdout_change(
            holdout_joined,
            best_score=float(
                best_baseline.get("mean_test_macro_f1", np.nan)
            ),
            path=figures / "figure_7_structured_holdout_changes.png",
        )
        _save_notebook_holdouts(
            holdouts,
            holdout_controls,
            figures / "figure_03_structured_holdout_subplots.png",
        )
        holdouts_hloss = _decorate_hloss_models(holdouts_all)
        controls_hloss = _decorate_hloss_models(holdout_controls_all)
        supported_hloss = holdouts_hloss[
            holdouts_hloss["class_supported_by_training_head"].astype(bool)
        ].copy()
        supported_hloss["holdout_task"] = (
            supported_hloss["holdout"] + " / " + supported_hloss["task"]
        )
        _save_grouped_bar(
            supported_hloss,
            x="holdout_task",
            series="model",
            value="target_recall",
            title="Biological holdouts: hierarchy loss 0 versus 0.2",
            ylabel="Target-class recall",
            path=(
                figures
                / "data_holdout_target_recall_by_model_hloss_comparison.png"
            ),
        )
        _save_holdout_change(
            _decorate_hloss_models(holdout_joined_all),
            best_score=float(
                best_baseline.get("mean_test_macro_f1", np.nan)
            ),
            path=(
                figures
                / "figure_7_structured_holdout_changes_hloss_comparison.png"
            ),
        )
        _save_notebook_holdouts(
            holdouts_hloss,
            controls_hloss,
            figures
            / "figure_03_structured_holdout_subplots_hloss_comparison.png",
        )

    confusion_created = _save_validation_selected_confusions(
        best_baseline_runs,
        figures / "figure_04_validation_selected_confusion_matrices.png",
    )
    comparison_baseline_runs = baseline_all[
        baseline_all["model"].eq(best_baseline.get("model"))
        & baseline_all["loss_recipe"].eq(
            best_baseline.get("loss_recipe")
        )
    ].copy()
    confusion_hloss_created = _save_hloss_confusion_comparison(
        comparison_baseline_runs,
        figures
        / "figure_04_validation_selected_confusion_matrices_hloss_comparison.png",
    )

    _save_workflow(figures / "figure_1_study_workflow.png")
    representative_created = _save_representative_images(
        split_frames=split_frames,
        data_root=data_root,
        path=figures / "figure_2_representative_images.png",
    )
    transformations_created = _save_transformation_examples(
        split_frames=split_frames,
        data_root=data_root,
        path=figures / "figure_3_transformation_examples.png",
    )
    manuscript_artifacts = {
        "methods_table_1_dataset_composition": not dataset_composition.empty,
        "methods_figure_1_study_overview": (
            figures / "figure_1_study_workflow.png"
        ).is_file(),
        "methods_figure_2_representative_images": representative_created,
        "methods_table_2_model_training_configurations": (
            not model_training.empty
        ),
        "methods_table_3_experimental_ablations": (
            not experimental_ablations.empty
        ),
        "methods_resolution_loss_schedule": (
            len(resolution_schedule) == len(RESOLUTION_LOSS_LEVELS)
        ),
        "methods_figure_3_transformation_examples": transformations_created,
        "results_table_1_dataset_composition": not dataset_composition.empty,
        "results_table_2_baseline_performance": not baseline_by_task.empty,
        "results_figure_1_study_workflow": (
            figures / "figure_1_study_workflow.png"
        ).is_file(),
        "results_figure_3_transformation_examples": transformations_created,
        "results_figure_4_baseline_with_ci": (
            figures / "figure_4_baseline_comparison_with_95ci.png"
        ).is_file(),
        "results_figure_5_visual_performance": (
            figures / "figure_5_visual_ablation_performance.png"
        ).is_file(),
        "results_figure_6_matched_vs_original": (
            figures / "figure_6_matched_vs_original_performance.png"
        ).is_file(),
        "results_table_3_holdout_definitions": (
            not holdout_definitions.empty
        ),
        "results_table_4_full_and_cohort_metrics": (
            not holdout_joined.empty
        ),
        "results_figure_7_holdout_changes": (
            figures / "figure_7_structured_holdout_changes.png"
        ).is_file(),
        "results_figure_8_interaction_contact_sheet": (
            figures / "figure_08_visual_interaction_contact_sheet.png"
        ).is_file() if not interactions.empty else True,
        "results_figure_9_resolution_controls": (
            figures / "figure_09_resolution_three_way_comparison.png"
        ).is_file() if (
            not visual.empty and not baseline_condition_summary.empty
        ) else True,
        "notebook_confusion_matrices": (
            confusion_created
            or not any(
                (Path(path) / "confusion_matrix_best_genus.csv").is_file()
                for path in best_baseline_runs.get("run_dir", [])
            )
        ),
        "notebook_confusion_matrices_hloss_comparison": (
            confusion_hloss_created
            or not any(
                (Path(path) / "confusion_matrix_best_genus.csv").is_file()
                for path in comparison_baseline_runs.get("run_dir", [])
            )
        ),
    }

    summary = {
        "completed_runs": int(len(runs)),
        "completed_by_stage": (
            runs.groupby("stage").size().astype(int).to_dict()
            if not runs.empty
            else {}
        ),
        "models": sorted(runs["model"].dropna().unique().tolist())
        if not runs.empty
        else [],
        "best_baseline": best_baseline,
        "best_baseline_run_count": int(len(best_baseline_runs)),
        "split_root": str(split_root),
        "data_root": str(data_root) if data_root is not None else None,
        "style": configured_style,
        "style_path": str(style_path or DEFAULT_STYLE_PATH),
        "representative_images_created": representative_created,
        "transformation_examples_created": transformations_created,
        "manuscript_artifacts": manuscript_artifacts,
        "all_manuscript_artifacts_ready": all(
            manuscript_artifacts.values()
        ),
        "tables": sorted(path.name for path in tables.glob("*.csv")),
        "figures": sorted(path.name for path in figures.glob("*.png")),
        "figure_source_directories": sorted(
            path.name for path in FIGURE_SOURCE_ROOT.iterdir()
            if path.is_dir()
        ),
    }
    (summary_dir / "paper_results_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-result", default="paper_result")
    parser.add_argument(
        "--split-root",
        default=None,
        help="Project root containing split_csv, or the split_csv directory.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Dataset root used to resolve raw, segmented, and mask paths.",
    )
    parser.add_argument(
        "--style",
        default=str(DEFAULT_STYLE_PATH),
        help="Editable YAML containing figure colours and display settings.",
    )
    args = parser.parse_args()
    summary = build_report(
        Path(args.paper_result).resolve(),
        split_root=(
            Path(args.split_root).resolve() if args.split_root else None
        ),
        data_root=(
            Path(args.data_root).expanduser().resolve()
            if args.data_root
            else None
        ),
        style_path=Path(args.style).resolve() if args.style else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
