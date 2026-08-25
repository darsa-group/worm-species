#!/usr/bin/env python3
"""Build the editable main and supplementary ablation figures.

The figures are deliberately generated outside the notebook as well, so the
same completed-run inputs create the same PNG, PDF, SVG, and source CSV files.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "path"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from PIL import Image

from scripts.build_adult_taxon_ablation_results import collect_adult_taxon_metrics
from scripts.build_paper_results import (
    _loss_name,
    collect_runs,
    dataset_composition_table,
    load_split_frames,
)
from src.worm_species.data.transforms import build_split_transform
from src.worm_species.training.metrics import classification_metric_summary


# Okabe-Ito: colour-blind safe, with marker/line redundancy added per plot.
PALETTE = (
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
    "#D55E00", "#F0E442", "#000000",
)
HEATMAP_CMAP = "cividis"
TASK_ORDER = ("genus", "species", "age")
TASK_LABELS = {
    "genus": "Genus",
    "species": "Species",
    "age": "Developmental stage",
}
BASELINE_RECIPE = _loss_name({"genus": 1.0, "species": 0.5, "age": 2.0})
PAPER_LOSS_WEIGHTS = {"genus": 1.0, "species": 0.5, "age": 2.0}
PAPER_SEEDS = tuple(range(40, 2941, 100))
TEST_COHORT = "independent_test"
TAXON_MODEL = "convnext_base"
T_CRITICAL = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
ABLATION_BOOTSTRAP_RESAMPLES = 10_000
ABLATION_BOOTSTRAP_SEED = 20260804
ABLATION_INTERVAL_LEVEL = 0.95
INPUT_SIZE = 224
FIGURE_DPI = 600

VISUAL_EXAMPLE_CONDITIONS = (
    ("Original", {"transform": "original"}),
    (
        "Gaussian blur 50%",
        {
            "transform": "gaussian_blur_percent",
            "parameters": {"percent": 50, "max_sigma": 64.0},
        },
    ),
    (
        "22 px retained",
        {"transform": "resolution_loss", "parameters": {"percent": 90}},
    ),
    (
        "Greyscale",
        {"transform": "saturation", "parameters": {"retention": 0.0}},
    ),
    (
        "Binary silhouette",
        {"transform": "binary_mask", "parameters": {"threshold": 5.0 / 255.0}},
    ),
    (
        "8x8 patch shuffle",
        {
            "transform": "patch_shuffle",
            "parameters": {"grid_size": 8, "seed": 2026},
        },
    ),
)


def _deterministic_jitter(
    values: Iterable[object], *, width: float = 0.08
) -> np.ndarray:
    """Stable jitter keyed by seed/identity rather than row ordering."""
    offsets = []
    for value in values:
        digest = hashlib.sha256(str(value).encode("utf-8")).digest()
        unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
        offsets.append((unit - 0.5) * 2.0 * width)
    return np.asarray(offsets, dtype=float)


def _interval_text(mean: float, half_width: float) -> str:
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(half_width):
        return f"{mean:.1%}"
    low = max(0.0, mean - half_width)
    high = min(1.0, mean + half_width)
    return f"{mean:.1%}\n[{low:.1%}, {high:.1%}]"


def _slug(value: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return text or "unnamed"


def _display_model(value: object) -> str:
    labels = {
        "resnet50": "ResNet-50",
        "convnext_base": "ConvNeXt-Base",
        "vit_b_16": "ViT-B/16",
        "dinov3_vitb16": "DINOv3 ViT-B/16",
        "dinov3_convnext_base": "DINOv3 ConvNeXt-Base",
    }
    return labels.get(str(value), str(value).replace("_", " "))


def _supported(frame: pd.DataFrame) -> pd.DataFrame:
    if "class_supported_by_training_head" not in frame:
        return frame.copy()
    supported = frame["class_supported_by_training_head"]
    if supported.dtype == bool:
        mask = supported
    else:
        mask = supported.astype(str).str.lower().isin({"true", "1", "yes"})
    return frame[mask].copy()


def _h0(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "hierarchy_loss_weight" not in frame:
        return frame.copy()
    weight = pd.to_numeric(frame["hierarchy_loss_weight"], errors="coerce").fillna(0.0)
    return frame[weight.eq(0.0)].copy()


def _paper_design_only(
    frame: pd.DataFrame,
    *,
    require_loss_recipe: bool = False,
) -> pd.DataFrame:
    """Keep only h=0, the one paper loss recipe, and publication seeds."""
    if frame.empty:
        return frame.copy()
    if require_loss_recipe and "hierarchy_loss_weight" not in frame:
        raise ValueError(
            "Paper figures require hierarchy_loss_weight; refusing to mix "
            "unknown hierarchy-loss configurations."
        )
    if require_loss_recipe:
        hierarchy_values = pd.to_numeric(
            frame["hierarchy_loss_weight"], errors="coerce"
        )
        if hierarchy_values.isna().any():
            raise ValueError(
                "Some runs have no recoverable hierarchy-loss setting; "
                "rerun collection from directories containing config.json."
            )
    selected = _h0(frame)
    if selected.empty:
        return selected
    if "seed" not in selected:
        raise ValueError("Paper figures require an explicit seed column")
    seeds = pd.to_numeric(selected["seed"], errors="coerce")
    if require_loss_recipe and seeds.isna().any():
        raise ValueError(
            "Some runs have no recoverable seed; rerun collection from "
            "directories containing config.json."
        )
    selected = selected[seeds.isin(PAPER_SEEDS)].copy()
    if "loss_recipe" in selected:
        recipes = selected["loss_recipe"].replace("", np.nan)
        if require_loss_recipe and recipes.isna().any():
            raise ValueError(
                "Some runs have no recoverable loss recipe; rerun collection "
                "from directories containing config.json."
            )
        selected = selected[recipes.eq(BASELINE_RECIPE)].copy()
    else:
        weight_columns = {
            task: f"{task}_weight" for task in PAPER_LOSS_WEIGHTS
        }
        if all(column in selected for column in weight_columns.values()):
            mask = pd.Series(True, index=selected.index)
            for task, expected in PAPER_LOSS_WEIGHTS.items():
                values = pd.to_numeric(
                    selected[weight_columns[task]], errors="coerce"
                )
                mask &= np.isclose(values, expected)
            selected = selected[mask].copy()
        elif require_loss_recipe:
            raise ValueError(
                "Paper figures require loss_recipe or explicit genus/species/age "
                "weight columns; refusing to mix loss-weight configurations."
            )
    return selected


def _paper_design_manifest() -> dict:
    return {
        "seeds": list(PAPER_SEEDS),
        "hierarchy_loss_weight": 0.0,
        "loss_recipe": BASELINE_RECIPE,
        "loss_weights": PAPER_LOSS_WEIGHTS,
    }


def _model_only(frame: pd.DataFrame, model: str, *, context: str) -> pd.DataFrame:
    """Fail closed when a figure's required model identity is unavailable."""
    if frame.empty:
        return frame.copy()
    if "model" not in frame:
        raise ValueError(f"{context} require an explicit model column")
    return frame[frame["model"].astype(str).eq(model)].copy()


def _attach_run_design_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover design columns from run configs for older collected metrics."""
    result = frame.copy()
    if result.empty or "run_dir" not in result:
        return result
    metadata: dict[str, dict[str, object]] = {}
    for run_dir_value in result["run_dir"].dropna().astype(str).unique():
        config_path = Path(run_dir_value) / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        multi_task = dict(config.get("multi_task", {}) or {})
        weights = dict(multi_task.get("loss_weights", {}) or {})
        numeric_weights = {
            task: float(weights.get(task, 0.0))
            for task in ("genus", "species", "age")
        }
        hierarchy = dict(multi_task.get("hierarchy_loss", {}) or {})
        hierarchy_weight = (
            float(hierarchy.get("weight", 0.0))
            if bool(hierarchy.get("enabled", False)) else 0.0
        )
        metadata[run_dir_value] = {
            "seed": config.get("seed"),
            "hierarchy_loss_weight": hierarchy_weight,
            "loss_recipe": _loss_name(numeric_weights),
            **{
                f"{task}_weight": value
                for task, value in numeric_weights.items()
            },
        }
    if not metadata:
        return result
    run_dirs = result["run_dir"].astype(str)
    for column in (
        "seed", "hierarchy_loss_weight", "loss_recipe",
        "genus_weight", "species_weight", "age_weight",
    ):
        recovered = run_dirs.map(
            lambda run_dir: metadata.get(run_dir, {}).get(column, np.nan)
        )
        if column in result:
            existing = result[column].replace("", np.nan)
            result[column] = existing.where(existing.notna(), recovered)
        else:
            result[column] = recovered
    return result


def seed_summary(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    """Calculate uncertainty from independent seed-level observations."""
    columns = [*groups, "mean", "standard_deviation", "ci95", "number_of_seeds"]
    if frame.empty or value not in frame:
        return pd.DataFrame(columns=columns)
    clean = frame.dropna(subset=[value]).copy()
    if clean.empty:
        return pd.DataFrame(columns=columns)
    seed_groups = [*groups, "seed"] if "seed" in clean and "seed" not in groups else groups
    per_seed = clean.groupby(seed_groups, dropna=False, as_index=False)[value].mean()
    result = per_seed.groupby(groups, dropna=False)[value].agg(
        mean="mean", standard_deviation="std", number_of_seeds="count"
    ).reset_index()
    result["ci95"] = result.apply(
        lambda row: (
            T_CRITICAL.get(int(row["number_of_seeds"]) - 1, 1.96)
            * float(row["standard_deviation"])
            / math.sqrt(int(row["number_of_seeds"]))
            if int(row["number_of_seeds"]) >= 2
            and pd.notna(row["standard_deviation"])
            else np.nan
        ),
        axis=1,
    )
    return result[columns]


def _save_bundle(
    fig: plt.Figure,
    *,
    output_dir: Path,
    source_root: Path,
    name: str,
    sources: dict[str, pd.DataFrame],
    details: dict | None = None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = source_root / name
    source_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("svg", "pdf", "png"):
        path = output_dir / f"{name}.{extension}"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        paths.append(str(path))
    for stem, frame in sources.items():
        frame.to_csv(source_dir / f"{stem}.csv", index=False)
    manifest = {
        "figure": name,
        "formats": paths,
        "raster_dpi": FIGURE_DPI,
        "source_files": sorted(f"{stem}.csv" for stem in sources),
        **(details or {}),
    }
    (source_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plt.close(fig)
    return paths


def prepare_baseline_frame(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or "stage" not in runs:
        return pd.DataFrame()
    frame = _paper_design_only(
        runs[runs["stage"].eq("baseline")], require_loss_recipe=True
    )
    metric_columns = {
        "test_mean_macro_f1": "All tasks",
        "test_genus_macro_f1": "Genus",
        "test_species_macro_f1": "Species",
        "test_age_macro_f1": "Developmental stage",
    }
    available = [column for column in metric_columns if column in frame]
    if frame.empty or not available:
        return pd.DataFrame()
    long = frame.melt(
        id_vars=[column for column in ("model", "seed", "loss_recipe", "run_dir") if column in frame],
        value_vars=available,
        var_name="metric_column",
        value_name="test_macro_f1",
    )
    long["task"] = long["metric_column"].map(metric_columns)
    long["model_label"] = long["model"].map(_display_model)
    return long.dropna(subset=["test_macro_f1"])


def save_baseline_overview(frame: pd.DataFrame, output_dir: Path, source_root: Path) -> list[str]:
    if frame.empty:
        return []
    task_order = [item for item in ("All tasks", "Genus", "Species", "Developmental stage") if item in set(frame["task"])]
    model_order = frame["model"].drop_duplicates().tolist()
    summary = seed_summary(frame, ["model", "model_label", "task"], "test_macro_f1")
    fig = plt.figure(figsize=(max(13, 1.8 * len(model_order)), 13.0))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.15), hspace=0.36, wspace=0.33)
    ax = fig.add_subplot(grid[0, :])
    x = np.arange(len(model_order), dtype=float)
    width = 0.78 / max(1, len(task_order))
    offsets = (np.arange(len(task_order)) - (len(task_order) - 1) / 2) * width
    for index, task in enumerate(task_order):
        current = summary[summary["task"].eq(task)].set_index("model")
        means = np.asarray([current["mean"].get(model, np.nan) for model in model_order])
        errors = np.asarray([current["ci95"].get(model, np.nan) for model in model_order])
        errors = np.nan_to_num(errors, nan=0.0)
        bars = ax.bar(
            x + offsets[index], means, width=width * 0.92, yerr=errors,
            capsize=3, color=PALETTE[index], label=task, alpha=0.9,
        )
        raw = frame[frame["task"].eq(task)]
        for model_index, model in enumerate(model_order):
            selected = raw[raw["model"].eq(model)]
            values = selected["test_macro_f1"].to_numpy()
            jitter = _deterministic_jitter(
                selected.get("seed", selected.index), width=width * 0.18
            )
            ax.scatter(
                np.full(len(values), x[model_index] + offsets[index]) + jitter,
                values, s=14, color="#222222", alpha=0.5, zorder=4,
            )
        for model_index, bar in enumerate(bars):
            if np.isfinite(bar.get_height()):
                half_width = errors[model_index]
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        _interval_text(bar.get_height(), half_width),
                        ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(x, [_display_model(model) for model in model_order])
    ax.tick_params(axis="x", rotation=18)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Baseline test performance across all tasks and models")
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    confusion_sources = []
    convnext_runs = frame[frame["model"].eq("convnext_base")]["run_dir"].dropna().drop_duplicates()
    image = None
    for task_index, task in enumerate(TASK_ORDER):
        matrix_ax = fig.add_subplot(grid[1, task_index])
        matrices = []
        expected_labels = None
        for run_dir in convnext_runs:
            path = Path(run_dir) / f"confusion_matrix_best_{task}.csv"
            if not path.is_file():
                continue
            current = pd.read_csv(path, index_col=0)
            labels = [str(value) for value in current.index]
            if expected_labels is None:
                expected_labels = labels
            if labels != expected_labels or [str(value) for value in current.columns] != expected_labels:
                raise ValueError(f"Inconsistent {task} confusion-matrix labels across seeds")
            values = current.to_numpy(dtype=float)
            totals = values.sum(axis=1, keepdims=True)
            normalised = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
            matrices.append(normalised)
            seed_value = frame.loc[frame["run_dir"].eq(str(run_dir)), "seed"].iloc[0]
            for row_index, true_label in enumerate(labels):
                for column_index, predicted_label in enumerate(labels):
                    confusion_sources.append({
                        "task": task,
                        "seed": seed_value,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "row_normalized_fraction": normalised[row_index, column_index],
                    })
        if not matrices:
            matrix_ax.text(0.5, 0.5, "Awaiting completed\nConvNeXt-Base runs", ha="center", va="center")
            matrix_ax.set_axis_off()
            continue
        stacked = np.stack(matrices)
        mean_matrix = np.mean(stacked, axis=0)
        if len(matrices) >= 2:
            half_width = (
                T_CRITICAL.get(len(matrices) - 1, 1.96)
                * np.std(stacked, axis=0, ddof=1)
                / math.sqrt(len(matrices))
            )
        else:
            half_width = np.full_like(mean_matrix, np.nan)
        image = matrix_ax.imshow(
            mean_matrix, vmin=0, vmax=1, cmap=HEATMAP_CMAP, aspect="auto"
        )
        for row_index in range(mean_matrix.shape[0]):
            for column_index in range(mean_matrix.shape[1]):
                value = mean_matrix[row_index, column_index]
                interval = half_width[row_index, column_index]
                contrast = "black" if value > 0.58 else "white"
                matrix_ax.text(
                    column_index,
                    row_index,
                    _interval_text(value, interval),
                    ha="center",
                    va="center",
                    fontsize=4.6,
                    color=contrast,
                )
                confusion_sources.append({
                    "task": task,
                    "seed": "summary",
                    "true_label": expected_labels[row_index],
                    "predicted_label": expected_labels[column_index],
                    "row_normalized_fraction": value,
                    "ci95_low": max(0.0, value - interval) if np.isfinite(interval) else np.nan,
                    "ci95_high": min(1.0, value + interval) if np.isfinite(interval) else np.nan,
                    "number_of_seeds": len(matrices),
                })
        matrix_ax.set_title(f"{TASK_LABELS[task]} confusion", fontweight="bold")
        matrix_ax.set_xlabel("Predicted")
        matrix_ax.set_ylabel("True")
        matrix_ax.set_xticks(range(len(expected_labels)), expected_labels, rotation=90, fontsize=6)
        matrix_ax.set_yticks(range(len(expected_labels)), expected_labels, fontsize=6)
    if image is not None:
        fig.colorbar(
            image,
            ax=fig.axes[1:],
            fraction=0.018,
            pad=0.015,
            label="Mean row-normalized percentage",
            format=PercentFormatter(1.0),
        )
    fig.suptitle("Test performance and ConvNeXt-Base confusion matrices", fontsize=16, fontweight="bold")
    fig.subplots_adjust(top=0.94, bottom=0.13, left=0.08, right=0.96)
    return _save_bundle(
        fig, output_dir=output_dir, source_root=source_root,
        name="figure_01_all_models_all_tasks",
        sources={
            "plot_data": frame,
            "seed_summary": summary,
            "convnext_confusion_matrices_by_seed": pd.DataFrame(confusion_sources),
        },
        details={
            "metric": "test macro-F1",
            "split": "test",
            "uncertainty": "95% t interval across seeds",
            "confusion_matrices": "mean of per-seed row-normalized test confusion matrices from best checkpoints",
            "baseline_recipe": BASELINE_RECIPE,
            **_paper_design_manifest(),
        },
    )


def prepare_developmental_stage_diagnostics(
    baseline_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read seed-level age reports and row-normalized confusion matrices."""
    metric_rows: list[dict] = []
    confusion_rows: list[dict] = []
    if baseline_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    runs = baseline_frame[
        [column for column in ("model", "model_label", "seed", "run_dir") if column in baseline_frame]
    ].drop_duplicates()
    for row in runs.itertuples(index=False):
        run_dir = Path(row.run_dir)
        report_path = run_dir / "classification_report_best_age.csv"
        matrix_path = run_dir / "confusion_matrix_best_age.csv"
        if report_path.is_file():
            report = pd.read_csv(report_path, index_col=0)
            values = {}
            for label in ("Adult", "Juvenile"):
                if label in report.index:
                    values.update({
                        f"{label} precision": report.loc[label, "precision"],
                        f"{label} recall": report.loc[label, "recall"],
                        f"{label} F1": report.loc[label, "f1-score"],
                    })
            if "accuracy" in report.index:
                values["Accuracy"] = report.loc["accuracy", "f1-score"]
            if "macro avg" in report.index:
                values.update({
                    "Balanced accuracy": report.loc["macro avg", "recall"],
                    "Macro precision": report.loc["macro avg", "precision"],
                    "Macro recall": report.loc["macro avg", "recall"],
                    "Macro F1": report.loc["macro avg", "f1-score"],
                })
            if "weighted avg" in report.index:
                values["Weighted F1"] = report.loc["weighted avg", "f1-score"]
            for metric, score in values.items():
                metric_rows.append({
                    "model": row.model,
                    "model_label": row.model_label,
                    "seed": row.seed,
                    "metric": metric,
                    "score": float(score),
                    "run_dir": str(run_dir),
                })
        if matrix_path.is_file():
            matrix = pd.read_csv(matrix_path, index_col=0)
            labels = [str(value) for value in matrix.index]
            values = matrix.to_numpy(dtype=float)
            totals = values.sum(axis=1, keepdims=True)
            normalised = np.divide(
                values, totals, out=np.zeros_like(values), where=totals > 0
            )
            for true_index, true_label in enumerate(labels):
                for predicted_index, predicted_label in enumerate(labels):
                    confusion_rows.append({
                        "model": row.model,
                        "model_label": row.model_label,
                        "seed": row.seed,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "row_normalized_fraction": normalised[
                            true_index, predicted_index
                        ],
                    })
    return pd.DataFrame(metric_rows), pd.DataFrame(confusion_rows)


def save_developmental_stage_diagnostics(
    metrics: pd.DataFrame,
    confusions: pd.DataFrame,
    output_dir: Path,
    source_root: Path,
) -> list[str]:
    if metrics.empty:
        return []
    metric_order = [
        "Accuracy", "Balanced accuracy", "Macro precision", "Macro recall",
        "Macro F1", "Weighted F1", "Adult precision", "Adult recall",
        "Adult F1", "Juvenile precision", "Juvenile recall", "Juvenile F1",
    ]
    metric_order = [value for value in metric_order if value in set(metrics["metric"])]
    model_order = metrics["model"].drop_duplicates().tolist()
    summary = seed_summary(metrics, ["model", "model_label", "metric"], "score")
    figure_height = max(10.0, 0.48 * len(metric_order) + 6.5)
    fig = plt.figure(figsize=(16.5, figure_height))
    grid = fig.add_gridspec(2, max(1, len(model_order)), height_ratios=(1.45, 1.0), hspace=0.34, wspace=0.34)
    ax = fig.add_subplot(grid[0, :])
    y = np.arange(len(metric_order), dtype=float)
    offsets = np.linspace(-0.22, 0.22, max(1, len(model_order)))
    for model_index, model in enumerate(model_order):
        current_summary = summary[summary["model"].eq(model)].set_index("metric")
        current_raw = metrics[metrics["model"].eq(model)]
        for metric_index, metric in enumerate(metric_order):
            raw = current_raw[current_raw["metric"].eq(metric)]
            if metric not in current_summary.index:
                continue
            row = current_summary.loc[metric]
            ax.scatter(
                raw["score"],
                np.full(len(raw), y[metric_index] + offsets[model_index])
                + _deterministic_jitter(raw.get("seed", raw.index), width=0.045),
                s=14,
                alpha=0.28,
                color=PALETTE[model_index % len(PALETTE)],
                marker=("o", "s", "^", "D", "P")[model_index % 5],
            )
            ax.errorbar(
                row["mean"],
                y[metric_index] + offsets[model_index],
                xerr=0.0 if pd.isna(row["ci95"]) else row["ci95"],
                fmt=("o", "s", "^", "D", "P")[model_index % 5],
                markersize=6,
                capsize=3,
                color=PALETTE[model_index % len(PALETTE)],
                label=_display_model(model) if metric_index == 0 else None,
                zorder=4,
            )
    ax.set_yticks(y, metric_order)
    ax.set_ylim(len(metric_order) - 0.4, -0.6)
    ax.set_xlim(0, 1.02)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Independent-test score")
    ax.set_title("A. Developmental-stage metrics across baseline seeds", loc="left", fontweight="bold")
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, ncol=max(1, len(model_order)))

    confusion_summary_rows = []
    for model_index, model in enumerate(model_order):
        matrix_ax = fig.add_subplot(grid[1, model_index])
        current = confusions[confusions["model"].eq(model)]
        labels = list(dict.fromkeys(current.get("true_label", pd.Series(dtype=str)).astype(str)))
        if current.empty or not labels:
            matrix_ax.text(0.5, 0.5, "No saved age confusion matrices", ha="center", va="center")
            matrix_ax.set_axis_off()
            continue
        matrices = []
        for _, seed_frame in current.groupby("seed"):
            pivot = seed_frame.pivot(index="true_label", columns="predicted_label", values="row_normalized_fraction").reindex(index=labels, columns=labels)
            matrices.append(pivot.to_numpy(dtype=float))
        stacked = np.stack(matrices)
        mean = stacked.mean(axis=0)
        half = (
            T_CRITICAL.get(len(matrices) - 1, 1.96)
            * stacked.std(axis=0, ddof=1)
            / math.sqrt(len(matrices))
            if len(matrices) >= 2
            else np.full_like(mean, np.nan)
        )
        matrix_ax.imshow(mean, vmin=0, vmax=1, cmap=HEATMAP_CMAP)
        for true_index, true_label in enumerate(labels):
            for predicted_index, predicted_label in enumerate(labels):
                value = mean[true_index, predicted_index]
                interval = half[true_index, predicted_index]
                matrix_ax.text(
                    predicted_index,
                    true_index,
                    _interval_text(value, interval),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if value > 0.58 else "white",
                )
                confusion_summary_rows.append({
                    "model": model,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "mean": value,
                    "ci95_low": max(0.0, value - interval) if np.isfinite(interval) else np.nan,
                    "ci95_high": min(1.0, value + interval) if np.isfinite(interval) else np.nan,
                    "number_of_seeds": len(matrices),
                })
        matrix_ax.set_xticks(range(len(labels)), labels)
        matrix_ax.set_yticks(range(len(labels)), labels)
        matrix_ax.set_xlabel("Predicted stage")
        if model_index == 0:
            matrix_ax.set_ylabel("True stage")
        matrix_ax.set_title(f"{chr(66 + model_index)}. {_display_model(model)}", fontweight="bold")
    fig.suptitle("Why is developmental-stage performance low?", fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.012, "Points are training seeds; whiskers and confusion-cell intervals are 95% t intervals across seeds. All metrics use the independent test split.", ha="center", fontsize=8.5)
    fig.subplots_adjust(top=0.94, bottom=0.07, left=0.16, right=0.98)
    return _save_bundle(
        fig,
        output_dir=output_dir,
        source_root=source_root,
        name="figure_01b_developmental_stage_diagnostics",
        sources={
            "seed_metrics": metrics,
            "seed_summary": summary,
            "confusion_matrices_by_seed": confusions,
            "confusion_matrix_summary": pd.DataFrame(confusion_summary_rows),
        },
        details={
            "split": "independent test",
            "uncertainty": "95% t interval across training seeds",
            "metric_unit": "image",
            "purpose": "diagnose Adult/Juvenile precision-recall asymmetry and directional confusions",
            **_paper_design_manifest(),
        },
    )


def _summarise_line(frame: pd.DataFrame, x: str, series: str | None = None) -> pd.DataFrame:
    groups = [x] if series is None else [series, x]
    return seed_summary(frame, groups, "test_mean_macro_f1")


def visual_uniform_chance_reference(
    runs: pd.DataFrame,
    split_root: Path,
) -> pd.DataFrame:
    """Expected macro-F1 for uniform predictions on the fixed test labels."""
    columns = [
        "task", "class_count", "test_n", "expected_uniform_macro_f1",
        "method",
    ]
    if runs.empty or "run_dir" not in runs:
        return pd.DataFrame(columns=columns)
    label_maps = {}
    for run_dir in runs["run_dir"].dropna().drop_duplicates():
        path = Path(run_dir) / "label_to_index_by_task.json"
        if path.is_file():
            label_maps = json.loads(path.read_text(encoding="utf-8"))
            if label_maps:
                break
    if not label_maps:
        return pd.DataFrame(columns=columns)
    test = load_split_frames(split_root).get("test", pd.DataFrame())
    target_columns = {
        "genus": "genus",
        "species": "species_label",
        "age": "life_stage",
    }
    rows = []
    for task in TASK_ORDER:
        labels = label_maps.get(task, {})
        if not isinstance(labels, dict) or not labels:
            continue
        class_count = len(labels)
        column = target_columns[task]
        observed = (
            test[column].astype(str)
            if not test.empty and column in test
            else pd.Series(dtype=str)
        )
        observed = observed[observed.isin(set(map(str, labels)))]
        counts = observed.value_counts()
        if counts.sum():
            prevalence = np.asarray(
                [counts.get(str(label), 0) / counts.sum() for label in labels],
                dtype=float,
            )
            class_f1 = np.divide(
                2.0 * prevalence,
                class_count * prevalence + 1.0,
                out=np.zeros_like(prevalence),
                where=(class_count * prevalence + 1.0) > 0,
            )
            chance = float(class_f1.mean())
        else:
            # Fail-soft for old/synthetic bundles without aligned split labels.
            chance = 1.0 / class_count
        rows.append({
            "task": task,
            "class_count": class_count,
            "test_n": int(counts.sum()),
            "expected_uniform_macro_f1": chance,
            "method": (
                "expected per-class F1 under uniform random prediction using "
                "the fixed independent-test class prevalence"
            ),
        })
    if rows:
        rows.append({
            "task": "mean",
            "class_count": np.nan,
            "test_n": int(sum(row["test_n"] for row in rows)),
            "expected_uniform_macro_f1": float(np.mean([
                row["expected_uniform_macro_f1"] for row in rows
            ])),
            "method": "unweighted mean of the three task-specific chance macro-F1 values",
        })
    return pd.DataFrame(rows, columns=columns)


def _draw_line_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    x: str,
    title: str,
    xlabel: str,
    series: str | None = None,
    order: Iterable[str] | None = None,
) -> pd.DataFrame:
    summary = _summarise_line(frame, x, series)
    series_values = [None] if series is None else list(order or frame[series].drop_duplicates())
    for index, value in enumerate(series_values):
        current = summary if value is None else summary[summary[series].eq(value)]
        current = current.sort_values(x)
        if current.empty:
            continue
        label = None if value is None else str(value)
        ax.errorbar(
            current[x], current["mean"], yerr=current["ci95"].fillna(0.0),
            marker="o", markersize=4.5, linewidth=2, capsize=2.5,
            color=PALETTE[index % len(PALETTE)], label=label,
        )
        raw = frame if value is None else frame[frame[series].eq(value)]
        unique_x = np.sort(
            pd.to_numeric(frame[x], errors="coerce").dropna().unique()
        )
        positive_steps = np.diff(unique_x)
        positive_steps = positive_steps[positive_steps > 0]
        jitter_width = (
            float(positive_steps.min()) * 0.08
            if positive_steps.size
            else 0.08
        )
        identities = raw.get("seed", raw.index)
        ax.scatter(
            pd.to_numeric(raw[x], errors="coerce")
            + _deterministic_jitter(identities, width=jitter_width),
            raw["test_mean_macro_f1"],
            s=13,
            color=PALETTE[index % len(PALETTE)],
            alpha=0.32,
            zorder=3,
        )
        for _, row in current.iterrows():
            if np.isfinite(row["mean"]):
                ax.annotate(
                    _interval_text(float(row["mean"]), float(row["ci95"])),
                    (row[x], row["mean"]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=5.7,
                    color=PALETTE[index % len(PALETTE)],
                )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test mean macro-F1")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    chance_values = pd.to_numeric(frame.get("chance", np.nan), errors="coerce")
    if np.ndim(chance_values) and pd.Series(chance_values).notna().any():
        chance = float(pd.Series(chance_values).dropna().mean())
        ax.axhline(
            chance,
            color="#000000",
            linestyle=(0, (4, 2)),
            linewidth=1.1,
            alpha=0.8,
            zorder=0,
        )
        ax.text(
            0.99,
            chance,
            f"uniform chance {chance:.1%}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
        )
    if series is not None:
        ax.legend(frameon=False, fontsize=8, ncol=2)
    return summary


def prepare_convnext_visual_frames(
    runs: pd.DataFrame, model: str = "convnext_base",
) -> dict[str, pd.DataFrame]:
    if runs.empty or "stage" not in runs:
        return {}
    visual = _paper_design_only(
        runs[runs["stage"].eq("visual_ablation")],
        require_loss_recipe=True,
    )
    interactions = _paper_design_only(
        runs[runs["stage"].eq("visual_interactions")],
        require_loss_recipe=True,
    )
    visual = visual[visual["model"].eq(model)].copy()
    interactions = interactions[interactions["model"].eq(model)].copy()
    if visual.empty:
        return {}

    original = visual[
        visual["transform"].eq("resolution_loss")
        & pd.to_numeric(visual["percent"], errors="coerce").eq(0.0)
    ].copy()
    gaussian = visual[visual["transform"].eq("gaussian_blur_percent")].copy()
    gaussian["level"] = pd.to_numeric(gaussian["percent"], errors="coerce")
    original_gaussian = original.copy()
    original_gaussian["level"] = 0.0
    gaussian = pd.concat([original_gaussian, gaussian], ignore_index=True)

    resolution = visual[visual["transform"].eq("resolution_loss")].copy()
    resolution["loss_percent"] = pd.to_numeric(
        resolution["percent"], errors="coerce"
    )
    resolution["level"] = resolution["loss_percent"].map(
        lambda percent: max(
            1, int(round(INPUT_SIZE * (1.0 - float(percent) / 100.0)))
        )
        if pd.notna(percent)
        else np.nan
    )

    colour = visual[visual["transform"].eq("saturation")].copy()
    colour["level"] = 1.0
    colour["category"] = "Colour removed"
    mask = visual[visual["transform"].eq("binary_mask")].copy()
    mask["level"] = 0.0
    mask["category"] = "Binary mask only"
    original_colour = original.copy()
    original_colour["level"] = 2.0
    original_colour["category"] = "Original RGB"
    colour = pd.concat([colour, mask, original_colour], ignore_index=True)

    patch = visual[visual["transform"].eq("patch_shuffle")].copy()
    patch["level"] = pd.to_numeric(patch["grid_size"], errors="coerce")
    original_patch = original.copy()
    original_patch["level"] = 1.0
    patch = pd.concat([original_patch, patch], ignore_index=True)

    interaction_parts = []
    gaussian_only = gaussian[
        pd.to_numeric(gaussian["level"], errors="coerce").isin(
            (0.0, 25.0, 50.0, 75.0, 100.0)
        )
    ].copy()
    gaussian_only["series"] = "100% colour (Gaussian only)"
    interaction_parts.append(gaussian_only)
    anchors = {
        "0% colour": colour[colour["category"].eq("Colour removed")],
        "2×2 patches": patch[patch["level"].eq(2.0)],
        "4×4 patches": patch[patch["level"].eq(4.0)],
        "8×8 patches": patch[patch["level"].eq(8.0)],
        "16×16 patches": patch[patch["level"].eq(16.0)],
    }
    conditions = (
        ("0% colour", "saturation", 0.0),
        ("2×2 patches", "patch_shuffle", 2.0),
        ("4×4 patches", "patch_shuffle", 4.0),
        ("8×8 patches", "patch_shuffle", 8.0),
        ("16×16 patches", "patch_shuffle", 16.0),
    )
    for label, transform, level in conditions:
        current = interactions[
            interactions["paired_transform"].eq(transform)
            & pd.to_numeric(interactions["paired_level"], errors="coerce").eq(level)
        ].copy()
        current["level"] = pd.to_numeric(current["gaussian_percent"], errors="coerce")
        anchor = anchors[label].copy()
        anchor["level"] = 0.0
        current = pd.concat([anchor, current], ignore_index=True)
        current["series"] = label
        interaction_parts.append(current)
    interaction = pd.concat(interaction_parts, ignore_index=True)
    return {
        "gaussian": gaussian.dropna(subset=["level", "test_mean_macro_f1"]),
        "resolution": resolution.dropna(subset=["level", "test_mean_macro_f1"]),
        "colour": colour.dropna(subset=["level", "test_mean_macro_f1"]),
        "patch": patch.dropna(subset=["level", "test_mean_macro_f1"]),
        "interaction": interaction.dropna(subset=["level", "test_mean_macro_f1"]),
    }


def prepare_visual_ablation_example(
    split_root: Path,
    data_root: Path,
    *,
    sample_seed: int = 2026,
    fallback_gallery: Path | None = None,
) -> tuple[list[tuple[str, np.ndarray]], pd.DataFrame]:
    """Create a reproducible example strip using the exact paper transforms."""
    split_path = Path(split_root) / "split_csv" / "test_split.csv"
    if not split_path.is_file():
        split_path = Path(split_root) / "test_split.csv"
    image_source = None
    provenance = {}
    if split_path.is_file() and Path(data_root).is_dir():
        frame = pd.read_csv(split_path)
        required = {"barcode", "rel_path_seg"}
        if required.issubset(frame):
            available = frame[
                frame["rel_path_seg"].map(
                    lambda value: (Path(data_root) / str(value)).is_file()
                )
            ].copy()
            barcodes = available["barcode"].dropna().astype(str).unique()
            if len(barcodes) >= 5:
                # Match the first individual in Figure 7's deterministic sample.
                generator = np.random.default_rng(sample_seed)
                selected_barcodes = generator.choice(
                    barcodes, size=5, replace=False
                )
                barcode = str(selected_barcodes[0])
                candidates = available[
                    available["barcode"].astype(str).eq(barcode)
                ]
                row = candidates.iloc[int(generator.integers(0, len(candidates)))]
                image_path = Path(data_root) / str(row["rel_path_seg"])
                image_source = Image.open(image_path).convert("RGB")
                provenance = {
                    "barcode": row["barcode"],
                    "relative_image_path": row["rel_path_seg"],
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    "example_source": "original segmented image",
                    "example_source_file": str(image_path),
                }
    if image_source is None and fallback_gallery is not None:
        gallery_path = Path(fallback_gallery)
        if gallery_path.is_file():
            gallery_text = gallery_path.read_text(encoding="utf-8")
            matches = re.findall(
                r"data:image/png;base64,\s*([^\"]+)", gallery_text, flags=re.DOTALL
            )
            if matches:
                encoded = re.sub(r"\s+", "", matches[0])
                image_source = Image.open(
                    io.BytesIO(base64.b64decode(encoded))
                ).convert("RGB")
                provenance = {
                    "barcode": "Aporrectodea_longa_Juvenile_29",
                    "relative_image_path": (
                        "01_Segmented/Aporrectodea_longa_Juvenile_29/"
                        "Aporrectodea_longa_Juvenile_29_5/"
                        "Aporrectodea_longa_Juvenile_29_5_seg.jpg"
                    ),
                    "sha256": (
                        "723f294024b035f86b70f4483594fd96d87cc777ec2180a97adf8aa207e1c75e"
                    ),
                    "example_source": "lossless original panel embedded in prior Figure 7 SVG",
                    "example_source_file": str(gallery_path),
                    "example_source_file_sha256": hashlib.sha256(
                        gallery_path.read_bytes()
                    ).hexdigest(),
                }
    if image_source is None:
        return [], pd.DataFrame()

    images = []
    source_rows = []
    for label, condition in VISUAL_EXAMPLE_CONDITIONS:
        transform = build_split_transform(
            split="test",
            preprocessing={
                "image_size": INPUT_SIZE,
                "normalisation": {"enabled": False},
            },
            condition=condition,
        )
        transformed = transform(image_source).permute(1, 2, 0).numpy()
        images.append((label, np.clip(transformed, 0.0, 1.0)))
        source_rows.append({
            "condition": label,
            "condition_config": json.dumps(condition, sort_keys=True),
            **provenance,
            "sample_seed": sample_seed,
            "split": "independent test",
            "model_input_side_pixels": INPUT_SIZE,
        })
    return images, pd.DataFrame(source_rows)


def _draw_visual_example_strip(
    fig: plt.Figure,
    subplot_spec,
    images: list[tuple[str, np.ndarray]],
) -> None:
    if not images:
        ax = fig.add_subplot(subplot_spec)
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "Representative image unavailable under the configured data root",
            ha="center",
            va="center",
            fontsize=9,
            color="#666666",
        )
        return
    strip = subplot_spec.subgridspec(1, len(images), wspace=0.06)
    axes = []
    for index, (label, transformed) in enumerate(images):
        ax = fig.add_subplot(strip[0, index])
        ax.imshow(transformed)
        ax.set_title(label, fontsize=8.5, fontweight="bold", pad=4)
        ax.set_axis_off()
        axes.append(ax)
    axes[0].text(
        -0.08,
        1.28,
        "E. What each ablation does to the same independent-test image",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def save_convnext_visual_figure(
    frames: dict[str, pd.DataFrame],
    output_dir: Path,
    source_root: Path,
    *,
    model: str,
    resolution_scale: str = "linear",
    chance_reference: pd.DataFrame | None = None,
    visual_example: tuple[list[tuple[str, np.ndarray]], pd.DataFrame] | None = None,
) -> list[str]:
    if not frames:
        return []
    if resolution_scale not in {"linear", "log2"}:
        raise ValueError("resolution_scale must be 'linear' or 'log2'")
    fig = plt.figure(figsize=(15.5, 13.4))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.0, 1.0, 0.46),
        hspace=0.48,
        wspace=0.28,
    )
    axes = {
        "gaussian": fig.add_subplot(grid[0, 0]),
        "resolution": fig.add_subplot(grid[0, 1]),
        "colour": fig.add_subplot(grid[1, 0]),
        "patch": fig.add_subplot(grid[1, 1]),
    }
    summaries = []
    specs = (
        ("gaussian", "A. Gaussian blur", "Gaussian blur (%)"),
        ("resolution", "B. Resolution retained", "Intermediate side (px)"),
        ("colour", "C. Colour and silhouette", "Input retained"),
        ("patch", "D. Patch shuffling", "Patch grid"),
    )
    for panel, title, xlabel in specs:
        summary = _draw_line_panel(axes[panel], frames[panel], x="level", title=title, xlabel=xlabel)
        summaries.append(summary.assign(panel=panel))
    axes["colour"].set_xticks(
        [0, 1, 2], ["Binary mask\nonly", "0% colour", "Original\nRGB"]
    )
    patch_ticks = sorted(frames["patch"]["level"].dropna().unique())
    axes["patch"].set_xticks(patch_ticks, [f"{int(value)}×{int(value)}" for value in patch_ticks])
    if resolution_scale == "log2":
        axes["resolution"].set_xscale("log", base=2)
    axes["resolution"].invert_xaxis()
    axes["resolution"].xaxis.set_major_formatter(
        matplotlib.ticker.ScalarFormatter()
    )
    axes["resolution"].set_xticks(
        sorted(frames["resolution"]["level"].dropna().unique(), reverse=True)
    )
    example_images, example_source = visual_example or ([], pd.DataFrame())
    _draw_visual_example_strip(fig, grid[2, :], example_images)
    fig.suptitle(
        f"Visual ablations for {_display_model(model)}",
        fontsize=16, fontweight="bold", y=0.995,
    )
    plot_data = pd.concat(
        [
            frame.assign(panel=panel)
            for panel, frame in frames.items()
            if panel != "interaction"
        ],
        ignore_index=True,
    )
    fig.subplots_adjust(top=0.96, bottom=0.045, left=0.07, right=0.98)
    return _save_bundle(
        fig, output_dir=output_dir, source_root=source_root,
        name=(
            "figure_02_convnext_visual_ablation"
            if resolution_scale == "linear"
            else "figure_02b_convnext_visual_ablation_resolution_log2"
        ),
        sources={
            "plot_data": plot_data,
            "seed_summary": pd.concat(summaries, ignore_index=True),
            "chance_reference": chance_reference
            if chance_reference is not None
            else pd.DataFrame(),
            "representative_visual_example": example_source,
        },
        details={
            "model": model,
            "metric": "test mean macro-F1",
            "split": "test",
            "uncertainty": "95% t interval across seeds",
            "resolution_x": "intermediate pixels per side",
            "resolution_scale": resolution_scale,
            "representative_example": (
                "same deterministically selected independent-test image rendered "
                "with the exact model-input transforms"
            ),
            "chance": "expected macro-F1 under uniform random prediction on the fixed test label distribution",
            **_paper_design_manifest(),
        },
    )


def save_mixed_visual_seed_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    source_root: Path,
    *,
    model: str,
    chance_reference: pd.DataFrame | None = None,
) -> list[str]:
    """Paired-seed small multiples for Gaussian x cue interactions."""
    if frame.empty:
        return []
    order = [
        "100% colour (Gaussian only)",
        "0% colour",
        "2×2 patches",
        "4×4 patches",
        "8×8 patches",
        "16×16 patches",
    ]
    observed = [value for value in order if value in set(frame["series"])]
    blur_levels = sorted(pd.to_numeric(frame["level"], errors="coerce").dropna().unique())
    summary = seed_summary(frame, ["level", "series"], "test_mean_macro_f1")
    fig, axes = plt.subplots(
        1,
        len(blur_levels),
        figsize=(max(14.0, 3.7 * len(blur_levels)), 6.4),
        sharey=True,
        squeeze=False,
    )
    for level_index, level in enumerate(blur_levels):
        ax = axes[0, level_index]
        current = frame[pd.to_numeric(frame["level"], errors="coerce").eq(level)]
        x_positions = np.arange(len(observed), dtype=float)
        for seed, seed_frame in current.groupby("seed", dropna=False):
            seed_frame = seed_frame.set_index("series")
            present = [value for value in observed if value in seed_frame.index]
            if len(present) >= 2:
                xs = [observed.index(value) for value in present]
                ys = [float(seed_frame.loc[value]["test_mean_macro_f1"]) for value in present]
                ax.plot(xs, ys, color="#777777", linewidth=0.45, alpha=0.14, zorder=1)
        for series_index, series_value in enumerate(observed):
            raw = current[current["series"].eq(series_value)]
            offsets = _deterministic_jitter(raw.get("seed", raw.index), width=0.10)
            ax.scatter(
                np.full(len(raw), x_positions[series_index]) + offsets,
                raw["test_mean_macro_f1"],
                s=17,
                alpha=0.36,
                color=PALETTE[series_index % len(PALETTE)],
                marker=("o", "s", "^", "v", "D", "P")[series_index],
                zorder=2,
            )
            row = summary[
                pd.to_numeric(summary["level"], errors="coerce").eq(level)
                & summary["series"].eq(series_value)
            ]
            if not row.empty:
                row = row.iloc[0]
                ax.errorbar(
                    x_positions[series_index],
                    row["mean"],
                    yerr=0.0 if pd.isna(row["ci95"]) else row["ci95"],
                    fmt="o",
                    color="#000000",
                    markerfacecolor=PALETTE[series_index % len(PALETTE)],
                    capsize=3,
                    markersize=6,
                    zorder=4,
                )
        chance = pd.to_numeric(current.get("chance", np.nan), errors="coerce")
        if np.ndim(chance) and pd.Series(chance).notna().any():
            chance_value = float(pd.Series(chance).dropna().mean())
            ax.axhline(
                chance_value,
                color="#000000",
                linestyle=(0, (4, 2)),
                linewidth=1.0,
            )
            ax.text(
                0.98,
                chance_value,
                f"chance {chance_value:.1%}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=6.5,
            )
        ax.set_title(f"Gaussian blur {level:g}%", fontweight="bold")
        ax.set_xticks(x_positions, observed, rotation=62, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        if level_index == 0:
            ax.set_ylabel("Test mean macro-F1")
        ax.set_ylim(0, 1.02)
    fig.suptitle(
        f"Paired-seed mixed visual ablations — {_display_model(model)}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Each translucent point is one training seed; thin lines connect the same seed across cue conditions. Black-edged points and whiskers are seed means and 95% t intervals.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.86, bottom=0.34, left=0.055, right=0.995, wspace=0.16)
    return _save_bundle(
        fig,
        output_dir=output_dir,
        source_root=source_root,
        name="figure_02c_mixed_visual_seed_comparison",
        sources={
            "plot_data": frame,
            "seed_summary": summary,
            "chance_reference": chance_reference
            if chance_reference is not None
            else pd.DataFrame(),
        },
        details={
            "model": model,
            "metric": "test mean macro-F1",
            "split": "test",
            "pairing_unit": "training seed",
            "uncertainty": "95% t interval across seeds",
            "chance": "expected macro-F1 under uniform random prediction on the fixed test label distribution",
            **_paper_design_manifest(),
        },
    )


def prepare_taxon_stage_holdout_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    """Normalise exhaustive taxon-stage metrics for paired comparisons."""
    frame = _supported(_attach_run_design_metadata(metrics))
    if frame.empty:
        return frame
    frame = frame[frame["training_regime"].isin({"adult_combo_withheld", "full_data_control"})].copy()
    frame["system"] = frame["training_regime"].map({
        "adult_combo_withheld": "Ablated training",
        "full_data_control": "Full-data baseline",
    })
    frame["macro_f1"] = pd.to_numeric(frame.get("macro_f1", np.nan), errors="coerce")
    fallback = pd.to_numeric(
        frame.get("target_recall", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    primary = pd.to_numeric(
        frame.get("target_recall_image", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    frame["target_recall"] = primary.fillna(fallback)
    if "combo_label" not in frame:
        frame["combo_label"] = frame.get("holdout", "unknown").astype(str).str.replace("_", " ")
    return frame


def attach_taxon_individual_counts(
    frame: pd.DataFrame,
    split_root: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach unique-individual counts for each plotted species-stage cohort."""
    inventory = dataset_composition_table(load_split_frames(split_root))
    count_columns = [
        "genus", "species", "stage", "individuals", "test_individuals",
    ]
    if inventory.empty:
        empty_columns = [
            "overall_individuals" if column == "individuals" else column
            for column in count_columns
        ]
        return frame.copy(), pd.DataFrame(columns=empty_columns)
    inventory = inventory[
        inventory["stage"].astype(str).isin(("Adult", "Juvenile"))
        & inventory["genus"].notna()
        & inventory["species"].notna()
    ][count_columns].copy()
    inventory = inventory.rename(columns={"individuals": "overall_individuals"})
    if frame.empty:
        return frame.copy(), inventory
    result = frame.merge(
        inventory,
        on=["genus", "species", "stage"],
        how="left",
        validate="many_to_one",
    )
    return result, inventory


def pair_taxon_metrics(frame: pd.DataFrame, *, cohort: str = TEST_COHORT) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if "cohort" not in frame:
        raise ValueError(
            "Taxon-stage metrics must identify their cohort; refusing to mix "
            "non-test results into the test-only figures."
        )
    selected = _paper_design_only(frame)
    if cohort:
        selected = selected[selected["cohort"].eq(cohort)].copy()
    candidates = [
        "model", "seed", "hierarchy_loss_weight", "holdout", "cohort", "task",
        "genus", "species", "stage", "combo_label",
    ]
    keys = [column for column in candidates if column in selected]
    values = [
        column
        for column in (
            "macro_f1", "target_recall", "target_precision", "target_f1",
            "target_specificity", "target_average_precision",
            "target_roc_auc", "chance",
        )
        if column in selected
    ]
    baseline = selected[selected["system"].eq("Full-data baseline")][[*keys, *values]].copy()
    ablated = selected[selected["system"].eq("Ablated training")][[*keys, *values]].copy()
    if baseline.empty or ablated.empty:
        return pd.DataFrame()
    baseline = baseline.groupby(keys, dropna=False, as_index=False)[values].mean()
    ablated = ablated.groupby(keys, dropna=False, as_index=False)[values].mean()
    baseline = baseline.rename(columns={column: f"baseline_{column}" for column in values})
    ablated = ablated.rename(columns={column: f"ablated_{column}" for column in values})
    paired = ablated.merge(baseline, on=keys, how="inner")
    for metric in (
        "macro_f1", "target_recall", "target_precision", "target_f1",
        "target_specificity", "target_average_precision", "target_roc_auc",
    ):
        left, right = f"ablated_{metric}", f"baseline_{metric}"
        if left in paired and right in paired:
            paired[f"delta_{metric}"] = paired[left] - paired[right]
    chance_columns = [column for column in ("baseline_chance", "ablated_chance") if column in paired]
    if len(chance_columns) == 2:
        mismatch = ~np.isclose(
            paired["baseline_chance"], paired["ablated_chance"],
            equal_nan=True,
        )
        if mismatch.any():
            raise ValueError(
                "Baseline and ablated runs disagree on the derived chance "
                "value for a matched task."
            )
        paired["chance"] = paired["baseline_chance"]
    elif chance_columns:
        paired["chance"] = paired[chance_columns[0]]
    return paired


def save_species_target_metric_figure(
    paired: pd.DataFrame,
    output_dir: Path,
    source_root: Path,
    *,
    species_ablation: str,
) -> list[str]:
    """Precision/recall/F1 view for the configurable main-text species."""
    metric_specs = (
        ("target_precision", "Precision"),
        ("target_recall", "Recall"),
        ("target_f1", "F1"),
    )
    available = [
        (metric, label)
        for metric, label in metric_specs
        if f"baseline_{metric}" in paired and f"ablated_{metric}" in paired
    ]
    if paired.empty or not available:
        return []
    rows = []
    for metric, label in available:
        for system, prefix in (
            ("Full-data baseline", "baseline"),
            ("Ablated training", "ablated"),
        ):
            current = paired[
                [column for column in ("seed", "stage", "task") if column in paired]
            ].copy()
            current["metric"] = label
            current["system"] = system
            current["score"] = pd.to_numeric(
                paired[f"{prefix}_{metric}"], errors="coerce"
            )
            rows.append(current)
    plot_data = pd.concat(rows, ignore_index=True).dropna(subset=["score"])
    if plot_data.empty:
        return []
    summary = seed_summary(
        plot_data,
        ["stage", "task", "metric", "system"],
        "score",
    )
    stages = _ordered_stages(plot_data["stage"].dropna().unique())
    fig, axes = plt.subplots(
        len(stages),
        len(TASK_ORDER),
        figsize=(18, max(5.6, 4.8 * len(stages))),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    x = np.arange(len(available), dtype=float)
    offsets = {"Full-data baseline": -0.13, "Ablated training": 0.13}
    markers = {"Full-data baseline": "D", "Ablated training": "o"}
    colours = {"Full-data baseline": PALETTE[0], "Ablated training": PALETTE[1]}
    for stage_index, stage in enumerate(stages):
        for task_index, task in enumerate(TASK_ORDER):
            ax = axes[stage_index, task_index]
            current = plot_data[
                plot_data["stage"].eq(stage) & plot_data["task"].eq(task)
            ]
            current_summary = summary[
                summary["stage"].eq(stage) & summary["task"].eq(task)
            ]
            for metric_index, (_, metric_label) in enumerate(available):
                for system in offsets:
                    raw = current[
                        current["metric"].eq(metric_label)
                        & current["system"].eq(system)
                    ]
                    row = current_summary[
                        current_summary["metric"].eq(metric_label)
                        & current_summary["system"].eq(system)
                    ]
                    ax.scatter(
                        np.full(len(raw), x[metric_index] + offsets[system])
                        + _deterministic_jitter(raw.get("seed", raw.index), width=0.045),
                        raw["score"],
                        s=14,
                        alpha=0.3,
                        marker=markers[system],
                        color=colours[system],
                    )
                    if not row.empty:
                        row = row.iloc[0]
                        ax.errorbar(
                            x[metric_index] + offsets[system],
                            row["mean"],
                            yerr=0.0 if pd.isna(row["ci95"]) else row["ci95"],
                            fmt=markers[system],
                            color=colours[system],
                            capsize=3,
                            markersize=6,
                            label=system if stage_index == task_index == metric_index == 0 else None,
                        )
            ax.set_xticks(x, [label for _, label in available])
            ax.set_ylim(0, 1.02)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(axis="y", alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if stage_index == 0:
                ax.set_title(TASK_LABELS[task], fontweight="bold")
            if task_index == 0:
                ax.set_ylabel(f"{stage}\nIndependent-test score")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
    fig.suptitle(
        f"Target-class metrics after withholding {species_ablation.replace('_', ' ')}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.text(0.5, 0.012, "Points are matched training seeds; whiskers are 95% t intervals. Precision is calculated one-vs-rest on the complete independent test split.", ha="center", fontsize=8.5)
    fig.subplots_adjust(top=0.89, bottom=0.08, left=0.09, right=0.985, hspace=0.25, wspace=0.16)
    return _save_bundle(
        fig,
        output_dir=output_dir,
        source_root=source_root,
        name="figure_03b_species_ablation_precision_recall_f1",
        sources={"plot_data": plot_data, "seed_summary": summary},
        details={
            "species_ablation": species_ablation,
            "split": "independent test",
            "metric_unit": "image",
            "precision_scope": "one-vs-rest on the complete test split",
            "uncertainty": "95% t interval across seeds",
            **_paper_design_manifest(),
        },
    )


def _read_test_predictions(run_dir: object) -> pd.DataFrame:
    path = Path(str(run_dir)) / "test_predictions_best.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def prepare_biological_question_frame(
    taxon_frame: pd.DataFrame,
    split_root: Path,
) -> pd.DataFrame:
    """Derive four cross-cohort questions from retained full-test predictions."""
    columns = [
        "question", "holdout", "genus", "species", "heldout_stage",
        "evaluated_stage", "task", "seed", "baseline_score",
        "ablated_score", "delta", "test_images", "baseline_target_precision",
        "ablated_target_precision", "baseline_target_recall",
        "ablated_target_recall", "baseline_target_f1", "ablated_target_f1",
    ]
    if taxon_frame.empty or "run_dir" not in taxon_frame:
        return pd.DataFrame(columns=columns)
    test = load_split_frames(split_root).get("test", pd.DataFrame())
    if test.empty or "filename" not in test:
        return pd.DataFrame(columns=columns)
    identities = [
        column for column in ("seed", "holdout", "genus", "species", "stage")
        if column in taxon_frame
    ]
    run_rows = taxon_frame[
        [*identities, "system", "run_dir"]
    ].drop_duplicates()
    baseline = run_rows[run_rows["system"].eq("Full-data baseline")].rename(
        columns={"run_dir": "baseline_run_dir"}
    )
    ablated = run_rows[run_rows["system"].eq("Ablated training")].rename(
        columns={"run_dir": "ablated_run_dir"}
    )
    paired_runs = ablated.merge(
        baseline,
        on=identities,
        how="inner",
        suffixes=("_ablated", "_baseline"),
    )
    prediction_cache: dict[str, pd.DataFrame] = {}
    output_rows = []
    for run in paired_runs.itertuples(index=False):
        baseline_dir = str(run.baseline_run_dir)
        ablated_dir = str(run.ablated_run_dir)
        if baseline_dir not in prediction_cache:
            prediction_cache[baseline_dir] = _read_test_predictions(baseline_dir)
        if ablated_dir not in prediction_cache:
            prediction_cache[ablated_dir] = _read_test_predictions(ablated_dir)
        baseline_predictions = prediction_cache[baseline_dir]
        ablated_predictions = prediction_cache[ablated_dir]
        if baseline_predictions.empty or ablated_predictions.empty:
            continue
        question_specs = [
            (
                "Direct withheld cohort",
                str(run.stage),
                lambda frame: frame["species_label"].astype(str).eq(str(run.species))
                & frame["life_stage"].astype(str).eq(str(run.stage)),
            ),
        ]
        if str(run.stage) == "Adult":
            question_specs.append((
                "Adult removed → evaluate Juvenile",
                "Juvenile",
                lambda frame: frame["species_label"].astype(str).eq(str(run.species))
                & frame["life_stage"].astype(str).eq("Juvenile"),
            ))
        if str(run.stage) == "Juvenile":
            question_specs.append((
                "Juvenile removed → evaluate Adult",
                "Adult",
                lambda frame: frame["species_label"].astype(str).eq(str(run.species))
                & frame["life_stage"].astype(str).eq("Adult"),
            ))
        question_specs.append((
            "Within-genus spillover",
            str(run.stage),
            lambda frame: frame["genus"].astype(str).eq(str(run.genus))
            & frame["life_stage"].astype(str).eq(str(run.stage))
            & ~frame["species_label"].astype(str).eq(str(run.species)),
        ))
        for task in TASK_ORDER:
            baseline_task = baseline_predictions[baseline_predictions["task"].eq(task)]
            ablated_task = ablated_predictions[ablated_predictions["task"].eq(task)]
            if baseline_task.empty or ablated_task.empty:
                continue
            for prediction_frame in (baseline_task, ablated_task):
                if "filename" not in prediction_frame:
                    break
            else:
                baseline_joined = baseline_task.merge(test, on="filename", how="inner")
                ablated_joined = ablated_task.merge(test, on="filename", how="inner")
                target_label_by_stage = {
                    "genus": str(run.genus),
                    "species": str(run.species),
                }
                for question, evaluated_stage, selector in question_specs:
                    baseline_cohort = baseline_joined[selector(baseline_joined)]
                    ablated_cohort = ablated_joined[selector(ablated_joined)]
                    if baseline_cohort.empty or ablated_cohort.empty:
                        continue
                    baseline_score = float(
                        baseline_cohort["true_label"].astype(str).eq(
                            baseline_cohort["predicted_label"].astype(str)
                        ).mean()
                    )
                    ablated_score = float(
                        ablated_cohort["true_label"].astype(str).eq(
                            ablated_cohort["predicted_label"].astype(str)
                        ).mean()
                    )
                    target_label = (
                        evaluated_stage if task == "age"
                        else target_label_by_stage.get(task)
                    )
                    baseline_target = {}
                    ablated_target = {}
                    if target_label is not None:
                        for prediction_frame, destination in (
                            (baseline_task, baseline_target),
                            (ablated_task, ablated_target),
                        ):
                            labels = list(dict.fromkeys(
                                [
                                    *prediction_frame["true_label"].astype(str).tolist(),
                                    *prediction_frame["predicted_label"].astype(str).tolist(),
                                ]
                            ))
                            if target_label in labels:
                                mapping = {label: index for index, label in enumerate(labels)}
                                destination.update(classification_metric_summary(
                                    prediction_frame["true_label"].astype(str).map(mapping).to_numpy(),
                                    prediction_frame["predicted_label"].astype(str).map(mapping).to_numpy(),
                                    target_index=mapping[target_label],
                                ))
                    output_rows.append({
                        "question": question,
                        "holdout": run.holdout,
                        "genus": run.genus,
                        "species": run.species,
                        "heldout_stage": run.stage,
                        "evaluated_stage": evaluated_stage,
                        "task": task,
                        "seed": run.seed,
                        "baseline_score": baseline_score,
                        "ablated_score": ablated_score,
                        "delta": ablated_score - baseline_score,
                        "test_images": int(len(baseline_cohort)),
                        "baseline_target_precision": baseline_target.get("target_precision"),
                        "ablated_target_precision": ablated_target.get("target_precision"),
                        "baseline_target_recall": baseline_target.get("target_recall"),
                        "ablated_target_recall": ablated_target.get("target_recall"),
                        "baseline_target_f1": baseline_target.get("target_f1"),
                        "ablated_target_f1": ablated_target.get("target_f1"),
                    })
    return pd.DataFrame(output_rows, columns=columns)


def save_biological_question_figure(
    frame: pd.DataFrame,
    output_dir: Path,
    source_root: Path,
) -> list[str]:
    if frame.empty:
        return []
    question_order = [
        "Direct withheld cohort",
        "Adult removed → evaluate Juvenile",
        "Juvenile removed → evaluate Adult",
        "Within-genus spillover",
    ]
    questions = [value for value in question_order if value in set(frame["question"])]
    per_seed = frame.groupby(
        ["question", "task", "seed"], dropna=False, as_index=False
    )["delta"].mean()
    summary = seed_summary(per_seed, ["question", "task"], "delta")
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), sharex=True, sharey=True, squeeze=False)
    x = np.arange(len(TASK_ORDER), dtype=float)
    for question_index, question in enumerate(questions):
        ax = axes.flat[question_index]
        current = per_seed[per_seed["question"].eq(question)]
        current_summary = summary[summary["question"].eq(question)].set_index("task")
        for task_index, task in enumerate(TASK_ORDER):
            raw = current[current["task"].eq(task)]
            ax.scatter(
                np.full(len(raw), x[task_index])
                + _deterministic_jitter(raw.get("seed", raw.index), width=0.09),
                raw["delta"],
                s=20,
                alpha=0.34,
                marker=("o", "s", "^")[task_index],
                color=PALETTE[task_index],
            )
            if task in current_summary.index:
                row = current_summary.loc[task]
                ax.errorbar(
                    x[task_index], row["mean"],
                    yerr=0.0 if pd.isna(row["ci95"]) else row["ci95"],
                    fmt=("o", "s", "^")[task_index], color=PALETTE[task_index],
                    markeredgecolor="black", capsize=4, markersize=8, zorder=4,
                )
        ax.axhline(0.0, color="#000000", linestyle=":", linewidth=1.1)
        ax.set_xticks(x, [TASK_LABELS[task] for task in TASK_ORDER])
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{chr(65 + question_index)}. {question}", loc="left", fontweight="bold")
        if question_index % 2 == 0:
            ax.set_ylabel("Change from matched full-data model")
    for unused in range(len(questions), 4):
        axes.flat[unused].set_axis_off()
    fig.suptitle("What transfers when a species-stage cohort is removed?", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.018, "Each point is a training seed after first averaging eligible species within that seed. Negative values indicate worse independent-test cohort accuracy after ablation; whiskers are 95% t intervals across seeds.", ha="center", fontsize=8.5)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.09, right=0.98, hspace=0.27, wspace=0.16)
    return _save_bundle(
        fig,
        output_dir=output_dir,
        source_root=source_root,
        name="figure_04_biological_transfer_questions",
        sources={
            "prediction_derived_metrics": frame,
            "per_seed_species_average": per_seed,
            "seed_summary": summary,
        },
        details={
            "split": "independent test only",
            "metric": "change in image-level cohort accuracy",
            "pairing_unit": "matched training seed",
            "precision_recall_f1_source": "complete-test one-vs-rest hard predictions",
            "uncertainty": "95% t interval across seed-level species averages",
            **_paper_design_manifest(),
        },
    )


def shared_variance_effect_summary(
    paired: pd.DataFrame,
    *,
    identity_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Put normal, ablated, and chance recall on the normal-model SD scale."""
    identities = [column for column in identity_columns if column in paired]
    columns = [
        *identities, "task", "n_seeds", "normal_mean_recall",
        "ablated_mean_recall", "chance", "number_of_classes", "chance_method",
        "shared_normal_sd", "m_total", "m_lost", "m_retained",
        "m_total_ci95_low", "m_total_ci95_high",
        "m_lost_ci95_low", "m_lost_ci95_high",
        "m_retained_ci95_low", "m_retained_ci95_high",
        "margin_additive_check_error", "d_total", "d_ablation", "d_retained",
        "d_total_ci95_low", "d_total_ci95_high",
        "d_ablation_ci95_low", "d_ablation_ci95_high",
        "d_retained_ci95_low", "d_retained_ci95_high",
        "standardised_additive_check_error",
    ]
    required = {
        "task", "seed", "chance", "baseline_target_recall",
        "ablated_target_recall",
    }
    if paired.empty or not required.issubset(paired.columns):
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    group_columns = [*identities, "task"]
    grouped = paired.groupby(group_columns, dropna=False)
    for identity, group in grouped:
        identity_values = identity if isinstance(identity, tuple) else (identity,)
        identity_record = dict(zip(group_columns, identity_values))
        clean = group.dropna(subset=[
            "seed", "chance", "baseline_target_recall",
            "ablated_target_recall",
        ]).copy()
        if clean.empty:
            continue
        per_seed = clean.groupby("seed", as_index=False).agg(
            normal_recall=("baseline_target_recall", "mean"),
            ablated_recall=("ablated_target_recall", "mean"),
            chance=("chance", "mean"),
        )
        chance_values = np.unique(
            np.round(per_seed["chance"].to_numpy(dtype=float), 12)
        )
        if len(chance_values) != 1:
            raise ValueError(
                "Shared-variance effects require one fixed chance value per "
                f"species-stage/task group; found {chance_values.tolist()}"
            )
        normal_scores = per_seed["normal_recall"].to_numpy(dtype=float)
        ablated_scores = per_seed["ablated_recall"].to_numpy(dtype=float)
        normal_mean = float(normal_scores.mean())
        ablated_mean = float(ablated_scores.mean())
        chance = float(chance_values[0])
        reciprocal = 1.0 / chance if chance > 0.0 else np.nan
        number_of_classes = (
            int(round(reciprocal))
            if np.isfinite(reciprocal)
            and np.isclose(reciprocal, round(reciprocal))
            else np.nan
        )
        m_total = normal_mean - chance
        m_lost = normal_mean - ablated_mean
        m_retained = ablated_mean - chance
        margin_additive_error = m_total - (m_lost + m_retained)
        shared_sd = (
            float(normal_scores.std(ddof=1)) if normal_scores.size >= 2
            else np.nan
        )
        if np.isfinite(shared_sd) and shared_sd > 0.0:
            d_total = (normal_mean - chance) / shared_sd
            d_ablation = (normal_mean - ablated_mean) / shared_sd
            d_retained = (ablated_mean - chance) / shared_sd
            standardised_additive_error = d_total - (d_ablation + d_retained)
        else:
            d_total = d_ablation = d_retained = np.nan
            standardised_additive_error = np.nan
        intervals = _paired_seed_bootstrap_intervals(
            normal_scores,
            ablated_scores,
            chance,
            identity=identity_record,
        )
        rows.append({
            **identity_record,
            "n_seeds": int(normal_scores.size),
            "normal_mean_recall": normal_mean,
            "ablated_mean_recall": ablated_mean,
            "chance": chance,
            "number_of_classes": number_of_classes,
            "chance_method": "uniform random prediction across K classes: 1/K",
            "shared_normal_sd": shared_sd,
            "m_total": m_total,
            "m_lost": m_lost,
            "m_retained": m_retained,
            **intervals,
            "margin_additive_check_error": float(margin_additive_error),
            "d_total": float(d_total),
            "d_ablation": float(d_ablation),
            "d_retained": float(d_retained),
            "standardised_additive_check_error": float(
                standardised_additive_error
            ),
        })
    result = pd.DataFrame(rows)
    for column in columns:
        if column not in result:
            result[column] = np.nan
    return result[columns]


def _paired_seed_bootstrap_intervals(
    normal_scores: np.ndarray,
    ablated_scores: np.ndarray,
    chance: float,
    *,
    identity: dict[str, object],
) -> dict[str, float]:
    """Pointwise percentile intervals from matched-seed bootstrap samples."""
    metric_names = (
        "m_total", "m_lost", "m_retained",
        "d_total", "d_ablation", "d_retained",
    )
    empty = {
        f"{metric}_{bound}": np.nan
        for metric in metric_names
        for bound in ("ci95_low", "ci95_high")
    }
    if normal_scores.size < 2 or normal_scores.size != ablated_scores.size:
        return empty

    identity_text = json.dumps(
        {
            str(key): None if pd.isna(value) else str(value)
            for key, value in identity.items()
        },
        sort_keys=True,
    )
    identity_seed = int.from_bytes(
        hashlib.sha256(identity_text.encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(ABLATION_BOOTSTRAP_SEED ^ identity_seed)
    indices = rng.integers(
        0,
        normal_scores.size,
        size=(ABLATION_BOOTSTRAP_RESAMPLES, normal_scores.size),
    )
    sampled_normal = normal_scores[indices]
    sampled_ablated = ablated_scores[indices]
    normal_mean = sampled_normal.mean(axis=1)
    ablated_mean = sampled_ablated.mean(axis=1)
    normal_sd = sampled_normal.std(axis=1, ddof=1)
    samples = {
        "m_total": normal_mean - chance,
        "m_lost": normal_mean - ablated_mean,
        "m_retained": ablated_mean - chance,
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        samples.update({
            "d_total": (normal_mean - chance) / normal_sd,
            "d_ablation": (normal_mean - ablated_mean) / normal_sd,
            "d_retained": (ablated_mean - chance) / normal_sd,
        })
    alpha = (1.0 - ABLATION_INTERVAL_LEVEL) / 2.0
    result = empty.copy()
    for metric, values in samples.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            low, high = np.quantile(finite, (alpha, 1.0 - alpha))
            result[f"{metric}_ci95_low"] = float(low)
            result[f"{metric}_ci95_high"] = float(high)
    return result


def _recall_seed_data(paired: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model", "seed", "holdout", "cohort", "task", "genus", "species",
        "stage", "combo_label", "overall_individuals", "test_individuals",
        "chance", "baseline_target_recall", "ablated_target_recall",
    ]
    return paired[[column for column in columns if column in paired]].copy()


def _individual_count_identity_columns(*columns: str) -> tuple[str, ...]:
    return (*columns, "overall_individuals", "test_individuals")


def _cohort_count_label(base: str, row: pd.Series) -> str:
    overall = pd.to_numeric(row.get("overall_individuals"), errors="coerce")
    test = pd.to_numeric(row.get("test_individuals"), errors="coerce")
    if pd.isna(overall) or pd.isna(test):
        return base
    return (
        f"{base}\n{int(overall):,} individuals overall; "
        f"{int(test):,} in test"
    )


def _ablation_figure_details() -> dict[str, str]:
    return {
        "evaluation_unit": "image",
        "reported_split": "independent test only",
        "individual_count_unit": "unique biological individual (barcode)",
        "individual_count_scope": "matching species-stage cohort",
        "overall_individuals": "unique barcodes across training, validation, and test splits",
        "test_individuals": "unique barcodes in the independent test split",
    }


def _ablation_interval_details() -> dict[str, object]:
    return {
        "interval": "pointwise 95% paired-seed percentile bootstrap interval",
        "interval_source": "variation across matched baseline/ablated training seeds",
        "interval_is_class_based": False,
        "interval_resamples": ABLATION_BOOTSTRAP_RESAMPLES,
        "interval_random_seed": ABLATION_BOOTSTRAP_SEED,
        "interval_pairing_unit": "matched training seed",
        "chance_reference": "1/K from the saved task label map",
        "chance_is_interval_source": False,
        "interval_visualisation": (
            "whiskers show retained and total position intervals; lost-gap "
            "interval bounds are preserved in the source CSV"
        ),
    }


def _ablation_figure_note(prefix: str) -> str:
    return (
        f"{prefix} Performance is image-level target recall on the independent "
        "test split only; displayed counts are unique biological individuals "
        "(barcodes) in each species-stage cohort."
    )


def _ordered_stages(values: Iterable[object]) -> list[str]:
    observed = [str(value) for value in values if pd.notna(value)]
    preferred = [stage for stage in ("Adult", "Juvenile") if stage in observed]
    return [*preferred, *sorted(set(observed).difference(preferred))]


def _configure_capacity_axis(ax: plt.Axes) -> None:
    ax.axvspan(-0.2, 0.2, color="#777777", alpha=0.06, zorder=0)
    ax.axvline(0.0, color="#555555", linestyle=":", linewidth=1.1, zorder=0)
    ax.axvline(0.8, color="#AAAAAA", linestyle="--", linewidth=0.7, zorder=0)
    ax.axvline(2.0, color="#AAAAAA", linestyle="--", linewidth=0.7, zorder=0)
    ax.grid(axis="x", alpha=0.16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Shared effect size (normal-model SD units)")


def _draw_capacity_row(
    ax: plt.Axes,
    row: pd.Series,
    y: float,
    *,
    annotate: bool,
) -> None:
    total = float(row["d_total"])
    lost = float(row["d_ablation"])
    retained = float(row["d_retained"])
    if not all(np.isfinite(value) for value in (total, lost, retained)):
        ax.text(0.0, y, "undefined: normal seed SD = 0", fontsize=8, va="center")
        return
    ax.plot([0.0, retained], [y, y], color="#009E73", linewidth=5.0,
            solid_capstyle="round", alpha=0.75, zorder=2)
    ax.plot([retained, total], [y, y], color="#D55E00", linewidth=5.0,
            solid_capstyle="round", alpha=0.75, zorder=2)
    ax.scatter(0.0, y, marker="|", s=115, color="#555555", linewidths=1.5,
               zorder=4)
    ax.scatter(retained, y, marker="o", s=54, color="#E69F00",
               edgecolors="white", linewidths=0.8, zorder=4)
    ax.scatter(total, y, marker="D", s=52, facecolors="white",
               edgecolors="#0072B2", linewidths=1.5, zorder=4)
    _draw_horizontal_interval(ax, row, "d_retained", y, "#E69F00")
    _draw_horizontal_interval(ax, row, "d_total", y, "#0072B2")
    if annotate:
        ax.annotate(
            f"retained {retained:.2f}", (retained, y), xytext=(0, -15),
            textcoords="offset points", ha="center", va="top", fontsize=7.5,
            color="#6B4E00",
        )
        ax.annotate(
            f"total {total:.2f}", (total, y), xytext=(0, 13),
            textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
            color="#00547F",
        )
        ax.annotate(
            f"lost {lost:.2f}", ((retained + total) / 2.0, y),
            xytext=(0, 5), textcoords="offset points", ha="center",
            va="bottom", fontsize=7.2, color="#8C3A00",
        )


def save_paired_estimation_figure(
    paired: pd.DataFrame,
    output_dir: Path,
    source_root: Path,
    *,
    name: str,
    title: str,
    details: dict,
) -> list[str]:
    if paired.empty:
        return []
    effect_summary = shared_variance_effect_summary(
        paired,
        identity_columns=_individual_count_identity_columns("stage"),
    )
    if effect_summary.empty:
        return []
    stages = _ordered_stages(effect_summary["stage"].dropna().unique())
    y = np.arange(len(stages), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True, squeeze=False)
    for task_index, task in enumerate(TASK_ORDER):
        ax = axes[0, task_index]
        current = effect_summary[effect_summary["task"].eq(task)].set_index("stage")
        for stage_index, stage in enumerate(stages):
            if stage in current.index:
                row = current.loc[stage]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                _draw_capacity_row(ax, row, y[stage_index], annotate=True)
        _configure_capacity_axis(ax)
        ax.set_title(TASK_LABELS[task], loc="left", fontweight="bold")
        ax.set_yticks(y)
        if task_index == 0:
            stage_labels = []
            for stage in stages:
                rows = effect_summary[effect_summary["stage"].eq(stage)]
                stage_labels.append(
                    _cohort_count_label(stage, rows.iloc[0])
                    if not rows.empty else stage
                )
            ax.set_yticklabels(stage_labels, fontsize=9.5, fontweight="bold")
            ax.set_ylabel("Life stage", labelpad=12)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylim(len(stages) - 0.5, -0.5)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], marker="|", color="#555555", linestyle="none",
                   markersize=12, label="Chance = 0"),
            Line2D([0], [0], marker="o", color="#E69F00", linestyle="none",
                   markeredgecolor="white", label=r"Ablated: $d_{retained}$"),
            Line2D([0], [0], marker="D", color="#0072B2", linestyle="none",
                   markerfacecolor="white", label=r"Normal: $d_{total}$"),
            Line2D([0], [0], color="#D55E00", linewidth=5,
                   label=r"Gap: $d_{ablation}$"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=4, frameon=False,
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    fig.text(
        0.5, 0.018,
        _ablation_figure_note(
            r"All positions use the normal model's sample SD: "
            r"$d_{total}=d_{ablation}+d_{retained}$. The publication design "
            r"uses 30 seeds per stage; guides mark $d=0.8$ and $d=2$. "
            r"Whiskers are pointwise 95% paired-seed bootstrap intervals; "
            r"gap intervals are retained in the source CSV."
        ),
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.09, right=0.98, wspace=0.14)
    return _save_bundle(
        fig, output_dir=output_dir, source_root=source_root, name=name,
        sources={
            "recall_seed_data": _recall_seed_data(paired),
            "shared_variance_effects": effect_summary,
        },
        details={
            "metric": "target_recall",
            "effect_framework": "shared normal-model variance",
            "shared_standard_deviation": "sample SD of normal-model seed recalls (ddof=1)",
            "d_total": "(normal mean recall - chance) / normal recall SD",
            "d_ablation": "(normal mean recall - ablated mean recall) / normal recall SD",
            "d_retained": "(ablated mean recall - chance) / normal recall SD",
            "additive_identity": "d_total = d_ablation + d_retained",
            **_ablation_interval_details(),
            "inference": "descriptive effect sizes with confidence intervals; no p-value",
            "cohort": TEST_COHORT,
            "split": "test",
            **_ablation_figure_details(),
            **_paper_design_manifest(),
            **details,
        },
    )


def _raw_margin_summary(effect_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "combo_label", "genus", "species", "stage", "task", "n_seeds",
        "overall_individuals", "test_individuals",
        "normal_mean_recall", "ablated_mean_recall", "chance",
        "number_of_classes", "chance_method", "m_total", "m_lost",
        "m_retained", "m_total_ci95_low", "m_total_ci95_high",
        "m_lost_ci95_low", "m_lost_ci95_high",
        "m_retained_ci95_low", "m_retained_ci95_high",
        "margin_additive_check_error",
    ]
    return effect_summary[
        [column for column in columns if column in effect_summary]
    ].copy()


def _configure_margin_axis(ax: plt.Axes) -> None:
    ax.axvline(0.0, color="#555555", linestyle=":", linewidth=1.1, zorder=0)
    ax.grid(axis="x", alpha=0.16)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Target-recall margin above chance")


def _draw_margin_row(
    ax: plt.Axes,
    row: pd.Series,
    y: float,
    *,
    annotate: bool,
) -> None:
    total = float(row["m_total"])
    lost = float(row["m_lost"])
    retained = float(row["m_retained"])
    ax.plot([0.0, retained], [y, y], color="#009E73", linewidth=5.0,
            solid_capstyle="round", alpha=0.75, zorder=2)
    ax.plot([retained, total], [y, y], color="#D55E00", linewidth=5.0,
            solid_capstyle="round", alpha=0.75, zorder=2)
    ax.scatter(0.0, y, marker="|", s=115, color="#555555", linewidths=1.5,
               zorder=4)
    ax.scatter(retained, y, marker="o", s=54, color="#E69F00",
               edgecolors="white", linewidths=0.8, zorder=4)
    ax.scatter(total, y, marker="D", s=52, facecolors="white",
               edgecolors="#0072B2", linewidths=1.5, zorder=4)
    _draw_horizontal_interval(ax, row, "m_retained", y, "#E69F00")
    _draw_horizontal_interval(ax, row, "m_total", y, "#0072B2")
    if annotate:
        ax.annotate(
            f"retained {retained:+.1%}", (retained, y), xytext=(0, -15),
            textcoords="offset points", ha="center", va="top", fontsize=7.5,
            color="#6B4E00",
        )
        ax.annotate(
            f"total {total:+.1%}", (total, y), xytext=(0, 13),
            textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
            color="#00547F",
        )
        ax.annotate(
            f"lost {lost:+.1%}", ((retained + total) / 2.0, y),
            xytext=(0, 5), textcoords="offset points", ha="center",
            va="bottom", fontsize=7.2, color="#8C3A00",
        )


def _draw_horizontal_interval(
    ax: plt.Axes,
    row: pd.Series,
    metric: str,
    y: float,
    colour: str,
) -> None:
    center = float(row.get(metric, np.nan))
    low = float(row.get(f"{metric}_ci95_low", np.nan))
    high = float(row.get(f"{metric}_ci95_high", np.nan))
    if not all(np.isfinite(value) for value in (center, low, high)):
        return
    ax.errorbar(
        center,
        y,
        xerr=np.asarray([[max(0.0, center - low)], [max(0.0, high - center)]]),
        fmt="none",
        ecolor=colour,
        elinewidth=1.2,
        capsize=3.0,
        capthick=1.0,
        alpha=0.95,
        zorder=3,
    )


def _raw_margin_details() -> dict[str, object]:
    return {
        "metric": "target_recall",
        "effect_framework": "additive raw target-recall margins",
        "m_total": "normal mean recall - chance",
        "m_lost": "normal mean recall - ablated mean recall",
        "m_retained": "ablated mean recall - chance",
        "additive_identity": "M_total = M_lost + M_retained",
        "chance_formula": "1/K",
        "chance_metric": "target recall under uniform random prediction",
        "chance_source": "K derived from each run's saved label_to_index_by_task.json",
        "random_prediction_strategy": "uniform probability over K task classes",
        "class_imbalance_note": "for class recall under uniform random prediction, chance remains 1/K even when true classes are imbalanced",
        "accuracy_weighted_recall_note": "not plotted; those metrics require chance from true class proportions and prediction probabilities",
        **_ablation_interval_details(),
        "inference": "descriptive margins with confidence intervals; no p-value",
        "cohort": TEST_COHORT,
        "split": "test",
        **_ablation_figure_details(),
        **_paper_design_manifest(),
    }


def save_species_margin_figure(
    paired: pd.DataFrame,
    output_dir: Path,
    source_root: Path,
    *,
    species_ablation: str,
) -> list[str]:
    if paired.empty:
        return []
    full_summary = shared_variance_effect_summary(
        paired,
        identity_columns=_individual_count_identity_columns("stage"),
    )
    if full_summary.empty:
        return []
    summary = _raw_margin_summary(full_summary)
    stages = _ordered_stages(summary["stage"].dropna().unique())
    y = np.arange(len(stages), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True, squeeze=False)
    for task_index, task in enumerate(TASK_ORDER):
        ax = axes[0, task_index]
        current = summary[summary["task"].eq(task)].set_index("stage")
        for stage_index, stage in enumerate(stages):
            if stage not in current.index:
                continue
            row = current.loc[stage]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            _draw_margin_row(ax, row, y[stage_index], annotate=True)
        _configure_margin_axis(ax)
        ax.set_title(TASK_LABELS[task], loc="left", fontweight="bold")
        ax.set_yticks(y)
        if task_index == 0:
            stage_labels = []
            for stage in stages:
                rows = summary[summary["stage"].eq(stage)]
                stage_labels.append(
                    _cohort_count_label(stage, rows.iloc[0])
                    if not rows.empty else stage
                )
            ax.set_yticklabels(stage_labels, fontsize=9.5, fontweight="bold")
            ax.set_ylabel("Life stage", labelpad=12)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylim(len(stages) - 0.5, -0.5)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], marker="|", color="#555555", linestyle="none",
                   markersize=12, label="Chance = 0 margin"),
            Line2D([0], [0], marker="o", color="#E69F00", linestyle="none",
                   markeredgecolor="white", label=r"Ablated: $M_{retained}$"),
            Line2D([0], [0], marker="D", color="#0072B2", linestyle="none",
                   markerfacecolor="white", label=r"Normal: $M_{total}$"),
            Line2D([0], [0], color="#D55E00", linewidth=5,
                   label=r"Gap: $M_{lost}$"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=4, frameon=False,
    )
    fig.suptitle(
        f"Raw recall margins for {species_ablation.replace('_', ' ')} — independent test cohort",
        fontsize=16, fontweight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.018,
        _ablation_figure_note(
            r"$M_{total}=M_{lost}+M_{retained}$; chance is derived as $1/K$ "
            r"from each task's saved class map. Whiskers are pointwise 95% "
            r"paired-seed bootstrap intervals; gap intervals are retained in "
            r"the source CSV."
        ),
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.09, right=0.98, wspace=0.14)
    return _save_bundle(
        fig, output_dir=output_dir, source_root=source_root,
        name="figure_05_species_ablation_raw_margins",
        sources={
            "recall_seed_data": _recall_seed_data(paired),
            "raw_margin_decomposition": summary,
        },
        details={
            "model": TAXON_MODEL,
            "species_ablation": species_ablation,
            **_raw_margin_details(),
        },
    )


def species_effect_summary(
    paired: pd.DataFrame,
) -> pd.DataFrame:
    """Shared-variance recall effects for every species-stage holdout."""
    return shared_variance_effect_summary(
        paired,
        identity_columns=_individual_count_identity_columns(
            "combo_label", "genus", "species", "stage"
        ),
    )


def save_all_species_effect_figure(
    paired: pd.DataFrame,
    output_dir: Path,
    source_root: Path,
) -> list[str]:
    """Show shared-variance recall capacity for every species-stage row."""
    if paired.empty or "combo_label" not in paired:
        return []
    summary = species_effect_summary(paired)
    if summary.empty:
        return []
    order_columns = [
        column for column in ("stage", "genus", "species", "combo_label")
        if column in paired
    ]
    order_frame = paired[order_columns].drop_duplicates()
    sort_columns = [
        column for column in ("stage", "genus", "species")
        if column in order_columns
    ]
    if sort_columns:
        order_frame = order_frame.sort_values(sort_columns)
    combinations = order_frame["combo_label"].astype(str).tolist()
    combinations = list(dict.fromkeys(combinations))
    y = np.arange(len(combinations), dtype=float)
    figure_height = max(9.0, 0.62 * len(combinations) + 3.0)
    fig, axes = plt.subplots(
        1, 3, figsize=(20, figure_height), sharey=True, squeeze=False,
    )
    for task_index, task in enumerate(TASK_ORDER):
        ax = axes[0, task_index]
        current = summary[summary["task"].eq(task)].set_index("combo_label")
        for combo_index, combo in enumerate(combinations):
            if combo not in current.index:
                continue
            row = current.loc[combo]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            _draw_capacity_row(ax, row, y[combo_index], annotate=False)
        _configure_capacity_axis(ax)
        ax.set_title(TASK_LABELS[task], loc="left", fontweight="bold")
        ax.set_yticks(y)
        if task_index == 0:
            labels = []
            for combo in combinations:
                rows = summary[summary["combo_label"].astype(str).eq(combo)]
                labels.append(
                    _cohort_count_label(combo, rows.iloc[0])
                    if not rows.empty else combo
                )
            ax.set_yticklabels(labels, fontsize=7.6)
            ax.set_ylabel("Species-stage holdout", labelpad=12)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylim(len(combinations) - 0.5, -0.5)
    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="|", color="#555555", linestyle="none",
               markersize=12, label="Chance = 0"),
        Line2D([0], [0], marker="o", color="#E69F00", linestyle="none",
               markeredgecolor="white", label=r"Ablated: $d_{retained}$"),
        Line2D([0], [0], marker="D", color="#0072B2", linestyle="none",
               markerfacecolor="white", label=r"Normal: $d_{total}$"),
        Line2D([0], [0], color="#D55E00", linewidth=5,
               label=r"Gap: $d_{ablation}$"),
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.935),
        ncol=4, frameon=False,
    )
    fig.suptitle(
        "Supplementary: ConvNeXt-Base recall capacity by species and life stage",
        fontsize=16, fontweight="bold", y=0.992,
    )
    fig.text(
        0.5, 0.018,
        _ablation_figure_note(
            r"Chance is 0; the ablated point is $d_{retained}$, the normal "
            r"point is $d_{total}$, and their gap is $d_{ablation}$. All use "
            r"the normal model's seed SD across the 30-seed publication design. "
            r"Whiskers are pointwise 95% paired-seed bootstrap intervals; "
            r"gap intervals are retained in the source CSV."
        ),
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.18, right=0.98, wspace=0.12)
    return _save_bundle(
        fig, output_dir=output_dir, source_root=source_root,
        name="supplementary_figure_01_all_species_effects",
        sources={
            "recall_seed_data": _recall_seed_data(paired),
            "shared_variance_effects": summary,
        },
        details={
            "metric": "target_recall",
            "model": TAXON_MODEL,
            "display_unit": "species-stage holdout",
            "effect_framework": "shared normal-model variance",
            "shared_standard_deviation": "sample SD of normal-model seed recalls (ddof=1)",
            "d_total": "(normal mean recall - chance) / normal recall SD",
            "d_ablation": "(normal mean recall - ablated mean recall) / normal recall SD",
            "d_retained": "(ablated mean recall - chance) / normal recall SD",
            "additive_identity": "d_total = d_ablation + d_retained",
            **_ablation_interval_details(),
            "inference": "descriptive effect sizes with confidence intervals; no p-value",
            "cohort": TEST_COHORT,
            "split": "test",
            **_ablation_figure_details(),
            **_paper_design_manifest(),
            "holdouts": sorted(paired.get("holdout", pd.Series(dtype=str)).dropna().unique().tolist()),
        },
    )


def save_all_species_margin_figure(
    paired: pd.DataFrame,
    output_dir: Path,
    source_root: Path,
) -> list[str]:
    if paired.empty or "combo_label" not in paired:
        return []
    full_summary = species_effect_summary(paired)
    if full_summary.empty:
        return []
    summary = _raw_margin_summary(full_summary)
    order_columns = [
        column for column in ("stage", "genus", "species", "combo_label")
        if column in paired
    ]
    order_frame = paired[order_columns].drop_duplicates()
    sort_columns = [
        column for column in ("stage", "genus", "species")
        if column in order_columns
    ]
    if sort_columns:
        order_frame = order_frame.sort_values(sort_columns)
    combinations = list(dict.fromkeys(
        order_frame["combo_label"].astype(str).tolist()
    ))
    y = np.arange(len(combinations), dtype=float)
    figure_height = max(9.0, 0.62 * len(combinations) + 3.0)
    fig, axes = plt.subplots(
        1, 3, figsize=(20, figure_height), sharey=True, squeeze=False,
    )
    for task_index, task in enumerate(TASK_ORDER):
        ax = axes[0, task_index]
        current = summary[summary["task"].eq(task)].set_index("combo_label")
        for combo_index, combo in enumerate(combinations):
            if combo not in current.index:
                continue
            row = current.loc[combo]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            _draw_margin_row(ax, row, y[combo_index], annotate=False)
        _configure_margin_axis(ax)
        ax.set_title(TASK_LABELS[task], loc="left", fontweight="bold")
        ax.set_yticks(y)
        if task_index == 0:
            labels = []
            for combo in combinations:
                rows = summary[summary["combo_label"].astype(str).eq(combo)]
                labels.append(
                    _cohort_count_label(combo, rows.iloc[0])
                    if not rows.empty else combo
                )
            ax.set_yticklabels(labels, fontsize=7.6)
            ax.set_ylabel("Species-stage holdout", labelpad=12)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylim(len(combinations) - 0.5, -0.5)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], marker="|", color="#555555", linestyle="none",
                   markersize=12, label="Chance = 0 margin"),
            Line2D([0], [0], marker="o", color="#E69F00", linestyle="none",
                   markeredgecolor="white", label=r"Ablated: $M_{retained}$"),
            Line2D([0], [0], marker="D", color="#0072B2", linestyle="none",
                   markerfacecolor="white", label=r"Normal: $M_{total}$"),
            Line2D([0], [0], color="#D55E00", linewidth=5,
                   label=r"Gap: $M_{lost}$"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=4, frameon=False,
    )
    fig.suptitle(
        "Supplementary: raw ConvNeXt-Base recall margins by species and life stage",
        fontsize=16, fontweight="bold", y=0.992,
    )
    fig.text(
        0.5, 0.018,
        _ablation_figure_note(
            r"$M_{total}=M_{lost}+M_{retained}$. Chance is derived separately "
            r"for each task as $1/K$ under uniform random prediction. Whiskers "
            r"are pointwise 95% paired-seed bootstrap intervals; gap intervals "
            r"are retained in the source CSV."
        ),
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.18, right=0.98, wspace=0.12)
    return _save_bundle(
        fig, output_dir=output_dir, source_root=source_root,
        name="supplementary_figure_02_all_species_raw_margins",
        sources={
            "recall_seed_data": _recall_seed_data(paired),
            "raw_margin_decomposition": summary,
        },
        details={
            "model": TAXON_MODEL,
            "display_unit": "species-stage holdout",
            **_raw_margin_details(),
            "holdouts": sorted(
                paired.get("holdout", pd.Series(dtype=str)).dropna().unique().tolist()
            ),
        },
    )


REPRESENTATIVE_CONDITIONS = (
    ("Original", {"transform": "original"}),
    ("Gaussian 50%", {"transform": "gaussian_blur_percent", "parameters": {"percent": 50, "max_sigma": 64.0}}),
    ("Resolution loss 50%", {"transform": "resolution_loss", "parameters": {"percent": 50}}),
    ("Colour removed", {"transform": "saturation", "parameters": {"retention": 0.0}}),
    ("Patch 8x8", {"transform": "patch_shuffle", "parameters": {"grid_size": 8, "seed": 2026}}),
    ("Patch 16x16", {"transform": "patch_shuffle", "parameters": {"grid_size": 16, "seed": 2026}}),
    ("Binary mask only", {"transform": "binary_mask", "parameters": {"threshold": 5.0 / 255.0}}),
    ("Gaussian50 + colour", {"transform": "composed", "parameters": {"operations": [
        {"transform": "gaussian_blur_percent", "parameters": {"percent": 50, "max_sigma": 64.0}},
        {"transform": "saturation", "parameters": {"retention": 0.0}},
    ]}}),
    ("Gaussian50 + patch8", {"transform": "composed", "parameters": {"operations": [
        {"transform": "gaussian_blur_percent", "parameters": {"percent": 50, "max_sigma": 64.0}},
        {"transform": "patch_shuffle", "parameters": {"grid_size": 8, "seed": 2026}},
    ]}}),
    ("Gaussian50 + patch16", {"transform": "composed", "parameters": {"operations": [
        {"transform": "gaussian_blur_percent", "parameters": {"percent": 50, "max_sigma": 64.0}},
        {"transform": "patch_shuffle", "parameters": {"grid_size": 16, "seed": 2026}},
    ]}}),
)


def save_representative_transformations(
    *,
    split_root: Path,
    data_root: Path,
    output_dir: Path,
    source_root: Path,
    sample_seed: int = 2026,
) -> list[str]:
    """Render the exact paper transforms for five reproducibly sampled worms."""
    split_path = Path(split_root) / "split_csv" / "test_split.csv"
    if not split_path.is_file():
        split_path = Path(split_root) / "test_split.csv"
    if not split_path.is_file() or not Path(data_root).is_dir():
        return []
    frame = pd.read_csv(split_path)
    required = {"barcode", "rel_path_seg"}
    if not required.issubset(frame):
        return []
    available = frame[
        frame["rel_path_seg"].map(lambda value: (Path(data_root) / str(value)).is_file())
    ].copy()
    barcodes = available["barcode"].dropna().astype(str).unique()
    if len(barcodes) < 5:
        return []
    generator = np.random.default_rng(sample_seed)
    selected_barcodes = generator.choice(barcodes, size=5, replace=False)
    rows = []
    for barcode in selected_barcodes:
        candidates = available[available["barcode"].astype(str).eq(str(barcode))]
        rows.append(candidates.iloc[int(generator.integers(0, len(candidates)))])

    fig, axes = plt.subplots(5, len(REPRESENTATIVE_CONDITIONS), figsize=(24, 12))
    source_rows = []
    for row_index, row in enumerate(rows):
        image_path = Path(data_root) / str(row["rel_path_seg"])
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        source_rows.append({
            "row": row_index + 1,
            "barcode": row["barcode"],
            "relative_image_path": row["rel_path_seg"],
            "sha256": digest,
            "sample_seed": sample_seed,
        })
        image_source = Image.open(image_path).convert("RGB")
        for column_index, (label, condition) in enumerate(REPRESENTATIVE_CONDITIONS):
            transform = build_split_transform(
                split="test",
                preprocessing={"image_size": 224, "normalisation": {"enabled": False}},
                condition=condition,
            )
            transformed = transform(image_source).permute(1, 2, 0).numpy()
            ax = axes[row_index, column_index]
            ax.imshow(np.clip(transformed, 0.0, 1.0))
            ax.set_axis_off()
            if row_index == 0:
                ax.set_title(label, fontsize=9, fontweight="bold")
            if column_index == 0:
                ax.text(-0.08, 0.5, f"Worm {row_index + 1}", transform=ax.transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle("Representative test-set transformations", fontsize=16, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.94, bottom=0.015, wspace=0.025, hspace=0.04)
    return _save_bundle(
        fig,
        output_dir=output_dir,
        source_root=source_root,
        name="figure_07_representative_transformations",
        sources={"selected_test_worms": pd.DataFrame(source_rows)},
        details={
            "split": "test",
            "sample_seed": sample_seed,
            "sampling": "five unique barcodes, one image per barcode, without replacement",
            "conditions": [label for label, _ in REPRESENTATIVE_CONDITIONS],
        },
    )


def build_holdout_visual_notebook_figures(
    paper_root: Path,
    output_dir: Path | None = None,
    taxon_stage_root: Path | None = None,
    *,
    visual_model: str = "convnext_base",
    species_ablation: str = "Aporrectodea_longa",
    split_root: Path = Path("."),
    data_root: Path = Path("../petridish-worm-images"),
) -> dict:
    paper_root = Path(paper_root)
    output_dir = Path(output_dir or paper_root / "notebook_holdout_visual_figures")
    source_root = output_dir / "figure_sources"
    taxon_stage_root = Path(taxon_stage_root or paper_root.parent / "adult_taxon_ablation_result")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(paper_root)
    baseline_frame = prepare_baseline_frame(runs)
    age_metric_frame, age_confusion_frame = (
        prepare_developmental_stage_diagnostics(baseline_frame)
    )
    visual_frames = prepare_convnext_visual_frames(runs, visual_model)
    chance_reference = visual_uniform_chance_reference(runs, split_root)
    chance_rows = chance_reference[chance_reference["task"].eq("mean")]
    visual_chance = (
        float(chance_rows.iloc[0]["expected_uniform_macro_f1"])
        if not chance_rows.empty
        else np.nan
    )
    for frame in visual_frames.values():
        frame["chance"] = visual_chance
    visual_example = prepare_visual_ablation_example(
        split_root,
        data_root,
        fallback_gallery=output_dir / "figure_07_representative_transformations.svg",
    )
    taxon_frame = _paper_design_only(
        _model_only(
            prepare_taxon_stage_holdout_frame(
                collect_adult_taxon_metrics(taxon_stage_root)
            ),
            TAXON_MODEL,
            context="Figures 3 through 6",
        ),
        require_loss_recipe=True,
    )
    paired = pair_taxon_metrics(taxon_frame)
    paired, taxon_individual_counts = attach_taxon_individual_counts(
        paired, split_root
    )
    biological_question_frame = prepare_biological_question_frame(
        taxon_frame, split_root
    )
    species_paired = paired[
        paired.get("species", pd.Series(index=paired.index, dtype=str)).astype(str).eq(species_ablation)
    ].copy() if not paired.empty else paired

    figures = {
        "baseline_all_models_tasks": save_baseline_overview(baseline_frame, output_dir, source_root),
        "developmental_stage_diagnostics": save_developmental_stage_diagnostics(
            age_metric_frame,
            age_confusion_frame,
            output_dir,
            source_root,
        ),
        "convnext_visual_ablation": save_convnext_visual_figure(
            visual_frames,
            output_dir,
            source_root,
            model=visual_model,
            resolution_scale="linear",
            chance_reference=chance_reference,
            visual_example=visual_example,
        ),
        "convnext_visual_ablation_resolution_log2": save_convnext_visual_figure(
            visual_frames,
            output_dir,
            source_root,
            model=visual_model,
            resolution_scale="log2",
            chance_reference=chance_reference,
            visual_example=visual_example,
        ),
        "mixed_visual_seed_comparison": save_mixed_visual_seed_figure(
            visual_frames.get("interaction", pd.DataFrame()),
            output_dir,
            source_root,
            model=visual_model,
            chance_reference=chance_reference,
        ),
        "species_ablation": save_paired_estimation_figure(
            species_paired, output_dir, source_root,
            name="figure_03_species_ablation",
            title=(
                f"ConvNeXt-Base species ablation: {species_ablation.replace('_', ' ')} "
                "— independent test cohort"
            ),
            details={
                "species_ablation": species_ablation,
                "model": TAXON_MODEL,
                "split": "test",
                "cohort": TEST_COHORT,
            },
        ),
        "species_ablation_precision_recall_f1": save_species_target_metric_figure(
            species_paired,
            output_dir,
            source_root,
            species_ablation=species_ablation,
        ),
        "biological_transfer_questions": save_biological_question_figure(
            biological_question_frame,
            output_dir,
            source_root,
        ),
        "all_data_ablations": save_all_species_effect_figure(
            paired, output_dir, source_root,
        ),
        "species_ablation_raw_margins": save_species_margin_figure(
            species_paired, output_dir, source_root,
            species_ablation=species_ablation,
        ),
        "all_data_ablations_raw_margins": save_all_species_margin_figure(
            paired, output_dir, source_root,
        ),
        "representative_transformations": save_representative_transformations(
            split_root=split_root,
            data_root=data_root,
            output_dir=output_dir,
            source_root=source_root,
        ),
    }
    manifest = {
        "paper_root": str(paper_root),
        "taxon_stage_root": str(taxon_stage_root),
        "output_dir": str(output_dir),
        "visual_model": visual_model,
        "taxon_model": TAXON_MODEL,
        "species_ablation": species_ablation,
        "split_root": str(split_root),
        "data_root": str(data_root),
        "paper_design": _paper_design_manifest(),
        "baseline_rows": int(len(baseline_frame)),
        "developmental_stage_metric_rows": int(len(age_metric_frame)),
        "developmental_stage_confusion_rows": int(len(age_confusion_frame)),
        "visual_chance_reference": chance_reference.to_dict(orient="records"),
        "taxon_stage_rows": int(len(taxon_frame)),
        "paired_taxon_rows": int(len(paired)),
        "taxon_individual_counts": taxon_individual_counts.to_dict(
            orient="records"
        ),
        "species_paired_rows": int(len(species_paired)),
        "biological_question_rows": int(len(biological_question_frame)),
        "figures": figures,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-result", type=Path, default=Path("paper_result"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--taxon-stage-result", type=Path)
    parser.add_argument("--visual-model", default="convnext_base")
    parser.add_argument("--species-ablation", default="Aporrectodea_longa")
    parser.add_argument("--split-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("../petridish-worm-images"))
    args = parser.parse_args()
    manifest = build_holdout_visual_notebook_figures(
        args.paper_result,
        args.output_dir,
        args.taxon_stage_result,
        visual_model=args.visual_model,
        species_ablation=args.species_ablation,
        split_root=args.split_root,
        data_root=args.data_root,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
