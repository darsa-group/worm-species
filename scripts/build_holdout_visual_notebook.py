#!/usr/bin/env python3
"""Build the seven analysis figures used by the ablation notebook.

The figures are deliberately generated outside the notebook as well, so the
same completed-run inputs create the same PNG, PDF, SVG, and source CSV files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from PIL import Image

from scripts.build_adult_taxon_ablation_results import collect_adult_taxon_metrics
from scripts.build_paper_results import _loss_name, collect_runs
from src.worm_species.data.transforms import build_split_transform


PALETTE = (
    "#3B5B92", "#B35C1E", "#3D7A57", "#8A4F7D", "#7A6A2F",
    "#4B7F8C", "#A04747", "#5E5E9A", "#98703D", "#487A73",
)
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
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(str(path))
    for stem, frame in sources.items():
        frame.to_csv(source_dir / f"{stem}.csv", index=False)
    manifest = {
        "figure": name,
        "formats": paths,
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
            values = raw.loc[raw["model"].eq(model), "test_macro_f1"].to_numpy()
            jitter = np.linspace(-width * 0.16, width * 0.16, max(1, len(values)))
            ax.scatter(
                np.full(len(values), x[model_index] + offsets[index]) + jitter,
                values, s=14, color="#222222", alpha=0.5, zorder=4,
            )
        for bar in bars:
            if np.isfinite(bar.get_height()):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
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
        mean_matrix = np.mean(np.stack(matrices), axis=0)
        image = matrix_ax.imshow(mean_matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        matrix_ax.set_title(f"{TASK_LABELS[task]} confusion", fontweight="bold")
        matrix_ax.set_xlabel("Predicted")
        matrix_ax.set_ylabel("True")
        matrix_ax.set_xticks(range(len(expected_labels)), expected_labels, rotation=90, fontsize=6)
        matrix_ax.set_yticks(range(len(expected_labels)), expected_labels, fontsize=6)
    if image is not None:
        fig.colorbar(image, ax=fig.axes[1:], fraction=0.018, pad=0.015, label="Mean row-normalized fraction")
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


def _summarise_line(frame: pd.DataFrame, x: str, series: str | None = None) -> pd.DataFrame:
    groups = [x] if series is None else [series, x]
    return seed_summary(frame, groups, "test_mean_macro_f1")


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
        ax.scatter(raw[x], raw["test_mean_macro_f1"], s=11,
                   color=PALETTE[index % len(PALETTE)], alpha=0.25)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test mean macro-F1")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
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
    resolution["level"] = pd.to_numeric(resolution["percent"], errors="coerce")

    colour = visual[visual["transform"].eq("saturation")].copy()
    colour["level"] = 0.0
    colour["category"] = "Colour removed"
    mask = visual[visual["transform"].eq("binary_mask")].copy()
    mask["level"] = 1.0
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
    gaussian_only = gaussian.copy()
    gaussian_only["series"] = "100% colour (Gaussian only)"
    interaction_parts.append(gaussian_only)
    anchors = {
        "0% colour": colour[colour["level"].eq(0.0)],
        "8×8 patches": patch[patch["level"].eq(8.0)],
        "16×16 patches": patch[patch["level"].eq(16.0)],
    }
    conditions = (
        ("0% colour", "saturation", 0.0),
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


def save_convnext_visual_figure(
    frames: dict[str, pd.DataFrame], output_dir: Path, source_root: Path, *, model: str,
) -> list[str]:
    if not frames:
        return []
    fig = plt.figure(figsize=(15.5, 15.5))
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.42), hspace=0.38, wspace=0.28)
    axes = {
        "gaussian": fig.add_subplot(grid[0, 0]),
        "resolution": fig.add_subplot(grid[0, 1]),
        "colour": fig.add_subplot(grid[1, 0]),
        "patch": fig.add_subplot(grid[1, 1]),
        "interaction": fig.add_subplot(grid[2, :]),
    }
    summaries = []
    specs = (
        ("gaussian", "A. Gaussian blur", "Gaussian blur (%)"),
        ("resolution", "B. Resolution loss", "Resolution loss (%)"),
        ("colour", "C. Colour and silhouette", "Input retained"),
        ("patch", "D. Patch shuffling", "Patch grid"),
    )
    for panel, title, xlabel in specs:
        summary = _draw_line_panel(axes[panel], frames[panel], x="level", title=title, xlabel=xlabel)
        summaries.append(summary.assign(panel=panel))
    axes["colour"].set_xticks(
        [0, 1, 2], ["Colour\nremoved", "Binary mask\nonly", "Original\nRGB"]
    )
    patch_ticks = sorted(frames["patch"]["level"].dropna().unique())
    axes["patch"].set_xticks(patch_ticks, [f"{int(value)}×{int(value)}" for value in patch_ticks])
    interaction_order = (
        "100% colour (Gaussian only)", "0% colour", "8×8 patches", "16×16 patches",
    )
    interaction_summary = _draw_line_panel(
        axes["interaction"], frames["interaction"], x="level",
        title="E. Combined cue ablations", xlabel="Gaussian blur (%)",
        series="series", order=interaction_order,
    )
    summaries.append(interaction_summary.assign(panel="interaction"))
    fig.suptitle(
        f"Visual ablations for {_display_model(model)}",
        fontsize=16, fontweight="bold", y=0.995,
    )
    plot_data = pd.concat(
        [frame.assign(panel=panel) for panel, frame in frames.items()], ignore_index=True
    )
    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.07, right=0.98)
    return _save_bundle(
        fig, output_dir=output_dir, source_root=source_root,
        name="figure_02_convnext_visual_ablation",
        sources={"plot_data": plot_data, "seed_summary": pd.concat(summaries, ignore_index=True)},
        details={
            "model": model,
            "metric": "test mean macro-F1",
            "split": "test",
            "uncertainty": "95% t interval across seeds",
            "interaction_series": list(interaction_order),
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
    values = [column for column in ("macro_f1", "target_recall", "chance") if column in selected]
    baseline = selected[selected["system"].eq("Full-data baseline")][[*keys, *values]].copy()
    ablated = selected[selected["system"].eq("Ablated training")][[*keys, *values]].copy()
    if baseline.empty or ablated.empty:
        return pd.DataFrame()
    baseline = baseline.groupby(keys, dropna=False, as_index=False)[values].mean()
    ablated = ablated.groupby(keys, dropna=False, as_index=False)[values].mean()
    baseline = baseline.rename(columns={column: f"baseline_{column}" for column in values})
    ablated = ablated.rename(columns={column: f"ablated_{column}" for column in values})
    paired = ablated.merge(baseline, on=keys, how="inner")
    for metric in ("macro_f1", "target_recall"):
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
        "margin_additive_check_error", "d_total", "d_ablation", "d_retained",
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


def _recall_seed_data(paired: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model", "seed", "holdout", "cohort", "task", "genus", "species",
        "stage", "combo_label", "chance", "baseline_target_recall",
        "ablated_target_recall",
    ]
    return paired[[column for column in columns if column in paired]].copy()


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
    ax.plot([0.0, retained], [y, y], color="#3D7A57", linewidth=5.0,
            solid_capstyle="round", alpha=0.75, zorder=2)
    ax.plot([retained, total], [y, y], color="#A04747", linewidth=5.0,
            solid_capstyle="round", alpha=0.75, zorder=2)
    ax.scatter(0.0, y, marker="|", s=115, color="#555555", linewidths=1.5,
               zorder=4)
    ax.scatter(retained, y, marker="o", s=54, color="#B35C1E",
               edgecolors="white", linewidths=0.8, zorder=4)
    ax.scatter(total, y, marker="D", s=52, facecolors="white",
               edgecolors="#3B5B92", linewidths=1.5, zorder=4)
    if annotate:
        ax.annotate(
            f"retained {retained:.2f}", (retained, y), xytext=(0, -15),
            textcoords="offset points", ha="center", va="top", fontsize=7.5,
            color="#8A4A17",
        )
        ax.annotate(
            f"total {total:.2f}", (total, y), xytext=(0, 13),
            textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
            color="#29456F",
        )
        ax.annotate(
            f"lost {lost:.2f}", ((retained + total) / 2.0, y),
            xytext=(0, 5), textcoords="offset points", ha="center",
            va="bottom", fontsize=7.2, color="#823D3D",
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
        paired, identity_columns=("stage",)
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
            ax.set_yticklabels(stages, fontsize=11, fontweight="bold")
            ax.set_ylabel("Life stage", labelpad=12)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylim(len(stages) - 0.5, -0.5)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], marker="|", color="#555555", linestyle="none",
                   markersize=12, label="Chance = 0"),
            Line2D([0], [0], marker="o", color="#B35C1E", linestyle="none",
                   markeredgecolor="white", label=r"Ablated: $d_{retained}$"),
            Line2D([0], [0], marker="D", color="#3B5B92", linestyle="none",
                   markerfacecolor="white", label=r"Normal: $d_{total}$"),
            Line2D([0], [0], color="#A04747", linewidth=5,
                   label=r"Gap: $d_{ablation}$"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=4, frameon=False,
    )
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.99)
    fig.text(
        0.5, 0.018,
        r"Target recall only. All positions use the normal model's sample SD: "
        r"$d_{total}=d_{ablation}+d_{retained}$. The publication design uses 30 seeds per stage; guides mark $d=0.8$ and $d=2$.",
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
            "inference": "descriptive effect sizes; no p-value or confidence interval",
            "cohort": TEST_COHORT,
            "split": "test",
            **_paper_design_manifest(),
            **details,
        },
    )


def _raw_margin_summary(effect_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "combo_label", "genus", "species", "stage", "task", "n_seeds",
        "normal_mean_recall", "ablated_mean_recall", "chance",
        "number_of_classes", "chance_method", "m_total", "m_lost",
        "m_retained", "margin_additive_check_error",
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
    ax.plot([0.0, retained], [y, y], color="#3D7A57", linewidth=5.0,
            solid_capstyle="round", alpha=0.75, zorder=2)
    ax.plot([retained, total], [y, y], color="#A04747", linewidth=5.0,
            solid_capstyle="round", alpha=0.75, zorder=2)
    ax.scatter(0.0, y, marker="|", s=115, color="#555555", linewidths=1.5,
               zorder=4)
    ax.scatter(retained, y, marker="o", s=54, color="#B35C1E",
               edgecolors="white", linewidths=0.8, zorder=4)
    ax.scatter(total, y, marker="D", s=52, facecolors="white",
               edgecolors="#3B5B92", linewidths=1.5, zorder=4)
    if annotate:
        ax.annotate(
            f"retained {retained:+.1%}", (retained, y), xytext=(0, -15),
            textcoords="offset points", ha="center", va="top", fontsize=7.5,
            color="#8A4A17",
        )
        ax.annotate(
            f"total {total:+.1%}", (total, y), xytext=(0, 13),
            textcoords="offset points", ha="center", va="bottom", fontsize=7.5,
            color="#29456F",
        )
        ax.annotate(
            f"lost {lost:+.1%}", ((retained + total) / 2.0, y),
            xytext=(0, 5), textcoords="offset points", ha="center",
            va="bottom", fontsize=7.2, color="#823D3D",
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
        "inference": "descriptive margins; no p-value or confidence interval",
        "cohort": TEST_COHORT,
        "split": "test",
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
        paired, identity_columns=("stage",)
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
            ax.set_yticklabels(stages, fontsize=11, fontweight="bold")
            ax.set_ylabel("Life stage", labelpad=12)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylim(len(stages) - 0.5, -0.5)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], marker="|", color="#555555", linestyle="none",
                   markersize=12, label="Chance = 0 margin"),
            Line2D([0], [0], marker="o", color="#B35C1E", linestyle="none",
                   markeredgecolor="white", label=r"Ablated: $M_{retained}$"),
            Line2D([0], [0], marker="D", color="#3B5B92", linestyle="none",
                   markerfacecolor="white", label=r"Normal: $M_{total}$"),
            Line2D([0], [0], color="#A04747", linewidth=5,
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
        r"ConvNeXt-Base target recall. $M_{total}=M_{lost}+M_{retained}$; chance is derived as $1/K$ from each task's saved class map.",
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
        identity_columns=("combo_label", "genus", "species", "stage"),
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
            ax.set_yticklabels(combinations, fontsize=8.5)
            ax.set_ylabel("Species-stage holdout", labelpad=12)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylim(len(combinations) - 0.5, -0.5)
    from matplotlib.lines import Line2D

    legend = [
        Line2D([0], [0], marker="|", color="#555555", linestyle="none",
               markersize=12, label="Chance = 0"),
        Line2D([0], [0], marker="o", color="#B35C1E", linestyle="none",
               markeredgecolor="white", label=r"Ablated: $d_{retained}$"),
        Line2D([0], [0], marker="D", color="#3B5B92", linestyle="none",
               markerfacecolor="white", label=r"Normal: $d_{total}$"),
        Line2D([0], [0], color="#A04747", linewidth=5,
               label=r"Gap: $d_{ablation}$"),
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.935),
        ncol=4, frameon=False,
    )
    fig.suptitle(
        "ConvNeXt-Base recall capacity by species and life stage — independent test cohort",
        fontsize=16, fontweight="bold", y=0.992,
    )
    fig.text(
        0.5, 0.018,
        r"Target recall only. Chance is 0; the ablated point is $d_{retained}$, the normal point is $d_{total}$, and their gap is $d_{ablation}$. All use the normal model's seed SD across the 30-seed publication design.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.18, right=0.98, wspace=0.12)
    return _save_bundle(
        fig, output_dir=output_dir, source_root=source_root,
        name="figure_04_all_data_ablations",
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
            "inference": "descriptive effect sizes; no p-value or confidence interval",
            "cohort": TEST_COHORT,
            "split": "test",
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
            ax.set_yticklabels(combinations, fontsize=8.5)
            ax.set_ylabel("Species-stage holdout", labelpad=12)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0, 0].set_ylim(len(combinations) - 0.5, -0.5)
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[
            Line2D([0], [0], marker="|", color="#555555", linestyle="none",
                   markersize=12, label="Chance = 0 margin"),
            Line2D([0], [0], marker="o", color="#B35C1E", linestyle="none",
                   markeredgecolor="white", label=r"Ablated: $M_{retained}$"),
            Line2D([0], [0], marker="D", color="#3B5B92", linestyle="none",
                   markerfacecolor="white", label=r"Normal: $M_{total}$"),
            Line2D([0], [0], color="#A04747", linewidth=5,
                   label=r"Gap: $M_{lost}$"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=4, frameon=False,
    )
    fig.suptitle(
        "Raw ConvNeXt-Base recall margins by species and life stage — independent test cohort",
        fontsize=16, fontweight="bold", y=0.992,
    )
    fig.text(
        0.5, 0.018,
        r"$M_{total}=M_{lost}+M_{retained}$. Chance is derived separately for each task as $1/K$ under uniform random prediction.",
        ha="center", fontsize=9,
    )
    fig.subplots_adjust(top=0.87, bottom=0.08, left=0.18, right=0.98, wspace=0.12)
    return _save_bundle(
        fig, output_dir=output_dir, source_root=source_root,
        name="figure_06_all_data_ablations_raw_margins",
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
    visual_frames = prepare_convnext_visual_frames(runs, visual_model)
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
    species_paired = paired[
        paired.get("species", pd.Series(index=paired.index, dtype=str)).astype(str).eq(species_ablation)
    ].copy() if not paired.empty else paired

    figures = {
        "baseline_all_models_tasks": save_baseline_overview(baseline_frame, output_dir, source_root),
        "convnext_visual_ablation": save_convnext_visual_figure(
            visual_frames, output_dir, source_root, model=visual_model,
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
        "taxon_stage_rows": int(len(taxon_frame)),
        "paired_taxon_rows": int(len(paired)),
        "species_paired_rows": int(len(species_paired)),
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
