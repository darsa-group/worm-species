#!/usr/bin/env python3
"""Build clearly named paper tables and figures from the full ablation suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
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
    ColourRetention,
    GaussianBlurPercent,
    PatchShuffle,
    ResolutionLoss,
)

DEFAULT_STYLE_PATH = (
    PROJECT_ROOT / "dev" / "paper_report_style.yaml"
)
REPORT_STYLE: dict[str, Any] = {}


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
        condition = dict(config.get("input_condition", {}) or {})
        holdout = dict(config.get("data_holdout", {}) or {})
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
            "holdout": holdout.get("name"),
            "holdout_question": holdout.get("question"),
            "best_epoch": summary.get("best_epoch"),
            "best_val_score": summary.get("best_val_score"),
            "test_mean_macro_f1": summary.get("test_mean_macro_f1"),
            "test_genus_macro_f1": summary.get("test_genus_macro_f1"),
            "test_species_macro_f1": summary.get("test_species_macro_f1"),
            "test_age_macro_f1": summary.get("test_age_macro_f1"),
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
        frame.insert(0, "model", config["model"]["name"])
        frame.insert(1, "seed", config.get("seed"))
        frame["run_dir"] = str(path.parents[1])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_original_cross_conditions(paper_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(
        (paper_root / "runs" / "visual_ablation").rglob(
            "condition_matrix_evaluation/task_metrics.csv"
        )
    ):
        frame = pd.read_csv(path)
        config = _read_json(path.parents[1] / "config.json")
        frame["seed"] = config.get("seed")
        frame["loss_recipe"] = _loss_name(
            dict(
                (config.get("multi_task", {}) or {}).get(
                    "loss_weights", {}
                )
                or {}
            )
        )
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
    return (
        selected.groupby(
            [
                "run_name",
                "model",
                "seed",
                "loss_recipe",
                "train_condition",
                "train_transform",
                "percent",
                "grid_size",
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
    ][["model", "loss_recipe", "test_mean_macro_f1"]].copy()
    if source.empty:
        return frame
    source = (
        source.groupby(["model", "loss_recipe"], as_index=False)[
            "test_mean_macro_f1"
        ]
        .mean()
    )
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
        "learning_rate",
        "batch_size",
        "selection_metric",
        "pretrained",
    ]
    for values, group in source.groupby(group_columns, dropna=False):
        (
            model,
            loss_recipe,
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
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
            "model_selection_criterion": selection_metric,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["backbone", "loss_recipe"]
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
        if transform == "saturation":
            removed = "All colour information"
        elif transform == "patch_shuffle":
            removed = f"Global layout using a {int(row['grid_size'])}x{int(row['grid_size'])} shuffled grid"
        elif transform == "gaussian_blur_percent":
            removed = f"Fine texture; blur severity {int(row['percent'])}%"
        else:
            removed = f"Spatial resolution; information loss {int(row['percent'])}%"
        rows.append({
            "ablation": transform,
            "training_condition": row["condition"],
            "test_condition": "matched condition and original images",
            "removed_information": removed,
            "scientific_question": questions[transform],
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
            ["model", "loss_recipe", "seed", f"test_{task}_macro_f1"]
        ].rename(columns={f"test_{task}_macro_f1": "macro_f1"})
        source["task"] = task
        task_frames.append(source)
    long = pd.concat(task_frames, ignore_index=True)
    return _ci_summary(
        long,
        groups=["model", "loss_recipe", "task"],
        value="macro_f1",
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
        full, on=["model", "seed", "holdout"], how="left"
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
    pivot = frame.pivot_table(index=x, columns=series, values=value, aggfunc="mean")
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.legend(title=series.replace("_", " "), bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=int(_style("dpi", 300)))
    plt.close(ax.figure)


def _save_lines(
    frame: pd.DataFrame,
    *,
    x: str,
    value: str,
    title: str,
    xlabel: str,
    ylabel: str = "Matched-test mean macro-F1",
    path: Path,
) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for model, group in frame.groupby("model"):
        ordered = group.sort_values(x)
        ax.plot(ordered[x], ordered[value], marker="o", label=model)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=int(_style("dpi", 300)))
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
    fig.savefig(
        path, dpi=int(_style("dpi", 300)), bbox_inches="tight"
    )
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
    fig.savefig(
        path, dpi=int(_style("dpi", 300)), bbox_inches="tight"
    )
    plt.close(fig)
    return True


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


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
        ("Blur 25%", GaussianBlurPercent(25)(tensor)),
        ("Blur 50%", GaussianBlurPercent(50)(tensor)),
        ("Blur 100%", GaussianBlurPercent(100)(tensor)),
        ("Resolution loss 25%", ResolutionLoss(25)(tensor)),
        ("Resolution loss 50%", ResolutionLoss(50)(tensor)),
        ("Resolution loss 100%", ResolutionLoss(100)(tensor)),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for ax, (title, transformed) in zip(axes.flat, examples):
        ax.imshow(_tensor_to_image(transformed))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Visual ablations applied to the same individual", fontsize=14)
    fig.tight_layout()
    fig.savefig(
        path, dpi=int(_style("dpi", 300)), bbox_inches="tight"
    )
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
    ax.set_title("Baseline comparison with seed-level 95% t confidence intervals")
    ax.legend(title="Loss recipe", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=int(_style("dpi", 300)))
    plt.close(fig)


def _save_visual_overview(
    visual: pd.DataFrame,
    best_score: float,
    path: Path,
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
    for ax, (transform, x_column, xlabel, title) in zip(axes.flat, panels):
        frame = visual[visual["transform"].eq(transform)]
        for model, group in frame.groupby("model"):
            ordered = group.sort_values(x_column)
            ax.plot(
                ordered[x_column],
                ordered["test_mean_macro_f1"],
                marker="o",
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
    colour_ax = axes.flat[3]
    colour = visual[visual["transform"].eq("saturation")]
    colour_ax.bar(
        colour["model"],
        colour["test_mean_macro_f1"],
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
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Visual-information ablations compared with the best baseline")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path, dpi=int(_style("dpi", 300)))
    plt.close(fig)


def matched_vs_original_table(
    visual: pd.DataFrame,
    original_test: pd.DataFrame,
) -> pd.DataFrame:
    if visual.empty or original_test.empty:
        return pd.DataFrame()
    matched = visual[
        ["model", "seed", "condition", "test_mean_macro_f1"]
    ].rename(columns={"test_mean_macro_f1": "matched_mean_macro_f1"})
    original = original_test[
        ["model", "seed", "train_condition", "mean_macro_f1"]
    ].rename(
        columns={
            "train_condition": "condition",
            "mean_macro_f1": "original_mean_macro_f1",
        }
    )
    matched["seed"] = matched["seed"].astype("string")
    original["seed"] = original["seed"].astype("string")
    result = matched.merge(
        original, on=["model", "seed", "condition"], how="inner"
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
    fig.savefig(path, dpi=int(_style("dpi", 300)))
    plt.close(fig)


def _save_holdout_change(
    joined: pd.DataFrame,
    *,
    best_score: float,
    path: Path,
) -> None:
    if joined.empty:
        return
    summary = (
        joined.groupby(["holdout", "model"], as_index=False)
        .agg(
            full_test_mean_macro_f1=("test_mean_macro_f1", "mean"),
            cohort_target_recall=("target_recall", "mean"),
        )
    )
    summary["change_from_best_baseline"] = (
        summary["full_test_mean_macro_f1"] - best_score
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    pivot_delta = summary.pivot(
        index="holdout",
        columns="model",
        values="change_from_best_baseline",
    )
    pivot_recall = summary.pivot(
        index="holdout",
        columns="model",
        values="cohort_target_recall",
    )
    pivot_delta.plot(kind="bar", ax=axes[0])
    axes[0].axhline(
        0,
        color=_style("best_baseline_colour", "#111111"),
        linewidth=1,
    )
    axes[0].set_ylabel("Full-test macro-F1 change")
    axes[0].set_title("Change relative to the best baseline")
    pivot_recall.plot(kind="bar", ax=axes[1])
    axes[1].set_ylabel("Held-out target recall")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Recognition within the preserved held-out cohort")
    for ax in axes:
        ax.set_xlabel("")
        ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=int(_style("dpi", 300)))
    plt.close(fig)


def build_report(
    paper_root: Path,
    *,
    split_root: Path | None = None,
    data_root: Path | None = None,
    style_path: Path | None = None,
) -> dict:
    configured_style = configure_report_style(style_path)
    paper_root.mkdir(parents=True, exist_ok=True)
    if split_root is None:
        split_root = Path(__file__).resolve().parents[1]
    tables = paper_root / "tables"
    figures = paper_root / "figures"
    summary_dir = paper_root / "summary"
    for directory in (tables, figures, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(paper_root)
    holdouts = collect_holdouts(paper_root)
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
    cross.to_csv(tables / "visual_original_image_metrics.csv", index=False)

    baseline = runs[runs["stage"].eq("baseline")] if not runs.empty else runs
    visual = runs[runs["stage"].eq("visual_ablation")] if not runs.empty else runs
    dataset_composition = dataset_composition_table(split_frames)
    model_training = model_training_table(runs)
    experimental_ablations = experimental_ablation_table(runs)
    holdout_definitions = collect_holdout_definitions(paper_root)
    holdout_joined = joined_holdout_metrics(runs, holdouts)
    baseline_by_task = baseline_task_table(baseline)
    best_baseline, best_baseline_runs = best_baseline_configuration(baseline)
    baseline.to_csv(tables / "baseline_metrics.csv", index=False)
    visual.to_csv(tables / "visual_ablation_metrics.csv", index=False)
    original_test = _original_test_condition_summary(cross)
    original_test.to_csv(
        tables / "visual_trained_ablation_tested_on_original_metrics.csv",
        index=False,
    )
    matched_original = matched_vs_original_table(visual, original_test)
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
    (summary_dir / "best_baseline_configuration.json").write_text(
        json.dumps(best_baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not baseline.empty:
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
    if not visual.empty:
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
        )
        resolution = visual[visual["transform"].eq("resolution_loss")]
        resolution = _with_original_anchor(
            resolution, baseline, x="percent", anchor=0
        )
        _save_lines(
            resolution,
            x="percent",
            value="test_mean_macro_f1",
            title="Spatial-resolution information loss",
            xlabel="Resolution discarded (%)",
            path=figures / "visual_resolution_loss_by_model.png",
        )
        _save_visual_overview(
            visual,
            float(best_baseline.get("mean_test_macro_f1", np.nan)),
            figures / "figure_5_visual_ablation_performance.png",
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
            xlabel="Training resolution discarded (%)",
            ylabel="Original-test mean macro-F1",
            path=figures / "visual_resolution_loss_original_test_by_model.png",
        )
        _save_matched_vs_original(
            matched_original,
            figures / "figure_6_matched_vs_original_performance.png",
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
