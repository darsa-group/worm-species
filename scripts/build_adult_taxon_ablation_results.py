#!/usr/bin/env python3
"""Aggregate exhaustive Adult taxon holdouts and render paired h-loss figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STYLE = PROJECT_ROOT / "dev" / "paper_report_style.yaml"
TASKS = ("genus", "species", "age")
TASK_LABELS = {
    "genus": "Genus recall",
    "species": "Species recall",
    "age": "Adult-stage recall",
}
COHORTS = ("development_withheld", "independent_test")
COHORT_LABELS = {
    "development_withheld": "Removed development cohort",
    "independent_test": "Independent matching test cohort",
}
T_CRITICAL = {
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _slug(value: str) -> str:
    return "_".join(
        part for part in "".join(
            character.lower() if character.isalnum() else " "
            for character in value
        ).split()
        if part
    )


def configure_style(path: Path | None) -> dict[str, Any]:
    style_path = path or DEFAULT_STYLE
    style = (
        yaml.safe_load(style_path.read_text(encoding="utf-8")) or {}
        if style_path.is_file()
        else {}
    )
    if not isinstance(style, dict):
        raise ValueError(f"Report style must be a mapping: {style_path}")
    palette = style.get(
        "palette",
        ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"],
    )
    if not isinstance(palette, list) or not palette:
        raise ValueError("Report style palette must be a non-empty list")
    plt.rcParams.update({
        "figure.dpi": int(style.get("dpi", 300)),
        "savefig.dpi": int(style.get("dpi", 300)),
        "font.size": float(style.get("font_size", 10)),
        "axes.grid": False,
    })
    style["palette"] = palette
    return style


def _save_formats(fig: plt.Figure, path: Path, dpi: int) -> list[str]:
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        target = path.with_suffix(f".{suffix}")
        fig.savefig(target, dpi=dpi, bbox_inches="tight")
        outputs.append(str(target))
    plt.close(fig)
    return outputs


def _hierarchy_weight(config: dict[str, Any]) -> float:
    hierarchy = (
        (config.get("multi_task", {}) or {}).get("hierarchy_loss", {})
        or {}
    )
    return (
        float(hierarchy.get("weight", 0.0))
        if bool(hierarchy.get("enabled", False))
        else 0.0
    )


def _control_definitions(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions = (
        ((config.get("evaluation", {}) or {}).get(
            "data_holdout_controls", {}
        ) or {}).get("definitions", [])
    )
    return {
        str(item["name"]): item
        for item in definitions
        if isinstance(item, dict) and item.get("name")
    }


def _collect_stage(
    paper_root: Path,
    *,
    stage: str,
    evaluation_directory: str,
    training_regime: str,
) -> pd.DataFrame:
    frames = []
    pattern = f"{evaluation_directory}/task_metrics.csv"
    for path in sorted((paper_root / "runs" / stage).rglob(pattern)):
        frame = _safe_read_csv(path)
        if frame.empty:
            continue
        run_dir = path.parents[1]
        config = _read_json(run_dir / "config.json")
        if not config:
            continue
        label_maps = _read_json(run_dir / "label_to_index_by_task.json")
        holdout = config.get("data_holdout", {}) or {}
        controls = _control_definitions(config)
        weight = _hierarchy_weight(config)
        frame = frame.copy()
        frame["model"] = (config.get("model", {}) or {}).get("name")
        frame["seed"] = config.get("seed")
        frame["hierarchy_loss_weight"] = weight
        frame["hierarchy_loss_label"] = f"h={weight:g}"
        frame["training_regime"] = training_regime
        frame["run_dir"] = str(run_dir)
        for index, row in frame.iterrows():
            definition = (
                holdout
                if training_regime == "adult_combo_withheld"
                else controls.get(str(row.get("holdout")), {})
            )
            where = (
                definition.get("evaluation_where")
                or definition.get("where")
                or {}
            )
            frame.at[index, "genus"] = where.get("genus")
            frame.at[index, "species"] = where.get("species")
            frame.at[index, "stage"] = where.get("age")
        frame["combo_label"] = (
            frame["species"].fillna(frame["holdout"])
            .astype(str)
            .str.replace("_", " ", regex=False)
        )
        frame["chance"] = frame["task"].map({
            task: 1.0 / len(labels)
            for task, labels in label_maps.items()
            if isinstance(labels, dict) and labels
        })
        frame["target_recall_image"] = frame.get(
            "target_recall_image",
            frame.get("target_recall", np.nan),
        )
        if "target_recall" in frame:
            frame["target_recall_image"] = frame[
                "target_recall_image"
            ].fillna(frame["target_recall"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_adult_taxon_metrics(paper_root: Path) -> pd.DataFrame:
    controls = _collect_stage(
        paper_root,
        stage="adult_taxon_baseline",
        evaluation_directory="data_holdout_control_evaluation",
        training_regime="full_data_control",
    )
    withheld = _collect_stage(
        paper_root,
        stage="adult_taxon_holdouts",
        evaluation_directory="data_holdout_evaluation",
        training_regime="adult_combo_withheld",
    )
    frames = [frame for frame in (controls, withheld) if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def seed_summary(
    frame: pd.DataFrame,
    *,
    groups: list[str],
    value: str,
) -> pd.DataFrame:
    columns = [*groups, "mean", "std", "n_seeds", "ci95"]
    if frame.empty or value not in frame:
        return pd.DataFrame(columns=columns)
    clean = frame.dropna(subset=[value]).copy()
    if clean.empty:
        return pd.DataFrame(columns=columns)
    if "seed" in clean and "seed" not in groups:
        clean = (
            clean.groupby([*groups, "seed"], dropna=False, as_index=False)
            [value]
            .mean()
        )
    summary = (
        clean.groupby(groups, dropna=False)[value]
        .agg(mean="mean", std="std", n_seeds="count")
        .reset_index()
    )
    summary["ci95"] = [
        (
            T_CRITICAL.get(int(count) - 1, 1.96)
            * float(std)
            / math.sqrt(int(count))
            if int(count) > 1 and pd.notna(std)
            else np.nan
        )
        for std, count in zip(summary["std"], summary["n_seeds"])
    ]
    return summary[columns]


def paired_ablation_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model", "seed", "hierarchy_loss_weight", "holdout", "cohort",
        "task", "genus", "species", "stage", "combo_label",
    ]
    if metrics.empty:
        return pd.DataFrame()
    controls = metrics[
        metrics["training_regime"].eq("full_data_control")
    ][[*columns, "target_recall_image", "chance"]].rename(columns={
        "target_recall_image": "full_data_target_recall",
        "chance": "full_data_chance",
    })
    withheld = metrics[
        metrics["training_regime"].eq("adult_combo_withheld")
    ][[
        *columns,
        "target_recall_image",
        "class_supported_by_training_head",
        "chance",
        "run_dir",
    ]].rename(columns={
        "target_recall_image": "withheld_target_recall",
        "chance": "withheld_chance",
    })
    paired = withheld.merge(controls, on=columns, how="inner")
    paired["target_recall_difference"] = (
        paired["withheld_target_recall"]
        - paired["full_data_target_recall"]
    )
    return paired


def observed_adult_combinations(split_root: Path | None) -> pd.DataFrame:
    columns = [
        "genus", "species", "stage", "holdout",
        "training_images", "training_individuals",
        "validation_images", "validation_individuals",
        "test_images", "test_individuals",
    ]
    if split_root is None:
        return pd.DataFrame(columns=columns)
    candidates = (split_root, split_root / "split_csv")
    split_dir = next(
        (
            candidate for candidate in candidates
            if all((candidate / name).is_file() for name in (
                "train_split.csv", "val_split.csv", "test_split.csv"
            ))
        ),
        None,
    )
    if split_dir is None:
        return pd.DataFrame(columns=columns)
    frames = {
        "training": pd.read_csv(split_dir / "train_split.csv"),
        "validation": pd.read_csv(split_dir / "val_split.csv"),
        "test": pd.read_csv(split_dir / "test_split.csv"),
    }
    keys = ["genus", "species_label", "life_stage"]
    result = None
    for split, frame in frames.items():
        adult = frame[
            frame["life_stage"].astype(str).eq("Adult")
            & frame["genus"].notna()
            & frame["species_label"].notna()
        ]
        counts = (
            adult.groupby(keys, as_index=False)
            .agg(
                **{
                    f"{split}_images": ("barcode", "size"),
                    f"{split}_individuals": ("barcode", "nunique"),
                }
            )
        )
        result = counts if result is None else result.merge(
            counts, on=keys, how="inner"
        )
    if result is None:
        return pd.DataFrame(columns=columns)
    result = result.rename(columns={
        "species_label": "species",
        "life_stage": "stage",
    })
    result["holdout"] = "adult_" + result["species"].map(_slug)
    return result[columns].sort_values(["genus", "species"])


def _series_metadata(
    frame: pd.DataFrame,
    *,
    compare_hloss: bool,
    palette: list[str],
) -> list[tuple[str, float, str, str, str]]:
    models = sorted(frame["model"].dropna().astype(str).unique())
    weights = (
        sorted(frame["hierarchy_loss_weight"].dropna().astype(float).unique())
        if compare_hloss else [0.0]
    )
    rows = []
    for model_index, model in enumerate(models):
        for weight in weights:
            rows.append((
                model,
                weight,
                f"{model} · h={weight:g}" if compare_hloss else model,
                palette[model_index % len(palette)],
                "s" if weight > 0 else "o",
            ))
    return rows


def _write_figure_sources(
    source_root: Path,
    stem: str,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    extra: dict[str, Any],
) -> None:
    directory = source_root / stem
    directory.mkdir(parents=True, exist_ok=True)
    data.to_csv(directory / "plot_data.csv", index=False)
    summary.to_csv(directory / "seed_summary.csv", index=False)
    (directory / "manifest.json").write_text(
        json.dumps({
            "figure": stem,
            "plot_data": "plot_data.csv",
            "seed_summary": "seed_summary.csv",
            "rows": int(len(data)),
            **extra,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _plot_matrix(
    *,
    frame: pd.DataFrame,
    value: str,
    figures_dir: Path,
    source_root: Path,
    stem: str,
    title: str,
    xlabel: str,
    compare_hloss: bool,
    style: dict[str, Any],
    chance_column: str | None = None,
    zero_reference: bool = False,
) -> bool:
    selected = frame.copy()
    if (
        selected.empty
        or value not in selected.columns
        or "hierarchy_loss_weight" not in selected.columns
    ):
        return False
    if not compare_hloss:
        selected = selected[
            selected["hierarchy_loss_weight"].eq(0.0)
        ]
    if selected.empty:
        return False
    groups = [
        "cohort", "task", "holdout", "combo_label", "model",
        "hierarchy_loss_weight",
    ]
    summary = seed_summary(selected, groups=groups, value=value)
    if summary.empty:
        return False
    combinations = (
        selected[["genus", "species", "combo_label"]]
        .drop_duplicates()
        .sort_values(["genus", "species"])["combo_label"]
        .tolist()
    )
    series = _series_metadata(
        selected,
        compare_hloss=compare_hloss,
        palette=style["palette"],
    )
    fig, axes = plt.subplots(
        len(COHORTS),
        len(TASKS),
        figsize=(20, max(11, len(combinations) * 1.25)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    y = np.arange(len(combinations), dtype=float)
    offsets = np.linspace(-0.32, 0.32, max(1, len(series)))
    for row_index, cohort in enumerate(COHORTS):
        for column_index, task in enumerate(TASKS):
            ax = axes[row_index, column_index]
            subset = selected[
                selected["cohort"].eq(cohort)
                & selected["task"].eq(task)
            ]
            summary_subset = summary[
                summary["cohort"].eq(cohort)
                & summary["task"].eq(task)
            ]
            for series_index, (
                model, weight, label, colour, marker
            ) in enumerate(series):
                current = summary_subset[
                    summary_subset["model"].eq(model)
                    & summary_subset["hierarchy_loss_weight"].eq(weight)
                ].set_index("combo_label")
                positions = y + offsets[series_index]
                means = np.asarray([
                    current["mean"].get(combo, np.nan)
                    for combo in combinations
                ], dtype=float)
                errors = np.asarray([
                    current["ci95"].get(combo, np.nan)
                    for combo in combinations
                ], dtype=float)
                ax.errorbar(
                    means,
                    positions,
                    xerr=errors,
                    fmt=marker,
                    markersize=4.5,
                    capsize=2,
                    color=colour,
                    markerfacecolor=("none" if weight > 0 else colour),
                    label=label,
                )
                raw = subset[
                    subset["model"].eq(model)
                    & subset["hierarchy_loss_weight"].eq(weight)
                ]
                for combo_index, combo in enumerate(combinations):
                    observations = raw[
                        raw["combo_label"].eq(combo)
                    ][value].dropna()
                    ax.scatter(
                        observations,
                        np.full(len(observations), positions[combo_index]),
                        s=9,
                        alpha=0.22,
                        color=colour,
                    )
            if zero_reference:
                ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
            elif chance_column and chance_column in subset:
                chance = subset[chance_column].dropna()
                if not chance.empty:
                    ax.axvline(
                        float(chance.mean()),
                        color=style.get("chance_colour", "#777777"),
                        linestyle=":",
                        linewidth=1.2,
                    )
            ax.set_title(
                f"{COHORT_LABELS[cohort]}\n{TASK_LABELS[task]}"
            )
            ax.grid(axis="x", alpha=0.2)
            ax.set_yticks(y, combinations if column_index == 0 else [])
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)
    if zero_reference:
        for ax in axes.ravel():
            ax.set_xlim(-1, 1)
    else:
        for ax in axes.ravel():
            ax.set_xlim(0, 1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(5, max(1, len(series))),
        frameon=False,
    )
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    _write_figure_sources(
        source_root,
        stem,
        selected,
        summary,
        extra={
            "value": value,
            "uncertainty_unit": "seed",
            "confidence_interval": "95% t interval",
            "seed_points_shown": True,
            "hierarchy_comparison": compare_hloss,
        },
    )
    _save_formats(
        fig,
        figures_dir / stem,
        int(style.get("dpi", 300)),
    )
    return True


def build_adult_taxon_report(
    paper_root: Path,
    *,
    split_root: Path | None = None,
    style_path: Path | None = None,
) -> dict[str, Any]:
    style = configure_style(style_path)
    tables = paper_root / "tables"
    figures = paper_root / "figures"
    figure_sources = paper_root / "figure_sources"
    summary_dir = paper_root / "summary"
    latex = paper_root / "latex"
    for directory in (tables, figures, figure_sources, summary_dir, latex):
        directory.mkdir(parents=True, exist_ok=True)

    metrics = collect_adult_taxon_metrics(paper_root)
    paired = paired_ablation_differences(metrics)
    inventory = observed_adult_combinations(split_root)
    metrics.to_csv(tables / "all_adult_taxon_metrics.csv", index=False)
    paired.to_csv(
        tables / "paired_adult_taxon_ablation_differences.csv",
        index=False,
    )
    inventory.to_csv(
        tables / "adult_taxon_combination_inventory.csv",
        index=False,
    )
    summary_groups = [
        "training_regime", "cohort", "task", "holdout", "combo_label",
        "model", "hierarchy_loss_weight",
    ]
    metric_summary = seed_summary(
        metrics,
        groups=summary_groups,
        value="target_recall_image",
    )
    difference_summary = seed_summary(
        paired,
        groups=[
            "cohort", "task", "holdout", "combo_label", "model",
            "hierarchy_loss_weight",
        ],
        value="target_recall_difference",
    )
    metric_summary.to_csv(
        tables / "adult_taxon_target_recall_seed_summary.csv",
        index=False,
    )
    difference_summary.to_csv(
        tables / "adult_taxon_ablation_difference_seed_summary.csv",
        index=False,
    )
    metric_summary.to_latex(
        latex / "adult_taxon_target_recall_seed_summary.tex",
        index=False,
        float_format="%.3f",
    )
    difference_summary.to_latex(
        latex / "adult_taxon_ablation_difference_seed_summary.tex",
        index=False,
        float_format="%.3f",
    )

    withheld = metrics[
        metrics["training_regime"].eq("adult_combo_withheld")
    ] if not metrics.empty else metrics
    figures_created = {}
    for stem, compare_hloss in (
        ("figure_01_adult_taxon_target_recall", False),
        ("figure_01_adult_taxon_target_recall_hloss_comparison", True),
    ):
        figures_created[stem] = _plot_matrix(
            frame=withheld,
            value="target_recall_image",
            figures_dir=figures,
            source_root=figure_sources,
            stem=stem,
            title=(
                "Adult genus–species combination holdouts"
                + (": hierarchy loss 0 versus 0.2" if compare_hloss else " (h=0)")
            ),
            xlabel="Target-class recall",
            compare_hloss=compare_hloss,
            style=style,
            chance_column="chance",
        )
    for stem, compare_hloss in (
        ("figure_02_adult_taxon_ablation_effect", False),
        ("figure_02_adult_taxon_ablation_effect_hloss_comparison", True),
    ):
        figures_created[stem] = _plot_matrix(
            frame=paired,
            value="target_recall_difference",
            figures_dir=figures,
            source_root=figure_sources,
            stem=stem,
            title=(
                "Adult-combination data-ablation effect relative to "
                "matched full-data controls"
                + (": hierarchy loss 0 versus 0.2" if compare_hloss else " (h=0)")
            ),
            xlabel="Recall difference: withheld − full-data control",
            compare_hloss=compare_hloss,
            style=style,
            zero_reference=True,
        )

    configured_holdouts = sorted(
        metrics["holdout"].dropna().astype(str).unique().tolist()
    ) if not metrics.empty else []
    expected_holdouts = sorted(inventory["holdout"].tolist())
    manifest = {
        "schema_version": 1,
        "paper_result": str(paper_root),
        "completed_metric_runs": int(
            metrics["run_dir"].nunique() if not metrics.empty else 0
        ),
        "configured_holdouts": configured_holdouts,
        "expected_holdouts_from_splits": expected_holdouts,
        "exhaustive_split_coverage": (
            configured_holdouts == expected_holdouts
            if expected_holdouts else None
        ),
        "hierarchy_loss_weights": (
            sorted(metrics["hierarchy_loss_weight"].dropna().unique().tolist())
            if not metrics.empty else []
        ),
        "figures": figures_created,
        "figure_formats": ["png", "pdf", "svg"],
        "uncertainty_unit": "seed",
        "formal_significance_testing": False,
    }
    (summary_dir / "adult_taxon_report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-result",
        type=Path,
        default=Path("adult_taxon_ablation_result"),
    )
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT)
    # Accepted for compatibility with the shared pipeline report-job contract.
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--style", type=Path, default=DEFAULT_STYLE)
    args = parser.parse_args()
    manifest = build_adult_taxon_report(
        args.paper_result,
        split_root=args.split_root,
        style_path=args.style,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
