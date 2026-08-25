#!/usr/bin/env python3
"""Aggregate Adult/Juvenile taxon holdouts and render comparisons."""

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
COHORTS = ("development_withheld", "independent_test")
TASK_LABELS = {
    "genus": "Genus recall",
    "species": "Species recall",
    "age": "Developmental-stage recall",
}
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


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _slug(value: str) -> str:
    return "_".join(
        "".join(
            character.lower() if character.isalnum() else " "
            for character in value
        ).split()
    )


def _hierarchy_weight(config: dict[str, Any]) -> float:
    hierarchy = (
        (config.get("multi_task", {}) or {}).get("hierarchy_loss", {})
        or {}
    )
    if not bool(hierarchy.get("enabled", False)):
        return 0.0
    return float(hierarchy.get("weight", 0.0))


def _loss_recipe(config: dict[str, Any]) -> tuple[str, dict[str, float]]:
    weights = dict(
        ((config.get("multi_task", {}) or {}).get("loss_weights", {}) or {})
    )
    numeric = {
        task: float(weights.get(task, 0.0))
        for task in ("genus", "species", "age")
    }
    label = "_".join(
        f"{task}-{numeric[task]:g}"
        for task in ("genus", "species", "age")
    )
    return label, numeric


def _control_definitions(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
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
    frames: list[pd.DataFrame] = []
    pattern = f"{evaluation_directory}/task_metrics.csv"
    for path in sorted((paper_root / "runs" / stage).rglob(pattern)):
        frame = _read_csv(path)
        if frame.empty:
            continue
        enriched = _read_csv(
            path.with_name("target_class_metrics_full_test.csv")
        )
        if not enriched.empty:
            merge_keys = [
                column
                for column in ("holdout", "cohort", "task", "target_label")
                if column in frame and column in enriched
            ]
            added_columns = [
                column
                for column in enriched
                if column not in merge_keys and column not in frame
            ]
            if merge_keys and added_columns:
                frame = frame.merge(
                    enriched[[*merge_keys, *added_columns]],
                    on=merge_keys,
                    how="left",
                    validate="many_to_one",
                )
        run_dir = path.parents[1]
        config = _read_json(run_dir / "config.json")
        if not config:
            continue
        maps = _read_json(run_dir / "label_to_index_by_task.json")
        controls = _control_definitions(config)
        configured_holdout = config.get("data_holdout", {}) or {}
        weight = _hierarchy_weight(config)
        loss_recipe, loss_weights = _loss_recipe(config)
        frame = frame.copy()
        frame["model"] = (config.get("model", {}) or {}).get("name")
        frame["seed"] = config.get("seed")
        frame["hierarchy_loss_weight"] = weight
        frame["hierarchy_loss_label"] = f"h={weight:g}"
        frame["loss_recipe"] = loss_recipe
        frame["genus_weight"] = loss_weights["genus"]
        frame["species_weight"] = loss_weights["species"]
        frame["age_weight"] = loss_weights["age"]
        frame["training_regime"] = training_regime
        frame["run_dir"] = str(run_dir)
        for index, row in frame.iterrows():
            definition = (
                configured_holdout
                if training_regime != "full_data_control"
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
            frame["stage"].fillna("Unknown").astype(str)
            + ": "
            + frame["species"].fillna(frame["holdout"])
            .astype(str).str.replace("_", " ", regex=False)
        )
        frame["chance"] = frame["task"].map({
            task: 1.0 / len(labels)
            for task, labels in maps.items()
            if isinstance(labels, dict) and labels
        })
        if "target_recall_image" not in frame:
            frame["target_recall_image"] = frame.get(
                "target_recall", np.nan
            )
        elif "target_recall" in frame:
            frame["target_recall_image"] = frame[
                "target_recall_image"
            ].fillna(frame["target_recall"])
        if "class_supported_by_training_head" not in frame:
            frame["class_supported_by_training_head"] = True
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
    transfer = _collect_stage(
        paper_root,
        stage="biological_transfer_holdouts",
        evaluation_directory="data_holdout_evaluation",
        training_regime="biological_transfer_withheld",
    )
    frames = [
        item for item in (controls, withheld, transfer) if not item.empty
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def observed_taxon_stage_combinations(
    split_root: Path | None,
) -> pd.DataFrame:
    columns = [
        "genus", "species", "stage", "holdout",
        "training_images", "training_individuals",
        "validation_images", "validation_individuals",
        "test_images", "test_individuals",
    ]
    if split_root is None:
        return pd.DataFrame(columns=columns)
    split_dir = next(
        (
            candidate
            for candidate in (split_root, split_root / "split_csv")
            if all((candidate / name).is_file() for name in (
                "train_split.csv", "val_split.csv", "test_split.csv"
            ))
        ),
        None,
    )
    if split_dir is None:
        return pd.DataFrame(columns=columns)
    result: pd.DataFrame | None = None
    for split, filename in (
        ("training", "train_split.csv"),
        ("validation", "val_split.csv"),
        ("test", "test_split.csv"),
    ):
        frame = pd.read_csv(split_dir / filename)
        selected = frame[
            frame["life_stage"].astype(str).isin(("Adult", "Juvenile"))
            & frame["genus"].notna()
            & frame["species_label"].notna()
        ]
        counts = (
            selected.groupby(
                ["genus", "species_label", "life_stage"], as_index=False
            )
            .agg(**{
                f"{split}_images": ("barcode", "size"),
                f"{split}_individuals": ("barcode", "nunique"),
            })
        )
        result = (
            counts if result is None
            else result.merge(
                counts,
                on=["genus", "species_label", "life_stage"],
                how="inner",
            )
        )
    if result is None:
        return pd.DataFrame(columns=columns)
    result = result.rename(columns={
        "species_label": "species",
        "life_stage": "stage",
    })
    result["holdout"] = (
        result["stage"].astype(str).str.lower()
        + "_"
        + result["species"].map(_slug)
    )
    return result[columns].sort_values(["stage", "genus", "species"])


def observed_adult_combinations(split_root: Path | None) -> pd.DataFrame:
    """Backward-compatible Adult-only view of the combined inventory."""
    frame = observed_taxon_stage_combinations(split_root)
    return frame[frame["stage"].eq("Adult")].reset_index(drop=True)


def paired_ablation_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    keys = [
        "model", "seed", "hierarchy_loss_weight", "holdout", "cohort",
        "task", "genus", "species", "stage", "combo_label",
    ]
    controls = metrics[
        metrics["training_regime"].eq("full_data_control")
    ][[*keys, "target_recall_image", "chance"]].rename(columns={
        "target_recall_image": "full_data_target_recall",
        "chance": "full_data_chance",
    })
    withheld = metrics[
        metrics["training_regime"].eq("adult_combo_withheld")
    ][[
        *keys,
        "target_recall_image",
        "class_supported_by_training_head",
        "chance",
        "run_dir",
    ]].rename(columns={
        "target_recall_image": "withheld_target_recall",
        "chance": "withheld_chance",
    })
    paired = withheld.merge(controls, on=keys, how="inner")
    paired["target_recall_difference"] = (
        paired["withheld_target_recall"]
        - paired["full_data_target_recall"]
    )
    return paired


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


def _configure_style(path: Path | None) -> dict[str, Any]:
    style_path = path or DEFAULT_STYLE
    style = (
        yaml.safe_load(style_path.read_text(encoding="utf-8")) or {}
        if style_path.is_file() else {}
    )
    if not isinstance(style, dict):
        raise ValueError(f"Report style must be a mapping: {style_path}")
    palette = style.get("palette", ["#4C78A8", "#F58518", "#54A24B"])
    if not isinstance(palette, list) or not palette:
        raise ValueError("Report style palette must be a non-empty list")
    style["palette"] = palette
    plt.rcParams.update({
        "figure.dpi": int(style.get("dpi", 300)),
        "savefig.dpi": int(style.get("dpi", 300)),
        "font.size": float(style.get("font_size", 10)),
        "axes.grid": False,
    })
    return style


def _save_formats(fig: plt.Figure, path: Path, dpi: int) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _write_figure_sources(
    root: Path,
    stem: str,
    data: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    value: str,
    hierarchy_comparison: bool,
) -> None:
    directory = root / stem
    directory.mkdir(parents=True, exist_ok=True)
    data.to_csv(directory / "plot_data.csv", index=False)
    summary.to_csv(directory / "seed_summary.csv", index=False)
    (directory / "manifest.json").write_text(
        json.dumps({
            "figure": stem,
            "value": value,
            "rows": int(len(data)),
            "uncertainty_unit": "seed",
            "confidence_interval": "95% t interval",
            "seed_points_shown": True,
            "hierarchy_comparison": hierarchy_comparison,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _plot_matrix(
    frame: pd.DataFrame,
    *,
    value: str,
    stem: str,
    figures: Path,
    sources: Path,
    style: dict[str, Any],
    compare_hloss: bool,
    difference: bool,
) -> bool:
    if (
        frame.empty
        or value not in frame
        or "hierarchy_loss_weight" not in frame
    ):
        return False
    selected = frame.copy()
    if not compare_hloss:
        selected = selected[selected["hierarchy_loss_weight"].eq(0.0)]
    if selected.empty:
        return False
    groups = [
        "cohort", "task", "holdout", "combo_label", "model",
        "hierarchy_loss_weight",
    ]
    summary = seed_summary(selected, groups=groups, value=value)
    combinations = (
        selected[["stage", "genus", "species", "combo_label"]]
        .drop_duplicates()
        .sort_values(["stage", "genus", "species"])["combo_label"]
        .tolist()
    )
    models = sorted(selected["model"].dropna().astype(str).unique())
    weights = (
        sorted(selected["hierarchy_loss_weight"].dropna().unique())
        if compare_hloss else [0.0]
    )
    series = [
        (model, weight)
        for model in models
        for weight in weights
    ]
    fig, axes = plt.subplots(
        2,
        3,
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
            for series_index, (model, weight) in enumerate(series):
                colour = style["palette"][models.index(model) % len(style["palette"])]
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
                label = (
                    f"{model} · h={weight:g}"
                    if compare_hloss else model
                )
                ax.errorbar(
                    means,
                    positions,
                    xerr=errors,
                    fmt="s" if weight > 0 else "o",
                    markersize=4.5,
                    capsize=2,
                    color=colour,
                    markerfacecolor="none" if weight > 0 else colour,
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
            if difference:
                ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
                ax.set_xlim(-1, 1)
            else:
                chances = subset["chance"].dropna()
                if not chances.empty:
                    ax.axvline(
                        float(chances.mean()),
                        color=style.get("chance_colour", "#777777"),
                        linestyle=":",
                        linewidth=1.2,
                    )
                ax.set_xlim(0, 1)
            ax.set_title(f"{COHORT_LABELS[cohort]}\n{TASK_LABELS[task]}")
            ax.grid(axis="x", alpha=0.2)
            ax.set_yticks(y, combinations if column_index == 0 else [])
            ax.invert_yaxis()
            ax.set_xlabel(
                "Recall difference: withheld − control"
                if difference else "Target-class recall"
            )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(3, max(1, len(series))),
        frameon=False,
    )
    title = (
        "Taxon-stage combination ablation effect"
        if difference else "Adult and Juvenile taxon-stage holdouts"
    )
    title += ": hierarchy loss 0 versus 0.2" if compare_hloss else " (h=0)"
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    _write_figure_sources(
        sources,
        stem,
        selected,
        summary,
        value=value,
        hierarchy_comparison=compare_hloss,
    )
    _save_formats(fig, figures / stem, int(style.get("dpi", 300)))
    return True


def build_adult_taxon_report(
    paper_root: Path,
    *,
    split_root: Path | None = None,
    style_path: Path | None = None,
) -> dict[str, Any]:
    style = _configure_style(style_path)
    directories = {
        name: paper_root / name
        for name in ("tables", "figures", "figure_sources", "summary", "latex")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    metrics = collect_adult_taxon_metrics(paper_root)
    paired = paired_ablation_differences(metrics)
    inventory = observed_taxon_stage_combinations(split_root)
    metrics.to_csv(
        directories["tables"] / "all_adult_taxon_metrics.csv", index=False
    )
    paired.to_csv(
        directories["tables"] / "paired_adult_taxon_ablation_differences.csv",
        index=False,
    )
    inventory.to_csv(
        directories["tables"] / "adult_taxon_combination_inventory.csv",
        index=False,
    )
    metric_summary = seed_summary(
        metrics,
        groups=[
            "training_regime", "cohort", "task", "holdout", "combo_label",
            "model", "hierarchy_loss_weight",
        ],
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
    for name, frame in (
        ("adult_taxon_target_recall_seed_summary", metric_summary),
        ("adult_taxon_ablation_difference_seed_summary", difference_summary),
    ):
        frame.to_csv(directories["tables"] / f"{name}.csv", index=False)
        frame.to_latex(
            directories["latex"] / f"{name}.tex",
            index=False,
            float_format="%.3f",
        )
    withheld = (
        metrics[metrics["training_regime"].eq("adult_combo_withheld")]
        if not metrics.empty else metrics
    )
    figure_specs = (
        ("figure_01_adult_taxon_target_recall", withheld,
         "target_recall_image", False, False),
        ("figure_01_adult_taxon_target_recall_hloss_comparison", withheld,
         "target_recall_image", True, False),
        ("figure_02_adult_taxon_ablation_effect", paired,
         "target_recall_difference", False, True),
        ("figure_02_adult_taxon_ablation_effect_hloss_comparison", paired,
         "target_recall_difference", True, True),
    )
    figures_created = {
        stem: _plot_matrix(
            frame,
            value=value,
            stem=stem,
            figures=directories["figures"],
            sources=directories["figure_sources"],
            style=style,
            compare_hloss=compare_hloss,
            difference=difference,
        )
        for stem, frame, value, compare_hloss, difference in figure_specs
    }
    configured = sorted(
        metrics["holdout"].dropna().astype(str).unique().tolist()
    ) if not metrics.empty else []
    expected = sorted(inventory["holdout"].tolist())
    manifest = {
        "schema_version": 1,
        "paper_result": str(paper_root),
        "completed_metric_runs": int(
            metrics["run_dir"].nunique() if not metrics.empty else 0
        ),
        "configured_holdouts": configured,
        "expected_holdouts_from_splits": expected,
        "exhaustive_split_coverage": configured == expected if expected else None,
        "hierarchy_loss_weights": sorted(
            metrics["hierarchy_loss_weight"].dropna().unique().tolist()
        ) if not metrics.empty else [],
        "figures": figures_created,
        "figure_formats": ["png", "pdf", "svg"],
        "uncertainty_unit": "seed",
        "formal_significance_testing": False,
    }
    (directories["summary"] / "adult_taxon_report_manifest.json").write_text(
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
