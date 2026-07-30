"""Completed-runs-only report for task-specific generalisation experiments."""

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
from sklearn.decomposition import PCA


TASKS = ("genus", "species", "age")
TASK_LABELS = {
    "genus": "Genus",
    "species": "Species",
    "age": "Developmental stage",
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
ARCHITECTURE_ORDER = (
    "shared_heads",
    "single_task_genus",
    "single_task_species",
    "single_task_age",
    "split_taxonomy_age",
    "split_joint_sampler",
    "split_pcgrad",
    "split_age_supcon",
    "split_joint_sampler_pcgrad",
    "split_full",
    "split_species_adversary",
)
HOLDOUT_LABELS = {
    ("juvenile_aporrectodea_longa", "age"):
        "Juvenile A. longa: juvenile recall",
    ("juvenile_aporrectodea_longa", "species"):
        "Juvenile A. longa: species recall",
    ("juvenile_allolobophora_chlorotica", "age"):
        "Juvenile A. chlorotica: juvenile recall",
    ("juvenile_allolobophora_chlorotica", "species"):
        "Juvenile A. chlorotica: species recall",
    ("juvenile_genus_aporrectodea", "age"):
        "Juvenile Aporrectodea: juvenile recall",
    ("juvenile_genus_aporrectodea", "genus"):
        "Juvenile Aporrectodea: genus recall",
    ("unseen_species_aporrectodea_longa_for_genus", "genus"):
        "Unseen A. longa: genus recall",
}
PALETTE = (
    "#3B5B92",
    "#B35C1E",
    "#3D7A57",
    "#8A4F7D",
    "#7A6A2F",
    "#4B7F8C",
    "#A04747",
    "#5E5E9A",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def architecture_name(config: dict[str, Any]) -> str:
    model = config.get("model", {}) or {}
    data = config.get("data", {}) or {}
    training = config.get("training", {}) or {}
    loss = config.get("loss", {}) or {}
    architecture = str(
        model.get("multitask_architecture", "shared_heads")
    )
    if architecture == "single_task":
        return f"single_task_{model.get('target_task', 'unknown')}"
    if architecture != "split_taxonomy_age":
        return architecture
    sampler = (data.get("sampler", {}) or {}).get("type", "default")
    strategy = (
        (training.get("gradient_strategy", {}) or {})
        .get("type", "standard")
    )
    contrastive = bool(
        ((loss.get("age_supervised_contrastive", {}) or {})
         .get("enabled", False))
    )
    adversary = bool(
        ((model.get("age_species_adversary", {}) or {})
         .get("enabled", False))
    )
    if adversary:
        return "split_species_adversary"
    if (
        sampler == "joint_species_stage"
        and strategy == "pcgrad"
        and contrastive
    ):
        return "split_full"
    if sampler == "joint_species_stage" and strategy == "pcgrad":
        return "split_joint_sampler_pcgrad"
    if sampler == "joint_species_stage":
        return "split_joint_sampler"
    if strategy == "pcgrad":
        return "split_pcgrad"
    if contrastive:
        return "split_age_supcon"
    return "split_taxonomy_age"


def discover_completed_runs(
    results_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows = []
    holdout_frames = []
    gradient_frames = []
    for config_path in sorted(results_root.rglob("config.json")):
        run_dir = config_path.parent
        metrics_path = run_dir / "test_metrics_best.json"
        if not metrics_path.is_file():
            metrics_path = run_dir / "test_metrics.json"
        if not metrics_path.is_file():
            continue
        config = _read_json(config_path)
        metrics = _read_json(metrics_path)
        if not config or not metrics:
            continue
        model_cfg = config.get("model", {}) or {}
        data_cfg = config.get("data", {}) or {}
        training_cfg = config.get("training", {}) or {}
        loss_cfg = config.get("loss", {}) or {}
        holdout_cfg = config.get("data_holdout", {}) or {}
        sampler = (data_cfg.get("sampler", {}) or {}).get(
            "type", "default"
        )
        strategy = (
            (training_cfg.get("gradient_strategy", {}) or {})
            .get("type", "standard")
        )
        contrastive = bool(
            ((loss_cfg.get("age_supervised_contrastive", {}) or {})
             .get("enabled", False))
        )
        adversary = bool(
            ((model_cfg.get("age_species_adversary", {}) or {})
             .get("enabled", False))
        )
        holdout = (
            str(holdout_cfg.get("name"))
            if holdout_cfg.get("enabled", False)
            else "original_baseline"
        )
        row = {
            "architecture": architecture_name(config),
            "sampler": sampler,
            "gradient_strategy": strategy,
            "contrastive_enabled": contrastive,
            "adversary_enabled": adversary,
            "model": model_cfg.get("name"),
            "seed": config.get("seed"),
            "holdout": holdout,
            "run_dir": str(run_dir),
            "test_mean_macro_f1": metrics.get("mean_macro_f1"),
            "test_genus_macro_f1": metrics.get("genus_macro_f1"),
            "test_species_macro_f1": metrics.get("species_macro_f1"),
            "test_age_macro_f1": metrics.get("age_macro_f1"),
            "species_adversary_accuracy": metrics.get(
                "species_adversary_accuracy"
            ),
            "age_supcon_valid_anchor_count": metrics.get(
                "age_supcon_valid_anchor_count"
            ),
            "age_supcon_valid_anchor_proportion": metrics.get(
                "age_supcon_valid_anchor_proportion"
            ),
        }
        run_rows.append(row)

        holdout_path = (
            run_dir / "data_holdout_evaluation" / "task_metrics.csv"
        )
        if holdout_path.is_file():
            try:
                frame = pd.read_csv(holdout_path)
            except pd.errors.EmptyDataError:
                frame = pd.DataFrame()
            if not frame.empty:
                if "cohort" in frame:
                    frame = frame[
                        frame["cohort"].eq("independent_test")
                    ].copy()
                for key in (
                    "architecture", "model", "seed", "run_dir",
                ):
                    frame[key] = row[key]
                holdout_frames.append(frame)

        gradient_path = run_dir / "gradient_diagnostics.csv"
        if gradient_path.is_file():
            frame = pd.read_csv(gradient_path)
            if not frame.empty:
                for key in (
                    "architecture", "model", "seed", "run_dir",
                ):
                    frame[key] = row[key]
                gradient_frames.append(frame)
    runs = pd.DataFrame(run_rows)
    holdouts = (
        pd.concat(holdout_frames, ignore_index=True)
        if holdout_frames else pd.DataFrame()
    )
    gradients = (
        pd.concat(gradient_frames, ignore_index=True)
        if gradient_frames else pd.DataFrame()
    )
    return runs, holdouts, gradients


def _ci95(std: float, count: int) -> float:
    if count < 2 or not math.isfinite(std):
        return float("nan")
    critical = T_CRITICAL.get(count - 1, 1.96)
    return float(critical * std / math.sqrt(count))


def _bootstrap_ci95(values: np.ndarray) -> tuple[float, float]:
    """Deterministic seed-level percentile interval for sufficiently large n."""
    generator = np.random.default_rng(2026)
    samples = generator.choice(
        values,
        size=(10_000, len(values)),
        replace=True,
    ).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def seed_summary(
    frame: pd.DataFrame,
    *,
    groups: list[str],
    value: str,
) -> pd.DataFrame:
    columns = [
        *groups, "mean", "standard_deviation",
        "ci95", "ci95_lower", "ci95_upper", "ci_method",
        "number_of_seeds",
    ]
    if frame.empty or value not in frame:
        return pd.DataFrame(columns=columns)
    clean = frame.dropna(subset=[value]).copy()
    if clean.empty:
        return pd.DataFrame(columns=columns)
    # Every input may contain multiple observations per seed (notably one
    # gradient row per diagnostic step). Collapse those before uncertainty is
    # calculated so images, batches, or steps never become pseudo-replicates.
    if "seed" in clean and "seed" not in groups:
        clean = (
            clean.groupby([*groups, "seed"], dropna=False, as_index=False)
            [value]
            .mean()
        )
    result = (
        clean.groupby(groups, dropna=False)[value]
        .agg(mean="mean", standard_deviation="std", number_of_seeds="count")
        .reset_index()
    )
    intervals = []
    for key, row in result.iterrows():
        mask = pd.Series(True, index=clean.index)
        for group in groups:
            mask &= clean[group].eq(row[group])
        values = clean.loc[mask, value].dropna().to_numpy(dtype=float)
        if len(values) >= 5:
            lower, upper = _bootstrap_ci95(values)
            method = "seed_bootstrap_percentile"
        elif len(values) >= 2:
            half_width = _ci95(float(np.std(values, ddof=1)), len(values))
            lower = float(np.mean(values) - half_width)
            upper = float(np.mean(values) + half_width)
            method = "seed_t_interval"
        else:
            lower = upper = float("nan")
            method = "unavailable"
        intervals.append((lower, upper, method))
    result["ci95_lower"] = [item[0] for item in intervals]
    result["ci95_upper"] = [item[1] for item in intervals]
    result["ci_method"] = [item[2] for item in intervals]
    result["ci95"] = (
        (result["ci95_upper"] - result["ci95_lower"]) / 2
    )
    return result[columns]


def task_performance_rows(runs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "architecture", "model", "seed", "holdout", "task",
        "macro_f1", "run_dir",
    ]
    rows = []
    for row in runs.itertuples():
        for task in TASKS:
            rows.append({
                "architecture": row.architecture,
                "model": row.model,
                "seed": row.seed,
                "holdout": row.holdout,
                "task": task,
                "macro_f1": getattr(row, f"test_{task}_macro_f1"),
                "run_dir": row.run_dir,
            })
    return pd.DataFrame(rows, columns=columns)


def paired_differences(
    holdouts: pd.DataFrame,
    baseline_architecture: str = "shared_heads",
) -> pd.DataFrame:
    if holdouts.empty:
        return pd.DataFrame()
    keys = ["model", "seed", "holdout", "task", "target_label"]
    baseline = holdouts[
        holdouts["architecture"].eq(baseline_architecture)
    ][[*keys, "target_recall"]].rename(
        columns={"target_recall": "baseline_target_recall"}
    )
    enhanced = holdouts[
        ~holdouts["architecture"].eq(baseline_architecture)
    ]
    paired = enhanced.merge(baseline, on=keys, how="inner")
    paired["target_recall_difference"] = (
        paired["target_recall"]
        - paired["baseline_target_recall"]
    )
    return paired


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _ordered_architectures(values: pd.Series) -> list[str]:
    available = set(values.dropna().astype(str))
    return [
        architecture for architecture in ARCHITECTURE_ORDER
        if architecture in available
    ] + sorted(available - set(ARCHITECTURE_ORDER))


def plot_architecture_comparison(
    tasks: pd.DataFrame,
    holdouts: pd.DataFrame,
    path: Path,
) -> bool:
    if tasks.empty:
        return False
    baseline = tasks[tasks["holdout"].eq("original_baseline")]
    if baseline.empty:
        return False
    architectures = _ordered_architectures(baseline["architecture"])
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax = axes[0]
    x = np.arange(len(architectures), dtype=float)
    offsets = np.linspace(-0.22, 0.22, len(TASKS))
    for task_index, task in enumerate(TASKS):
        frame = baseline[baseline["task"].eq(task)]
        summary = seed_summary(
            frame, groups=["architecture"], value="macro_f1"
        ).set_index("architecture")
        positions = x + offsets[task_index]
        ax.errorbar(
            positions,
            [summary["mean"].get(item, np.nan) for item in architectures],
            yerr=[summary["ci95"].get(item, np.nan) for item in architectures],
            fmt="o",
            capsize=4,
            label=TASK_LABELS[task],
            color=PALETTE[task_index],
        )
        for architecture_index, architecture in enumerate(architectures):
            seeds = frame[
                frame["architecture"].eq(architecture)
            ]["macro_f1"].dropna()
            ax.scatter(
                np.full(len(seeds), positions[architecture_index]),
                seeds,
                color=PALETTE[task_index],
                alpha=0.45,
                s=18,
            )
    ax.set_xticks(x, architectures, rotation=28, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1")
    ax.set_title("(a) Baseline test performance")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1]
    relevant = holdouts[
        holdouts.apply(
            lambda row: (row.get("holdout"), row.get("task"))
            in HOLDOUT_LABELS,
            axis=1,
        )
    ].copy() if not holdouts.empty else holdouts
    if not relevant.empty:
        relevant["evaluation"] = [
            HOLDOUT_LABELS[(row.holdout, row.task)]
            for row in relevant.itertuples()
        ]
        evaluations = list(dict.fromkeys(HOLDOUT_LABELS.values()))
        architectures_h = _ordered_architectures(
            relevant["architecture"]
        )
        y = np.arange(len(evaluations), dtype=float)
        offsets_h = np.linspace(
            -0.28, 0.28, max(1, len(architectures_h))
        )
        for index, architecture in enumerate(architectures_h):
            frame = relevant[
                relevant["architecture"].eq(architecture)
            ]
            summary = seed_summary(
                frame, groups=["evaluation"], value="target_recall"
            ).set_index("evaluation")
            positions = y + offsets_h[index]
            ax.errorbar(
                [summary["mean"].get(item, np.nan) for item in evaluations],
                positions,
                xerr=[summary["ci95"].get(item, np.nan) for item in evaluations],
                fmt="o",
                capsize=3,
                label=architecture,
                color=PALETTE[index % len(PALETTE)],
            )
            for evaluation_index, evaluation in enumerate(evaluations):
                seeds = frame[
                    frame["evaluation"].eq(evaluation)
                ]["target_recall"].dropna()
                ax.scatter(
                    seeds,
                    np.full(len(seeds), positions[evaluation_index]),
                    color=PALETTE[index % len(PALETTE)],
                    alpha=0.35,
                    s=14,
                )
        for boundary in (1.5, 3.5, 5.5):
            ax.axhline(boundary, color="#DDDDDD", linewidth=0.8)
        ax.set_yticks(y, evaluations)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Target-class recall")
        ax.grid(axis="x", alpha=0.2)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    else:
        ax.text(0.5, 0.5, "No completed holdout metrics", ha="center")
        ax.set_axis_off()
    ax.set_title("(b) Structured-holdout target recall")
    fig.suptitle("Architecture comparison with seed-level observations")
    fig.tight_layout()
    _save_figure(fig, path)
    return True


def plot_effect_differences(paired: pd.DataFrame, path: Path) -> bool:
    if paired.empty:
        return False
    paired = paired.copy()
    paired["evaluation"] = [
        HOLDOUT_LABELS.get(
            (row.holdout, row.task),
            f"{row.holdout}: {row.task}",
        )
        for row in paired.itertuples()
    ]
    summary = seed_summary(
        paired,
        groups=["architecture", "evaluation"],
        value="target_recall_difference",
    )
    if summary.empty:
        return False
    rows = [
        (architecture, evaluation)
        for architecture in _ordered_architectures(
            summary["architecture"]
        )
        for evaluation in HOLDOUT_LABELS.values()
        if not summary[
            summary["architecture"].eq(architecture)
            & summary["evaluation"].eq(evaluation)
        ].empty
    ]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.34 * len(rows))))
    y = np.arange(len(rows), dtype=float)
    indexed = summary.set_index(["architecture", "evaluation"])
    means = [indexed["mean"].get(row, np.nan) for row in rows]
    cis = [indexed["ci95"].get(row, np.nan) for row in rows]
    colours = [
        PALETTE[
            _ordered_architectures(summary["architecture"]).index(row[0])
            % len(PALETTE)
        ]
        for row in rows
    ]
    for index, (mean, ci, colour) in enumerate(
        zip(means, cis, colours)
    ):
        ax.errorbar(mean, y[index], xerr=ci, fmt="o", capsize=3, color=colour)
    ax.axvline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_yticks(
        y,
        [f"{architecture} — {evaluation}" for architecture, evaluation in rows],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Target-recall difference from matched shared-head seed")
    ax.set_title("Effect of each modelling change on biological generalisation")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, path)
    return True


def plot_age_comparison(
    tasks: pd.DataFrame,
    holdouts: pd.DataFrame,
    path: Path,
) -> bool:
    architectures = (
        "shared_heads",
        "single_task_age",
        "split_taxonomy_age",
        "split_joint_sampler",
        "split_pcgrad",
        "split_full",
    )
    rows = []
    baseline = tasks[
        tasks["holdout"].eq("original_baseline")
        & tasks["task"].eq("age")
        & tasks["architecture"].isin(architectures)
    ]
    for row in baseline.itertuples():
        rows.append({
            "architecture": row.architecture,
            "seed": row.seed,
            "target": "Ordinary test age macro-F1",
            "value": row.macro_f1,
        })
    age_holdouts = (
        holdouts[
            holdouts["task"].eq("age")
            & holdouts["architecture"].isin(architectures)
        ]
        if not holdouts.empty else holdouts
    )
    target_names = {
        "juvenile_aporrectodea_longa": "Juvenile A. longa recall",
        "juvenile_allolobophora_chlorotica":
            "Juvenile A. chlorotica recall",
        "juvenile_genus_aporrectodea":
            "Juvenile Aporrectodea recall",
    }
    for row in age_holdouts.itertuples():
        if row.holdout in target_names:
            rows.append({
                "architecture": row.architecture,
                "seed": row.seed,
                "target": target_names[row.holdout],
                "value": row.target_recall,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return False
    targets = [
        "Ordinary test age macro-F1",
        *target_names.values(),
    ]
    present_architectures = [
        item for item in architectures
        if item in set(frame["architecture"])
    ]
    fig, ax = plt.subplots(figsize=(13, 7))
    y = np.arange(len(targets), dtype=float)
    offsets = np.linspace(
        -0.28, 0.28, max(1, len(present_architectures))
    )
    for index, architecture in enumerate(present_architectures):
        selected = frame[frame["architecture"].eq(architecture)]
        summary = seed_summary(
            selected, groups=["target"], value="value"
        ).set_index("target")
        positions = y + offsets[index]
        ax.errorbar(
            [summary["mean"].get(target, np.nan) for target in targets],
            positions,
            xerr=[summary["ci95"].get(target, np.nan) for target in targets],
            fmt="o",
            capsize=3,
            label=architecture,
            color=PALETTE[index % len(PALETTE)],
        )
        for target_index, target in enumerate(targets):
            seeds = selected[
                selected["target"].eq(target)
            ]["value"].dropna()
            ax.scatter(
                seeds,
                np.full(len(seeds), positions[target_index]),
                color=PALETTE[index % len(PALETTE)],
                alpha=0.35,
                s=15,
            )
    ax.set_yticks(y, targets)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 or target recall")
    ax.set_title("Age-specific generalisation comparison")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, path)
    return True


def plot_gradient_interactions(
    gradients: pd.DataFrame,
    path: Path,
) -> bool:
    required = {
        "step",
        "architecture",
        "genus_gradient_norm",
        "species_gradient_norm",
        "age_gradient_norm",
        "genus_species_cosine",
        "genus_age_cosine",
        "species_age_cosine",
    }
    if gradients.empty or not required.issubset(gradients):
        return False
    selected_names = {
        "shared_heads",
        "split_taxonomy_age",
        "split_pcgrad",
        "split_joint_sampler_pcgrad",
    }
    selected = gradients[
        gradients["architecture"].isin(selected_names)
    ]
    if selected.empty:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    for architecture_index, (architecture, frame) in enumerate(
        selected.groupby("architecture")
    ):
        colour = PALETTE[architecture_index % len(PALETTE)]
        for task, linestyle in zip(TASKS, ("-", "--", ":")):
            metric = f"{task}_gradient_norm"
            for _, seed_frame in frame.groupby("seed"):
                seed_frame = seed_frame.sort_values("step")
                axes[0].plot(
                    seed_frame["step"],
                    seed_frame[metric],
                    color=colour,
                    linestyle=linestyle,
                    linewidth=0.5,
                    alpha=0.16,
                )
            summary = seed_summary(
                frame,
                groups=["step"],
                value=metric,
            ).sort_values("step")
            axes[0].plot(
                summary["step"],
                summary["mean"],
                color=colour,
                linestyle=linestyle,
                linewidth=1.5,
                label=f"{architecture}: {task}",
            )
            axes[0].fill_between(
                summary["step"].to_numpy(dtype=float),
                summary["ci95_lower"].to_numpy(dtype=float),
                summary["ci95_upper"].to_numpy(dtype=float),
                color=colour,
                alpha=0.07,
            )
        for pair, linestyle in zip(
            ("genus_species", "genus_age", "species_age"),
            ("-", "--", ":"),
        ):
            metric = f"{pair}_cosine"
            for _, seed_frame in frame.groupby("seed"):
                seed_frame = seed_frame.sort_values("step")
                axes[1].plot(
                    seed_frame["step"],
                    seed_frame[metric],
                    color=colour,
                    linestyle=linestyle,
                    linewidth=0.5,
                    alpha=0.16,
                )
            summary = seed_summary(
                frame,
                groups=["step"],
                value=metric,
            ).sort_values("step")
            axes[1].plot(
                summary["step"],
                summary["mean"],
                color=colour,
                linestyle=linestyle,
                linewidth=1.5,
                label=f"{architecture}: {pair.replace('_', '–')}",
            )
            axes[1].fill_between(
                summary["step"].to_numpy(dtype=float),
                summary["ci95_lower"].to_numpy(dtype=float),
                summary["ci95_upper"].to_numpy(dtype=float),
                color=colour,
                alpha=0.07,
            )
    axes[0].set_title("(a) Task-gradient norms over training")
    axes[0].set_xlabel("Optimisation step")
    axes[0].set_ylabel("Gradient L2 norm (mean and 95% seed CI)")
    axes[0].grid(alpha=0.2)
    axes[1].set_title("(b) Pairwise task-gradient cosine similarities")
    axes[1].set_xlabel("Optimisation step")
    axes[1].set_ylabel("Cosine similarity (mean and 95% seed CI)")
    axes[1].axhline(0, color="#333333", linestyle="--", linewidth=1)
    axes[1].grid(alpha=0.2)
    for ax in axes:
        ax.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    _save_figure(fig, path)
    return True


def plot_embedding_diagnostics(
    runs: pd.DataFrame,
    path: Path,
) -> bool:
    if runs.empty:
        return False
    candidates = runs.copy()
    candidates["embedding_path"] = candidates["run_dir"].map(
        lambda value: str(Path(value) / "age_embeddings_best.npz")
    )
    candidates = candidates[
        candidates["embedding_path"].map(lambda value: Path(value).is_file())
    ]
    if candidates.empty:
        return False
    candidates["_priority"] = candidates["architecture"].map(
        lambda value: 0 if value == "split_full" else 1
    )
    row = candidates.sort_values(
        ["_priority", "architecture", "seed"]
    ).iloc[0]
    run_dir = Path(row["run_dir"])
    metadata_path = run_dir / "age_embeddings_best_metadata.csv"
    if not metadata_path.is_file():
        return False
    embeddings = np.load(
        run_dir / "age_embeddings_best.npz"
    )["embeddings"]
    metadata = pd.read_csv(metadata_path)
    if len(embeddings) != len(metadata) or len(embeddings) < 3:
        return False
    reducer = PCA(
        n_components=2,
        svd_solver="randomized",
        random_state=2026,
    )
    coordinates = reducer.fit_transform(embeddings)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, column, title in (
        (axes[0], "developmental_stage", "Developmental stage"),
        (axes[1], "species", "Species"),
    ):
        values = metadata[column].fillna("<MISSING>").astype(str)
        for index, value in enumerate(sorted(values.unique())):
            mask = values.eq(value).to_numpy()
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=16,
                alpha=0.65,
                color=PALETTE[index % len(PALETTE)],
                label=value,
            )
        ax.set_title(title)
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle(
        f"Descriptive age-embedding projection: "
        f"{row['architecture']}, seed {row['seed']}\n"
        "PCA randomized solver, n_components=2, random_state=2026; "
        "visual separation is not formal evidence"
    )
    fig.tight_layout()
    _save_figure(fig, path)
    parameters = {
        "method": "PCA",
        "n_components": 2,
        "svd_solver": "randomized",
        "random_state": 2026,
        "architecture": row["architecture"],
        "seed": int(row["seed"]),
        "explained_variance_ratio": (
            reducer.explained_variance_ratio_.tolist()
        ),
        "interpretation": "descriptive only",
    }
    path.with_name(path.name + "_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def _comparison_sentence(
    *,
    enhanced: str,
    shared: pd.DataFrame,
    candidate: pd.DataFrame,
    metric_label: str,
) -> str:
    left = shared[["seed", "value"]].rename(
        columns={"value": "shared"}
    )
    right = candidate[["seed", "value"]].rename(
        columns={"value": "enhanced"}
    )
    paired = right.merge(left, on="seed", how="inner").dropna()
    if paired.empty:
        return (
            f"No matched completed seeds were available for {enhanced} "
            f"versus shared_heads on {metric_label}."
        )
    differences = paired["enhanced"] - paired["shared"]
    consistent = bool((differences > 0).all() or (differences < 0).all())
    direction = (
        "increased" if float(differences.mean()) > 0 else "decreased"
        if float(differences.mean()) < 0 else "did not change"
    )
    qualifier = (
        f"The direction was consistent across all {len(paired)} seeds."
        if consistent else
        f"The direction was not consistent across the {len(paired)} seeds."
    )
    if len(differences) >= 5:
        lower, upper = _bootstrap_ci95(differences.to_numpy(dtype=float))
        uncertainty = (
            f"The seed-bootstrap 95% CI for the paired change was "
            f"[{lower:+.3f}, {upper:+.3f}]."
        )
    elif len(differences) >= 2:
        half_width = _ci95(
            float(differences.std(ddof=1)),
            len(differences),
        )
        uncertainty = (
            f"The paired-change SD was {differences.std(ddof=1):.3f} "
            f"and its seed-level 95% t-CI was "
            f"[{differences.mean() - half_width:+.3f}, "
            f"{differences.mean() + half_width:+.3f}]."
        )
    else:
        uncertainty = (
            "A 95% interval is unavailable from a single completed seed."
        )
    return (
        f"{enhanced} {direction} "
        f"{metric_label} from {paired['shared'].mean():.3f} to "
        f"{paired['enhanced'].mean():.3f}, an absolute change of "
        f"{differences.mean():+.3f}. {uncertainty} {qualifier}"
    )


def generate_results_summary(
    tasks: pd.DataFrame,
    holdouts: pd.DataFrame,
    gradients: pd.DataFrame,
) -> str:
    sections: list[tuple[str, list[str]]] = []
    baseline_age = tasks[
        tasks["holdout"].eq("original_baseline")
        & tasks["task"].eq("age")
    ][["architecture", "seed", "macro_f1"]].rename(
        columns={"macro_f1": "value"}
    )

    overall = []
    for architecture in _ordered_architectures(
        tasks["architecture"] if not tasks.empty else pd.Series(dtype=str)
    ):
        frame = tasks[
            tasks["architecture"].eq(architecture)
            & tasks["holdout"].eq("original_baseline")
        ]["macro_f1"].dropna()
        if not frame.empty:
            overall.append(
                f"{architecture} had mean task macro-F1 "
                f"{frame.mean():.3f} (SD {frame.std():.3f}) across "
                f"{tasks[tasks['architecture'].eq(architecture)]['seed'].nunique()} "
                "seed(s)."
            )
    sections.append(("Overall classification", overall))

    age_lines = []
    shared_age = baseline_age[
        baseline_age["architecture"].eq("shared_heads")
    ]
    for architecture in (
        "single_task_age", "split_taxonomy_age",
        "split_joint_sampler", "split_pcgrad",
        "split_joint_sampler_pcgrad", "split_full",
    ):
        candidate = baseline_age[
            baseline_age["architecture"].eq(architecture)
        ]
        if not candidate.empty and not shared_age.empty:
            age_lines.append(_comparison_sentence(
                enhanced=architecture,
                shared=shared_age,
                candidate=candidate,
                metric_label="ordinary-test age macro-F1",
            ))
    age_holdouts = holdouts[
        holdouts["task"].eq("age")
    ].copy() if not holdouts.empty else holdouts
    for holdout, label in (
        (
            "juvenile_aporrectodea_longa",
            "juvenile-stage recall for the juvenile A. longa holdout",
        ),
        (
            "juvenile_allolobophora_chlorotica",
            "juvenile-stage recall for the juvenile A. chlorotica holdout",
        ),
        (
            "juvenile_genus_aporrectodea",
            "juvenile-stage recall for the juvenile Aporrectodea holdout",
        ),
    ):
        evaluation = age_holdouts[
            age_holdouts["holdout"].eq(holdout)
        ][["architecture", "seed", "target_recall"]].rename(
            columns={"target_recall": "value"}
        )
        shared = evaluation[
            evaluation["architecture"].eq("shared_heads")
        ]
        for architecture in (
            "single_task_age",
            "split_taxonomy_age",
            "split_joint_sampler_pcgrad",
            "split_full",
        ):
            candidate = evaluation[
                evaluation["architecture"].eq(architecture)
            ]
            if not shared.empty and not candidate.empty:
                age_lines.append(_comparison_sentence(
                    enhanced=architecture,
                    shared=shared,
                    candidate=candidate,
                    metric_label=label,
                ))
    sections.append(("Developmental-stage generalisation", age_lines))

    taxonomic = []
    for task in ("genus", "species"):
        frame = tasks[
            tasks["holdout"].eq("original_baseline")
            & tasks["task"].eq(task)
        ]
        if not frame.empty:
            summary = seed_summary(
                frame, groups=["architecture"], value="macro_f1"
            )
            if not summary.empty:
                best = summary.sort_values("mean", ascending=False).iloc[0]
                taxonomic.append(
                    f"The highest completed {task} mean macro-F1 was "
                    f"{best['mean']:.3f} for {best['architecture']} "
                    f"using {int(best['number_of_seeds'])} seed(s)."
                )
    taxonomic_holdouts = holdouts[
        holdouts["task"].isin(("genus", "species"))
    ] if not holdouts.empty else holdouts
    for (holdout, task), label in HOLDOUT_LABELS.items():
        if task not in {"genus", "species"}:
            continue
        evaluation = taxonomic_holdouts[
            taxonomic_holdouts["holdout"].eq(holdout)
            & taxonomic_holdouts["task"].eq(task)
        ][["architecture", "seed", "target_recall"]].rename(
            columns={"target_recall": "value"}
        )
        shared = evaluation[
            evaluation["architecture"].eq("shared_heads")
        ]
        for architecture in ("split_taxonomy_age", "split_full"):
            candidate = evaluation[
                evaluation["architecture"].eq(architecture)
            ]
            if not shared.empty and not candidate.empty:
                taxonomic.append(_comparison_sentence(
                    enhanced=architecture,
                    shared=shared,
                    candidate=candidate,
                    metric_label=label,
                ))
    sections.append(("Taxonomic generalisation", taxonomic))

    conflict_lines = []
    single_age = baseline_age[
        baseline_age["architecture"].eq("single_task_age")
    ]
    if not shared_age.empty and not single_age.empty:
        conflict_lines.append(_comparison_sentence(
            enhanced="single_task_age",
            shared=shared_age,
            candidate=single_age,
            metric_label="ordinary-test age macro-F1",
        ))
        paired_age = (
            single_age[["seed", "value"]]
            .merge(
                shared_age[["seed", "value"]],
                on="seed",
                suffixes=("_single", "_shared"),
            )
        )
        if not paired_age.empty:
            mean_change = float(
                (
                    paired_age["value_single"]
                    - paired_age["value_shared"]
                ).mean()
            )
            interpretation = (
                "This direction is consistent with reduced negative transfer, "
                "but the comparison does not establish causality."
                if mean_change > 0 else
                "The age-only diagnostic did not improve this metric, so it "
                "does not support reduced negative transfer on the ordinary "
                "test split."
            )
            conflict_lines.append(interpretation)
    if not gradients.empty:
        for architecture in _ordered_architectures(
            gradients["architecture"]
        ):
            frame = gradients[
                gradients["architecture"].eq(architecture)
            ]
            columns = [
                column for column in (
                    "genus_species_cosine",
                    "genus_age_cosine",
                    "species_age_cosine",
                )
                if column in frame
            ]
            if columns:
                values = frame[columns].stack().dropna()
                if not values.empty:
                    conflict_lines.append(
                        f"{architecture} had mean pairwise gradient cosine "
                        f"{values.mean():.3f}; "
                        f"{(values < 0).mean():.1%} of recorded values were "
                        "negative. Negative values are consistent with "
                        "conflicting task gradients but do not establish a "
                        "causal mechanism."
                    )
    sections.append(("Negative-transfer diagnosis", conflict_lines))

    def mechanism_section(name: str, architectures: tuple[str, ...]) -> None:
        lines = []
        aggregate_holdout = (
            holdouts.groupby(
                ["architecture", "seed"],
                as_index=False,
            )["target_recall"].mean()
            .rename(columns={"target_recall": "value"})
            if not holdouts.empty
            else pd.DataFrame()
        )
        shared_holdout = (
            aggregate_holdout[
                aggregate_holdout["architecture"].eq("shared_heads")
            ]
            if not aggregate_holdout.empty else aggregate_holdout
        )
        for architecture in architectures:
            candidate = baseline_age[
                baseline_age["architecture"].eq(architecture)
            ]
            if not candidate.empty and not shared_age.empty:
                lines.append(_comparison_sentence(
                    enhanced=architecture,
                    shared=shared_age,
                    candidate=candidate,
                    metric_label="ordinary-test age macro-F1",
                ))
            holdout_candidate = (
                aggregate_holdout[
                    aggregate_holdout["architecture"].eq(architecture)
                ]
                if not aggregate_holdout.empty else aggregate_holdout
            )
            if (
                not shared_holdout.empty
                and not holdout_candidate.empty
            ):
                lines.append(_comparison_sentence(
                    enhanced=architecture,
                    shared=shared_holdout,
                    candidate=holdout_candidate,
                    metric_label=(
                        "mean recall across the seven structured-holdout "
                        "target evaluations"
                    ),
                ))
        sections.append((name, lines))

    mechanism_section(
        "Effect of joint sampling",
        ("split_joint_sampler", "split_joint_sampler_pcgrad", "split_full"),
    )
    mechanism_section(
        "Effect of PCGrad",
        ("split_pcgrad", "split_joint_sampler_pcgrad", "split_full"),
    )
    mechanism_section(
        "Effect of age contrastive learning",
        ("split_age_supcon", "split_full"),
    )
    sections.append((
        "Limitations",
        [
            "All uncertainty estimates use seed-level observations rather than "
            "images as independent replicates.",
            "With only three seeds, intervals are wide and formal significance "
            "testing would be exploratory; no causal claim is made.",
            "Ordinary predefined random splits primarily measure interpolation, "
            "whereas structured holdouts probe biological generalisation.",
            "Embedding projections are descriptive and must not be interpreted "
            "as formal evidence of deconfounding.",
        ],
    ))
    lines = ["# Task-specific multitask generalisation results", ""]
    for title, paragraphs in sections:
        lines.extend([f"## {title}", ""])
        lines.extend(paragraphs or [
            "No completed runs with the required metrics were available."
        ])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_latex_outputs(
    *,
    latex_dir: Path,
    architecture_summary: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    markdown_summary: str,
) -> None:
    latex_dir.mkdir(parents=True, exist_ok=True)
    architecture_summary.to_latex(
        latex_dir / "architecture_summary.tex",
        index=False,
        float_format="%.3f",
    )
    holdout_summary.to_latex(
        latex_dir / "holdout_target_recall_summary.tex",
        index=False,
        float_format="%.3f",
    )
    prose = []
    for line in markdown_summary.splitlines():
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            prose.append(
                "\\subsection*{" + _latex_escape(line[3:]) + "}"
            )
        elif line:
            prose.append(_latex_escape(line) + "\n")
    prose.extend([
        "\\paragraph{Figure A caption.} Architecture performance on the "
        "ordinary test split and structured biological holdouts. Points show "
        "seed observations and intervals are seed-level 95\\% confidence intervals.",
        "\\paragraph{Figure B caption.} Matched-seed target-recall differences "
        "from the shared-head model. The vertical line marks zero change.",
        "\\paragraph{Figure C caption.} Age-specific ordinary-test and juvenile "
        "holdout comparisons with seed-level observations.",
        "\\paragraph{Figure D caption.} Shared-parameter gradient norms and "
        "pairwise cosine similarities; negative cosine values indicate gradient conflict.",
        "\\paragraph{Figure E caption.} Descriptive PCA views of age embeddings "
        "coloured by developmental stage and species; separation is not formal evidence.",
    ])
    (latex_dir / "results_summary.tex").write_text(
        "\n\n".join(prose) + "\n",
        encoding="utf-8",
    )


def build_generalisation_report(
    results_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    latex_dir = output_dir / "latex"
    figures_dir.mkdir(exist_ok=True)
    latex_dir.mkdir(exist_ok=True)
    runs, holdouts, gradients = discover_completed_runs(results_root)
    architecture_columns = [
        "architecture", "sampler", "gradient_strategy",
        "contrastive_enabled", "adversary_enabled", "model", "seed",
        "test_mean_macro_f1", "test_genus_macro_f1",
        "test_species_macro_f1", "test_age_macro_f1",
    ]
    architecture_summary = (
        runs.reindex(columns=architecture_columns)
        if not runs.empty
        else pd.DataFrame(columns=architecture_columns)
    )
    holdout_columns = [
        "architecture", "model", "seed", "holdout", "task",
        "target_label", "target_n", "target_recall",
    ]
    holdout_table = (
        holdouts.reindex(columns=holdout_columns)
        if not holdouts.empty
        else pd.DataFrame(columns=holdout_columns)
    )
    tasks = task_performance_rows(runs)
    task_summary = seed_summary(
        tasks,
        groups=["architecture", "model", "holdout", "task"],
        value="macro_f1",
    )
    holdout_summary = seed_summary(
        holdout_table,
        groups=[
            "architecture", "model", "holdout", "task", "target_label",
        ],
        value="target_recall",
    )
    gradient_value_columns = [
        column for column in (
            "genus_gradient_norm", "species_gradient_norm",
            "age_gradient_norm", "genus_species_cosine",
            "genus_age_cosine", "species_age_cosine",
        )
        if column in gradients
    ]
    gradient_summary_rows = []
    for value in gradient_value_columns:
        summary = seed_summary(
            gradients,
            groups=["architecture", "model"],
            value=value,
        )
        summary["metric"] = value
        gradient_summary_rows.append(summary)
    gradient_summary = (
        pd.concat(gradient_summary_rows, ignore_index=True)
        if gradient_summary_rows else pd.DataFrame()
    )

    runs.to_csv(output_dir / "all_generalisation_runs.csv", index=False)
    architecture_summary.to_csv(
        output_dir / "architecture_summary.csv", index=False
    )
    holdout_table.to_csv(
        output_dir / "holdout_target_recall.csv", index=False
    )
    task_summary.to_csv(
        output_dir / "task_performance_summary.csv", index=False
    )
    holdout_summary.to_csv(
        output_dir / "holdout_target_recall_summary.csv", index=False
    )
    gradient_summary.to_csv(
        output_dir / "gradient_conflict_summary.csv", index=False
    )
    gradients.to_csv(
        output_dir / "gradient_diagnostics_all.csv", index=False
    )

    paired = paired_differences(holdout_table)
    paired.to_csv(
        output_dir / "paired_holdout_differences.csv", index=False
    )
    figures = {
        "figure_a_architecture_comparison": plot_architecture_comparison(
            tasks, holdout_table,
            figures_dir / "figure_a_architecture_comparison",
        ),
        "figure_b_modelling_change_effects": plot_effect_differences(
            paired,
            figures_dir / "figure_b_modelling_change_effects",
        ),
        "figure_c_age_specific": plot_age_comparison(
            tasks, holdout_table,
            figures_dir / "figure_c_age_specific",
        ),
        "figure_d_gradient_interactions": plot_gradient_interactions(
            gradients,
            figures_dir / "figure_d_gradient_interactions",
        ),
        "figure_e_embedding_diagnostics": plot_embedding_diagnostics(
            runs,
            figures_dir / "figure_e_embedding_diagnostics",
        ),
    }
    markdown_summary = generate_results_summary(
        tasks, holdout_table, gradients
    )
    (output_dir / "results_summary.md").write_text(
        markdown_summary, encoding="utf-8"
    )
    _write_latex_outputs(
        latex_dir=latex_dir,
        architecture_summary=architecture_summary,
        holdout_summary=holdout_summary,
        markdown_summary=markdown_summary,
    )
    manifest = {
        "results_root": str(results_root),
        "output_dir": str(output_dir),
        "completed_runs": int(len(runs)),
        "architectures": (
            sorted(runs["architecture"].dropna().unique().tolist())
            if not runs.empty else []
        ),
        "figures": figures,
        "statistics_unit": "seed",
        "formal_significance_testing": False,
        "embedding_interpretation": "descriptive only",
    }
    (output_dir / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/generalisation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/generalisation_report"),
    )
    args = parser.parse_args()
    manifest = build_generalisation_report(
        args.results_root,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
