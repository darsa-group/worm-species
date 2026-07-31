"""Performance-iteration tables, figures, provenance checks, and interpretation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_LABELS = {"genus": "Genus", "species": "Species", "age": "Developmental stage"}
STRUCTURED_ENDPOINTS = {
    ("juvenile_aporrectodea_longa", "age"): "Juvenile A. longa — juvenile recall",
    ("juvenile_allolobophora_chlorotica", "age"): "Juvenile A. chlorotica — juvenile recall",
    ("juvenile_genus_aporrectodea", "age"): "Juvenile Aporrectodea — juvenile recall",
    ("unseen_species_aporrectodea_longa_for_genus", "genus"): "Unseen A. longa — genus recall",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_signature(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _family(config: dict) -> str:
    architecture = (config.get("model", {}) or {}).get(
        "multitask_architecture", "shared_heads"
    )
    multiview = bool(((config.get("data", {}) or {}).get("multiview", {}) or {}).get("enabled", False))
    differential = bool((config.get("optimizer", {}) or {}).get("learning_rates"))
    staged = bool(((config.get("training", {}) or {}).get("staged_unfreezing", {}) or {}).get("enabled", False))
    genus_supcon = bool(((config.get("loss", {}) or {}).get("genus_supervised_contrastive", {}) or {}).get("enabled", False))
    taxonomy = bool(((config.get("loss", {}) or {}).get("taxonomy_consistency", {}) or {}).get("enabled", False))
    age_supcon = bool(((config.get("loss", {}) or {}).get("age_supervised_contrastive", {}) or {}).get("enabled", False))
    cross_sampler = ((config.get("data", {}) or {}).get("sampler", {}) or {}).get("type") == "cross_species_stage_contrastive"
    ensemble = bool(((config.get("evaluation", {}) or {}).get("checkpoint_ensemble", {}) or {}).get("enabled", False))
    if all((multiview, differential, staged, genus_supcon, taxonomy, age_supcon, cross_sampler, ensemble)):
        return "performance_full"
    if multiview:
        return "multiview_training"
    if "multiview" in (config.get("data", {}) or {}) and bool(
        ((config.get("evaluation", {}) or {}).get("individual_level", {}) or {}).get(
            "enabled", False
        )
    ):
        return "multiview_inference"
    if genus_supcon:
        return "genus_supcon"
    if taxonomy:
        return "taxonomy_consistency"
    if differential:
        return "differential_lr"
    if staged:
        return "staged_unfreezing"
    if cross_sampler:
        return "cross_species_age_batches"
    if ensemble:
        return "checkpoint_ensemble"
    return str(architecture)


def _validate_task_contract(config: dict, predictions: pd.DataFrame, run_dir: Path) -> None:
    architecture = (config.get("model", {}) or {}).get("multitask_architecture", "shared_heads")
    observed = set(predictions["task"].dropna().astype(str))
    if architecture == "single_task":
        expected = {(config.get("model", {}) or {}).get("target_task")}
        if observed != expected:
            raise ValueError(
                f"{run_dir}: single-task predictions violate task contract; "
                f"expected {expected}, observed {observed}"
            )


def discover_performance_runs(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordinary, holdouts, verification = [], [], []
    for config_path in sorted(results_root.rglob("config.json")):
        run_dir = config_path.parent
        completion = run_dir / "completion_manifest.json"
        prediction_path = run_dir / "predictions_best.csv"
        individual_path = run_dir / "individual_predictions_best.csv"
        metric_path = run_dir / "test_metrics_best.json"
        if not all(path.exists() for path in (completion, prediction_path, individual_path, metric_path)):
            continue
        config = _read_json(config_path)
        predictions = pd.read_csv(prediction_path)
        _validate_task_contract(config, predictions, run_dir)
        metrics = _read_json(metric_path)
        provenance = _read_json(run_dir / "runtime_provenance.json")
        family = _family(config)
        backbone = str((config.get("model", {}) or {}).get("name"))
        seed = int(config["seed"])
        holdout_name = (config.get("data_holdout", {}) or {}).get("name", "original_baseline")
        split_hash = json.dumps(provenance.get("split_hashes", {}), sort_keys=True)
        if holdout_name == "original_baseline":
            for task in ("genus", "species", "age"):
                for level in ("image", "individual"):
                    if f"{task}_{level}_macro_f1" not in metrics:
                        continue
                    ordinary.append({
                        "run_name": run_dir.name, "architecture": family,
                        "backbone": backbone,
                        "seed": seed, "task": task, "level": level,
                        "accuracy": metrics.get(f"{task}_{level}_accuracy"),
                        "balanced_accuracy": metrics.get(f"{task}_{level}_balanced_accuracy"),
                        "macro_f1": metrics.get(f"{task}_{level}_macro_f1"),
                        "n": metrics.get(f"{task}_{level}_n"),
                        "genus_species_agreement_rate": metrics.get(
                            "individual_genus_species_agreement_rate"
                        ),
                        "split_hash": split_hash,
                    })
        holdout_path = run_dir / "data_holdout_evaluation" / "task_metrics.csv"
        if holdout_path.exists():
            frame = pd.read_csv(holdout_path)
            for _, row in frame.iterrows():
                endpoint = STRUCTURED_ENDPOINTS.get((str(row["holdout"]), str(row["task"])))
                if endpoint is None:
                    continue
                holdouts.append({
                    "run_name": run_dir.name, "architecture": family,
                    "backbone": backbone,
                    "seed": seed, "holdout": row["holdout"], "task": row["task"],
                    "endpoint": endpoint, "target_label": row.get("target_label"),
                    "target_n_images": row.get("target_n_images"),
                    "target_n_individuals": row.get("target_n_individuals"),
                    "target_recall_image": row.get("target_recall_image"),
                    "target_recall_individual": row.get("target_recall_individual"),
                    "split_hash": split_hash,
                })
        model_parameters = _read_json(run_dir / "model_parameters.json")
        complete = _read_json(completion)
        verification.append({
            "run_name": run_dir.name,
            "architecture": family,
            "backbone": backbone,
            "effective_config": json.dumps(config, sort_keys=True),
            "config_hash": provenance.get("resolved_config_hash"),
            "parameter_count": model_parameters.get("total_parameters"),
            "split_hash": split_hash,
            "checkpoint": str(run_dir / "best_model.pt"),
            "checkpoint_hash": complete.get("checkpoint_hash") or _sha(run_dir / "best_model.pt"),
            "prediction_hash": complete.get("prediction_hash") or _sha(prediction_path),
            "metric_hash": complete.get("metric_hash") or _sha(metric_path),
        })
    return (
        pd.DataFrame(ordinary, columns=[
            "run_name", "architecture", "backbone", "seed", "task", "level",
            "accuracy", "balanced_accuracy", "macro_f1", "n", "split_hash",
            "genus_species_agreement_rate",
        ]),
        pd.DataFrame(holdouts, columns=[
            "run_name", "architecture", "backbone", "seed", "holdout", "task",
            "endpoint", "target_label", "target_n_images", "target_n_individuals",
            "target_recall_image", "target_recall_individual", "split_hash",
        ]),
        pd.DataFrame(verification, columns=[
            "run_name", "architecture", "backbone", "effective_config", "config_hash",
            "parameter_count", "split_hash", "checkpoint", "checkpoint_hash",
            "prediction_hash", "metric_hash",
        ]),
    )


def _ci(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().astype(float)
    mean = float(clean.mean()) if len(clean) else float("nan")
    if len(clean) < 2:
        return mean, 0.0
    critical = 4.303 if len(clean) == 3 else 1.96
    return mean, critical * float(clean.std(ddof=1)) / math.sqrt(len(clean))


def _save_figure(fig, base: Path) -> list[str]:
    paths = []
    for suffix in ("svg", "pdf", "png"):
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _plot_ordinary(frame: pd.DataFrame, base: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    frame = frame.copy()
    if not frame.empty:
        frame["variant"] = frame["architecture"] + " · " + frame["backbone"]
    architectures = sorted(frame["variant"].unique()) if not frame.empty else []
    for axis, task in zip(axes, ("genus", "species", "age"), strict=True):
        task_frame = frame[frame["task"] == task]
        for level, marker, offset in (("image", "o", -0.12), ("individual", "D", 0.12)):
            for index, architecture in enumerate(architectures):
                values = task_frame[(task_frame["variant"] == architecture) & (task_frame["level"] == level)]["macro_f1"]
                mean, error = _ci(values)
                axis.scatter(np.full(len(values), index + offset), values, alpha=0.65, marker=marker)
                if len(values):
                    axis.errorbar(index + offset, mean, yerr=error, color="black", marker=marker, capsize=3)
        axis.set_title(TASK_LABELS[task] + " macro-F1")
        axis.set_xticks(range(len(architectures)), architectures, rotation=35, ha="right")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Macro-F1")
    axes[-1].scatter([], [], marker="o", label="Image")
    axes[-1].scatter([], [], marker="D", label="Individual")
    axes[-1].legend()
    return _save_figure(fig, base)


def _plot_holdouts(frame: pd.DataFrame, base: Path) -> list[str]:
    fig, axis = plt.subplots(figsize=(10, 6))
    endpoints = list(STRUCTURED_ENDPOINTS.values())
    frame = frame.copy()
    if not frame.empty:
        frame["variant"] = frame["architecture"] + " · " + frame["backbone"]
    architectures = sorted(frame["variant"].unique()) if not frame.empty else []
    offsets = (
        np.linspace(-0.25, 0.25, len(architectures))
        if architectures else []
    )
    for architecture, offset in zip(architectures, offsets, strict=True):
        for y, endpoint in enumerate(endpoints):
            values = frame[(frame["variant"] == architecture) & (frame["endpoint"] == endpoint)]["target_recall_individual"]
            mean, error = _ci(values)
            axis.scatter(values, np.full(len(values), y + offset), alpha=0.55)
            if len(values):
                axis.errorbar(mean, y + offset, xerr=error, marker="o", capsize=3, label=architecture if y == 0 else None)
    axis.set_yticks(range(len(endpoints)), endpoints)
    axis.set_xlim(0, 1)
    axis.set_xlabel("Individual-level recall")
    axis.grid(axis="x", alpha=0.2)
    if architectures:
        axis.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    return _save_figure(fig, base)


def paired_differences(holdouts: pd.DataFrame) -> pd.DataFrame:
    if holdouts.empty:
        return pd.DataFrame()
    shared = holdouts[holdouts["architecture"] == "shared_heads"]
    candidates = holdouts[holdouts["architecture"] != "shared_heads"]
    keys = ["backbone", "seed", "holdout", "task", "endpoint", "split_hash"]
    paired = candidates.merge(
        shared[keys + ["target_recall_individual"]],
        on=keys,
        how="inner",
        suffixes=("", "_shared_heads"),
        validate="many_to_one",
    )
    if paired.empty:
        return paired
    paired["difference"] = (
        paired["target_recall_individual"]
        - paired["target_recall_individual_shared_heads"]
    )
    return paired


def _plot_paired(frame: pd.DataFrame, base: Path) -> list[str]:
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.axvline(0, color="black", linewidth=1)
    if not frame.empty:
        labels = frame["architecture"] + " · " + frame["backbone"] + " | " + frame["endpoint"]
        unique = list(dict.fromkeys(labels))
        mapping = {label: index for index, label in enumerate(unique)}
        axis.scatter(frame["difference"], [mapping[label] for label in labels], alpha=0.7)
        axis.set_yticks(range(len(unique)), unique)
    axis.set_xlabel("Candidate − shared_heads individual recall")
    axis.grid(axis="x", alpha=0.2)
    return _save_figure(fig, base)


def _claim(values: pd.Series, positive: str, unavailable: str) -> str:
    values = values.dropna().astype(float)
    if len(values) < 3:
        return unavailable + " Fewer than three matched seeds are available."
    mean = float(values.mean())
    variability = float(values.std(ddof=1))
    if not (values.gt(0).all() or values.lt(0).all()) or abs(mean) < variability:
        return unavailable + " Seed directions disagree or the mean change is smaller than seed variability."
    return positive.format(mean=mean)


def _ordinary_pair(
    ordinary: pd.DataFrame,
    candidate: str,
    reference: str,
    value: str = "macro_f1",
) -> pd.Series:
    selected = ordinary[ordinary["level"] == "individual"]
    keys = ["backbone", "seed", "task", "split_hash"]
    candidate_rows = selected[selected["architecture"] == candidate]
    reference_rows = selected[selected["architecture"] == reference]
    merged = candidate_rows.merge(
        reference_rows[keys + [value]], on=keys, how="inner",
        suffixes=("", "_reference"), validate="one_to_one",
    )
    return merged[value] - merged[f"{value}_reference"] if not merged.empty else pd.Series(dtype=float)


def _write_summary(
    ordinary: pd.DataFrame,
    holdouts: pd.DataFrame,
    paired: pd.DataFrame,
    ensemble: pd.DataFrame,
    output_dir: Path,
) -> None:
    aggregation = pd.DataFrame()
    if not ordinary.empty:
        pivot = ordinary.pivot_table(index=["run_name", "architecture", "seed", "task"], columns="level", values="macro_f1").dropna()
        aggregation = pivot.get("individual", pd.Series(dtype=float)) - pivot.get("image", pd.Series(dtype=float))
    genus_pairs = paired[(paired.get("architecture") == "genus_supcon") & (paired.get("holdout") == "unseen_species_aporrectodea_longa_for_genus")] if not paired.empty else pd.DataFrame()
    multiview = _ordinary_pair(
        ordinary, "multiview_training", "multiview_inference"
    )
    taxonomy_agreement = _ordinary_pair(
        ordinary[ordinary["task"] == "genus"],
        "taxonomy_consistency",
        "split_taxonomy_age",
        value="genus_species_agreement_rate",
    )
    variability_changes = []
    individual = ordinary[ordinary["level"] == "individual"]
    for (backbone, task), group in individual.groupby(["backbone", "task"]):
        differential_values = group[group["architecture"] == "differential_lr"]["macro_f1"].dropna()
        reference_values = group[group["architecture"] == "split_taxonomy_age"]["macro_f1"].dropna()
        if len(differential_values) == len(reference_values) == 3:
            variability_changes.append(float(reference_values.std(ddof=1) - differential_values.std(ddof=1)))
    ensemble_stability = []
    for (architecture, backbone, task), group in ensemble.groupby(
        ["architecture", "backbone", "task"]
    ) if not ensemble.empty else []:
        ensemble_values = group["individual_macro_f1"].dropna()
        best_values = individual[
            (individual["architecture"] == architecture)
            & (individual["backbone"] == backbone)
            & (individual["task"] == task)
        ]["macro_f1"].dropna()
        if len(ensemble_values) == len(best_values) == 3:
            ensemble_stability.append(
                float(best_values.std(ddof=1) - ensemble_values.std(ddof=1))
            )
    lines = [
        "# Performance-focused generalisation summary", "",
        "- Did individual-level aggregation improve performance? " + _claim(aggregation, "Yes: individual aggregation improved macro-F1 by {mean:.4f} on average.", "No robust improvement is claimed."),
        "- Did multi-view training improve over inference-only aggregation? " + _claim(multiview, "Yes: multi-view training improved individual macro-F1 by {mean:.4f}.", "No robust improvement is claimed."),
        "- Did differential learning rates reduce seed variability? " + _claim(pd.Series(variability_changes, dtype=float), "Yes: the standard deviation decreased by {mean:.4f} on average.", "No robust reduction is claimed."),
        "- Did genus SupCon improve unseen-species genus recall? " + _claim(genus_pairs.get("difference", pd.Series(dtype=float)), "Yes: recall improved by {mean:.4f} versus shared heads.", "No robust improvement is claimed."),
        "- Did taxonomy consistency improve genus/species agreement? " + _claim(taxonomy_agreement, "Yes: agreement improved by {mean:.4f}.", "No robust improvement is claimed."),
        "- Did checkpoint ensembling improve stability? " + _claim(pd.Series(ensemble_stability, dtype=float), "Yes: seed-level standard deviation decreased by {mean:.4f}.", "No robust improvement is claimed."),
        "",
        "Interpretation guardrails: claims require three matched seeds, consistent direction, and a mean paired change at least as large as seed-level variability. Image/individual disagreements are reported rather than resolved in favour of one level.",
    ]
    (output_dir / "performance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    escaped = [line.replace("_", "\\_").replace("−", "--") for line in lines if line and not line.startswith("#")]
    (output_dir / "performance_summary.tex").write_text("\n".join("\\noindent " + line + "\\par" for line in escaped) + "\n", encoding="utf-8")


def _write_latex_tables(
    ordinary: pd.DataFrame,
    holdouts: pd.DataFrame,
    paired: pd.DataFrame,
    ensemble: pd.DataFrame,
    output_dir: Path,
) -> None:
    tables = {
        "individual_level_metrics.tex": ordinary,
        "individual_holdout_target_recall.tex": holdouts,
        "paired_architecture_differences.tex": paired,
        "checkpoint_ensemble_metrics.tex": ensemble,
    }
    for filename, frame in tables.items():
        (output_dir / filename).write_text(
            frame.to_latex(
                index=False,
                escape=True,
                float_format=lambda value: f"{value:.4f}",
            ),
            encoding="utf-8",
        )


def build_performance_report(results_root: Path, output_dir: Path) -> dict:
    ordinary, holdouts, verification = discover_performance_runs(results_root)
    paired = paired_differences(holdouts)
    ensemble_rows = []
    for path in sorted(results_root.rglob("test_metrics_ensemble.json")):
        config = _read_json(path.parent / "config.json")
        metrics = _read_json(path)
        for task in ("genus", "species", "age"):
            ensemble_rows.append({
                "run_name": path.parent.name, "architecture": _family(config),
                "backbone": (config.get("model", {}) or {}).get("name"),
                "seed": config["seed"], "task": task,
                "individual_macro_f1": metrics.get(f"{task}_individual_macro_f1"),
                "checkpoint_epochs": json.dumps(metrics.get("checkpoint_epochs", [])),
            })
    ensemble = pd.DataFrame(ensemble_rows)
    ordinary.to_csv(output_dir / "individual_level_metrics.csv", index=False)
    holdouts.to_csv(output_dir / "individual_holdout_target_recall.csv", index=False)
    paired.to_csv(output_dir / "paired_architecture_differences.csv", index=False)
    ensemble.to_csv(output_dir / "checkpoint_ensemble_metrics.csv", index=False)
    verification.to_csv(output_dir / "verification_table.csv", index=False)
    comparison_rows = []
    if not verification.empty:
        signatures = {}
        for _, row in verification.iterrows():
            config = json.loads(row["effective_config"])
            signature = {
                "backbone": row["backbone"],
                "architecture": (config.get("model", {}) or {}).get("multitask_architecture"),
                "sampler": ((config.get("data", {}) or {}).get("sampler", {}) or {}).get("type", "default"),
                "multiview_training": bool(((config.get("data", {}) or {}).get("multiview", {}) or {}).get("enabled", False)),
                "gradient_strategy": ((config.get("training", {}) or {}).get("gradient_strategy", {}) or {}).get("type", "standard"),
                "staged_unfreezing": bool(((config.get("training", {}) or {}).get("staged_unfreezing", {}) or {}).get("enabled", False)),
                "differential_lr": (config.get("optimizer", {}) or {}).get("learning_rates"),
                "age_supcon": bool(((config.get("loss", {}) or {}).get("age_supervised_contrastive", {}) or {}).get("enabled", False)),
                "genus_supcon": bool(((config.get("loss", {}) or {}).get("genus_supervised_contrastive", {}) or {}).get("enabled", False)),
                "taxonomy_consistency": bool(((config.get("loss", {}) or {}).get("taxonomy_consistency", {}) or {}).get("enabled", False)),
                "checkpoint_ensemble": bool(((config.get("evaluation", {}) or {}).get("checkpoint_ensemble", {}) or {}).get("enabled", False)),
            }
            signatures[(row["architecture"], row["backbone"])] = _stable_signature(signature)
        reverse: dict[str, set[str]] = {}
        for (architecture, backbone), signature in signatures.items():
            reverse.setdefault(signature, set()).add(architecture)
        for (architecture, backbone), group in verification.groupby(["architecture", "backbone"]):
            config_rows = [json.loads(value) for value in group["effective_config"]]
            seeds = sorted({int(value["seed"]) for value in config_rows})
            signature = signatures[(architecture, backbone)]
            duplicates = sorted(reverse[signature] - {architecture})
            status = "valid"
            reason = "three matched seeds with provenance"
            if duplicates:
                status = "duplicated"
                reason = "same implemented mechanism as " + ", ".join(duplicates)
            elif len(seeds) < 3:
                status = "unsupported"
                reason = f"only {len(seeds)} completed seed(s)"
            comparison_rows.append({
                "architecture": architecture, "backbone": backbone,
                "completed_seeds": json.dumps(seeds), "status": status,
                "reason": reason, "mechanism_signature": signature,
            })
    comparison_audit = pd.DataFrame(comparison_rows)
    comparison_audit.to_csv(output_dir / "comparison_support.csv", index=False)
    figures_dir = output_dir / "performance_figures"
    figures_dir.mkdir(exist_ok=True)
    figures = {
        "ordinary_test": _plot_ordinary(ordinary, figures_dir / "figure_1_ordinary_test"),
        "structured_generalisation": _plot_holdouts(holdouts, figures_dir / "figure_2_structured_generalisation"),
        "paired_improvement": _plot_paired(paired, figures_dir / "figure_3_paired_improvement"),
    }
    if not ensemble.empty and not ordinary.empty:
        best = ordinary[ordinary["level"] == "individual"][
            ["run_name", "task", "macro_f1"]
        ].rename(columns={"macro_f1": "best_individual_macro_f1"})
        ensemble = ensemble.merge(
            best, on=["run_name", "task"], how="left", validate="one_to_one"
        )
        ensemble["difference_from_best"] = (
            ensemble["individual_macro_f1"]
            - ensemble["best_individual_macro_f1"]
        )
        ensemble.to_csv(output_dir / "checkpoint_ensemble_metrics.csv", index=False)
    _write_latex_tables(ordinary, holdouts, paired, ensemble, output_dir)
    _write_summary(ordinary, holdouts, paired, ensemble, output_dir)
    return {
        "completed_performance_runs": int(len(verification)),
        "performance_figures": figures,
        "family_assignment": "resolved_configuration",
        "single_task_contract": "fail_closed",
        "comparison_audit": str(output_dir / "comparison_support.csv"),
    }


__all__ = ["build_performance_report", "discover_performance_runs", "paired_differences"]
