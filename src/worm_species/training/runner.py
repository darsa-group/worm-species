"""Single-run canonical training lifecycle. This module never expands sweeps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from ..evaluation.condition_matrix import evaluate_condition_matrix
from ..evaluation.cue_suppression import evaluate_test_cue_suppression
from ..evaluation.data_holdout import evaluate_data_holdout
from ..evaluation.holdout_controls import evaluate_holdout_controls
from ..logging import create_wandb_logger
from ..models.multitask import build_multitask_model
from ..results.writing import save_json
from .checkpoints import build_checkpoint_payload
from .checkpoints import load_checkpoint
from .checkpoints import save_checkpoint
from .epochs import run_hierarchy_epoch
from .loaders import get_input_condition
from .loaders import make_profile_loaders
from .losses import build_child_to_parent_matrix
from .losses import build_criteria
from .metrics import score_for_selection
from .modes import TrainingProfile
from .modes import resolved_run_name
from .modes import stress_evaluation_enabled
from .reproducibility import set_seed

def initialise_wandb_run(
    cfg: dict,
    run_name: str,
    out_dir: Path,
    profile: TrainingProfile,
):
    return create_wandb_logger(cfg, run_name, out_dir, profile).run


def get_colour_metadata(cfg: dict) -> tuple[float, int]:
    retention = float(cfg.get("data", {}).get("colour_retention", 1.0))
    if not 0 <= retention <= 1:
        raise ValueError(
            "data.colour_retention must be between 0 and 1, got "
            f"{retention}."
        )
    return retention, int(round(retention * 100))


def make_experiment_run_name(cfg: dict, profile: TrainingProfile) -> str:
    return resolved_run_name(cfg, profile)


def run_test_evaluation(
    *,
    checkpoint_name: str,
    checkpoint_path: Path,
    write_legacy_outputs: bool,
    run_name: str,
    out_dir: Path,
    model: torch.nn.Module,
    bundle,
    criteria,
    device: torch.device,
    use_amp: bool,
    weights: dict[str, float],
    normalize: bool,
    hierarchy_cfg: dict,
    matrix: torch.Tensor | None,
    profile: TrainingProfile,
    wandb_logger,
    input_condition: dict,
) -> tuple[dict, dict[str, list[int]], dict[str, list[int]]]:
    """Evaluate one checkpoint on the test split and save its outputs."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics, true, pred = run_hierarchy_epoch(
        model,
        bundle.test_loader,
        criteria,
        None,
        device,
        False,
        None,
        use_amp,
        weights,
        normalize,
        hierarchy_cfg,
        matrix,
        profile.masked_labels,
    )

    wandb_condition = (
       f"original_{checkpoint_name}"
    )

    for task in bundle.target_cols:
        labels = list(range(len(bundle.index_to_label_by_task[task])))
        names = [bundle.index_to_label_by_task[task][index] for index in labels]
        y_true = np.asarray(true[task], dtype=int)
        y_pred = np.asarray(pred[task], dtype=int)

        report_path = (
            out_dir / f"classification_report_{checkpoint_name}_{task}.csv"
        )
        matrix_path = out_dir / f"confusion_matrix_{checkpoint_name}_{task}.csv"

        if not len(y_true):
            empty_report = pd.DataFrame(
                [{"note": "No labelled test examples for this task."}]
            )
            empty_report.to_csv(report_path, index=False)
            pd.DataFrame().to_csv(matrix_path)

            if write_legacy_outputs:
                empty_report.to_csv(
                    out_dir / f"classification_report_{task}.csv",
                    index=False,
                )
                pd.DataFrame().to_csv(out_dir / f"confusion_matrix_{task}.csv")
            continue

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
            output_dict=True,
            zero_division=0,
        )
        matrix_frame = confusion_matrix(y_true, y_pred, labels=labels)
        report_frame = pd.DataFrame(report).transpose()
        confusion_frame = pd.DataFrame(
            matrix_frame,
            index=names,
            columns=names,
        )

        report_frame.to_csv(report_path)
        confusion_frame.to_csv(matrix_path)

        if write_legacy_outputs:
            report_frame.to_csv(out_dir / f"classification_report_{task}.csv")
            confusion_frame.to_csv(out_dir / f"confusion_matrix_{task}.csv")

        wandb_logger.log_classification_report(
            condition=wandb_condition,
            task=task,
            report=report,
            metrics=test_metrics,
            train_condition=input_condition,
        )
        wandb_logger.log_confusion_matrix(
            condition=wandb_condition,
            task=task,
            y_true=y_true,
            y_pred=y_pred,
            class_names=names,
            title=f"Confusion Matrix ({checkpoint_name}, {task})",
        )

    metrics_path = out_dir / f"test_metrics_{checkpoint_name}.json"
    save_json(test_metrics, metrics_path)
    if write_legacy_outputs:
        save_json(test_metrics, out_dir / "test_metrics.json")

    wandb_logger.log_test_condition(
        wandb_condition,
        test_metrics,
        train_condition=input_condition,
    )
    print(
        f"[{run_name}] {checkpoint_name.capitalize()} checkpoint test "
        f"mean macro-F1: {test_metrics.get('mean_macro_f1', float('nan')):.4f}"
    )
    return test_metrics, true, pred


def run_one(cfg: dict, profile: TrainingProfile) -> dict:
    """Run exactly one resolved configuration; never generate another config."""
    set_seed(cfg["seed"])
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    print(f"Using device: {device_name}")
    print("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_condition = {
        "condition": "original",
        "feature": "baseline",
        "transform": "original",
        "strength": 0.0,
    }
    if profile.loader_mode == "condition":
        input_condition = get_input_condition(cfg)
        stress_enabled = stress_evaluation_enabled(cfg)
        if stress_enabled and input_condition["transform"] != "original":
            raise ValueError(
                "Fixed-RGB stress evaluation requires an original-trained "
                "input condition"
            )

    if profile.loader_mode == "standard":
        colour_retention, colour_percent = 1.0, 100
    else:
        colour_retention, colour_percent = get_colour_metadata(cfg)

    if profile.loader_mode == "colour":
        print(
            "Colour retention in data: "
            f"{cfg['data'].get('colour_retention', 1.0)}"
        )
        print(
            f"Colour retention: {colour_retention:.2f} "
            f"({colour_percent}% chromatic information retained)"
        )
    elif profile.loader_mode == "condition":
        print(
            f"Base colour retention: {colour_retention:.2f} "
            f"({colour_percent}% chromatic information retained)"
        )
        print(
            "Matched train/validation/test condition: "
            f"{input_condition['condition']} ({input_condition['transform']})"
        )

    run_name = make_experiment_run_name(cfg, profile)
    out_dir = Path(cfg["output"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config.json")

    wandb_logger = create_wandb_logger(cfg, run_name, out_dir, profile)
    bundle = make_profile_loaders(cfg, profile)
    save_json(bundle.split_summary, out_dir / "split_summary.json")
    save_json(
        bundle.label_to_index_by_task,
        out_dir / "label_to_index_by_task.json",
    )
    print(f"Split summary and label maps saved to {out_dir}")

    num_classes_by_task = {
        task: len(label_to_index)
        for task, label_to_index in bundle.label_to_index_by_task.items()
    }
    model = build_multitask_model(
        cfg,
        num_classes_by_task,
    ).to(device)
    model_parameter_counts = {
        "total_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
    }
    save_json(model_parameter_counts, out_dir / "model_parameters.json")
    print("Model built and moved to device.")

    criteria = build_criteria(
        bundle.train_df,
        bundle.target_cols,
        cfg["data"]["group_col"],
        bundle.label_to_index_by_task,
        device,
        use_class_weights=cfg.get("training", {}).get("class_weight", True),
    )
    weights = cfg.get("multi_task", {}).get(
        "loss_weights",
        {task: 1.0 for task in bundle.target_cols},
    )
    normalize = cfg.get("multi_task", {}).get(
        "normalize_loss_by_active_tasks", True
    )
    hierarchy_cfg = (
        cfg.get("multi_task", {}).get("hierarchy_loss", {})
        if profile.hierarchy
        else {}
    )
    matrix = None
    if hierarchy_cfg.get("enabled", False):
        parent_task = hierarchy_cfg.get("parent_task", "genus")
        child_task = hierarchy_cfg.get("child_task", "species")
        matrix = build_child_to_parent_matrix(
            bundle.label_to_index_by_task,
            parent_task,
            child_task,
            device,
            hierarchy_cfg.get("child_to_parent"),
        )
        print(
            f"Using hierarchy loss: {child_task} -> {parent_task} with weight "
            f"{hierarchy_cfg.get('weight', weights.get('hierarchy', 0.1))}"
        )

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )
    use_amp = cfg["training"].get("use_amp", True)
    scaler = torch.amp.GradScaler(
        enabled=use_amp and device.type == "cuda"
    )

    early = cfg.get("early_stopping", {})
    early_enabled = early.get("enabled", True)
    patience = early.get("patience", 3)
    min_delta = early.get("min_delta", 0.001)
    best = -float("inf")
    best_epoch = 0
    stale = 0
    history = []
    selection = cfg.get("multi_task", {}).get(
        "selection_metric", "mean_macro_f1"
    )
    interval = cfg["training"].get("val_interval", 3)

    print(
        f"Training for {cfg['training']['epochs']} epochs with early stopping: "
        f"{early_enabled}, patience: {patience}, min_delta: {min_delta}"
    )
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics, _, _ = run_hierarchy_epoch(
            model,
            bundle.train_loader,
            criteria,
            optimizer,
            device,
            True,
            scaler,
            use_amp,
            weights,
            normalize,
            hierarchy_cfg,
            matrix,
            profile.masked_labels,
        )
        validate = (
            epoch == 1
            or epoch % interval == 0
            or epoch == cfg["training"]["epochs"]
        )
        if validate:
            val_metrics = run_hierarchy_epoch(
                model,
                bundle.val_loader,
                criteria,
                None,
                device,
                False,
                None,
                use_amp,
                weights,
                normalize,
                hierarchy_cfg,
                matrix,
                profile.masked_labels,
            )[0]
        else:
            val_metrics = {}

        learning_rate = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        wandb_logger.log_epoch_metrics(
            epoch=epoch,
            learning_rate=learning_rate,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )

        if not validate:
            print(
                f"[{run_name}] Epoch {epoch:03d}/{cfg['training']['epochs']} | "
                f"train loss {train_metrics['loss']:.4f} | validation skipped"
            )
            continue

        if selection not in val_metrics:
            raise ValueError(
                f"multi_task.selection_metric={selection!r} is not available. "
                f"Available validation metrics: {list(val_metrics)}"
            )
        score = score_for_selection(val_metrics, selection)
        print(
            f"[{run_name}] Epoch {epoch:03d}/{cfg['training']['epochs']} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val {selection} {val_metrics[selection]:.4f} | "
            "complete exact-match "
            f"{val_metrics['complete_exact_match_accuracy']:.4f} "
            f"n={val_metrics['complete_exact_match_n']}"
        )
        for task in bundle.target_cols:
            print(
                f"    {task}: val macro-F1 "
                f"{val_metrics[f'{task}_macro_f1']:.4f} | val bal-acc "
                f"{val_metrics[f'{task}_balanced_accuracy']:.4f} | "
                f"n={val_metrics[f'{task}_n']}"
            )

        improved = score > best + min_delta
        if improved or epoch == 1:
            best = score
            best_epoch = epoch
            stale = 0
            payload = build_checkpoint_payload(
                profile=profile,
                model_state=model.state_dict(),
                cfg=cfg,
                label_to_index_by_task=bundle.label_to_index_by_task,
                index_to_label_by_task=bundle.index_to_label_by_task,
                best_val_score=best,
                selection_metric=selection,
                best_epoch=best_epoch,
                colour_retention=colour_retention,
                colour_percent=colour_percent,
                training_condition=input_condition,
            )
            save_checkpoint(payload, out_dir / "best_model.pt")
            wandb_logger.update_best(
                best_epoch=best_epoch,
                best_val_score=best,
                selection_metric=selection,
            )
            print(
                f"[{run_name}] New best model saved | best val {selection} "
                f"{best:.4f} at epoch {best_epoch}"
            )
        else:
            stale += 1
            print(
                f"[{run_name}] No improvement for {stale}/{patience} "
                "validation checks"
            )

        if early_enabled and stale >= patience:
            print(
                f"[{run_name}] Early stopping at epoch {epoch}. Best val "
                f"{selection} {best:.4f} at epoch {best_epoch}."
            )
            break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    payload = build_checkpoint_payload(
        profile=profile,
        model_state=model.state_dict(),
        cfg=cfg,
        label_to_index_by_task=bundle.label_to_index_by_task,
        index_to_label_by_task=bundle.index_to_label_by_task,
        best_val_score=best,
        selection_metric=selection,
        best_epoch=best_epoch,
        colour_retention=colour_retention,
        colour_percent=colour_percent,
        training_condition=input_condition,
    )
    save_checkpoint(payload, out_dir / "last_model.pt")
    wandb_logger.update_best(
        best_epoch=best_epoch,
        best_val_score=best,
        selection_metric=selection,
    )
    print(
        f"[{run_name}] Last model saved"
    )
    # Evaluate the final checkpoint first, then the best checkpoint. This leaves
    # ``model`` loaded with the best weights for stress and condition evaluation.
    last_test_metrics, _, _ = run_test_evaluation(
        checkpoint_name="last",
        checkpoint_path=out_dir / "last_model.pt",
        write_legacy_outputs=False,
        run_name=run_name,
        out_dir=out_dir,
        model=model,
        bundle=bundle,
        criteria=criteria,
        device=device,
        use_amp=use_amp,
        weights=weights,
        normalize=normalize,
        hierarchy_cfg=hierarchy_cfg,
        matrix=matrix,
        profile=profile,
        wandb_logger=wandb_logger,
        input_condition=input_condition,
    )
    test_metrics, true, pred = run_test_evaluation(
        checkpoint_name="best",
        checkpoint_path=out_dir / "best_model.pt",
        write_legacy_outputs=True,
        run_name=run_name,
        out_dir=out_dir,
        model=model,
        bundle=bundle,
        criteria=criteria,
        device=device,
        use_amp=use_amp,
        weights=weights,
        normalize=normalize,
        hierarchy_cfg=hierarchy_cfg,
        matrix=matrix,
        profile=profile,
        wandb_logger=wandb_logger,
        input_condition=input_condition,
    )

    data_holdout = evaluate_data_holdout(
        cfg=cfg,
        out_dir=out_dir,
        model=model,
        bundle=bundle,
        criteria=criteria,
        device=device,
        use_amp=use_amp,
        task_loss_weights=weights,
        normalize_loss_by_active_tasks=normalize,
        hierarchy_cfg=hierarchy_cfg,
        child_to_parent_matrix=matrix,
        use_masked_labels=profile.masked_labels,
    )
    holdout_controls = evaluate_holdout_controls(
        cfg=cfg,
        out_dir=out_dir,
        model=model,
        bundle=bundle,
        criteria=criteria,
        device=device,
        use_amp=use_amp,
        task_loss_weights=weights,
        normalize_loss_by_active_tasks=normalize,
        hierarchy_cfg=hierarchy_cfg,
        child_to_parent_matrix=matrix,
        use_masked_labels=profile.masked_labels,
    )

    if profile.loader_mode == "colour":
        test_mean_macro_f1 = float(
            test_metrics.get("mean_macro_f1", float("nan"))
        )
        if test_mean_macro_f1 >= 0.90:
            wandb_logger.alert(
                title="Test macro-F1 reached 0.90",
                text=(
                    f"Run {run_name} achieved best-checkpoint "
                    f"test/mean_macro_f1 = {test_mean_macro_f1:.4f}"
                ),
            )

    stress = {"enabled": False, "n_conditions": 0}
    if profile.stress_evaluation:
        stress = evaluate_test_cue_suppression(
            cfg=cfg,
            run_name=run_name,
            out_dir=out_dir,
            model=model,
            checkpoint_name="best",
            checkpoint_path=out_dir / "best_model.pt",
            baseline_metrics=test_metrics,
            test_loader_context=bundle.test_loader_context,
            criteria=criteria,
            target_cols=bundle.target_cols,
            device=device,
            use_amp=use_amp,
            task_loss_weights=weights,
            normalize_loss_by_active_tasks=normalize,
            hierarchy_cfg=hierarchy_cfg,
            child_to_parent_matrix=matrix,
            wandb_logger=wandb_logger,
        )
        stress = evaluate_test_cue_suppression(
            cfg=cfg,
            run_name=run_name,
            out_dir=out_dir,
            model=model,
            checkpoint_name="last",
            checkpoint_path=out_dir / "last_model.pt",
            baseline_metrics=test_metrics,
            test_loader_context=bundle.test_loader_context,
            criteria=criteria,
            target_cols=bundle.target_cols,
            device=device,
            use_amp=use_amp,
            task_loss_weights=weights,
            normalize_loss_by_active_tasks=normalize,
            hierarchy_cfg=hierarchy_cfg,
            child_to_parent_matrix=matrix,
            wandb_logger=wandb_logger,
        )

    condition_matrix = {"enabled": False, "n_conditions": 0, "n_task_rows": 0}
    evaluation = cfg.get("evaluation", {}) or {}
    canonical_matrix = (
        evaluation.get("condition_matrix", {}) or {}
        if isinstance(evaluation, dict)
        else {}
    )
    legacy_matrix = cfg.get("condition_matrix_evaluation", {}) or {}
    matrix_cfg = canonical_matrix
    if (
        not canonical_matrix.get("conditions")
        and isinstance(legacy_matrix, dict)
        and bool(legacy_matrix.get("enabled", False))
    ):
        matrix_cfg = legacy_matrix
    if bool(matrix_cfg.get("enabled", False)):
        condition_matrix = evaluate_condition_matrix(
            cfg=cfg,
            run_name=run_name,
            out_dir=out_dir,
            model=model,
            training_condition=input_condition,
            baseline_metrics=test_metrics,
            baseline_true=true,
            baseline_pred=pred,
            test_loader_context=bundle.test_loader_context,
            criteria=criteria,
            target_cols=bundle.target_cols,
            index_to_label_by_task=bundle.index_to_label_by_task,
            device=device,
            use_amp=use_amp,
            task_loss_weights=weights,
            normalize_loss_by_active_tasks=normalize,
            hierarchy_cfg=hierarchy_cfg,
            child_to_parent_matrix=matrix,
            use_masked_labels=profile.masked_labels,
            wandb_logger=wandb_logger,
        )

    result = {
        "run_name": run_name,
        "out_dir": str(out_dir),
        "best_val_score": best,
        "selection_metric": selection,
        **{f"test_{key}": value for key, value in test_metrics.items()},
        **{f"last_test_{key}": value for key, value in last_test_metrics.items()},
    }
    if profile.loader_mode == "colour":
        result = {
            "run_name": run_name,
            "out_dir": str(out_dir),
            "colour_retention": colour_retention,
            "colour_percent": colour_percent,
            "best_epoch": best_epoch,
            "best_val_score": best,
            "selection_metric": selection,
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **{
                f"last_test_{key}": value
                for key, value in last_test_metrics.items()
            },
        }
    elif profile.loader_mode == "condition":
        result = {
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
                    if key
                    not in {"condition", "feature", "transform", "strength"}
                },
                sort_keys=True,
            ),
            "best_epoch": best_epoch,
            "best_val_score": best,
            "selection_metric": selection,
            "cue_suppression_enabled": stress["enabled"],
            "cue_suppression_n_conditions": stress["n_conditions"],
            "cue_suppression_n_unique_evaluations": stress.get(
                "n_unique_evaluations", 0
            ),
            "condition_matrix_enabled": condition_matrix["enabled"],
            "condition_matrix_n_conditions": condition_matrix["n_conditions"],
            "condition_matrix_n_task_rows": condition_matrix["n_task_rows"],
            "condition_matrix_manifest_path": condition_matrix.get(
                "manifest_path"
            ),
            "condition_matrix_condition_metrics_path": condition_matrix.get(
                "condition_metrics_path"
            ),
            "condition_matrix_task_metrics_path": condition_matrix.get(
                "task_metrics_path"
            ),
            **{f"test_{key}": value for key, value in test_metrics.items()},
            **{
                f"last_test_{key}": value
                for key, value in last_test_metrics.items()
            },
        }
    if data_holdout.get("enabled", False):
        result["data_holdout"] = data_holdout
    if holdout_controls.get("enabled", False):
        result["data_holdout_controls"] = holdout_controls
    result.update(model_parameter_counts)

    if profile.run_summary:
        save_json(result, out_dir / "run_summary.json")

    summary = {
        "best_epoch": best_epoch,
        "best_val_score": best,
        "selection_metric": selection,
        f"best_test_{selection}": test_metrics.get(selection),
        f"last_test_{selection}": last_test_metrics.get(selection),
    }
    if profile.loader_mode in {"colour", "condition"}:
        summary.update({
            "colour_retention": colour_retention,
            "colour_percent": colour_percent,
        })
    if profile.loader_mode == "condition":
        summary.update({
            "train_condition": input_condition["condition"],
            "train_feature": input_condition["feature"],
            "train_transform": input_condition["transform"],
            "train_strength": input_condition.get("strength"),
        })
    if data_holdout.get("enabled", False):
        summary["data_holdout_name"] = data_holdout["name"]
        for row in data_holdout["tasks"]:
            summary[
                "data_holdout/"
                f"{row.get('cohort', 'independent_test')}/"
                f"{row['task']}/target_recall"
            ] = row["target_recall"]
    artifact_paths = [
        out_dir / "config.json",
        out_dir / "test_metrics.json",
        out_dir / "test_metrics_best.json",
        out_dir / "test_metrics_last.json",
        out_dir / "split_summary.json",
        out_dir / "label_to_index_by_task.json",
        out_dir / "model_parameters.json",
        out_dir / "run_summary.json",
        out_dir / "best_model.pt",
        *sorted(out_dir.glob("classification_report_*.csv")),
        *sorted(out_dir.glob("confusion_matrix_*.csv")),
        out_dir / "data_holdout_evaluation" / "summary.json",
        out_dir / "data_holdout_evaluation" / "task_metrics.csv",
        out_dir / "data_holdout_control_evaluation" / "summary.json",
        out_dir / "data_holdout_control_evaluation" / "task_metrics.csv",
    ]
    artifact_paths = [path for path in artifact_paths if path.exists()]
    wandb_logger.log_artifacts(
        artifact_paths,
        model_metadata={
            **summary,
            "training_condition": input_condition,
            "class_mappings": bundle.label_to_index_by_task,
        },
    )
    wandb_logger.finalise_run(status="completed", summary=summary)

    print("\nBest-checkpoint test metrics:")
    print(test_metrics)
    print("\nLast-checkpoint test metrics:")
    print(last_test_metrics)
    return result
