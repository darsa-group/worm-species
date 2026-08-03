"""Single-run canonical training lifecycle. This module never expands sweeps."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from ..evaluation.condition_matrix import evaluate_condition_matrix
from ..evaluation.cue_suppression import evaluate_test_cue_suppression
from ..evaluation.data_holdout import evaluate_data_holdout
from ..evaluation.predictions import collect_probability_predictions
from ..evaluation.predictions import ensemble_prediction_frames
from ..evaluation.predictions import aggregate_individual_probabilities
from ..evaluation.predictions import prediction_metrics
from ..evaluation.predictions import public_prediction_frame
from ..evaluation.predictions import structured_target_metrics
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
from .optimizers import StagedUnfreezer
from .optimizers import build_optimizer
from .metrics import score_for_selection
from .modes import TrainingProfile
from .modes import resolved_run_name
from .modes import stress_evaluation_enabled
from .reproducibility import set_seed


def _stable_json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    columns = sorted(frame.columns)
    payload = frame.loc[:, columns].fillna("<NA>").astype(str).sort_values(
        columns, kind="stable"
    ).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _validate_checkpoint_contract(checkpoint: dict, bundle, cfg: dict) -> None:
    if checkpoint.get("label_to_index_by_task") != bundle.label_to_index_by_task:
        raise RuntimeError(
            "Checkpoint label-to-index mappings do not match this run directory"
        )
    checkpoint_tasks = set((checkpoint.get("label_to_index_by_task") or {}).keys())
    if checkpoint_tasks != set(bundle.target_cols):
        raise RuntimeError(
            "Checkpoint task heads do not match the resolved active tasks"
        )
    if _stable_json_hash(checkpoint.get("cfg")) != _stable_json_hash(cfg):
        raise RuntimeError(
            "Checkpoint resolved configuration does not match the current run"
        )

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
    cfg: dict,
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
    age_supcon_cfg: dict,
    age_species_adversary_cfg: dict,
    genus_supcon_cfg: dict,
    taxonomy_consistency_cfg: dict,
    species_to_genus_matrix: torch.Tensor | None,
) -> tuple[dict, dict[str, list[int]], dict[str, list[int]]]:
    """Evaluate one checkpoint on the test split and save its outputs."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    _validate_checkpoint_contract(checkpoint, bundle, cfg)
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
        age_supcon_cfg=age_supcon_cfg,
        age_species_adversary_cfg=age_species_adversary_cfg,
        genus_supcon_cfg=genus_supcon_cfg,
        taxonomy_consistency_cfg=taxonomy_consistency_cfg,
        species_to_genus_matrix=species_to_genus_matrix,
    )
    image_predictions, individual_predictions, probability_metrics = (
        collect_probability_predictions(
            models=[model],
            loader=bundle.test_loader,
            tasks=bundle.target_cols,
            index_to_label_by_task=bundle.index_to_label_by_task,
            device=device,
            use_amp=use_amp,
            run_id=run_name,
            checkpoint=checkpoint_name,
            split="test",
            maximum_images_per_individual=(
                bundle.multiview_evaluation_max_images
            ),
        )
    )
    test_metrics.update(probability_metrics)
    public_prediction_frame(image_predictions).to_csv(
        out_dir / f"predictions_{checkpoint_name}.csv", index=False
    )
    public_prediction_frame(individual_predictions).to_csv(
        out_dir / f"individual_predictions_{checkpoint_name}.csv", index=False
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


def save_age_embedding_diagnostics(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    use_amp: bool,
    out_dir: Path,
) -> dict[str, str] | None:
    """Save a comparable best-checkpoint representation and its provenance.

    Prefer the normalised age projection when one exists. Otherwise retain the
    pre-classifier age feature. The manifest makes that distinction explicit,
    including models that do not have an age head.
    """
    model.eval()
    embedding_batches = []
    rows = []
    representation_type = None
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(
                enabled=use_amp and device.type == "cuda",
                device_type=device.type,
            ):
                outputs = model(images)
            embeddings = outputs.get("age_embedding")
            current_representation = "age_projection"
            if embeddings is None:
                embeddings = outputs.get("age_features")
                current_representation = (
                    "age_branch_features"
                    if outputs.get("age_logits") is not None
                    else "backbone_features_no_age_head"
                )
            if embeddings is None:
                return None
            if representation_type is None:
                representation_type = current_representation
            elif representation_type != current_representation:
                raise RuntimeError(
                    "Representation type changed while exporting embeddings"
                )
            embedding_batches.append(
                embeddings.detach().float().cpu().numpy()
            )
            label_names = batch.get(
                "metadata_label_names", batch.get("label_names", {})
            )
            paths = batch.get("path", [""] * len(images))
            for index in range(len(images)):
                rows.append({
                    "row": len(rows),
                    "developmental_stage": (
                        label_names.get(
                            "age", ["<MISSING>"] * len(images)
                        )[index]
                    ),
                    "species": (
                        label_names.get(
                            "species", ["<MISSING>"] * len(images)
                        )[index]
                    ),
                    "path": paths[index],
                })
    if not embedding_batches:
        return None
    embedding_path = out_dir / "age_embeddings_best.npz"
    metadata_path = out_dir / "age_embeddings_best_metadata.csv"
    manifest_path = out_dir / "age_embeddings_best_manifest.json"
    all_embeddings = np.concatenate(embedding_batches, axis=0)
    np.savez_compressed(
        embedding_path,
        embeddings=all_embeddings,
        representation_type=np.asarray(representation_type),
    )
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    save_json({
        "representation_type": representation_type,
        "normalised": representation_type == "age_projection",
        "feature_dimension": int(all_embeddings.shape[1]),
        "number_of_rows": int(all_embeddings.shape[0]),
        "source_output": (
            "age_embedding"
            if representation_type == "age_projection"
            else "age_features"
        ),
        "interpretation": "descriptive only",
    }, manifest_path)
    return {
        "embeddings": str(embedding_path),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "representation_type": str(representation_type),
    }


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
    config_hash = _stable_json_hash(cfg)
    existing_config_path = out_dir / "config.json"
    if existing_config_path.exists():
        existing_config = json.loads(existing_config_path.read_text())
        if _stable_json_hash(existing_config) != config_hash:
            raise RuntimeError(
                f"Refusing to reuse {out_dir} for a different resolved config"
            )
    save_json(cfg, out_dir / "config.json")

    wandb_logger = create_wandb_logger(cfg, run_name, out_dir, profile)
    bundle = make_profile_loaders(cfg, profile)
    save_json(bundle.split_summary, out_dir / "split_summary.json")
    save_json(
        bundle.label_to_index_by_task,
        out_dir / "label_to_index_by_task.json",
    )
    if bundle.sampler_summary is not None:
        bundle.sampler_summary.to_csv(
            out_dir / "joint_species_stage_sampler.csv",
            index=False,
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
    staged_unfreezer = StagedUnfreezer.from_config(cfg)
    initial_trainable_parameters = staged_unfreezer.initialise(model)
    if hasattr(model, "branch_mode_used"):
        print(
            "Split taxonomy-age branch mode: "
            f"{model.branch_mode_used}"
        )
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
    runtime_provenance = {
        "run_id": run_name,
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_status": _git_value("status", "--short"),
        "resolved_config_hash": config_hash,
        "split_hashes": {
            "train": _frame_hash(bundle.train_df),
            "validation": _frame_hash(bundle.val_df),
            "test": _frame_hash(bundle.test_df),
        },
        "model_class": type(model).__name__,
        "active_tasks": list(bundle.target_cols),
        "head_names": list(getattr(model, "heads", {}).keys()),
        "parameter_count": model_parameter_counts["total_parameters"],
        "sampler_class": type(bundle.train_loader.batch_sampler).__name__,
    }
    save_json(runtime_provenance, out_dir / "runtime_provenance.json")
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

    taxonomy_consistency_cfg = (
        (cfg.get("loss", {}) or {}).get("taxonomy_consistency", {}) or {}
    )
    genus_supcon_cfg = (
        (cfg.get("loss", {}) or {}).get(
            "genus_supervised_contrastive", {}
        ) or {}
    )
    species_to_genus_matrix = None
    if bool(taxonomy_consistency_cfg.get("enabled", False)):
        species_to_genus_matrix = build_child_to_parent_matrix(
            bundle.label_to_index_by_task,
            "genus",
            "species",
            device,
        )

    optimizer, optimizer_group_summary = build_optimizer(model, cfg)
    save_json(optimizer_group_summary, out_dir / "optimizer_parameter_groups.json")
    for group in optimizer_group_summary:
        print(
            "Optimizer group "
            f"{group['group_name']}: {group['number_of_parameters']} parameters, "
            f"lr={group['learning_rate']}, weight_decay={group['weight_decay']}"
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
    gradient_strategy_cfg = (
        cfg.get("training", {}).get("gradient_strategy", {}) or {}
    )
    gradient_diagnostics_cfg = (
        cfg.get("training", {}).get("gradient_diagnostics", {}) or {}
    )
    age_supcon_cfg = (
        cfg.get("loss", {}).get("age_supervised_contrastive", {}) or {}
    )
    age_species_adversary_cfg = (
        cfg.get("model", {}).get("age_species_adversary", {}) or {}
    )
    gradient_diagnostics_records: list[dict] = []
    global_step = 0
    ensemble_cfg = (
        (cfg.get("evaluation", {}) or {}).get("checkpoint_ensemble", {}) or {}
    )
    ensemble_enabled = bool(ensemble_cfg.get("enabled", False))
    ensemble_top_k = int(ensemble_cfg.get("top_k", 3))
    top_checkpoints: list[tuple[float, int, Path]] = []
    unfreezing_transitions = [{
        "epoch": 0,
        "trainable_parameters": int(initial_trainable_parameters),
    }]

    print(
        f"Training for {cfg['training']['epochs']} epochs with early stopping: "
        f"{early_enabled}, patience: {patience}, min_delta: {min_delta}"
    )
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        changed, trainable_count = staged_unfreezer.transition(
            model,
            optimizer,
            cfg,
            epoch - 1,
            scheduler=scheduler,
        )
        if changed:
            unfreezing_transitions.append({
                "epoch": epoch - 1,
                "trainable_parameters": trainable_count,
            })
            print(
                f"[{run_name}] Staged unfreezing epoch {epoch - 1}: "
                f"{trainable_count} trainable parameters"
            )
            for group in optimizer.param_groups:
                print(
                    f"    optimizer group {group.get('name', 'unnamed')}: "
                    f"{sum(parameter.numel() for parameter in group['params'])} "
                    f"parameters, lr={group['lr']}, "
                    f"weight_decay={group['weight_decay']}"
                )
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
            epoch=epoch,
            global_step_offset=global_step,
            gradient_strategy_cfg=gradient_strategy_cfg,
            gradient_diagnostics_cfg=gradient_diagnostics_cfg,
            gradient_diagnostics_records=gradient_diagnostics_records,
            age_supcon_cfg=age_supcon_cfg,
            age_species_adversary_cfg=age_species_adversary_cfg,
            genus_supcon_cfg=genus_supcon_cfg,
            taxonomy_consistency_cfg=taxonomy_consistency_cfg,
            species_to_genus_matrix=species_to_genus_matrix,
        )
        global_step += len(bundle.train_loader)
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
                epoch=epoch,
                age_supcon_cfg=age_supcon_cfg,
                age_species_adversary_cfg=age_species_adversary_cfg,
                genus_supcon_cfg=genus_supcon_cfg,
                taxonomy_consistency_cfg=taxonomy_consistency_cfg,
                species_to_genus_matrix=species_to_genus_matrix,
            )[0]
            individual_evaluation_enabled = bool(
                ((cfg.get("evaluation", {}) or {}).get(
                    "individual_level", {}
                ) or {}).get("enabled", False)
            )
            if individual_evaluation_enabled:
                _, _, val_probability_metrics = collect_probability_predictions(
                    models=[model],
                    loader=bundle.val_loader,
                    tasks=bundle.target_cols,
                    index_to_label_by_task=bundle.index_to_label_by_task,
                    device=device,
                    use_amp=use_amp,
                    run_id=run_name,
                    checkpoint=f"validation_epoch_{epoch:03d}",
                    split="validation",
                    maximum_images_per_individual=(
                        bundle.multiview_evaluation_max_images
                    ),
                )
                val_metrics.update(val_probability_metrics)
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

        if ensemble_enabled:
            candidate_dir = out_dir / "validation_checkpoints"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = candidate_dir / f"epoch_{epoch:03d}.pt"
            candidate_payload = build_checkpoint_payload(
                profile=profile,
                model_state=model.state_dict(),
                cfg=cfg,
                label_to_index_by_task=bundle.label_to_index_by_task,
                index_to_label_by_task=bundle.index_to_label_by_task,
                best_val_score=score,
                selection_metric=selection,
                best_epoch=epoch,
                colour_retention=colour_retention,
                colour_percent=colour_percent,
                training_condition=input_condition,
            )
            save_checkpoint(candidate_payload, candidate_path)
            top_checkpoints.append((float(score), int(epoch), candidate_path))
            top_checkpoints.sort(key=lambda item: (-item[0], item[1]))
            while len(top_checkpoints) > ensemble_top_k:
                _, _, rejected_path = top_checkpoints.pop()
                rejected_path.unlink(missing_ok=True)

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
    if gradient_diagnostics_cfg.get("enabled", False):
        pd.DataFrame(
            gradient_diagnostics_records,
            columns=[
                "epoch",
                "step",
                "genus_gradient_norm",
                "species_gradient_norm",
                "age_gradient_norm",
                "genus_species_cosine",
                "genus_age_cosine",
                "species_age_cosine",
            ],
        ).to_csv(out_dir / "gradient_diagnostics.csv", index=False)
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
        cfg=cfg,
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
        age_supcon_cfg=age_supcon_cfg,
        age_species_adversary_cfg=age_species_adversary_cfg,
        genus_supcon_cfg=genus_supcon_cfg,
        taxonomy_consistency_cfg=taxonomy_consistency_cfg,
        species_to_genus_matrix=species_to_genus_matrix,
    )
    test_metrics, true, pred = run_test_evaluation(
        cfg=cfg,
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
        age_supcon_cfg=age_supcon_cfg,
        age_species_adversary_cfg=age_species_adversary_cfg,
        genus_supcon_cfg=genus_supcon_cfg,
        taxonomy_consistency_cfg=taxonomy_consistency_cfg,
        species_to_genus_matrix=species_to_genus_matrix,
    )

    ensemble_metrics = None
    if ensemble_enabled:
        checkpoint_frames = []
        for _, checkpoint_epoch, checkpoint_path in top_checkpoints:
            checkpoint = load_checkpoint(checkpoint_path, map_location=device)
            _validate_checkpoint_contract(checkpoint, bundle, cfg)
            model.load_state_dict(checkpoint["model_state"])
            frame, _, _ = collect_probability_predictions(
                models=[model],
                loader=bundle.test_loader,
                tasks=bundle.target_cols,
                index_to_label_by_task=bundle.index_to_label_by_task,
                device=device,
                use_amp=use_amp,
                run_id=run_name,
                checkpoint=f"epoch_{checkpoint_epoch:03d}",
                split="test",
                maximum_images_per_individual=(
                    bundle.multiview_evaluation_max_images
                ),
            )
            checkpoint_frames.append(frame)
        ensemble_image = ensemble_prediction_frames(checkpoint_frames)
        ensemble_individual = aggregate_individual_probabilities(
            ensemble_image,
            maximum_images=bundle.multiview_evaluation_max_images,
        )
        ensemble_metrics = prediction_metrics(ensemble_image, ensemble_individual)
        ensemble_metrics["checkpoint_epochs"] = [
            epoch for _, epoch, _ in top_checkpoints
        ]
        ensemble_metrics["selection_scores"] = [
            score for score, _, _ in top_checkpoints
        ]
        save_json(ensemble_metrics, out_dir / "test_metrics_ensemble.json")
        public_prediction_frame(ensemble_image).to_csv(
            out_dir / "predictions_ensemble.csv", index=False
        )
        public_prediction_frame(ensemble_individual).to_csv(
            out_dir / "individual_predictions_ensemble.csv", index=False
        )

        holdout_ensemble_rows = []
        for cohort_name, loader in (bundle.data_holdout_loaders or {}).items():
            cohort_frames = []
            for _, checkpoint_epoch, checkpoint_path in top_checkpoints:
                checkpoint = load_checkpoint(checkpoint_path, map_location=device)
                _validate_checkpoint_contract(checkpoint, bundle, cfg)
                model.load_state_dict(checkpoint["model_state"])
                frame, _, _ = collect_probability_predictions(
                    models=[model], loader=loader, tasks=bundle.target_cols,
                    index_to_label_by_task=bundle.index_to_label_by_task,
                    device=device, use_amp=use_amp, run_id=run_name,
                    checkpoint=f"epoch_{checkpoint_epoch:03d}",
                    split="structured_holdout", holdout=cohort_name,
                    maximum_images_per_individual=(
                        bundle.multiview_evaluation_max_images
                    ),
                )
                cohort_frames.append(frame)
            image_frame = ensemble_prediction_frames(cohort_frames)
            individual_frame = aggregate_individual_probabilities(
                image_frame,
                maximum_images=bundle.multiview_evaluation_max_images,
            )
            for task in (cfg.get("data_holdout", {}) or {}).get("primary_tasks", []):
                task_image = image_frame[image_frame["task"] == task]
                task_individual = individual_frame[individual_frame["task"] == task]
                target = (
                    (cfg.get("data_holdout", {}) or {}).get("evaluation_where")
                    or (cfg.get("data_holdout", {}) or {}).get("where")
                    or {}
                ).get(task)
                holdout_ensemble_rows.append({
                    "holdout": (cfg.get("data_holdout", {}) or {}).get("name"),
                    "cohort": cohort_name,
                    "task": task,
                    "target_label": target,
                    **structured_target_metrics(
                        task_image, task_individual, target_label=target
                    ),
                })
        pd.DataFrame(holdout_ensemble_rows).to_csv(
            out_dir / "data_holdout_metrics_ensemble.csv", index=False
        )
        best_checkpoint = load_checkpoint(out_dir / "best_model.pt", map_location=device)
        _validate_checkpoint_contract(best_checkpoint, bundle, cfg)
        model.load_state_dict(best_checkpoint["model_state"])
    age_embedding_artifacts = save_age_embedding_diagnostics(
        model=model,
        loader=bundle.test_loader,
        device=device,
        use_amp=use_amp,
        out_dir=out_dir,
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
    if age_embedding_artifacts is not None:
        result["age_embedding_artifacts"] = age_embedding_artifacts
    result.update({
        "multitask_architecture": cfg.get("model", {}).get(
            "multitask_architecture", "shared_heads"
        ),
        "target_task": cfg.get("model", {}).get("target_task"),
        "branch_mode_used": getattr(model, "branch_mode_used", None),
        "sampler": cfg.get("data", {}).get("sampler", {}).get(
            "type", "default"
        ),
        "gradient_strategy": gradient_strategy_cfg.get("type", "standard"),
        "age_supervised_contrastive_enabled": bool(
            age_supcon_cfg.get("enabled", False)
        ),
        "species_adversary_enabled": bool(
            age_species_adversary_cfg.get("enabled", False)
        ),
        "genus_supervised_contrastive_enabled": bool(
            genus_supcon_cfg.get("enabled", False)
        ),
        "taxonomy_consistency_enabled": bool(
            taxonomy_consistency_cfg.get("enabled", False)
        ),
        "multiview_training_enabled": bool(
            (cfg.get("data", {}).get("multiview", {}) or {}).get("enabled", False)
        ),
        "resolved_config_hash": config_hash,
        "split_hashes": runtime_provenance["split_hashes"],
        "optimizer_parameter_groups": optimizer_group_summary,
        "staged_unfreezing_schedule": staged_unfreezer.resolved_schedule(),
        "staged_unfreezing_transitions": unfreezing_transitions,
        "checkpoint_ensemble_enabled": ensemble_enabled,
        "checkpoint_ensemble_epochs": (
            [epoch for _, epoch, _ in top_checkpoints]
            if ensemble_enabled else []
        ),
    })
    result.update(model_parameter_counts)
    final_optimizer_group_summary = [
        {
            "group_name": group.get("name", "unnamed"),
            "number_of_parameters": int(sum(
                parameter.numel() for parameter in group["params"]
            )),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in optimizer.param_groups
    ]
    result["optimizer_parameter_groups_final"] = final_optimizer_group_summary
    save_json(
        final_optimizer_group_summary,
        out_dir / "optimizer_parameter_groups_final.json",
    )

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
        out_dir / "runtime_provenance.json",
        out_dir / "optimizer_parameter_groups.json",
        out_dir / "optimizer_parameter_groups_final.json",
        out_dir / "run_summary.json",
        out_dir / "best_model.pt",
        out_dir / "gradient_diagnostics.csv",
        out_dir / "joint_species_stage_sampler.csv",
        out_dir / "age_embeddings_best.npz",
        out_dir / "age_embeddings_best_metadata.csv",
        out_dir / "age_embeddings_best_manifest.json",
        out_dir / "predictions_best.csv",
        out_dir / "predictions_last.csv",
        out_dir / "individual_predictions_best.csv",
        out_dir / "individual_predictions_last.csv",
        out_dir / "test_metrics_ensemble.json",
        out_dir / "predictions_ensemble.csv",
        out_dir / "individual_predictions_ensemble.csv",
        out_dir / "data_holdout_metrics_ensemble.csv",
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
    completion = {
        "status": "completed",
        "run_id": run_name,
        "resolved_config_hash": config_hash,
        "checkpoint_hash": _file_hash(out_dir / "best_model.pt"),
        "prediction_hash": _file_hash(out_dir / "predictions_best.csv"),
        "metric_hash": _file_hash(out_dir / "test_metrics_best.json"),
    }
    save_json(completion, out_dir / "completion_manifest.json")

    print("\nBest-checkpoint test metrics:")
    print(test_metrics)
    print("\nLast-checkpoint test metrics:")
    print(last_test_metrics)
    return result
