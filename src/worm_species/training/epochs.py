"""Hierarchy-capable epoch loop shared by training and fixed evaluations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from ..data.labels import MISSING_LABEL
from ..models.multitask import task_logits
from .gradients import (
    gradient_statistics,
    pcgrad_project,
    replace_shared_gradients,
    task_gradients,
    trainable_shared_parameters,
)
from .losses import (
    age_supervised_contrastive_loss,
    genus_supervised_contrastive_loss,
    hierarchy_consistency_loss,
    taxonomy_consistency_loss,
)
from .metrics import safe_metric


def _adversary_coefficient(config: dict, epoch: int) -> float:
    if not bool(config.get("enabled", False)):
        return 0.0
    maximum = float(config.get("max_weight", 0.1))
    warmup = int(config.get("warmup_epochs", 10))
    if warmup <= 0:
        return maximum
    return maximum * min(max(epoch, 0) / warmup, 1.0)


def run_hierarchy_epoch(
    model: nn.Module,
    loader: DataLoader,
    criteria: dict[str, nn.Module],
    optimizer,
    device: torch.device,
    train: bool,
    scaler=None,
    use_amp: bool = True,
    task_loss_weights: dict[str, float] | None = None,
    normalize_loss_by_active_tasks: bool = True,
    hierarchy_cfg: dict | None = None,
    child_to_parent_matrix: torch.Tensor | None = None,
    use_masked_labels: bool = True,
    *,
    epoch: int = 0,
    global_step_offset: int = 0,
    gradient_strategy_cfg: dict | None = None,
    gradient_diagnostics_cfg: dict | None = None,
    gradient_diagnostics_records: list[dict] | None = None,
    age_supcon_cfg: dict | None = None,
    age_species_adversary_cfg: dict | None = None,
    genus_supcon_cfg: dict | None = None,
    taxonomy_consistency_cfg: dict | None = None,
    species_to_genus_matrix: torch.Tensor | None = None,
):
    if train:
        model.train()
        sampler = getattr(loader, "sampler", None)
        if callable(getattr(sampler, "set_epoch", None)):
            sampler.set_epoch(epoch)
        batch_sampler = getattr(loader, "batch_sampler", None)
        if callable(getattr(batch_sampler, "set_epoch", None)):
            batch_sampler.set_epoch(epoch)
        dataset = getattr(loader, "dataset", None)
        if callable(getattr(dataset, "set_epoch", None)):
            dataset.set_epoch(epoch)
    else:
        model.eval()

    tasks = list(criteria.keys())
    task_loss_weights = task_loss_weights or {
        task: 1.0 for task in tasks
    }
    hierarchy_cfg = hierarchy_cfg or {}
    gradient_strategy_cfg = gradient_strategy_cfg or {}
    gradient_diagnostics_cfg = gradient_diagnostics_cfg or {}
    age_supcon_cfg = age_supcon_cfg or {}
    age_species_adversary_cfg = age_species_adversary_cfg or {}
    genus_supcon_cfg = genus_supcon_cfg or {}
    taxonomy_consistency_cfg = taxonomy_consistency_cfg or {}
    gradient_strategy = str(
        gradient_strategy_cfg.get("type", "standard")
    )
    if gradient_strategy not in {"standard", "pcgrad"}:
        raise ValueError(
            "training.gradient_strategy.type must be standard or pcgrad"
        )
    diagnostics_enabled = (
        train and bool(gradient_diagnostics_cfg.get("enabled", False))
    )
    diagnostics_interval = int(
        gradient_diagnostics_cfg.get("interval_steps", 100)
    )
    if diagnostics_enabled and diagnostics_interval <= 0:
        raise ValueError(
            "training.gradient_diagnostics.interval_steps must be positive"
        )
    pcgrad_enabled = train and gradient_strategy == "pcgrad"
    shared_parameters = (
        trainable_shared_parameters(model)
        if pcgrad_enabled or diagnostics_enabled
        else []
    )

    hierarchy_enabled = bool(hierarchy_cfg.get("enabled", False))
    hierarchy_parent_task = hierarchy_cfg.get("parent_task", "genus")
    hierarchy_child_task = hierarchy_cfg.get("child_task", "species")
    hierarchy_weight = float(
        hierarchy_cfg.get(
            "weight",
            task_loss_weights.get("hierarchy", 0.1),
        )
    )
    use_hierarchy_loss = (
        hierarchy_enabled
        and hierarchy_weight > 0.0
        and child_to_parent_matrix is not None
        and hierarchy_parent_task in tasks
        and hierarchy_child_task in tasks
    )
    use_supcon = bool(age_supcon_cfg.get("enabled", False))
    use_adversary = bool(age_species_adversary_cfg.get("enabled", False))
    use_genus_supcon = bool(genus_supcon_cfg.get("enabled", False))
    use_taxonomy_consistency = bool(
        taxonomy_consistency_cfg.get("enabled", False)
    )
    if train and callable(getattr(model, "set_adversary_coefficient", None)):
        model.set_adversary_coefficient(
            _adversary_coefficient(age_species_adversary_cfg, epoch)
        )

    losses = []
    task_losses = {task: [] for task in tasks}
    hierarchy_losses = []
    supcon_losses = []
    supcon_valid_anchors = 0
    supcon_candidate_anchors = 0
    supcon_cross_species_pairs = 0
    supcon_same_species_pairs = 0
    supcon_invalid_anchors = 0
    genus_supcon_losses = []
    genus_valid_anchors = 0
    genus_candidate_anchors = 0
    genus_cross_species_pairs = 0
    taxonomy_losses = []
    taxonomy_agreements = []
    adversary_losses = []
    adversary_correct = 0
    adversary_total = 0
    negative_pair_proportions = []
    all_true = {task: [] for task in tasks}
    all_pred = {task: [] for task in tasks}
    diagnostic_image_true = {task: [] for task in tasks}
    diagnostic_image_pred = {task: [] for task in tasks}

    complete_exact_correct = 0
    complete_exact_total = 0

    for batch_index, batch in enumerate(loader):
        x = batch["image"].to(device, non_blocking=True)
        y = {
            task: batch["labels"][task].to(device, non_blocking=True)
            for task in tasks
        }
        barcode_values = [str(value) for value in batch.get("barcode", [])]
        barcode_lookup = {
            value: index for index, value in enumerate(dict.fromkeys(barcode_values))
        }
        individual_labels = torch.tensor(
            [barcode_lookup[value] for value in barcode_values],
            dtype=torch.long,
            device=device,
        )

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(
                enabled=(use_amp and device.type == "cuda"),
                device_type=device.type,
            ):
                view_mask = batch.get("view_mask")
                if view_mask is not None:
                    view_mask = view_mask.to(device, non_blocking=True)
                outputs = (
                    model(x, view_mask=view_mask)
                    if view_mask is not None
                    else model(x)
                )
                logits_by_task = {}
                for task in tasks:
                    logits = task_logits(outputs, task)
                    if logits is None:
                        raise ValueError(
                            f"Model returned no logits for active task {task!r}"
                        )
                    logits_by_task[task] = logits

                total_loss = torch.zeros((), device=device)
                active_weight_sum = 0.0
                loss_by_task: dict[str, torch.Tensor | None] = {}
                task_objectives: dict[str, torch.Tensor] = {}

                for task in tasks:
                    missing = y[task] == MISSING_LABEL
                    if not use_masked_labels and missing.any():
                        raise ValueError(
                            "training.use_masked_labels=false encountered a "
                            f"missing {task!r} label; no rows were dropped"
                        )
                    valid = ~missing if use_masked_labels else torch.ones_like(
                        missing, dtype=torch.bool
                    )

                    if valid.any():
                        task_loss = criteria[task](
                            logits_by_task[task][valid], y[task][valid]
                        )
                        weight = float(task_loss_weights.get(task, 1.0))
                        weighted = weight * task_loss
                        total_loss = total_loss + weighted
                        active_weight_sum += weight
                        loss_by_task[task] = task_loss
                        task_objectives[task] = weighted
                    else:
                        loss_by_task[task] = None

                if use_hierarchy_loss:
                    hierarchy_valid = (
                        (y[hierarchy_parent_task] != MISSING_LABEL)
                        & (y[hierarchy_child_task] != MISSING_LABEL)
                    )
                    hierarchy_loss = hierarchy_consistency_loss(
                        parent_logits=logits_by_task[hierarchy_parent_task],
                        child_logits=logits_by_task[hierarchy_child_task],
                        child_to_parent_matrix=child_to_parent_matrix,
                        valid_mask=hierarchy_valid,
                    )
                    if hierarchy_loss is not None:
                        total_loss = (
                            total_loss + hierarchy_weight * hierarchy_loss
                        )
                        active_weight_sum += hierarchy_weight
                        loss_by_task["hierarchy"] = hierarchy_loss
                    else:
                        loss_by_task["hierarchy"] = None

                if active_weight_sum == 0:
                    continue
                if normalize_loss_by_active_tasks:
                    total_loss = total_loss / active_weight_sum
                    task_objectives = {
                        task: objective / active_weight_sum
                        for task, objective in task_objectives.items()
                    }

                supcon_loss = None
                supcon_stats = {
                    "valid_anchor_count": 0,
                    "candidate_anchor_count": 0,
                }
                age_embedding = outputs.get("age_embedding")
                if use_supcon:
                    if age_embedding is None:
                        raise ValueError(
                            "Age supervised-contrastive loss is enabled but "
                            "the model returned no age_embedding"
                        )
                    if "age" not in y:
                        raise ValueError(
                            "Age supervised-contrastive loss requires age labels"
                        )
                    supcon_loss, supcon_stats = (
                        age_supervised_contrastive_loss(
                            age_embedding,
                            y["age"],
                            species_labels=y.get("species"),
                            individual_labels=individual_labels,
                            temperature=float(
                                age_supcon_cfg.get("temperature", 0.07)
                            ),
                            cross_species_positives=bool(
                                age_supcon_cfg.get(
                                    "cross_species_positives", True
                                )
                            ),
                        )
                    )
                    if supcon_loss is not None:
                        total_loss = total_loss + float(
                            age_supcon_cfg.get("weight", 0.1)
                        ) * supcon_loss

                genus_supcon_loss = None
                genus_stats = {
                    "valid_anchor_count": 0,
                    "candidate_anchor_count": 0,
                    "cross_species_positive_pairs": 0,
                }
                if use_genus_supcon:
                    genus_embedding = outputs.get("genus_embedding")
                    if genus_embedding is None or not {"genus", "species"}.issubset(y):
                        raise ValueError(
                            "Genus supervised contrastive loss requires genus and "
                            "species labels plus genus_embedding"
                        )
                    genus_supcon_loss, genus_stats = genus_supervised_contrastive_loss(
                        genus_embedding,
                        y["genus"],
                        y["species"],
                        individual_labels,
                        temperature=float(genus_supcon_cfg.get("temperature", 0.07)),
                        cross_species_positives=bool(
                            genus_supcon_cfg.get("cross_species_positives", True)
                        ),
                    )
                    if genus_supcon_loss is not None:
                        total_loss = total_loss + float(
                            genus_supcon_cfg.get("weight", 0.05)
                        ) * genus_supcon_loss

                taxonomy_loss = None
                taxonomy_agreement = float("nan")
                if use_taxonomy_consistency:
                    if species_to_genus_matrix is None or not {"genus", "species"}.issubset(y):
                        raise ValueError(
                            "Taxonomy consistency requires genus/species heads and mapping"
                        )
                    taxonomy_valid = (
                        (y["genus"] != MISSING_LABEL)
                        & (y["species"] != MISSING_LABEL)
                    )
                    taxonomy_loss, taxonomy_agreement = taxonomy_consistency_loss(
                        logits_by_task["genus"],
                        logits_by_task["species"],
                        species_to_genus_matrix,
                        taxonomy_valid,
                        direction=str(
                            taxonomy_consistency_cfg.get("direction", "symmetric")
                        ),
                    )
                    if taxonomy_loss is not None:
                        total_loss = total_loss + float(
                            taxonomy_consistency_cfg.get("weight", 0.05)
                        ) * taxonomy_loss

                adversary_loss = None
                adversary_logits = outputs.get("species_adversary_logits")
                if use_adversary:
                    if adversary_logits is None or "species" not in y:
                        raise ValueError(
                            "Species adversary requires split model species "
                            "logits and species labels"
                        )
                    adversary_valid = y["species"] != MISSING_LABEL
                    if adversary_valid.any():
                        adversary_loss = F.cross_entropy(
                            adversary_logits[adversary_valid],
                            y["species"][adversary_valid],
                        )
                        total_loss = total_loss + float(
                            age_species_adversary_cfg.get("weight", 0.05)
                        ) * adversary_loss

            raw_task_gradients = None
            global_step = global_step_offset + batch_index + 1
            record_diagnostics = (
                diagnostics_enabled
                and global_step % diagnostics_interval == 0
            )
            if shared_parameters and (pcgrad_enabled or record_diagnostics):
                raw_task_gradients = task_gradients(
                    task_objectives,
                    shared_parameters,
                )
                if record_diagnostics:
                    statistics = gradient_statistics(
                        raw_task_gradients,
                        shared_parameters,
                    )
                    record = {
                        "epoch": int(epoch),
                        "step": int(global_step),
                        "genus_gradient_norm": statistics.get(
                            "genus_gradient_norm", float("nan")
                        ),
                        "species_gradient_norm": statistics.get(
                            "species_gradient_norm", float("nan")
                        ),
                        "age_gradient_norm": statistics.get(
                            "age_gradient_norm", float("nan")
                        ),
                        "genus_species_cosine": statistics.get(
                            "genus_species_cosine", float("nan")
                        ),
                        "genus_age_cosine": statistics.get(
                            "genus_age_cosine", float("nan")
                        ),
                        "species_age_cosine": statistics.get(
                            "species_age_cosine", float("nan")
                        ),
                    }
                    if gradient_diagnostics_records is not None:
                        gradient_diagnostics_records.append(record)

            if train:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    if pcgrad_enabled and raw_task_gradients is not None:
                        projected, negative_proportion = pcgrad_project(
                            raw_task_gradients, shared_parameters
                        )
                        replace_shared_gradients(
                            parameters=shared_parameters,
                            raw_task_gradients=raw_task_gradients,
                            projected_gradients=projected,
                        )
                        negative_pair_proportions.append(
                            negative_proportion
                        )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if pcgrad_enabled and raw_task_gradients is not None:
                        projected, negative_proportion = pcgrad_project(
                            raw_task_gradients, shared_parameters
                        )
                        replace_shared_gradients(
                            parameters=shared_parameters,
                            raw_task_gradients=raw_task_gradients,
                            projected_gradients=projected,
                        )
                        negative_pair_proportions.append(
                            negative_proportion
                        )
                    optimizer.step()

        losses.append(float(total_loss.item()))
        if use_hierarchy_loss and loss_by_task.get("hierarchy") is not None:
            hierarchy_losses.append(
                float(loss_by_task["hierarchy"].item())
            )
        if supcon_loss is not None:
            supcon_losses.append(float(supcon_loss.item()))
        supcon_valid_anchors += int(
            supcon_stats.get("valid_anchor_count", 0)
        )
        supcon_candidate_anchors += int(
            supcon_stats.get("candidate_anchor_count", 0)
        )
        supcon_cross_species_pairs += int(
            supcon_stats.get("cross_species_positive_pairs", 0)
        )
        supcon_same_species_pairs += int(
            supcon_stats.get("same_species_positive_pairs", 0)
        )
        supcon_invalid_anchors += int(
            supcon_stats.get("invalid_anchor_count", 0)
        )
        if genus_supcon_loss is not None:
            genus_supcon_losses.append(float(genus_supcon_loss.item()))
        genus_valid_anchors += int(genus_stats.get("valid_anchor_count", 0))
        genus_candidate_anchors += int(genus_stats.get("candidate_anchor_count", 0))
        genus_cross_species_pairs += int(
            genus_stats.get("cross_species_positive_pairs", 0)
        )
        if taxonomy_loss is not None:
            taxonomy_losses.append(float(taxonomy_loss.item()))
        if np.isfinite(taxonomy_agreement):
            taxonomy_agreements.append(float(taxonomy_agreement))
        if adversary_loss is not None:
            adversary_losses.append(float(adversary_loss.item()))
            adversary_valid = y["species"] != MISSING_LABEL
            adversary_predictions = adversary_logits.argmax(dim=1)
            adversary_correct += int(
                adversary_predictions[adversary_valid]
                .eq(y["species"][adversary_valid])
                .sum()
                .item()
            )
            adversary_total += int(adversary_valid.sum().item())

        complete_mask = torch.ones(
            x.shape[0], dtype=torch.bool, device=device
        )
        complete_correct = torch.ones(
            x.shape[0], dtype=torch.bool, device=device
        )
        for task in tasks:
            pred = logits_by_task[task].argmax(dim=1)
            valid = (
                y[task] != MISSING_LABEL
                if use_masked_labels
                else torch.ones_like(y[task], dtype=torch.bool)
            )
            complete_mask &= valid
            complete_correct &= pred.eq(y[task])
            if valid.any():
                if loss_by_task[task] is not None:
                    task_losses[task].append(
                        float(loss_by_task[task].item())
                    )
                all_true[task].extend(
                    y[task][valid].detach().cpu().numpy().tolist()
                )
                all_pred[task].extend(
                    pred[valid].detach().cpu().numpy().tolist()
                )
            image_logits = outputs.get("image_logits_by_task")
            if image_logits is not None and task in image_logits:
                per_view = image_logits[task]
                mask = (
                    view_mask
                    if view_mask is not None
                    else torch.ones(per_view.shape[:2], dtype=torch.bool, device=device)
                )
                repeated = y[task][:, None].expand_as(mask)
                diagnostic_valid = mask & (repeated != MISSING_LABEL)
                if diagnostic_valid.any():
                    diagnostic_image_true[task].extend(
                        repeated[diagnostic_valid].detach().cpu().tolist()
                    )
                    diagnostic_image_pred[task].extend(
                        per_view.argmax(dim=-1)[diagnostic_valid].detach().cpu().tolist()
                    )

        if complete_mask.any():
            complete_exact_total += int(complete_mask.sum().item())
            complete_exact_correct += int(
                (complete_correct & complete_mask).sum().item()
            )

    metrics = {
        f"{task}_loss": (
            float(task_losses[task][-1])
            if task_losses[task]
            else float("nan")
        )
        for task in tasks
    }
    metrics["loss"] = (
        float(np.mean(losses)) if losses else float("nan")
    )
    if use_hierarchy_loss:
        metrics["hierarchy_loss"] = (
            float(np.mean(hierarchy_losses))
            if hierarchy_losses
            else float("nan")
        )
    if use_supcon:
        metrics["age_supervised_contrastive_loss"] = (
            float(np.mean(supcon_losses))
            if supcon_losses else float("nan")
        )
        metrics["age_supcon_valid_anchor_count"] = (
            supcon_valid_anchors
        )
        metrics["age_supcon_candidate_anchor_count"] = (
            supcon_candidate_anchors
        )
        metrics["age_supcon_valid_anchor_proportion"] = (
            float(supcon_valid_anchors / supcon_candidate_anchors)
            if supcon_candidate_anchors else 0.0
        )
        metrics["valid_anchor_fraction"] = metrics[
            "age_supcon_valid_anchor_proportion"
        ]
        metrics["number_of_cross_species_positive_pairs"] = (
            supcon_cross_species_pairs
        )
        metrics["number_of_same_species_positive_pairs"] = (
            supcon_same_species_pairs
        )
        metrics["number_of_invalid_anchors"] = supcon_invalid_anchors
    if use_genus_supcon:
        metrics["genus_supcon_loss"] = (
            float(np.mean(genus_supcon_losses))
            if genus_supcon_losses else float("nan")
        )
        metrics["valid_genus_anchor_fraction"] = (
            float(genus_valid_anchors / genus_candidate_anchors)
            if genus_candidate_anchors else 0.0
        )
        metrics["cross_species_genus_positive_pairs"] = (
            genus_cross_species_pairs
        )
    if use_taxonomy_consistency:
        metrics["taxonomy_consistency_loss"] = (
            float(np.mean(taxonomy_losses))
            if taxonomy_losses else float("nan")
        )
        metrics["genus_species_agreement_rate"] = (
            float(np.mean(taxonomy_agreements))
            if taxonomy_agreements else float("nan")
        )
    if use_adversary:
        metrics["species_adversary_loss"] = (
            float(np.mean(adversary_losses))
            if adversary_losses else float("nan")
        )
        metrics["species_adversary_accuracy"] = (
            float(adversary_correct / adversary_total)
            if adversary_total else float("nan")
        )
        metrics["species_adversary_n"] = adversary_total
    if pcgrad_enabled:
        metrics["pcgrad_negative_pair_proportion"] = (
            float(np.mean(negative_pair_proportions))
            if negative_pair_proportions else 0.0
        )

    if train:
        metrics["individual_level_training_loss"] = metrics["loss"]
        for task in tasks:
            if diagnostic_image_true[task]:
                metrics[f"{task}_image_diagnostic_accuracy"] = safe_metric(
                    accuracy_score,
                    diagnostic_image_true[task],
                    diagnostic_image_pred[task],
                )
                metrics[f"{task}_image_diagnostic_macro_f1"] = float(
                    f1_score(
                        diagnostic_image_true[task],
                        diagnostic_image_pred[task],
                        average="macro",
                        zero_division=0,
                    )
                )
        return metrics, all_true, all_pred

    macro_f1_values = []
    for task in tasks:
        y_true = np.array(all_true[task], dtype=int)
        y_pred = np.array(all_pred[task], dtype=int)
        if len(y_true) == 0:
            metrics[f"{task}_loss"] = float("nan")
            metrics[f"{task}_n"] = 0
            metrics[f"{task}_accuracy"] = float("nan")
            metrics[f"{task}_balanced_accuracy"] = float("nan")
            metrics[f"{task}_macro_f1"] = float("nan")
            continue

        task_macro_f1 = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        macro_f1_values.append(task_macro_f1)
        metrics[f"{task}_loss"] = (
            float(np.mean(task_losses[task]))
            if task_losses[task] else float("nan")
        )
        metrics[f"{task}_n"] = int(len(y_true))
        metrics[f"{task}_accuracy"] = safe_metric(
            accuracy_score, y_true, y_pred
        )
        metrics[f"{task}_balanced_accuracy"] = safe_metric(
            balanced_accuracy_score, y_true, y_pred
        )
        metrics[f"{task}_macro_f1"] = float(task_macro_f1)

    metrics["mean_macro_f1"] = (
        float(np.mean(macro_f1_values))
        if macro_f1_values else float("nan")
    )
    metrics["complete_exact_match_accuracy"] = (
        float(complete_exact_correct / complete_exact_total)
        if complete_exact_total > 0
        else float("nan")
    )
    metrics["complete_exact_match_n"] = int(complete_exact_total)
    return metrics, all_true, all_pred
