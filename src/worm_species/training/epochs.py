"""The hierarchy-capable epoch loop shared verbatim by four trainers."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from ..data.labels import MISSING_LABEL
from .losses import hierarchy_consistency_loss
from .metrics import safe_metric


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
):
    if train:
        model.train()
    else:
        model.eval()

    tasks = list(criteria.keys())
    task_loss_weights = task_loss_weights or {task: 1.0 for task in tasks}
    hierarchy_cfg = hierarchy_cfg or {}
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

    losses = []
    task_losses = {task: [] for task in tasks}
    hierarchy_losses = []
    all_true = {task: [] for task in tasks}
    all_pred = {task: [] for task in tasks}

    complete_exact_correct = 0
    complete_exact_total = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = {
            task: batch["labels"][task].to(device, non_blocking=True)
            for task in tasks
        }

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(
                enabled=(use_amp and device.type == "cuda"),
                device_type=device.type,
            ):
                logits_by_task = model(x)

                total_loss = torch.zeros((), device=device)
                active_weight_sum = 0.0
                loss_by_task: dict[str, torch.Tensor | None] = {}

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
                        task_loss = criteria[task](logits_by_task[task][valid], y[task][valid])
                        weight = float(task_loss_weights.get(task, 1.0))
                        total_loss = total_loss + weight * task_loss
                        active_weight_sum += weight
                        loss_by_task[task] = task_loss
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
                        total_loss = total_loss + hierarchy_weight * hierarchy_loss
                        active_weight_sum += hierarchy_weight
                        loss_by_task["hierarchy"] = hierarchy_loss
                    else:
                        loss_by_task["hierarchy"] = None

                if active_weight_sum == 0:
                    continue

                if normalize_loss_by_active_tasks:
                    total_loss = total_loss / active_weight_sum

            if train:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

        losses.append(float(total_loss.item()))
        if use_hierarchy_loss and loss_by_task.get("hierarchy") is not None:
            hierarchy_losses.append(float(loss_by_task["hierarchy"].item()))

        complete_mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)
        complete_correct = torch.ones(x.shape[0], dtype=torch.bool, device=device)

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
                    task_losses[task].append(float(loss_by_task[task].item()))
                all_true[task].extend(y[task][valid].detach().cpu().numpy().tolist())
                all_pred[task].extend(pred[valid].detach().cpu().numpy().tolist())

        if complete_mask.any():
            complete_exact_total += int(complete_mask.sum().item())
            complete_exact_correct += int((complete_correct & complete_mask).sum().item())
    metrics = {}
    for task in tasks:
        metrics[f"{task}_loss"] = float(task_losses[task][-1]) if task_losses[task] else float("nan")
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    if use_hierarchy_loss:
        metrics["hierarchy_loss"] = (
            float(np.mean(hierarchy_losses)) if hierarchy_losses else float("nan")
        )

    if train:
        return metrics, all_true, all_pred
    else:
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

            task_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            macro_f1_values.append(task_macro_f1)

            metrics[f"{task}_loss"] = float(np.mean(task_losses[task])) if task_losses[task] else float("nan")
            metrics[f"{task}_n"] = int(len(y_true))
            metrics[f"{task}_accuracy"] = safe_metric(accuracy_score, y_true, y_pred)
            metrics[f"{task}_balanced_accuracy"] = safe_metric(balanced_accuracy_score, y_true, y_pred)
            metrics[f"{task}_macro_f1"] = float(task_macro_f1)

        metrics["mean_macro_f1"] = float(np.mean(macro_f1_values)) if macro_f1_values else float("nan")
        metrics["complete_exact_match_accuracy"] = (
            float(complete_exact_correct / complete_exact_total)
            if complete_exact_total > 0
            else float("nan")
        )
        metrics["complete_exact_match_n"] = int(complete_exact_total)

    return metrics, all_true, all_pred
