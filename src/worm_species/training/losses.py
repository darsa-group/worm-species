"""Class-balanced task losses and taxonomy-hierarchy consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


def compute_individual_class_weights(
    train_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    label_to_index: dict[str, int],
) -> torch.Tensor:
    labelled = train_df[train_df[target_col].notna()]
    counts = labelled.groupby(target_col)[group_col].nunique().to_dict()

    weights = []
    for label in label_to_index:
        weights.append(1.0 / max(float(counts.get(label, 1.0)), 1.0))

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def build_criteria(
    train_df: pd.DataFrame,
    target_cols: dict[str, str],
    group_col: str,
    label_to_index_by_task: dict[str, dict[str, int]],
    device: torch.device,
    *,
    use_class_weights: bool = True,
) -> dict[str, nn.Module]:
    """Build per-task losses, optionally preserving inverse-frequency weights."""
    criteria = {}

    for task, col in target_cols.items():
        class_weights = None
        if use_class_weights:
            class_weights = compute_individual_class_weights(
                train_df=train_df,
                target_col=col,
                group_col=group_col,
                label_to_index=label_to_index_by_task[task],
            ).to(device)

        criteria[task] = nn.CrossEntropyLoss(weight=class_weights)

    return criteria


def infer_parent_label_from_child_label(child_label: str) -> str:
    """Infer a genus-like parent from a space- or underscore-delimited label."""
    child_label = str(child_label).strip()

    if " " in child_label:
        return child_label.split()[0]
    if "_" in child_label:
        return child_label.split("_")[0]

    return child_label


def build_child_to_parent_matrix(
    label_to_index_by_task: dict[str, dict[str, int]],
    parent_task: str,
    child_task: str,
    device: torch.device,
    child_to_parent: dict[str, str] | None = None,
) -> torch.Tensor:
    """Build the exact legacy child-class to parent-class mapping matrix."""
    if parent_task not in label_to_index_by_task:
        raise ValueError(f"Parent task {parent_task!r} is not in label_to_index_by_task.")
    if child_task not in label_to_index_by_task:
        raise ValueError(f"Child task {child_task!r} is not in label_to_index_by_task.")

    parent_to_index = label_to_index_by_task[parent_task]
    child_to_index = label_to_index_by_task[child_task]
    child_to_parent = child_to_parent or {}

    matrix = torch.zeros(
        len(child_to_index),
        len(parent_to_index),
        dtype=torch.float32,
        device=device,
    )

    missing_parent_labels = []

    for child_label, child_index in child_to_index.items():
        parent_label = child_to_parent.get(
            child_label,
            infer_parent_label_from_child_label(child_label),
        )

        if parent_label not in parent_to_index:
            missing_parent_labels.append((child_label, parent_label))
            continue

        parent_index = parent_to_index[parent_label]
        matrix[child_index, parent_index] = 1.0

    if missing_parent_labels:
        examples = ", ".join(
            f"{child!r}->{parent!r}" for child, parent in missing_parent_labels[:10]
        )
        raise ValueError(
            f"Could not map {len(missing_parent_labels)} {child_task!r} labels "
            f"to valid {parent_task!r} labels. Examples: {examples}. "
            "Either make sure species labels start with the genus name, "
            "or provide multi_task.hierarchy_loss.child_to_parent in the config."
        )

    if not torch.all(matrix.sum(dim=1) == 1):
        raise ValueError(
            f"Each {child_task!r} class must map to exactly one {parent_task!r} class."
        )

    return matrix


def hierarchy_consistency_loss(
    parent_logits: torch.Tensor,
    child_logits: torch.Tensor,
    child_to_parent_matrix: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor | None:
    """Penalise parent/child disagreement on jointly labelled samples.

    The hierarchy calculation is performed in float32 because probability-space
    aggregation and logarithms are numerically unstable under float16 autocast.
    """
    if not torch.any(valid_mask):
        return None

    device_type = parent_logits.device.type

    # Explicitly exclude this calculation from AMP.
    with torch.autocast(device_type=device_type, enabled=False):
        parent_logits = parent_logits[valid_mask].float()
        child_logits = child_logits[valid_mask].float()

        child_to_parent_matrix = child_to_parent_matrix.to(
            device=child_logits.device,
            dtype=torch.float32,
        )

        # log_softmax avoids calculating softmax followed by log for the
        # directly predicted parent distribution.
        parent_log_probs = F.log_softmax(parent_logits, dim=1)
        parent_probs = parent_log_probs.exp()

        # Probability-space aggregation is required for the child-derived
        # parent distribution.
        child_probs = F.softmax(child_logits, dim=1)
        implied_parent_probs = child_probs @ child_to_parent_matrix

        # Aggregation may still produce exact zeros, even in float32.
        implied_parent_probs = implied_parent_probs.clamp_min(eps)

        # Clamping slightly changes the row sums, so restore a valid
        # probability distribution.
        implied_parent_probs = implied_parent_probs / implied_parent_probs.sum(
            dim=1,
            keepdim=True,
        )
        implied_parent_log_probs = implied_parent_probs.log()

        # KL(implied parent || predicted parent):
        # gradients update the parent head only.
        parent_loss = F.kl_div(
            parent_log_probs,
            implied_parent_log_probs.detach(),
            reduction="batchmean",
            log_target=True,
        )

        # KL(predicted parent || implied parent):
        # gradients update the child head only.
        child_loss = F.kl_div(
            implied_parent_log_probs,
            parent_log_probs.detach(),
            reduction="batchmean",
            log_target=True,
        )

        return 0.5 * (parent_loss + child_loss)


def ground_truth_taxonomic_mass_loss(
    child_logits: torch.Tensor,
    true_parent_labels: torch.Tensor,
    child_to_parent_matrix: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Penalise child probability mass outside each sample's true parent.

    Unlike :func:`hierarchy_consistency_loss`, this loss does not consume parent
    logits and therefore cannot pull the parent head towards an incorrect child
    prediction. Parents with no classes in the child vocabulary cannot define a
    child-mass target and are excluded. The calculation is forced to float32 for
    AMP stability.
    """
    if not torch.any(valid_mask):
        return None
    device_type = child_logits.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        logits = child_logits[valid_mask].float()
        parents = true_parent_labels[valid_mask].long()
        matrix = child_to_parent_matrix.to(device=logits.device, dtype=torch.bool)
        if matrix.ndim != 2 or matrix.shape[0] != logits.shape[1]:
            raise ValueError("child_to_parent_matrix shape does not match child logits")
        if parents.min() < 0 or parents.max() >= matrix.shape[1]:
            raise ValueError("True parent label is outside the parent vocabulary")
        allowed = matrix[:, parents].T
        supported = allowed.any(dim=1)
        if not torch.any(supported):
            return None
        logits = logits[supported]
        allowed = allowed[supported]
        log_probs = F.log_softmax(logits, dim=1)
        masked = log_probs.masked_fill(~allowed, -torch.inf)
        true_parent_log_mass = torch.logsumexp(masked, dim=1)
        return -true_parent_log_mass.mean()
