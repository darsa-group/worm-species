"""Class-balanced task losses and taxonomy-hierarchy consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from ..data.labels import MISSING_LABEL


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


def age_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    age_labels: torch.Tensor,
    *,
    species_labels: torch.Tensor | None = None,
    individual_labels: torch.Tensor | None = None,
    temperature: float = 0.07,
    cross_species_positives: bool = True,
) -> tuple[torch.Tensor | None, dict[str, float | int]]:
    """Supervised contrastive loss with cross-species age positives preferred."""
    if temperature <= 0:
        raise ValueError("Supervised-contrastive temperature must be positive")
    valid = age_labels != MISSING_LABEL
    if species_labels is not None:
        valid &= species_labels != MISSING_LABEL
    embeddings = embeddings[valid]
    ages = age_labels[valid]
    species = species_labels[valid] if species_labels is not None else None
    individuals = (
        individual_labels[valid] if individual_labels is not None else None
    )
    valid_count = int(len(ages))
    empty_stats = {
        "valid_anchor_count": 0,
        "valid_anchor_proportion": 0.0,
        "candidate_anchor_count": valid_count,
        "cross_species_positive_pairs": 0,
        "same_species_positive_pairs": 0,
        "invalid_anchor_count": valid_count,
    }
    if valid_count < 2:
        return None, empty_stats

    embeddings = F.normalize(embeddings.float(), dim=-1)
    logits = embeddings @ embeddings.T / float(temperature)
    identity = torch.eye(valid_count, dtype=torch.bool, device=logits.device)
    logits_mask = ~identity
    same_age = ages[:, None].eq(ages[None, :]) & logits_mask
    if individuals is not None:
        same_age &= individuals[:, None].ne(individuals[None, :])
    positive_mask = same_age
    cross_species = torch.zeros_like(same_age)
    if cross_species_positives and species is not None:
        cross_species = (
            same_age
            & species[:, None].ne(species[None, :])
        )
        has_cross_species = cross_species.any(dim=1, keepdim=True)
        positive_mask = torch.where(
            has_cross_species,
            cross_species,
            same_age,
        )

    valid_anchors = positive_mask.any(dim=1)
    valid_anchor_count = int(valid_anchors.sum().item())
    stats = {
        "valid_anchor_count": valid_anchor_count,
        "valid_anchor_proportion": (
            float(valid_anchor_count / valid_count)
            if valid_count else 0.0
        ),
        "candidate_anchor_count": valid_count,
        "cross_species_positive_pairs": int(cross_species.sum().item()),
        "same_species_positive_pairs": int(
            (same_age & ~cross_species).sum().item()
        ),
        "invalid_anchor_count": int(valid_count - valid_anchor_count),
    }
    if valid_anchor_count == 0:
        return None, stats

    negatives = ages[:, None].ne(ages[None, :])
    denominator_mask = positive_mask | negatives
    masked_logits = logits.masked_fill(~denominator_mask, -torch.inf)
    log_prob = masked_logits - torch.logsumexp(
        masked_logits, dim=1, keepdim=True
    )
    positive_counts = positive_mask.sum(dim=1).clamp_min(1)
    mean_positive_log_prob = (
        log_prob.masked_fill(~positive_mask, 0.0).sum(dim=1)
        / positive_counts
    )
    return -mean_positive_log_prob[valid_anchors].mean(), stats


def genus_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    genus_labels: torch.Tensor,
    species_labels: torch.Tensor,
    individual_labels: torch.Tensor,
    *,
    temperature: float = 0.07,
    cross_species_positives: bool = True,
) -> tuple[torch.Tensor | None, dict[str, float | int]]:
    """Contrast genera using different biological individuals as evidence."""
    if temperature <= 0:
        raise ValueError("Supervised-contrastive temperature must be positive")
    valid = (
        (genus_labels != MISSING_LABEL)
        & (species_labels != MISSING_LABEL)
        & (individual_labels != MISSING_LABEL)
    )
    embeddings = F.normalize(embeddings[valid].float(), dim=-1)
    genera = genus_labels[valid]
    species = species_labels[valid]
    individuals = individual_labels[valid]
    count = int(len(genera))
    empty = {
        "valid_anchor_count": 0,
        "candidate_anchor_count": count,
        "valid_anchor_fraction": 0.0,
        "cross_species_positive_pairs": 0,
    }
    if count < 2:
        return None, empty
    identity = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    different_individual = individuals[:, None].ne(individuals[None, :])
    same_genus = genera[:, None].eq(genera[None, :]) & ~identity & different_individual
    cross_species = same_genus & species[:, None].ne(species[None, :])
    if cross_species_positives:
        positive = torch.where(
            cross_species.any(dim=1, keepdim=True), cross_species, same_genus
        )
    else:
        positive = same_genus
    valid_anchor = positive.any(dim=1)
    valid_count = int(valid_anchor.sum().item())
    stats = {
        "valid_anchor_count": valid_count,
        "candidate_anchor_count": count,
        "valid_anchor_fraction": float(valid_count / count) if count else 0.0,
        "cross_species_positive_pairs": int(cross_species.sum().item()),
    }
    if not valid_count:
        return None, stats
    logits = embeddings @ embeddings.T / float(temperature)
    negatives = genera[:, None].ne(genera[None, :])
    denominator = positive | negatives
    masked_logits = logits.masked_fill(~denominator, -torch.inf)
    log_prob = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1).clamp_min(1)
    mean_positive = log_prob.masked_fill(~positive, 0.0).sum(dim=1) / positive_count
    return -mean_positive[valid_anchor].mean(), stats


def taxonomy_consistency_loss(
    genus_logits: torch.Tensor,
    species_logits: torch.Tensor,
    species_to_genus_matrix: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    direction: str = "symmetric",
    eps: float = 1e-8,
) -> tuple[torch.Tensor | None, float]:
    """Compare direct genus probabilities with species-implied probabilities."""
    if not torch.any(valid_mask):
        return None, float("nan")
    with torch.autocast(device_type=genus_logits.device.type, enabled=False):
        direct = torch.softmax(genus_logits[valid_mask].float(), dim=1)
        species = torch.softmax(species_logits[valid_mask].float(), dim=1)
        implied = species @ species_to_genus_matrix.float().to(species.device)
        direct = direct.clamp_min(eps)
        implied = implied.clamp_min(eps)
        direct = direct / direct.sum(dim=1, keepdim=True)
        implied = implied / implied.sum(dim=1, keepdim=True)
        direct_to_implied = (direct * (direct.log() - implied.log())).sum(dim=1).mean()
        implied_to_direct = (implied * (implied.log() - direct.log())).sum(dim=1).mean()
        normalized = direction.lower().replace("-", "_")
        if normalized in {"kl", "direct", "direct_genus_to_implied"}:
            loss = direct_to_implied
        elif normalized in {"symmetric", "symmetric_kl"}:
            loss = 0.5 * (direct_to_implied + implied_to_direct)
        elif normalized in {"jensen_shannon", "js"}:
            midpoint = 0.5 * (direct + implied)
            loss = 0.5 * (
                (direct * (direct.log() - midpoint.log())).sum(dim=1).mean()
                + (implied * (implied.log() - midpoint.log())).sum(dim=1).mean()
            )
        else:
            raise ValueError(
                "taxonomy consistency direction must be direct, symmetric, or "
                "jensen_shannon"
            )
        agreement = float(
            direct.argmax(dim=1).eq(implied.argmax(dim=1)).float().mean().item()
        )
        return loss, agreement
