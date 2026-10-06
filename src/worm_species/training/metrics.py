"""Small metric helpers whose edge-case semantics are shared by all trainers."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score


def safe_metric(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    default: float = float("nan"),
) -> float:
    if len(y_true) == 0:
        return default
    return float(metric_fn(y_true, y_pred))


def score_for_selection(
    metrics: dict,
    selection_metric: str,
    mode: str = "max",
) -> float:
    if mode not in {"min", "max"}:
        raise ValueError("selection mode must be 'min' or 'max'")
    value = float(metrics.get(selection_metric, float("nan")))
    if math.isnan(value):
        return -float("inf") if mode == "max" else float("inf")
    return value


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _expected_calibration_error(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Binary equal-width ECE for one target class."""
    if truth.size == 0 or probabilities.size != truth.size:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(truth.size)
    value = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == bins - 1
            else (probabilities >= lower) & (probabilities < upper)
        )
        if not selected.any():
            continue
        value += (
            float(selected.sum())
            / total
            * abs(float(truth[selected].mean()) - float(probabilities[selected].mean()))
        )
    return float(value)


def classification_metric_summary(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    *,
    target_index: int | None = None,
    target_probabilities: np.ndarray | list[float] | None = None,
) -> dict[str, float | int]:
    """Return aggregate and one-vs-rest target metrics from one prediction set.

    Probability metrics are emitted only when target-class probabilities are
    available and both target/non-target examples occur. Hard-label metrics
    remain available for historical runs that saved predictions but no logits.
    """
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    if truth.size == 0 or truth.size != prediction.size:
        return {"n": int(truth.size)}

    result: dict[str, float | int] = {"n": int(truth.size)}
    for average in ("macro", "micro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth,
            prediction,
            average=average,
            zero_division=0,
        )
        result.update({
            f"precision_{average}": float(precision),
            f"recall_{average}": float(recall),
            f"f1_{average}": float(f1),
        })

    if target_index is None:
        return result

    target_truth = truth == int(target_index)
    target_prediction = prediction == int(target_index)
    tp = int(np.sum(target_truth & target_prediction))
    fp = int(np.sum(~target_truth & target_prediction))
    fn = int(np.sum(target_truth & ~target_prediction))
    tn = int(np.sum(~target_truth & ~target_prediction))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    negative_predictive_value = _safe_divide(tn, tn + fn)
    f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall
        else float("nan")
    )
    result.update({
        "target_tp": tp,
        "target_fp": fp,
        "target_fn": fn,
        "target_tn": tn,
        "target_n": int(target_truth.sum()),
        "target_predicted_n": int(target_prediction.sum()),
        "target_precision": precision,
        "target_recall": recall,
        "target_specificity": specificity,
        "target_negative_predictive_value": negative_predictive_value,
        "target_f1": f1,
    })

    if target_probabilities is None:
        return result
    probabilities = np.asarray(target_probabilities, dtype=float)
    if probabilities.size != truth.size:
        raise ValueError(
            "target_probabilities must align one-to-one with y_true"
        )
    if np.unique(target_truth).size < 2:
        return result
    binary_truth = target_truth.astype(int)
    result.update({
        "target_roc_auc": float(roc_auc_score(binary_truth, probabilities)),
        "target_average_precision": float(
            average_precision_score(binary_truth, probabilities)
        ),
        "target_brier_score": float(
            brier_score_loss(binary_truth, probabilities)
        ),
        "target_ece_10bin": _expected_calibration_error(
            binary_truth, probabilities, bins=10
        ),
    })
    return result
