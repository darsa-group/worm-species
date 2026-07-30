"""Model factories and the shared multi-task classification head."""

from .factory import (
    DINOV3_MODEL_ALIASES,
    build_model,
    is_dinov3_model_name,
    resolve_dinov3_model_name,
)
from .multitask import (
    MultiTaskClassifier,
    SingleTaskClassifier,
    SplitTaxonomyAgeClassifier,
    TaskAttentionPooling,
    build_multitask_model,
    gradient_reverse,
    task_logits,
)

__all__ = [
    "DINOV3_MODEL_ALIASES",
    "MultiTaskClassifier",
    "SingleTaskClassifier",
    "SplitTaxonomyAgeClassifier",
    "TaskAttentionPooling",
    "build_model",
    "build_multitask_model",
    "gradient_reverse",
    "is_dinov3_model_name",
    "resolve_dinov3_model_name",
    "task_logits",
]
