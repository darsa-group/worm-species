from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .factory import build_model


LOGGER = logging.getLogger(__name__)
TASKS = ("genus", "species", "age")
TASK_LOGIT_KEYS = {
    "genus": "genus_logits",
    "species": "species_logits",
    "age": "age_logits",
}
STANDARD_OUTPUT_KEYS = (
    "genus_logits",
    "species_logits",
    "age_logits",
    "taxonomy_features",
    "age_features",
    "age_embedding",
    "species_adversary_logits",
)


def task_logits(
    outputs: dict[str, torch.Tensor | None],
    task: str,
) -> torch.Tensor | None:
    """Read canonical logits while accepting historical task-name keys."""
    canonical = outputs.get(TASK_LOGIT_KEYS[task])
    return canonical if canonical is not None else outputs.get(task)


def _standard_outputs(
    *,
    logits_by_task: dict[str, torch.Tensor],
    taxonomy_features: torch.Tensor | None,
    age_features: torch.Tensor | None,
    age_embedding: torch.Tensor | None = None,
    species_adversary_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    outputs: dict[str, torch.Tensor | None] = {
        "genus_logits": logits_by_task.get("genus"),
        "species_logits": logits_by_task.get("species"),
        "age_logits": logits_by_task.get("age"),
        "taxonomy_features": taxonomy_features,
        "age_features": age_features,
        "age_embedding": age_embedding,
        "species_adversary_logits": species_adversary_logits,
    }
    # Historical callers indexed outputs by task name. Keep these aliases while
    # all in-repository training and evaluation uses the canonical keys above.
    outputs.update(logits_by_task)
    return outputs


def _pool_feature_tensor(features: torch.Tensor) -> torch.Tensor:
    if features.ndim == 4:
        return torch.flatten(F.adaptive_avg_pool2d(features, 1), 1)
    if features.ndim == 3:
        return features.mean(dim=1)
    if features.ndim == 2:
        return features
    raise ValueError(
        "Backbone features must be a vector, token sequence, or spatial map; "
        f"got shape {tuple(features.shape)}"
    )


def _unwrap_features(features: Any) -> torch.Tensor:
    if isinstance(features, (tuple, list)):
        features = features[0]
    if not torch.is_tensor(features):
        raise TypeError(
            "Backbone must return a tensor or a tuple/list whose first item "
            "is a tensor."
        )
    return features


class TaskAttentionPooling(nn.Module):
    """Learn a task-specific weighted average over spatial/token features."""

    def __init__(self, feature_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.normalise = nn.LayerNorm(feature_dim)
        self.attention = nn.Linear(feature_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            tokens = features.flatten(2).transpose(1, 2)
        elif features.ndim == 3:
            tokens = features
        else:
            raise ValueError(
                "Task-attention pooling requires [N,C,H,W] or [N,L,C] "
                f"features, got {tuple(features.shape)}"
            )
        tokens = self.normalise(tokens)
        weights = torch.softmax(self.attention(tokens), dim=1)
        return self.dropout(torch.sum(weights * tokens, dim=1))


class ResidualAdapter(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        adapter_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.network(features)


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, coefficient: float):
        ctx.coefficient = float(coefficient)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.coefficient * gradient, None


def gradient_reverse(
    inputs: torch.Tensor,
    coefficient: float,
) -> torch.Tensor:
    return _GradientReversal.apply(inputs, coefficient)


class _FeatureModel(nn.Module):
    feature_dim: int

    @staticmethod
    def _remove_classifier_and_get_feature_dim(backbone: nn.Module) -> int:
        # timm models, including DINOv3, expose a uniform classifier API.
        get_classifier = getattr(backbone, "get_classifier", None)
        reset_classifier = getattr(backbone, "reset_classifier", None)
        if callable(get_classifier) and callable(reset_classifier):
            classifier = get_classifier()
            if isinstance(classifier, nn.Linear):
                feature_dim = classifier.in_features
            else:
                feature_dim = getattr(backbone, "num_features", None)
            if isinstance(feature_dim, int) and feature_dim > 0:
                reset_classifier(0)
                return feature_dim

        if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return feature_dim

        if hasattr(backbone, "classifier"):
            classifier = backbone.classifier
            if isinstance(classifier, nn.Linear):
                feature_dim = classifier.in_features
                backbone.classifier = nn.Identity()
                return feature_dim
            if isinstance(classifier, nn.Sequential):
                for index in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[index], nn.Linear):
                        feature_dim = classifier[index].in_features
                        classifier[index] = nn.Identity()
                        return feature_dim

        if hasattr(backbone, "heads") and hasattr(backbone.heads, "head"):
            head = backbone.heads.head
            if isinstance(head, nn.Linear):
                feature_dim = head.in_features
                backbone.heads.head = nn.Identity()
                return feature_dim

        if hasattr(backbone, "head") and isinstance(backbone.head, nn.Linear):
            feature_dim = backbone.head.in_features
            backbone.head = nn.Identity()
            return feature_dim

        raise ValueError(
            "Could not identify the final classifier layer. Add a case in "
            "_FeatureModel._remove_classifier_and_get_feature_dim for this model."
        )

    @staticmethod
    def _spatial_features(
        backbone: nn.Module,
        inputs: torch.Tensor,
    ) -> torch.Tensor | None:
        features_module = getattr(backbone, "features", None)
        if callable(features_module):
            return _unwrap_features(features_module(inputs))
        forward_features = getattr(backbone, "forward_features", None)
        if callable(forward_features):
            return _unwrap_features(forward_features(inputs))
        # torchvision ResNet exposes its spatial trunk as named modules.
        if all(
            hasattr(backbone, name)
            for name in (
                "conv1", "bn1", "relu", "maxpool",
                "layer1", "layer2", "layer3", "layer4",
            )
        ):
            x = backbone.maxpool(
                backbone.relu(backbone.bn1(backbone.conv1(inputs)))
            )
            for name in ("layer1", "layer2", "layer3", "layer4"):
                x = getattr(backbone, name)(x)
            return x
        return None


class MultiTaskClassifier(_FeatureModel):
    """Historical shared backbone and heads, with canonical output aliases."""

    def __init__(
        self,
        base_model: nn.Module,
        num_classes_by_task: dict[str, int],
        *,
        pooling_type: str = "global_average",
        pooling_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = base_model
        self.feature_dim = self._remove_classifier_and_get_feature_dim(
            self.backbone
        )
        self.heads = nn.ModuleDict({
            task: nn.Linear(self.feature_dim, num_classes)
            for task, num_classes in num_classes_by_task.items()
        })
        self.pooling_type = pooling_type
        self._attention_fallback_logged = False
        if pooling_type == "task_attention":
            self.taxonomy_pool = TaskAttentionPooling(
                self.feature_dim, pooling_dropout
            )
            self.age_pool = TaskAttentionPooling(
                self.feature_dim, pooling_dropout
            )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if self.pooling_type == "task_attention":
            spatial = self._spatial_features(self.backbone, inputs)
            if spatial is not None and spatial.ndim in {3, 4}:
                taxonomy_features = self.taxonomy_pool(spatial)
                age_features = self.age_pool(spatial)
            else:
                if not self._attention_fallback_logged:
                    LOGGER.info(
                        "Task-attention pooling unavailable for vector-only "
                        "backbone %s; using the vector directly.",
                        type(self.backbone).__name__,
                    )
                    self._attention_fallback_logged = True
                vector = _pool_feature_tensor(
                    _unwrap_features(self.backbone(inputs))
                )
                taxonomy_features = vector
                age_features = vector
        else:
            vector = _pool_feature_tensor(
                _unwrap_features(self.backbone(inputs))
            )
            taxonomy_features = vector
            age_features = vector

        logits_by_task = {
            task: head(
                age_features if task == "age" else taxonomy_features
            )
            for task, head in self.heads.items()
        }
        return _standard_outputs(
            logits_by_task=logits_by_task,
            taxonomy_features=taxonomy_features,
            age_features=age_features,
        )

    def shared_parameters(self) -> Iterable[nn.Parameter]:
        return self.backbone.parameters()


class SingleTaskClassifier(MultiTaskClassifier):
    """Shared preprocessing/backbone contract with exactly one task head."""

    def __init__(
        self,
        base_model: nn.Module,
        num_classes_by_task: dict[str, int],
        target_task: str,
        *,
        pooling_type: str = "global_average",
        pooling_dropout: float = 0.1,
    ) -> None:
        if target_task not in TASKS:
            raise ValueError(
                f"model.target_task must be one of {TASKS}, got {target_task!r}"
            )
        if target_task not in num_classes_by_task:
            raise ValueError(
                f"No label map is available for target task {target_task!r}"
            )
        super().__init__(
            base_model,
            {target_task: num_classes_by_task[target_task]},
            pooling_type=pooling_type,
            pooling_dropout=pooling_dropout,
        )
        self.target_task = target_task


class SplitTaxonomyAgeClassifier(_FeatureModel):
    """Separate taxonomy and developmental-stage feature pathways."""

    def __init__(
        self,
        base_model: nn.Module,
        num_classes_by_task: dict[str, int],
        *,
        model_name: str,
        branch_mode: str,
        adapter_dim: int,
        adapter_dropout: float,
        pooling_type: str,
        pooling_dropout: float,
        age_projection_enabled: bool,
        adversary_enabled: bool,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.branch_mode_requested = branch_mode
        self.adversary_coefficient = 0.0
        self.feature_dim = self._remove_classifier_and_get_feature_dim(
            base_model
        )
        can_duplicate = self._supports_final_stage_duplication(base_model)
        if branch_mode == "duplicated_final_stage" and not can_duplicate:
            raise ValueError(
                "branch_mode=duplicated_final_stage is only available for "
                "backbones with a separable ConvNeXt final stage"
            )
        self.branch_mode_used = (
            "duplicated_final_stage"
            if branch_mode in {"auto", "duplicated_final_stage"}
            and can_duplicate
            else "residual_adapter"
        )
        self.pooling_type = pooling_type
        self._attention_fallback_logged = False
        LOGGER.info(
            "split_taxonomy_age resolved branch_mode=%s for %s",
            self.branch_mode_used,
            model_name,
        )

        if self.branch_mode_used == "duplicated_final_stage":
            self._configure_duplicated_final_stage(base_model)
        else:
            self.backbone = base_model
            self.taxonomy_adapter = ResidualAdapter(
                self.feature_dim, adapter_dim, adapter_dropout
            )
            self.age_adapter = ResidualAdapter(
                self.feature_dim, adapter_dim, adapter_dropout
            )

        if pooling_type == "task_attention":
            self.taxonomy_pool = TaskAttentionPooling(
                self.feature_dim, pooling_dropout
            )
            self.age_pool = TaskAttentionPooling(
                self.feature_dim, pooling_dropout
            )

        self.heads = nn.ModuleDict({
            task: nn.Linear(self.feature_dim, num_classes)
            for task, num_classes in num_classes_by_task.items()
        })
        self.age_projection = (
            nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.GELU(),
                nn.Linear(256, 128),
            )
            if age_projection_enabled
            else None
        )
        self.species_adversary = (
            nn.Linear(
                self.feature_dim,
                num_classes_by_task["species"],
            )
            if adversary_enabled and "species" in num_classes_by_task
            else None
        )

    @staticmethod
    def _supports_final_stage_duplication(base_model: nn.Module) -> bool:
        features = getattr(base_model, "features", None)
        if isinstance(features, nn.Sequential) and len(features) == 8:
            return True
        stages = getattr(base_model, "stages", None)
        return (
            hasattr(base_model, "stem")
            and isinstance(stages, nn.Sequential)
            and len(stages) == 4
        )

    def _configure_duplicated_final_stage(
        self,
        base_model: nn.Module,
    ) -> None:
        features = getattr(base_model, "features", None)
        if isinstance(features, nn.Sequential) and len(features) == 8:
            modules = list(features.children())
            self.shared_backbone = nn.Sequential(*modules[:-1])
            self.taxonomy_final_stage = modules[-1]
            self.age_final_stage = copy.deepcopy(modules[-1])
            return
        stages = getattr(base_model, "stages")
        self.shared_backbone = nn.Sequential(
            base_model.stem,
            *list(stages.children())[:-1],
        )
        self.taxonomy_final_stage = list(stages.children())[-1]
        self.age_final_stage = copy.deepcopy(self.taxonomy_final_stage)

    def _branch_features(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.branch_mode_used == "duplicated_final_stage":
            shared = self.shared_backbone(inputs)
            taxonomy_raw = self.taxonomy_final_stage(shared)
            age_raw = self.age_final_stage(shared)
        else:
            raw = self._spatial_features(self.backbone, inputs)
            if raw is None:
                raw = _unwrap_features(self.backbone(inputs))
            if (
                self.pooling_type == "task_attention"
                and raw.ndim in {3, 4}
            ):
                taxonomy_vector = self.taxonomy_pool(raw)
                age_vector = self.age_pool(raw)
            else:
                if self.pooling_type == "task_attention":
                    if not self._attention_fallback_logged:
                        LOGGER.info(
                            "Task-attention pooling unavailable for "
                            "vector-only backbone %s; using the vector "
                            "directly.",
                            type(self.backbone).__name__,
                        )
                        self._attention_fallback_logged = True
                vector = _pool_feature_tensor(raw)
                taxonomy_vector = vector
                age_vector = vector
            return (
                self.taxonomy_adapter(taxonomy_vector),
                self.age_adapter(age_vector),
            )

        if self.pooling_type == "task_attention":
            taxonomy_vector = self.taxonomy_pool(taxonomy_raw)
            age_vector = self.age_pool(age_raw)
        else:
            taxonomy_vector = _pool_feature_tensor(taxonomy_raw)
            age_vector = _pool_feature_tensor(age_raw)
        return taxonomy_vector, age_vector

    def set_adversary_coefficient(self, coefficient: float) -> None:
        self.adversary_coefficient = float(coefficient)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        taxonomy_features, age_features = self._branch_features(inputs)
        logits_by_task = {
            task: head(
                age_features if task == "age" else taxonomy_features
            )
            for task, head in self.heads.items()
        }
        age_embedding = (
            F.normalize(self.age_projection(age_features), dim=-1)
            if self.age_projection is not None
            else None
        )
        species_adversary_logits = (
            self.species_adversary(
                gradient_reverse(
                    age_features,
                    self.adversary_coefficient,
                )
            )
            if self.species_adversary is not None
            else None
        )
        return _standard_outputs(
            logits_by_task=logits_by_task,
            taxonomy_features=taxonomy_features,
            age_features=age_features,
            age_embedding=age_embedding,
            species_adversary_logits=species_adversary_logits,
        )

    def shared_parameters(self) -> Iterable[nn.Parameter]:
        if self.branch_mode_used == "duplicated_final_stage":
            return self.shared_backbone.parameters()
        return self.backbone.parameters()


def _pooling_config(cfg: dict) -> tuple[str, float]:
    pooling = (cfg.get("model", {}) or {}).get("pooling", {}) or {}
    pooling_type = str(pooling.get("type", "global_average"))
    return pooling_type, float(pooling.get("dropout", 0.1))


def build_multitask_model(
    cfg: dict,
    num_classes_by_task: dict[str, int],
) -> nn.Module:
    model_cfg = cfg["model"]
    architecture = str(
        model_cfg.get("multitask_architecture", "shared_heads")
    )
    pooling_type, pooling_dropout = _pooling_config(cfg)
    temporary_num_classes = max(num_classes_by_task.values())
    base_model = build_model(
        name=model_cfg["name"],
        num_classes=temporary_num_classes,
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )
    if architecture == "shared_heads":
        return MultiTaskClassifier(
            base_model=base_model,
            num_classes_by_task=num_classes_by_task,
            pooling_type=pooling_type,
            pooling_dropout=pooling_dropout,
        )
    if architecture == "single_task":
        return SingleTaskClassifier(
            base_model=base_model,
            num_classes_by_task=num_classes_by_task,
            target_task=str(model_cfg.get("target_task", "")),
            pooling_type=pooling_type,
            pooling_dropout=pooling_dropout,
        )
    if architecture == "split_taxonomy_age":
        supcon = (
            (cfg.get("loss", {}) or {}).get(
                "age_supervised_contrastive", {}
            )
            or {}
        )
        adversary = model_cfg.get("age_species_adversary", {}) or {}
        return SplitTaxonomyAgeClassifier(
            base_model=base_model,
            num_classes_by_task=num_classes_by_task,
            model_name=str(model_cfg["name"]),
            branch_mode=str(model_cfg.get("branch_mode", "auto")),
            adapter_dim=int(model_cfg.get("adapter_dim", 256)),
            adapter_dropout=float(
                model_cfg.get("adapter_dropout", 0.2)
            ),
            pooling_type=pooling_type,
            pooling_dropout=pooling_dropout,
            age_projection_enabled=bool(supcon.get("enabled", False)),
            adversary_enabled=bool(adversary.get("enabled", False)),
        )
    raise ValueError(
        "model.multitask_architecture must be shared_heads, single_task, "
        f"or split_taxonomy_age; got {architecture!r}"
    )
