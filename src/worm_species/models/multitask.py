from __future__ import annotations

import torch
from torch import nn

from .factory import build_model


class MultiTaskClassifier(nn.Module):
    """Shared image backbone with one classification head per task."""

    def __init__(self, base_model: nn.Module, num_classes_by_task: dict[str, int]):
        super().__init__()
        self.backbone = base_model
        feature_dim = self._remove_classifier_and_get_feature_dim()
        self.heads = nn.ModuleDict({
            task: nn.Linear(feature_dim, num_classes)
            for task, num_classes in num_classes_by_task.items()
        })

    def _remove_classifier_and_get_feature_dim(self) -> int:
        # timm models, including DINOv3, expose a uniform classifier API.
        # reset_classifier(0) restores feature-extractor output while keeping
        # pooling and normalization behavior owned by the backbone.
        get_classifier = getattr(self.backbone, "get_classifier", None)
        reset_classifier = getattr(self.backbone, "reset_classifier", None)
        if callable(get_classifier) and callable(reset_classifier):
            classifier = get_classifier()
            if isinstance(classifier, nn.Linear):
                feature_dim = classifier.in_features
            else:
                feature_dim = getattr(self.backbone, "num_features", None)
            if isinstance(feature_dim, int) and feature_dim > 0:
                reset_classifier(0)
                return feature_dim

        if hasattr(self.backbone, "fc") and isinstance(self.backbone.fc, nn.Linear):
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            return feature_dim

        if hasattr(self.backbone, "classifier"):
            classifier = self.backbone.classifier
            if isinstance(classifier, nn.Linear):
                feature_dim = classifier.in_features
                self.backbone.classifier = nn.Identity()
                return feature_dim
            if isinstance(classifier, nn.Sequential):
                for index in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[index], nn.Linear):
                        feature_dim = classifier[index].in_features
                        classifier[index] = nn.Identity()
                        return feature_dim

        if hasattr(self.backbone, "heads") and hasattr(self.backbone.heads, "head"):
            head = self.backbone.heads.head
            if isinstance(head, nn.Linear):
                feature_dim = head.in_features
                self.backbone.heads.head = nn.Identity()
                return feature_dim

        if hasattr(self.backbone, "head") and isinstance(self.backbone.head, nn.Linear):
            feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
            return feature_dim

        raise ValueError(
            "Could not identify the final classifier layer. "
            "Add a case in MultiTaskClassifier._remove_classifier_and_get_feature_dim "
            "for your model."
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(inputs)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(
                torch.nn.functional.adaptive_avg_pool2d(features, 1), 1
            )
        return {task: head(features) for task, head in self.heads.items()}


def build_multitask_model(
    cfg: dict,
    num_classes_by_task: dict[str, int],
) -> nn.Module:
    temporary_num_classes = max(num_classes_by_task.values())
    base_model = build_model(
        name=cfg["model"]["name"],
        num_classes=temporary_num_classes,
        pretrained=cfg["model"].get("pretrained", True),
        freeze_backbone=cfg["model"].get("freeze_backbone", False),
    )
    return MultiTaskClassifier(
        base_model=base_model,
        num_classes_by_task=num_classes_by_task,
    )
