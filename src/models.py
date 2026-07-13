from __future__ import annotations

import torch.nn as nn
from torchvision import models


def _load_model(name: str, pretrained: bool):
    fn = getattr(models, name)

    if pretrained:
        try:
            return fn(weights="DEFAULT")
        except Exception:
            return fn(pretrained=True)

    try:
        return fn(weights=None)
    except Exception:
        return fn(pretrained=False)


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    model = _load_model(name, pretrained)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # ResNet family
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    # EfficientNet family
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
            return model

        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
            return model

    # Vision Transformer
    if hasattr(model, "heads"):
        if hasattr(model.heads, "head"):
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)
            return model

    # Swin Transformer
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Do not know how to replace classification head for model: {name}"
    )