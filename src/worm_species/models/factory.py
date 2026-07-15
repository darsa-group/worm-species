from __future__ import annotations

import torch.nn as nn
from torchvision import models


def _load_model(name: str, pretrained: bool) -> nn.Module:
    try:
        constructor = getattr(models, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown torchvision model: {name}") from exc

    if pretrained:
        try:
            return constructor(weights="DEFAULT")
        except TypeError:
            return constructor(pretrained=True)
    try:
        return constructor(weights=None)
    except TypeError:
        return constructor(pretrained=False)


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    model = _load_model(name, pretrained)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            for index in range(len(model.classifier) - 1, -1, -1):
                layer = model.classifier[index]
                if isinstance(layer, nn.Linear):
                    model.classifier[index] = nn.Linear(layer.in_features, num_classes)
                    return model
        if isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model

    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise ValueError(f"Do not know how to replace classification head for model: {name}")
