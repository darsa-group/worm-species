from __future__ import annotations

import torch.nn as nn
from torchvision import models


DINOV3_MODEL_ALIASES: dict[str, str] = {
    # Meta/PyTorch Hub names, mapped to the equivalent timm checkpoints.
    "dinov3_vits16": "vit_small_patch16_dinov3.lvd1689m",
    "dinov3_vits16plus": "vit_small_plus_patch16_dinov3.lvd1689m",
    "dinov3_vitb16": "vit_base_patch16_dinov3.lvd1689m",
    "dinov3_vitl16": "vit_large_patch16_dinov3.lvd1689m",
    "dinov3_vith16plus": "vit_huge_plus_patch16_dinov3.lvd1689m",
    "dinov3_vit7b16": "vit_7b_patch16_dinov3.lvd1689m",
    "dinov3_convnext_tiny": "convnext_tiny.dinov3_lvd1689m",
    "dinov3_convnext_small": "convnext_small.dinov3_lvd1689m",
    "dinov3_convnext_base": "convnext_base.dinov3_lvd1689m",
    "dinov3_convnext_large": "convnext_large.dinov3_lvd1689m",
    # Explicit satellite-pretrained aliases. The unqualified aliases above use
    # the web-image LVD-1689M weights, matching Meta's default examples.
    "dinov3_vitl16_sat493m": "vit_large_patch16_dinov3.sat493m",
    "dinov3_vit7b16_sat493m": "vit_7b_patch16_dinov3.sat493m",
}

DINOV3_TIMM_MODELS = frozenset(DINOV3_MODEL_ALIASES.values())


def resolve_dinov3_model_name(name: str) -> str | None:
    """Resolve a supported DINOv3 alias or canonical timm model name."""
    if name in DINOV3_MODEL_ALIASES:
        return DINOV3_MODEL_ALIASES[name]
    if name in DINOV3_TIMM_MODELS:
        return name
    return None


def is_dinov3_model_name(name: str) -> bool:
    return resolve_dinov3_model_name(name) is not None


def _load_torchvision_model(name: str, pretrained: bool) -> nn.Module:
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


def _load_dinov3_model(name: str, pretrained: bool) -> nn.Module:
    canonical_name = resolve_dinov3_model_name(name)
    if canonical_name is None:  # pragma: no cover - guarded by build_model
        raise ValueError(f"Unknown DINOv3 model: {name}")
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError(
            "DINOv3 models require timm>=1.0.20. Recreate the conda "
            "environment or install 'timm>=1.0.20'."
        ) from exc

    try:
        # DINOv3 checkpoints are feature backbones. Starting without a head
        # also prevents a mismatched ImageNet head from being downloaded.
        return timm.create_model(
            canonical_name,
            pretrained=pretrained,
            num_classes=0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load DINOv3 model {name!r} as {canonical_name!r}: {exc}"
        ) from exc


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    if is_dinov3_model_name(name):
        model = _load_dinov3_model(name, pretrained)
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False
        if not callable(getattr(model, "reset_classifier", None)):
            raise ValueError(
                f"DINOv3 model {name!r} does not expose timm's classifier API"
            )
        # Add the task head after freezing so freeze_backbone freezes only the
        # pretrained feature extractor, consistent with torchvision models.
        model.reset_classifier(num_classes)
        return model

    model = _load_torchvision_model(name, pretrained)
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
