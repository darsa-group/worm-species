from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.worm_species.config.validation import ConfigValidationError
from src.worm_species.config.validation import validate_config
from src.worm_species.models.factory import build_model
from src.worm_species.models.factory import resolve_dinov3_model_name
from src.worm_species.models.multitask import build_multitask_model


class _FakeTimmBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_features = 4
        self.projection = nn.Linear(3, self.num_features)
        self.head: nn.Module = nn.Identity()

    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int) -> None:
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.projection(inputs))


def _fake_timm_module(calls: list[dict[str, object]]) -> types.ModuleType:
    module = types.ModuleType("timm")

    def create_model(
        name: str, *, pretrained: bool, num_classes: int
    ) -> _FakeTimmBackbone:
        calls.append({
            "name": name,
            "pretrained": pretrained,
            "num_classes": num_classes,
        })
        model = _FakeTimmBackbone()
        model.reset_classifier(num_classes)
        return model

    module.create_model = create_model  # type: ignore[attr-defined]
    return module


class DinoV3ModelContracts(unittest.TestCase):
    def test_meta_alias_resolves_to_timm_checkpoint(self) -> None:
        self.assertEqual(
            resolve_dinov3_model_name("dinov3_vits16"),
            "vit_small_patch16_dinov3.lvd1689m",
        )
        self.assertEqual(
            resolve_dinov3_model_name("dinov3_vitl16_sat493m"),
            "vit_large_patch16_dinov3.sat493m",
        )
        self.assertIsNone(resolve_dinov3_model_name("dinov3_not_real"))

    def test_build_model_adds_trainable_head_after_freezing_backbone(self) -> None:
        calls: list[dict[str, object]] = []
        with patch.dict(sys.modules, {"timm": _fake_timm_module(calls)}):
            model = build_model(
                "dinov3_vits16",
                num_classes=3,
                pretrained=True,
                freeze_backbone=True,
            )

        self.assertEqual(calls, [{
            "name": "vit_small_patch16_dinov3.lvd1689m",
            "pretrained": True,
            "num_classes": 0,
        }])
        self.assertFalse(model.projection.weight.requires_grad)
        self.assertTrue(model.head.weight.requires_grad)
        self.assertEqual(model(torch.zeros(2, 3)).shape, (2, 3))

    def test_multitask_model_uses_pooled_dinov3_features(self) -> None:
        calls: list[dict[str, object]] = []
        with patch.dict(sys.modules, {"timm": _fake_timm_module(calls)}):
            model = build_multitask_model(
                {
                    "model": {
                        "name": "dinov3_vits16",
                        "pretrained": False,
                        "freeze_backbone": False,
                    }
                },
                {"genus": 2, "species": 3},
            )

        outputs = model(torch.zeros(2, 3))
        self.assertEqual(outputs["genus"].shape, (2, 2))
        self.assertEqual(outputs["species"].shape, (2, 3))
        self.assertIsInstance(model.backbone.get_classifier(), nn.Identity)

    def test_validation_accepts_dinov3_and_rejects_unknown_name(self) -> None:
        validate_config(
            {"model": {"name": "dinov3_vits16"}},
            workflow="saved",
            check_paths=False,
            check_model_registry=True,
        )
        with self.assertRaisesRegex(
            ConfigValidationError, "unknown torchvision or DINOv3 model"
        ):
            validate_config(
                {"model": {"name": "dinov3_not_real"}},
                workflow="saved",
                check_paths=False,
                check_model_registry=True,
            )


if __name__ == "__main__":
    unittest.main()
