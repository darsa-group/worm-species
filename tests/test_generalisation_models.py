from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.worm_species.models.multitask import (
    MultiTaskClassifier,
    SingleTaskClassifier,
    SplitTaxonomyAgeClassifier,
    STANDARD_OUTPUT_KEYS,
    build_multitask_model,
    gradient_reverse,
)


class _VectorBackbone(nn.Module):
    def __init__(self, feature_dim: int = 4) -> None:
        super().__init__()
        self.projection = nn.Linear(3, feature_dim)
        self.fc: nn.Module = nn.Linear(feature_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.projection(inputs))


class _ConvNeXtLikeBackbone(nn.Module):
    def __init__(self, feature_dim: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, feature_dim, kernel_size=1),
            nn.GELU(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feature_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).mean(dim=(-2, -1))
        return self.classifier(features)


def _split_model(
    *,
    branch_mode: str = "auto",
    pooling_type: str = "global_average",
    projection: bool = False,
    adversary: bool = False,
    backbone: nn.Module | None = None,
) -> SplitTaxonomyAgeClassifier:
    return SplitTaxonomyAgeClassifier(
        base_model=backbone or _ConvNeXtLikeBackbone(),
        num_classes_by_task={"genus": 2, "species": 3, "age": 2},
        model_name="convnext_tiny",
        branch_mode=branch_mode,
        adapter_dim=2,
        adapter_dropout=0.0,
        pooling_type=pooling_type,
        pooling_dropout=0.0,
        age_projection_enabled=projection,
        adversary_enabled=adversary,
    )


class GeneralisationModelTests(unittest.TestCase):
    def test_shared_heads_default_state_dict_remains_strictly_loadable(self) -> None:
        old_config = {
            "model": {
                "name": "fake",
                "pretrained": False,
                "freeze_backbone": False,
            }
        }
        explicit_config = {
            "model": {
                **old_config["model"],
                "multitask_architecture": "shared_heads",
            }
        }
        with patch(
            "src.worm_species.models.multitask.build_model",
            side_effect=lambda **_: _VectorBackbone(),
        ):
            original = build_multitask_model(
                old_config,
                {"genus": 2, "species": 3, "age": 2},
            )
            restored = build_multitask_model(
                explicit_config,
                {"genus": 2, "species": 3, "age": 2},
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(original.state_dict(), path)
            restored.load_state_dict(
                torch.load(path, map_location="cpu"),
                strict=True,
            )
        self.assertEqual(
            set(original.state_dict()),
            set(restored.state_dict()),
        )

    def test_all_models_return_the_standard_output_interface(self) -> None:
        models = [
            MultiTaskClassifier(
                _VectorBackbone(),
                {"genus": 2, "species": 3, "age": 2},
            ),
            SingleTaskClassifier(
                _VectorBackbone(),
                {"genus": 2, "species": 3, "age": 2},
                "age",
            ),
            _split_model(projection=True, adversary=True),
        ]
        for model in models:
            output = model(
                torch.randn(2, 3, 4, 4)
                if isinstance(model, SplitTaxonomyAgeClassifier)
                else torch.randn(2, 3)
            )
            self.assertTrue(set(STANDARD_OUTPUT_KEYS).issubset(output))

    def test_single_task_constructs_only_requested_head(self) -> None:
        model = SingleTaskClassifier(
            _VectorBackbone(),
            {"genus": 2, "species": 3, "age": 2},
            "age",
        )
        output = model(torch.randn(3, 3))
        self.assertEqual(set(model.heads), {"age"})
        self.assertIsNone(output["genus_logits"])
        self.assertIsNone(output["species_logits"])
        self.assertEqual(output["age_logits"].shape, (3, 2))

    def test_convnext_final_stages_do_not_alias_parameters(self) -> None:
        model = _split_model()
        self.assertEqual(model.branch_mode_used, "duplicated_final_stage")
        taxonomy_parameters = list(model.taxonomy_final_stage.parameters())
        age_parameters = list(model.age_final_stage.parameters())
        self.assertEqual(len(taxonomy_parameters), len(age_parameters))
        self.assertTrue(taxonomy_parameters)
        for taxonomy, age in zip(taxonomy_parameters, age_parameters):
            self.assertIsNot(taxonomy, age)
            self.assertNotEqual(taxonomy.data_ptr(), age.data_ptr())
            self.assertTrue(torch.equal(taxonomy, age))

    def test_vector_backbone_uses_residual_adapter_fallback(self) -> None:
        model = _split_model(
            backbone=_VectorBackbone(),
            branch_mode="auto",
            pooling_type="task_attention",
        )
        output = model(torch.randn(2, 3))
        self.assertEqual(model.branch_mode_used, "residual_adapter")
        self.assertEqual(output["genus_logits"].shape, (2, 2))
        self.assertEqual(output["age_logits"].shape, (2, 2))
        taxonomy_parameters = {
            id(parameter) for parameter in model.taxonomy_adapter.parameters()
        }
        age_parameters = {
            id(parameter) for parameter in model.age_adapter.parameters()
        }
        self.assertFalse(taxonomy_parameters & age_parameters)

    def test_task_attention_pooling_and_projection_are_task_specific(self) -> None:
        model = _split_model(
            pooling_type="task_attention",
            projection=True,
        )
        output = model(torch.randn(3, 3, 5, 5))
        self.assertEqual(output["taxonomy_features"].shape, (3, 4))
        self.assertEqual(output["age_features"].shape, (3, 4))
        self.assertEqual(output["age_embedding"].shape, (3, 128))
        self.assertTrue(torch.allclose(
            output["age_embedding"].norm(dim=-1),
            torch.ones(3),
            atol=1e-5,
        ))
        self.assertIsNot(model.taxonomy_pool, model.age_pool)

    def test_gradient_reversal_only_reverses_upstream_gradient(self) -> None:
        features = torch.tensor([[1.0, 2.0]], requires_grad=True)
        classifier = nn.Linear(2, 1, bias=False)
        classifier.weight.data.fill_(1.0)
        gradient_reverse(features, 0.25).sum().backward()
        self.assertTrue(torch.equal(
            features.grad,
            torch.full_like(features, -0.25),
        ))
        self.assertIsNotNone(classifier.weight)


if __name__ == "__main__":
    unittest.main()
