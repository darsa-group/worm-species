from __future__ import annotations

import unittest

import pandas as pd
import torch

from src.worm_species.training.losses import build_criteria


class TrainingLossConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "barcode": ["a", "b", "c", "d"],
                "genus": ["common", "common", "common", "rare"],
            }
        )
        self.targets = {"genus": "genus"}
        self.labels = {"genus": {"common": 0, "rare": 1}}

    def test_class_weight_true_preserves_balanced_loss(self) -> None:
        criterion = build_criteria(
            self.frame,
            self.targets,
            "barcode",
            self.labels,
            torch.device("cpu"),
            use_class_weights=True,
        )["genus"]

        self.assertIsNotNone(criterion.weight)
        self.assertTrue(torch.allclose(criterion.weight, torch.tensor([0.5, 1.5])))

    def test_class_weight_false_uses_unweighted_cross_entropy(self) -> None:
        criterion = build_criteria(
            self.frame,
            self.targets,
            "barcode",
            self.labels,
            torch.device("cpu"),
            use_class_weights=False,
        )["genus"]

        self.assertIsNone(criterion.weight)


if __name__ == "__main__":
    unittest.main()
