from __future__ import annotations

import unittest

import pandas as pd
import torch

from src.worm_species.training.losses import build_criteria, hierarchy_consistency_loss


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


class HierarchyConsistencyLossTests(unittest.TestCase):
    def test_hloss_penalises_parent_child_disagreement(self) -> None:
        # Child classes 0 and 1 belong to parent 0; child class 2 belongs to
        # parent 1. The child prediction therefore implies [0.5, 0.5].
        child_to_parent = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        child_logits = torch.log(torch.tensor([[0.3, 0.2, 0.5]]))
        valid_mask = torch.tensor([True])

        consistent_loss = hierarchy_consistency_loss(
            parent_logits=torch.log(torch.tensor([[0.5, 0.5]])),
            child_logits=child_logits,
            child_to_parent_matrix=child_to_parent,
            valid_mask=valid_mask,
        )
        inconsistent_loss = hierarchy_consistency_loss(
            parent_logits=torch.log(torch.tensor([[0.9, 0.1]])),
            child_logits=child_logits,
            child_to_parent_matrix=child_to_parent,
            valid_mask=valid_mask,
        )

        self.assertIsNotNone(consistent_loss)
        self.assertIsNotNone(inconsistent_loss)
        self.assertAlmostEqual(consistent_loss.item(), 0.0, places=6)
        self.assertGreater(inconsistent_loss.item(), consistent_loss.item())

    def test_hloss_remains_finite_for_extreme_disagreement(self) -> None:
        parent_logits = torch.tensor(
            [[1_000.0, -1_000.0]],
            requires_grad=True,
        )
        child_logits = torch.tensor(
            [[-1_000.0, -1_000.0, 1_000.0]],
            requires_grad=True,
        )
        child_to_parent = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        loss = hierarchy_consistency_loss(
            parent_logits=parent_logits,
            child_logits=child_logits,
            child_to_parent_matrix=child_to_parent,
            valid_mask=torch.tensor([True]),
        )

        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(parent_logits.grad)))
        self.assertTrue(torch.all(torch.isfinite(child_logits.grad)))


if __name__ == "__main__":
    unittest.main()
