from __future__ import annotations

import unittest

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.worm_species.data.samplers import JointSpeciesStageSampler
from src.worm_species.models.multitask import (
    MultiTaskClassifier,
    SingleTaskClassifier,
)
from src.worm_species.training.epochs import run_hierarchy_epoch
from src.worm_species.training.gradients import (
    gradient_statistics,
    pcgrad_project,
)
from src.worm_species.training.losses import (
    age_supervised_contrastive_loss,
)


class _VectorBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.fc: nn.Module = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.projection(inputs))


class _AgeDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.tensor(
                [float(index), 1.0, -1.0],
                dtype=torch.float32,
            ),
            "labels": {"age": torch.tensor(index % 2)},
        }


class _MultiTaskDataset(Dataset):
    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.tensor(
                [float(index) / 5.0, 1.0, -1.0],
                dtype=torch.float32,
            ),
            "labels": {
                "genus": torch.tensor(index % 2),
                "species": torch.tensor(index % 3),
                "age": torch.tensor((index // 2) % 2),
            },
        }


class JointSamplerTests(unittest.TestCase):
    def test_sampler_balances_combinations_and_individuals(self) -> None:
        frame = pd.DataFrame({
            "species": ["common"] * 10 + ["rare"] * 2,
            "stage": ["adult"] * 10 + ["juvenile"] * 2,
            "individual": (
                ["worm_many"] * 9 + ["worm_one"]
                + ["rare_one", "rare_two"]
            ),
        })
        sampler = JointSpeciesStageSampler(
            frame,
            species_col="species",
            stage_col="stage",
            group_col="individual",
            samples_per_epoch=6000,
            seed=19,
        )
        draws = list(sampler)
        combinations = frame.loc[draws, ["species", "stage"]].value_counts(
            normalize=True
        )
        self.assertAlmostEqual(
            combinations[("common", "adult")],
            0.5,
            delta=0.04,
        )
        common_draws = [
            index for index in draws
            if frame.loc[index, "species"] == "common"
        ]
        common_individuals = frame.loc[
            common_draws, "individual"
        ].value_counts(normalize=True)
        self.assertAlmostEqual(
            common_individuals["worm_many"],
            0.5,
            delta=0.05,
        )
        self.assertEqual(set(sampler.summary["individuals"]), {2})
        self.assertTrue(
            sampler.summary["effective_combination_probability"]
            .eq(0.5)
            .all()
        )

    def test_sampler_epoch_is_deterministic_but_changes(self) -> None:
        frame = pd.DataFrame({
            "species": ["a", "a", "b", "b"],
            "stage": ["j", "j", "a", "a"],
            "individual": ["1", "2", "3", "4"],
        })
        sampler = JointSpeciesStageSampler(
            frame,
            species_col="species",
            stage_col="stage",
            group_col="individual",
            samples_per_epoch=12,
            seed=7,
        )
        first = list(sampler)
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))


class ContrastiveAndGradientTests(unittest.TestCase):
    def test_supcon_skips_batches_without_positive_pairs(self) -> None:
        loss, stats = age_supervised_contrastive_loss(
            torch.randn(3, 5),
            torch.tensor([0, 1, 2]),
            species_labels=torch.tensor([0, 1, 2]),
        )
        self.assertIsNone(loss)
        self.assertEqual(stats["valid_anchor_count"], 0)
        self.assertEqual(stats["candidate_anchor_count"], 3)

    def test_supcon_prefers_valid_cross_species_age_pairs(self) -> None:
        embeddings = torch.randn(4, 6, requires_grad=True)
        loss, stats = age_supervised_contrastive_loss(
            embeddings,
            torch.tensor([0, 0, 1, 1]),
            species_labels=torch.tensor([0, 1, 0, 1]),
        )
        self.assertIsNotNone(loss)
        self.assertEqual(stats["valid_anchor_count"], 4)
        self.assertAlmostEqual(stats["valid_anchor_proportion"], 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_pcgrad_projects_conflicting_gradients(self) -> None:
        parameter = nn.Parameter(torch.zeros(2))
        gradients = {
            "genus": [torch.tensor([1.0, 0.0])],
            "age": [torch.tensor([-1.0, 1.0])],
        }
        projected, negative = pcgrad_project(gradients, [parameter])
        self.assertEqual(negative, 1.0)
        self.assertIsNotNone(projected[0])
        self.assertTrue(torch.isfinite(projected[0]).all())
        self.assertGreaterEqual(
            torch.dot(projected[0], torch.tensor([0.0, 1.0])).item(),
            0.0,
        )

    def test_pcgrad_leaves_non_conflicting_sum_unchanged(self) -> None:
        parameter = nn.Parameter(torch.zeros(2))
        gradients = {
            "genus": [torch.tensor([1.0, 0.0])],
            "age": [torch.tensor([1.0, 1.0])],
        }
        projected, negative = pcgrad_project(gradients, [parameter])
        self.assertEqual(negative, 0.0)
        self.assertTrue(torch.equal(
            projected[0],
            torch.tensor([2.0, 1.0]),
        ))

    def test_gradient_statistics_reports_norms_and_cosines(self) -> None:
        parameter = nn.Parameter(torch.zeros(2))
        statistics = gradient_statistics(
            {
                "genus": [torch.tensor([1.0, 0.0])],
                "species": [torch.tensor([0.0, 1.0])],
                "age": [torch.tensor([-1.0, 0.0])],
            },
            [parameter],
        )
        self.assertAlmostEqual(statistics["genus_gradient_norm"], 1.0)
        self.assertAlmostEqual(statistics["genus_species_cosine"], 0.0)
        self.assertAlmostEqual(statistics["genus_age_cosine"], -1.0)

    def test_single_task_epoch_only_reports_target_metrics(self) -> None:
        model = SingleTaskClassifier(
            _VectorBackbone(),
            {"genus": 2, "species": 3, "age": 2},
            "age",
        )
        loader = DataLoader(_AgeDataset(), batch_size=2)
        metrics, _, _ = run_hierarchy_epoch(
            model=model,
            loader=loader,
            criteria={"age": nn.CrossEntropyLoss()},
            optimizer=None,
            device=torch.device("cpu"),
            train=False,
            use_amp=False,
        )
        self.assertIn("age_macro_f1", metrics)
        self.assertNotIn("genus_macro_f1", metrics)
        self.assertNotIn("species_macro_f1", metrics)

    def test_epoch_records_gradient_diagnostics_at_interval(self) -> None:
        model = SingleTaskClassifier(
            _VectorBackbone(),
            {"age": 2},
            "age",
        )
        loader = DataLoader(_AgeDataset(), batch_size=2)
        records: list[dict] = []
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics, _, _ = run_hierarchy_epoch(
            model=model,
            loader=loader,
            criteria={"age": nn.CrossEntropyLoss()},
            optimizer=optimizer,
            device=torch.device("cpu"),
            train=True,
            use_amp=False,
            gradient_diagnostics_cfg={
                "enabled": True,
                "interval_steps": 1,
            },
            gradient_diagnostics_records=records,
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["step"], 1)
        self.assertTrue(torch.isfinite(
            torch.tensor(records[0]["age_gradient_norm"])
        ))
        self.assertIn("loss", metrics)

    def test_pcgrad_runs_through_multitask_epoch(self) -> None:
        model = MultiTaskClassifier(
            _VectorBackbone(),
            {"genus": 2, "species": 3, "age": 2},
        )
        loader = DataLoader(_MultiTaskDataset(), batch_size=3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metrics, _, _ = run_hierarchy_epoch(
            model=model,
            loader=loader,
            criteria={
                "genus": nn.CrossEntropyLoss(),
                "species": nn.CrossEntropyLoss(),
                "age": nn.CrossEntropyLoss(),
            },
            optimizer=optimizer,
            device=torch.device("cpu"),
            train=True,
            use_amp=False,
            gradient_strategy_cfg={"type": "pcgrad"},
        )
        self.assertIn("pcgrad_negative_pair_proportion", metrics)
        self.assertGreaterEqual(
            metrics["pcgrad_negative_pair_proportion"], 0.0
        )
        self.assertLessEqual(
            metrics["pcgrad_negative_pair_proportion"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
