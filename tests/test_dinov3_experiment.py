from __future__ import annotations

import unittest
from pathlib import Path

from src.worm_species.config.loading import load_config
from src.worm_species.config.normalization import normalize_config
from src.worm_species.config.sweeps import expand_sweep_items
from src.worm_species.config.validation import validate_config
from src.worm_species.evaluation.condition_matrix import (
    resolve_condition_matrix_conditions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "dinov3_rgb_stress.yaml"


class DinoV3RgbStressConfigContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = normalize_config(load_config(CONFIG))

    def test_plan_has_twelve_original_rgb_training_runs(self) -> None:
        validate_config(
            self.config,
            workflow="run_specs",
            check_paths=False,
            check_model_registry=False,
        )
        items = expand_sweep_items(self.config)
        self.assertEqual(len(items), 12)
        self.assertEqual(
            {item.parameter_values["model.name"] for item in items},
            {"dinov3_vitb16", "dinov3_convnext_base"},
        )
        self.assertEqual(
            {item.parameter_values["training.lr"] for item in items},
            {0.0005, 0.0001},
        )
        self.assertEqual(
            {
                item.parameter_values["multi_task.hierarchy_loss.weight"]
                for item in items
            },
            {0.0, 0.5, 1.0},
        )
        self.assertEqual({item.condition["name"] for item in items}, {"original"})

    def test_equal_task_weights_and_train_only_gaussian_augmentation(self) -> None:
        self.assertEqual(
            self.config["multi_task"]["loss_weights"],
            {"genus": 1.0, "species": 1.0, "age": 1.0},
        )
        gaussian = self.config["augmentation"]["gaussian_blur"]
        self.assertEqual(
            gaussian,
            {
                "enabled": True,
                "probability": 0.5,
                "kernel_size": 5,
                "sigma": [0.1, 2.0],
            },
        )

    def test_every_checkpoint_has_twenty_four_test_conditions(self) -> None:
        conditions = resolve_condition_matrix_conditions(self.config)
        self.assertEqual(len(conditions), 24)
        names = {condition["condition"] for condition in conditions}
        self.assertIn("original", names)
        self.assertIn("saturation_000pct", names)
        self.assertIn("gaussian_sigma_2", names)
        self.assertIn("patch_shuffle_grid_8", names)
        self.assertEqual(12 * len(conditions), 288)


if __name__ == "__main__":
    unittest.main()
