from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.worm_species.config.validation import validate_config
from src.worm_species.data.conditions import PatchShuffle
from src.worm_species.data.conditions import build_condition_transform
from src.worm_species.data.conditions import build_test_condition_transform
from src.worm_species.evaluation.cue_suppression import generate_test_cue_conditions
from src.worm_species.experiments.conditions import generate_conditions
from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs" / "experiments" / "patch_shuffle_matrix.yaml"
LOCAL_CLUSTER = ROOT / "configs" / "clusters" / "local.yaml"
MODELS = ("resnet18", "resnet50", "efficientnet_b0", "vit_b_16")
CONDITIONS = (
    "original",
    "patch_shuffle_grid_2",
    "patch_shuffle_grid_4",
)


class PatchShuffleMatrixContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_submission_config(EXPERIMENT, LOCAL_CLUSTER)
        cls.plan = plan_submission(cls.config)

    def test_explicit_config_is_valid_without_loading_model_weights(self) -> None:
        validate_config(
            self.config,
            workflow="run_specs",
            check_paths=False,
            check_model_registry=False,
        )
        self.assertTrue(self.config["model"]["pretrained"])
        self.assertFalse(self.config["model"]["freeze_backbone"])
        self.assertEqual(
            tuple(self.config["sweep"]["parameters"]["model.name"]),
            MODELS,
        )

    def test_four_models_by_three_conditions_produce_twelve_trainings(self) -> None:
        plan = self.plan
        self.assertEqual(plan.array_size, 12)
        self.assertEqual(plan.expected_internal_training_runs_per_task, 1)
        self.assertEqual(plan.models, MODELS)
        self.assertEqual(plan.conditions, CONDITIONS)
        self.assertEqual(
            [(spec.model, spec.training_condition) for spec in plan.run_specs],
            [(model, condition) for model in MODELS for condition in CONDITIONS],
        )
        self.assertEqual(len({spec.run_id for spec in plan.run_specs}), 12)
        self.assertEqual(len({spec.config_sha256 for spec in plan.run_specs}), 12)
        self.assertEqual(len({spec.output_relpath for spec in plan.run_specs}), 12)

    def test_each_task_is_one_pretrained_canonical_training_run(self) -> None:
        for spec in self.plan.run_specs:
            with self.subTest(model=spec.model, condition=spec.training_condition):
                resolved = spec.resolved_config
                self.assertTrue(resolved["model"]["pretrained"])
                self.assertFalse(resolved["model"]["freeze_backbone"])
                self.assertFalse(resolved["sweep"]["enabled"])
                self.assertFalse(resolved["colour_ablation"]["enabled"])
                self.assertFalse(resolved["matched_condition_training"]["enabled"])
                self.assertEqual(
                    spec.trainer_command,
                    (
                        "python",
                        "-m",
                        "worm_species.training",
                        "--config",
                        "resolved_run_config.yaml",
                        "--single-run",
                    ),
                )

    def test_matched_training_and_fixed_rgb_stress_remain_separate(self) -> None:
        original_specs = [
            spec
            for spec in self.plan.run_specs
            if spec.training_condition == "original"
        ]
        transformed_specs = [
            spec
            for spec in self.plan.run_specs
            if spec.training_condition != "original"
        ]
        self.assertEqual(len(original_specs), 4)
        self.assertEqual(len(transformed_specs), 8)

        for spec in original_specs:
            with self.subTest(model=spec.model, training="original"):
                self.assertEqual(spec.experiment_type, "matched_and_rgb_stress")
                self.assertTrue(
                    spec.resolved_config["test_cue_suppression"]["enabled"]
                )
                self.assertEqual(
                    [
                        condition["condition"]
                        for condition in generate_test_cue_conditions(
                            spec.resolved_config
                        )
                    ],
                    ["patch_shuffle_grid_2", "patch_shuffle_grid_4"],
                )

        for spec in transformed_specs:
            with self.subTest(
                model=spec.model, training=spec.training_condition
            ):
                self.assertEqual(spec.experiment_type, "matched_condition")
                self.assertFalse(
                    spec.resolved_config["test_cue_suppression"]["enabled"]
                )
                self.assertEqual(
                    spec.resolved_config["input_condition"]["condition"],
                    spec.training_condition,
                )

    def test_full_train_test_matrix_is_represented_without_extra_training(self) -> None:
        cells: set[tuple[str, str, str]] = set()
        for spec in self.plan.run_specs:
            cells.update(
                (spec.model, spec.training_condition, test_condition)
                for test_condition in CONDITIONS
            )

        self.assertEqual(len(cells), 36)
        for model in MODELS:
            for train_condition in CONDITIONS:
                for test_condition in CONDITIONS:
                    self.assertIn(
                        (model, train_condition, test_condition), cells
                    )
        self.assertEqual(self.plan.array_size, 12)

    def test_both_patch_grids_build_cpu_transforms_for_train_validation_and_test(self) -> None:
        conditions = {
            condition["condition"]: condition
            for condition in generate_conditions(self.config)
            if condition["transform"] == "patch_shuffle"
        }
        self.assertEqual(tuple(conditions), CONDITIONS[1:])
        pixels = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        image = Image.fromarray(pixels, mode="RGB")

        for grid in (2, 4):
            name = f"patch_shuffle_grid_{grid}"
            condition = conditions[name]
            with self.subTest(grid=grid):
                training = build_condition_transform(16, True, condition)
                validation = build_condition_transform(16, False, condition)
                fixed_rgb_test = build_test_condition_transform(16, condition)

                for transform in (training, validation, fixed_rgb_test):
                    shuffles = [
                        operation
                        for operation in transform.transforms
                        if isinstance(operation, PatchShuffle)
                    ]
                    self.assertEqual(len(shuffles), 1)
                    self.assertEqual(shuffles[0].grid_size, grid)
                    self.assertEqual(shuffles[0].seed, 2026)
                    output = transform(image)
                    self.assertEqual(tuple(output.shape), (3, 16, 16))
                    self.assertEqual(output.dtype, torch.float32)

                self.assertTrue(
                    torch.equal(validation(image), validation(image))
                )
                self.assertTrue(
                    torch.equal(fixed_rgb_test(image), fixed_rgb_test(image))
                )
                self.assertTrue(
                    torch.equal(validation(image), fixed_rgb_test(image))
                )


if __name__ == "__main__":
    unittest.main()
