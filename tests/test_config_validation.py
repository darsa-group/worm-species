from __future__ import annotations

import copy
import unittest
from pathlib import Path

from src.worm_species.config.inspect import inspection_summary
from src.worm_species.config.loading import load_config
from src.worm_species.config.validation import ConfigValidationError
from src.worm_species.config.validation import validate_config
from src.worm_species.config.validation import validate_override_items


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationValidationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = load_config(ROOT / "config.yaml")

    def assert_invalid(self, config: dict, expected: str, workflow: str = "saved") -> None:
        with self.assertRaises(ConfigValidationError) as caught:
            validate_config(
                config,
                workflow=workflow,
                check_paths=False,
                check_model_registry=False,
            )
        self.assertIn(expected, str(caught.exception))

    def test_current_two_model_baseline_is_valid_and_has_224_runs(self) -> None:
        validate_config(
            self.baseline,
            workflow="run_specs",
            check_paths=False,
            check_model_registry=False,
        )
        summary = inspection_summary(self.baseline, "run_specs")
        self.assertEqual(summary["expected_model_count"], 2)
        self.assertEqual(summary["expected_condition_count"], 112)
        self.assertEqual(summary["expected_total_run_count"], 224)

    def test_missing_training_keys_are_reported_together(self) -> None:
        self.assert_invalid({}, "seed: is required for training", workflow="training")

    def test_invalid_types_and_ranges_are_rejected(self) -> None:
        self.assert_invalid({"training": {"epochs": "many"}}, "training.epochs")
        self.assert_invalid(
            {"data": {"colour_retention": 1.01}},
            "data.colour_retention: must be <= 1",
        )
        self.assert_invalid(
            {"data": {"image_size": 0}},
            "data.image_size: must be > 0",
        )
        self.assert_invalid(
            {"cache": {"num_workers": 0}},
            "cache.num_workers: must be >= 1",
        )

    def test_supported_wandb_modes_are_explicit(self) -> None:
        for mode in ("online", "offline", "disabled", "dryrun", "run", "shared", None):
            with self.subTest(mode=mode):
                validate_config(
                    {"wandb": {"mode": mode}},
                    workflow="saved",
                    check_paths=False,
                    check_model_registry=False,
                )
        self.assert_invalid(
            {"wandb": {"mode": "sometimes"}},
            "wandb.mode: must be one of",
        )

    def test_task_loss_weights_are_known_finite_and_usable(self) -> None:
        tasks = {
            "data": {
                "target_cols": {
                    "genus": "genus",
                    "species": "species_label",
                    "age": "life_stage",
                }
            }
        }
        valid = copy.deepcopy(tasks)
        valid["multi_task"] = {
            "loss_weights": {"genus": 1.0, "species": 0.5, "age": 2.0}
        }
        validate_config(
            valid,
            workflow="saved",
            check_paths=False,
            check_model_registry=False,
        )

        invalid_cases = (
            ({"genus": -1.0}, "must be >= 0"),
            ({"genus": float("nan")}, "must be finite"),
            ({"unknown": 1.0}, "task is not present in data.target_cols"),
            (
                {"genus": 0.0, "species": 0.0, "age": 0.0},
                "at least one selected task weight must be greater than zero",
            ),
        )
        for weights, expected in invalid_cases:
            with self.subTest(weights=weights):
                config = copy.deepcopy(tasks)
                config["multi_task"] = {"loss_weights": weights}
                self.assert_invalid(config, expected)

    def test_hierarchy_tasks_must_be_distinct_selected_tasks(self) -> None:
        base = {
            "data": {"target_cols": {"genus": "genus", "species": "species_label"}},
            "multi_task": {
                "hierarchy_loss": {
                    "enabled": True,
                    "parent_task": "genus",
                    "child_task": "species",
                    "weight": 0.5,
                }
            },
        }
        validate_config(
            base,
            workflow="saved",
            check_paths=False,
            check_model_registry=False,
        )

        same = copy.deepcopy(base)
        same["multi_task"]["hierarchy_loss"]["child_task"] = "genus"
        self.assert_invalid(same, "parent_task and child_task must differ")

        missing = copy.deepcopy(base)
        missing["multi_task"]["hierarchy_loss"]["child_task"] = "age"
        self.assert_invalid(missing, "must name a task in data.target_cols")

    def test_unknown_override_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigValidationError, "unknown configuration"):
            validate_override_items(["training.not_a_real_key=1"])
        with self.assertRaisesRegex(ConfigValidationError, "training.profile"):
            validate_override_items(["training.profile=cue_suppression"])

    def test_unknown_training_profile_and_experiment_type_are_rejected(self) -> None:
        self.assert_invalid(
            {"training": {"profile": "legacy_script_name"}},
            "training.profile: is legacy-only",
        )
        self.assert_invalid(
            {"experiment": {"type": "combined_ambiguous_mode"}},
            "experiment.type: must be one of",
        )
        self.assert_invalid(
            {"training": {"mode": "script_specific_mode"}},
            "training.mode: must be one of",
        )

    def test_config_driven_feature_contradictions_are_rejected(self) -> None:
        self.assert_invalid(
            {
                "experiment": {"type": "standard"},
                "input_condition": {
                    "enabled": True,
                    "transform": "original",
                },
            },
            "experiment.type=standard cannot enable input_condition",
        )
        self.assert_invalid(
            {
                "experiment": {"type": "rgb_stress_test"},
                "test_cue_suppression": {"enabled": False},
            },
            "requires test_cue_suppression.enabled=true",
        )
        self.assert_invalid(
            {
                "colour_ablation": {"enabled": True},
                "test_cue_suppression": {"enabled": True},
            },
            "colour_ablation.enabled cannot be combined with fixed-RGB stress",
        )
        self.assert_invalid(
            {
                "colour_ablation": {"enabled": True},
                "input_condition": {"enabled": True, "transform": "original"},
            },
            "colour_ablation.enabled cannot be combined with input_condition",
        )

    def test_sweep_values_must_be_non_empty_lists(self) -> None:
        config = {
            "model": {"name": "resnet18"},
            "sweep": {"enabled": True, "parameters": {"model.name": []}},
        }
        self.assert_invalid(config, "must be a non-empty list", workflow="run_specs")

    def test_external_conditions_cannot_enable_internal_expansion(self) -> None:
        config = copy.deepcopy(self.baseline)
        config["input_condition"] = {
            "enabled": True,
            "condition": "original",
            "transform": "original",
        }
        self.assert_invalid(
            config,
            "must be false while external matched-condition run specifications",
            workflow="run_specs",
        )

    def test_unknown_and_invalid_transform_parameters_are_clear(self) -> None:
        self.assert_invalid(
            {"input_condition": {"enabled": True, "transform": "mystery"}},
            "unknown transformation 'mystery'",
        )
        self.assert_invalid(
            {
                "data": {"image_size": 224},
                "input_condition": {
                    "enabled": True,
                    "transform": "patch_shuffle",
                    "grid_size": 3,
                },
            },
            "must divide data.image_size=224",
        )
        self.assert_invalid(
            {
                "input_condition": {
                    "enabled": True,
                    "transform": "channel_shuffle",
                    "order": [0, 0, 1],
                }
            },
            "must be a permutation",
        )
        self.assert_invalid(
            {
                "input_condition": {
                    "enabled": False,
                    "transform": "grayscale",
                }
            },
            "must be original when input_condition.enabled=false",
        )
        self.assert_invalid(
            {
                "data": {"image_size": 224},
                "test_cue_suppression": {
                    "patch_shuffle": {"enabled": True, "grid_sizes": [3]}
                },
            },
            "must divide data.image_size=224",
        )

    def test_transform_catalogue_values_must_be_unique(self) -> None:
        duplicate_catalogues = (
            {
                "saturation": {
                    "enabled": True,
                    "values": [0.5, 0.5],
                }
            },
            {
                "channel_shuffle": {
                    "enabled": True,
                    "orders": [[2, 0, 1], [2, 0, 1]],
                }
            },
            {
                "gaussian_blur": {
                    "enabled": True,
                    "sigmas": [1, 1.0],
                }
            },
            {
                "bilateral_filter": {
                    "enabled": True,
                    "settings": [
                        {"diameter": 5, "sigma_colour": 25, "sigma_space": 25},
                        {"diameter": 5, "sigma_colour": 25.0, "sigma_space": 25.0},
                    ],
                }
            },
            {
                "patch_shuffle": {
                    "enabled": True,
                    "grid_sizes": [2, 2],
                }
            },
        )
        for catalogue in duplicate_catalogues:
            with self.subTest(catalogue=next(iter(catalogue))):
                self.assert_invalid(
                    {"test_cue_suppression": catalogue},
                    "must contain unique values",
                )


if __name__ == "__main__":
    unittest.main()
