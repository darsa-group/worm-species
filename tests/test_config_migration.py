from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import yaml

from src.worm_species.config.__main__ import main as config_main
from src.worm_species.config.inspect import main as inspect_main
from src.worm_species.config.loading import load_config
from src.worm_species.config.migrate import main as migrate_main
from src.worm_species.config.normalization import (
    normalize_config,
    normalize_config_with_report,
)
from src.worm_species.config.sweeps import expand_sweep_items
from src.worm_species.config.validation import (
    ConfigValidationError,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiments"


class LegacyNormalizationContracts(unittest.TestCase):
    def test_scalar_aliases_resolve_without_mutating_the_source(self) -> None:
        source = {
            "data": {"image_size": 224, "colour_retention": 0.5},
            "preprocessing": {"image_size": 384},
            "input_condition": {
                "enabled": False,
                "condition": "original",
                "transform": "original",
                "retention": 1.0,
            },
        }
        before = copy.deepcopy(source)

        result = normalize_config_with_report(source)
        normalized = result.config

        self.assertEqual(source, before)
        self.assertEqual(normalized["preprocessing"]["image_size"], 384)
        self.assertNotIn("image_size", normalized["data"])
        self.assertNotIn("colour_retention", normalized["data"])
        self.assertEqual(
            normalized["input_condition"],
            {
                "enabled": True,
                "name": "colour_050pct",
                "feature": "colour",
                "transform": "saturation",
                "strength": 0.5,
                "parameters": {"retention": 0.5},
            },
        )
        warning_paths = {warning.path for warning in result.warnings}
        self.assertIn("data.image_size", warning_paths)
        self.assertIn("data.colour_retention", warning_paths)
        self.assertIn("input_condition.*", warning_paths)

    def test_all_legacy_planner_sections_disappear_from_canonical_config(self) -> None:
        source = {
            "seed": 7,
            "sweep": {"enabled": True, "parameters": {"model.name": ["a", "b"]}},
            "matched_condition_training": {
                "enabled": True,
                "include_original": True,
                "condition_names": ["original", "grayscale", "patch_shuffle_grid_2"],
                "evaluate_original_model_on_all_test_conditions": True,
            },
            "test_cue_suppression": {
                "enabled": False,
                "saturation": {"enabled": False},
                "grayscale": {"enabled": True},
                "channel_shuffle": {"enabled": False},
                "bilateral_filter": {"enabled": False},
                "gaussian_blur": {"enabled": False},
                "patch_shuffle": {"enabled": True, "grid_sizes": [2], "seed": 2026},
            },
            "condition_matrix_evaluation": {
                "enabled": True,
                "condition_names": ["original", "patch_shuffle_grid_2"],
                "write_reports": True,
            },
        }
        normalized = normalize_config(source)
        for legacy in (
            "matched_condition_training",
            "test_cue_suppression",
            "condition_matrix_evaluation",
        ):
            self.assertNotIn(legacy, normalized)
        self.assertEqual(
            [item["name"] for item in normalized["sweep"]["conditions"]],
            ["original", "grayscale", "patch_shuffle_grid_2"],
        )
        self.assertEqual(len(expand_sweep_items(normalized)), 6)
        self.assertTrue(
            normalized["evaluation"]["test_conditions"][
                "evaluate_original_training"
            ]
        )
        self.assertEqual(
            [
                item["name"]
                for item in normalized["evaluation"]["condition_matrix"][
                    "conditions"
                ]
            ],
            ["original", "patch_shuffle_grid_2"],
        )

    def test_existing_experiment_counts_are_preserved_canonically(self) -> None:
        expected = {
            "dual_cue.yaml": 224,
            "colour_ablation.yaml": 202,
            "patch_shuffle_matrix.yaml": 12,
        }
        for filename, count in expected.items():
            with self.subTest(filename=filename):
                normalized = normalize_config(load_config(EXPERIMENTS / filename))
                self.assertEqual(len(expand_sweep_items(normalized)), count)
                for legacy in (
                    "colour_ablation",
                    "matched_condition_training",
                    "test_cue_suppression",
                    "condition_matrix_evaluation",
                ):
                    self.assertNotIn(legacy, normalized)

    def test_normalization_is_idempotent_and_silent(self) -> None:
        source = load_config(EXPERIMENTS / "dual_cue.yaml")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            once = normalize_config(source)
            twice = normalize_config(once)
        self.assertEqual(twice, once)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")


class CanonicalValidationContracts(unittest.TestCase):
    def assert_invalid(self, config: dict, expected: str) -> None:
        with self.assertRaisesRegex(ConfigValidationError, expected):
            validate_config(
                config,
                workflow="saved",
                check_paths=False,
                check_model_registry=False,
            )

    def test_preprocessing_and_augmentation_validation(self) -> None:
        cases = (
            ({"preprocessing": {"image_size": 0}}, "preprocessing.image_size"),
            (
                {"augmentation": {"horizontal_flip": {"probability": 1.1}}},
                "augmentation.horizontal_flip.probability",
            ),
            (
                {"augmentation": {"vertical_flip": {"probability": -0.1}}},
                "augmentation.vertical_flip.probability",
            ),
            (
                {"augmentation": {"rotation": {"degrees": -1}}},
                "augmentation.rotation.degrees",
            ),
            (
                {"preprocessing": {"normalisation": {"mean": [0.1], "std": [1, 2]}}},
                "equal lengths",
            ),
            (
                {"preprocessing": {"normalisation": {"mean": [0.1], "std": [0]}}},
                "greater than zero",
            ),
        )
        for config, expected in cases:
            with self.subTest(config=config):
                self.assert_invalid(config, expected)

    def test_canonical_nested_condition_parameters_validate(self) -> None:
        valid = {
            "preprocessing": {"image_size": 224},
            "input_condition": {
                "enabled": True,
                "name": "gaussian_sigma_2",
                "transform": "gaussian_blur",
                "parameters": {"sigma": 2.0},
            },
        }
        validate_config(
            valid,
            workflow="saved",
            check_paths=False,
            check_model_registry=False,
        )
        invalid = copy.deepcopy(valid)
        invalid["input_condition"]["parameters"]["sigma"] = 0
        self.assert_invalid(invalid, "input_condition.parameters.sigma")

    def test_evaluation_conditions_validate_but_do_not_expand_sweep(self) -> None:
        config = {
            "sweep": {
                "enabled": True,
                "parameters": {"model.name": ["a", "b"]},
            },
            "evaluation": {
                "test_conditions": {
                    "enabled": True,
                    "conditions": ["original", "grayscale"],
                }
            },
        }
        validate_config(
            config,
            workflow="saved",
            check_paths=False,
            check_model_registry=False,
        )
        self.assertEqual(len(expand_sweep_items(config)), 2)


class MigrationCommandContracts(unittest.TestCase):
    def test_migrate_prints_valid_yaml_and_never_modifies_source(self) -> None:
        source_payload = {
            "data": {"image_size": 16, "colour_retention": 1.0},
            "colour_ablation": {"enabled": False},
            "matched_condition_training": {"enabled": False},
            "test_cue_suppression": {"enabled": False, "saturation": {"enabled": False}},
            "condition_matrix_evaluation": {"enabled": False, "condition_names": ["original"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.yaml"
            original_text = yaml.safe_dump(source_payload, sort_keys=False)
            path.write_text(original_text, encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = migrate_main(["--config", str(path)])
            self.assertEqual(path.read_text(encoding="utf-8"), original_text)

        self.assertEqual(status, 0, stderr.getvalue())
        migrated = yaml.safe_load(stdout.getvalue())
        self.assertEqual(migrated["preprocessing"]["image_size"], 16)
        self.assertNotIn("image_size", migrated["data"])
        self.assertNotIn("colour_ablation", migrated)
        self.assertIn("Compatibility:", stderr.getvalue())

    def test_unified_module_dispatches_migration_and_inspection_reports_warnings(self) -> None:
        config_path = ROOT / "config.yaml"
        migrate_stdout = io.StringIO()
        with redirect_stdout(migrate_stdout), redirect_stderr(io.StringIO()):
            self.assertEqual(
                config_main(["migrate", "--config", str(config_path), "--format", "json"]),
                0,
            )
        self.assertIn("preprocessing", json.loads(migrate_stdout.getvalue()))

        inspect_stdout = io.StringIO()
        inspect_stderr = io.StringIO()
        with redirect_stdout(inspect_stdout), redirect_stderr(inspect_stderr):
            status = inspect_main([
                "--config", str(config_path),
                "--workflow", "training",
                "--format", "json",
            ])
        self.assertEqual(status, 0, inspect_stderr.getvalue())
        payload = json.loads(inspect_stdout.getvalue())
        self.assertTrue(payload["compatibility_warnings"])
        self.assertIn("preprocessing", payload["canonical_resolved_config"])


if __name__ == "__main__":
    unittest.main()
