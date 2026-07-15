from __future__ import annotations

import copy
import unittest
from decimal import Decimal

from src.worm_species.config.normalization import (
    ConfigNormalizationError,
    normalize_config,
)
from src.worm_species.config.ranges import (
    DecimalRange,
    RangeExpansionError,
    expand_decimal_range,
)
from src.worm_species.config.sweeps import (
    apply_sweep_item,
    expand_sweep_items,
    generate_sweep_configs,
)


def saturation_range() -> dict:
    return {
        "name_template": "saturation_{percent:03d}",
        "transform": "saturation",
        "parameter": "retention",
        "range": {"start": 1.0, "stop": 0.0, "step": -0.01},
    }


class DecimalRangeContracts(unittest.TestCase):
    def test_both_endpoints_are_included_without_float_drift(self) -> None:
        values = expand_decimal_range(
            {"start": 1.0, "stop": 0.0, "step": -0.01}
        )
        self.assertEqual(len(values), 101)
        self.assertEqual(values[0], Decimal("1.0"))
        self.assertEqual(values[1], Decimal("0.99"))
        self.assertEqual(values[-1], Decimal("0.0"))

    def test_non_divisible_interval_still_includes_stop_once(self) -> None:
        number_range = DecimalRange.from_mapping(
            {"start": 0, "stop": 1, "step": 0.3}
        )
        self.assertEqual(
            number_range.decimals(),
            (
                Decimal("0"),
                Decimal("0.3"),
                Decimal("0.6"),
                Decimal("0.9"),
                Decimal("1"),
            ),
        )

    def test_zero_and_wrong_direction_steps_are_rejected(self) -> None:
        cases = (
            ({"start": 1, "stop": 0, "step": 0}, "must not be zero"),
            ({"start": 1, "stop": 0, "step": 0.1}, "must be negative"),
            ({"start": 0, "stop": 1, "step": -0.1}, "must be positive"),
        )
        for raw, message in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(
                RangeExpansionError, message
            ):
                DecimalRange.from_mapping(raw)


class CanonicalNormalizationContracts(unittest.TestCase):
    def test_range_names_and_values_are_deterministic(self) -> None:
        source = {"sweep": {"enabled": True, "conditions": [saturation_range()]}}
        before = copy.deepcopy(source)

        normalized = normalize_config(source)
        conditions = normalized["sweep"]["conditions"]

        self.assertEqual(source, before)
        self.assertEqual(len(conditions), 101)
        self.assertEqual(conditions[0]["name"], "saturation_100")
        self.assertEqual(conditions[1]["name"], "saturation_099")
        self.assertEqual(conditions[-1]["name"], "saturation_000")
        self.assertEqual(conditions[0]["parameters"], {"retention": 1.0})
        self.assertEqual(conditions[-1]["parameters"], {"retention": 0.0})

    def test_normalization_is_idempotent_and_leaves_evaluation_untouched(self) -> None:
        source = {
            "sweep": {"enabled": True, "conditions": [saturation_range()]},
            "evaluation": {
                "test_conditions": {
                    "enabled": True,
                    "conditions": ["original", "grayscale"],
                }
            },
        }
        once = normalize_config(source)
        twice = normalize_config(once)
        self.assertEqual(twice, once)
        self.assertEqual(twice["evaluation"], source["evaluation"])
        self.assertIsNot(twice["evaluation"], source["evaluation"])

    def test_duplicate_explicit_or_generated_names_are_rejected(self) -> None:
        duplicate_explicit = {
            "sweep": {
                "conditions": [
                    {"name": "same", "transform": "original", "parameters": {}},
                    {"name": "same", "transform": "grayscale", "parameters": {}},
                ]
            }
        }
        duplicate_range = {
            "sweep": {
                "conditions": [{
                    "name_template": "same",
                    "transform": "saturation",
                    "parameter": "retention",
                    "range": {"start": 1, "stop": 0, "step": -1},
                }]
            }
        }
        for source in (duplicate_explicit, duplicate_range):
            with self.subTest(source=source), self.assertRaisesRegex(
                ConfigNormalizationError, "duplicate condition identifier"
            ):
                normalize_config(source)

    def test_empty_condition_dimension_is_rejected_instead_of_silently_skipped(self) -> None:
        with self.assertRaisesRegex(
            ConfigNormalizationError, "must be a non-empty list"
        ):
            normalize_config({"sweep": {"conditions": []}})


class GenericSweepContracts(unittest.TestCase):
    def test_complete_condition_objects_are_one_atomic_dimension(self) -> None:
        config = {
            "model": {"name": "base"},
            "sweep": {
                "enabled": True,
                "parameters": {"model.name": ["model_a", "model_b"]},
                "conditions": [
                    {"name": "original", "transform": "original", "parameters": {}},
                    {
                        "name": "gaussian_sigma_2",
                        "transform": "gaussian_blur",
                        "parameters": {"sigma": 2.0},
                    },
                    {
                        "name": "patch_shuffle_grid_4",
                        "transform": "patch_shuffle",
                        "parameters": {"grid_size": 4, "seed": 2026},
                    },
                ],
            },
        }

        before = copy.deepcopy(config)
        items = expand_sweep_items(config)
        expanded = generate_sweep_configs(config)

        self.assertEqual(config, before)
        self.assertEqual(items, expand_sweep_items(config))
        self.assertEqual(len(expanded), 6)
        self.assertEqual(
            [
                (item["model"]["name"], item["input_condition"]["name"])
                for item in expanded
            ],
            [
                (model, condition)
                for model in ("model_a", "model_b")
                for condition in (
                    "original",
                    "gaussian_sigma_2",
                    "patch_shuffle_grid_4",
                )
            ],
        )
        self.assertEqual(
            expanded[1]["input_condition"]["parameters"], {"sigma": 2.0}
        )
        self.assertNotIn("grid_size", expanded[1]["input_condition"]["parameters"])
        self.assertEqual(
            expanded[2]["input_condition"]["parameters"],
            {"grid_size": 4, "seed": 2026},
        )
        self.assertNotIn("sigma", expanded[2]["input_condition"]["parameters"])

    def test_two_models_by_saturation_range_creates_exactly_202_fits(self) -> None:
        config = {
            "sweep": {
                "enabled": True,
                "parameters": {"model.name": ["convnext_base", "vit_b_16"]},
                "conditions": [saturation_range()],
            }
        }
        items = expand_sweep_items(config)
        self.assertEqual(len(items), 202)
        self.assertEqual(items[0].parameter_values["model.name"], "convnext_base")
        self.assertEqual(items[0].condition["name"], "saturation_100")
        self.assertEqual(items[100].condition["name"], "saturation_000")
        self.assertEqual(items[101].parameter_values["model.name"], "vit_b_16")
        self.assertEqual(items[-1].condition["name"], "saturation_000")

    def test_evaluation_conditions_never_increase_training_fit_count(self) -> None:
        config = {
            "sweep": {
                "enabled": True,
                "parameters": {"model.name": ["a", "b"]},
            },
            "evaluation": {
                "test_conditions": {
                    "enabled": True,
                    "conditions": ["one", "two", "three", "four"],
                },
                "condition_matrix": {"enabled": True},
            },
        }
        self.assertEqual(len(expand_sweep_items(config)), 2)
        self.assertEqual(len(generate_sweep_configs(config)), 2)

    def test_resolved_item_can_disable_only_the_external_sweep(self) -> None:
        config = {
            "sweep": {
                "enabled": True,
                "parameters": {"model.name": ["a"]},
                "conditions": [
                    {"name": "original", "transform": "original", "parameters": {}}
                ],
            },
            "evaluation": {"test_conditions": {"enabled": True}},
        }
        item = expand_sweep_items(config)[0]
        resolved = apply_sweep_item(config, item, disable_sweep=True)
        self.assertFalse(resolved["sweep"]["enabled"])
        self.assertTrue(resolved["input_condition"]["enabled"])
        self.assertEqual(resolved["evaluation"], config["evaluation"])
        self.assertTrue(config["sweep"]["enabled"])
        self.assertNotIn("input_condition", config)

    def test_legacy_parameter_and_colour_expansion_order_is_unchanged(self) -> None:
        ordinary = {
            "sweep": {
                "enabled": True,
                "parameters": {
                    "model.name": ["a", "b"],
                    "training.lr": [0.1, 0.2],
                },
            }
        }
        expanded = generate_sweep_configs(ordinary)
        self.assertEqual(
            [(item["model"]["name"], item["training"]["lr"]) for item in expanded],
            [("a", 0.1), ("a", 0.2), ("b", 0.1), ("b", 0.2)],
        )

        colour = {
            "sweep": {
                "enabled": True,
                "parameters": {"model.name": ["a", "b"]},
            },
            "colour_ablation": {
                "enabled": True,
                "start_percent": 100,
                "stop_percent": 0,
                "step_percent": 1,
                "combine_with_sweep": True,
            },
        }
        colour_expanded = generate_sweep_configs(
            colour, include_colour_ablation=True
        )
        self.assertEqual(len(colour_expanded), 202)
        self.assertEqual(
            (
                colour_expanded[0]["model"]["name"],
                colour_expanded[0]["data"]["colour_retention"],
            ),
            ("a", 1.0),
        )
        self.assertEqual(
            (
                colour_expanded[100]["model"]["name"],
                colour_expanded[100]["data"]["colour_retention"],
            ),
            ("a", 0.0),
        )
        self.assertEqual(
            (
                colour_expanded[101]["model"]["name"],
                colour_expanded[101]["data"]["colour_retention"],
            ),
            ("b", 1.0),
        )
        self.assertEqual(
            (
                colour_expanded[-1]["model"]["name"],
                colour_expanded[-1]["data"]["colour_retention"],
            ),
            ("b", 0.0),
        )

    def test_disabled_empty_sweep_preserves_legacy_identity_result(self) -> None:
        config = {
            "sweep": {"enabled": False},
            "evaluation": {"test_conditions": {"conditions": ["a", "b"]}},
        }
        expanded = generate_sweep_configs(config)
        self.assertEqual(expanded, [config])
        self.assertIs(expanded[0], config)


if __name__ == "__main__":
    unittest.main()
