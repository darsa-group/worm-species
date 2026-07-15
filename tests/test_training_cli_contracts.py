from __future__ import annotations

import argparse
import copy
import contextlib
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml
import pandas as pd
import torch

from src.worm_species.training.checkpoints import build_checkpoint_payload
from src.worm_species.training.checkpoints import checkpoint_keys
from src.worm_species.training.checkpoints import load_checkpoint
from src.worm_species.training.checkpoints import save_checkpoint
from src.worm_species.training.cli import execute, resolve_plan
from src.worm_species.training.loaders import require_complete_task_labels
from src.worm_species.training.modes import get_profile
from src.worm_species.training.modes import infer_experiment_type
from src.worm_species.training.modes import resolve_configured_profile
from src.worm_species.training.runner import initialise_wandb_run


LEGACY_PROFILES = {
    "train_multitask_masked": "masked",
    "train_multitask_masked_hloss": "masked_hloss",
    "train_multitask_masked_hloss_wandb": "masked_hloss_wandb",
    "train_multitask_colour_ablation": "colour_ablation",
    "train_multitask_cue_suppression": "cue_suppression",
}


def minimal_config(output_dir: Path) -> dict:
    return {
        "seed": 7,
        "data": {
            "image_col": "rel_path_seg",
            "target_col": "barcode",
            "target_cols": {
                "genus": "genus",
                "species": "species_label",
                "age": "life_stage",
            },
            "colour_retention": 1.0,
        },
        "model": {"name": "resnet18"},
        "multi_task": {
            "loss_weights": {"genus": 1.0, "species": 1.0, "age": 1.0},
            "normalize_loss_by_active_tasks": True,
            "hierarchy_loss": {"enabled": True, "weight": 0.5},
        },
        "output": {"out_dir": str(output_dir)},
        "wandb": {"enabled": False},
        "sweep": {"enabled": False, "parameters": {}},
        "colour_ablation": {"enabled": False},
        "matched_condition_training": {"enabled": False},
        "test_cue_suppression": {"enabled": False},
    }


class CanonicalTrainingCliContracts(unittest.TestCase):
    def test_preferred_cli_resolves_features_without_a_named_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = minimal_config(root / "outputs")
            config["training"] = {
                "mode": "multitask",
                "use_masked_labels": True,
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            profile, configs, experiment_types = resolve_plan(
                str(config_path), [], [], None
            )

        self.assertEqual(profile.name, "configured")
        self.assertEqual(profile.loader_mode, "standard")
        self.assertTrue(profile.hierarchy)
        self.assertFalse(profile.wandb)
        self.assertTrue(profile.masked_labels)
        self.assertEqual(len(configs), 1)
        self.assertEqual(experiment_types, ["standard"])

    def test_explicit_switches_select_condition_and_colour_contracts(self) -> None:
        base = minimal_config(Path("outputs"))
        base["training"] = {"use_masked_labels": False}
        base["wandb"]["enabled"] = True

        stress = copy.deepcopy(base)
        stress["test_cue_suppression"]["enabled"] = True
        stress_profile = resolve_configured_profile(stress)
        self.assertEqual(stress_profile.loader_mode, "condition")
        self.assertTrue(stress_profile.stress_evaluation)
        self.assertTrue(stress_profile.wandb)
        self.assertFalse(stress_profile.masked_labels)

        matched_stress = copy.deepcopy(stress)
        matched_stress["input_condition"] = {
            "enabled": True,
            "condition": "original",
            "transform": "original",
        }
        self.assertEqual(
            infer_experiment_type(matched_stress), "matched_and_rgb_stress"
        )

        colour = copy.deepcopy(base)
        colour["colour_ablation"]["enabled"] = True
        colour_profile = resolve_configured_profile(colour)
        self.assertEqual(colour_profile.loader_mode, "colour")
        self.assertTrue(colour_profile.colour_sweep)
        self.assertFalse(colour_profile.stress_evaluation)

        matched = copy.deepcopy(base)
        matched["input_condition"] = {
            "enabled": True,
            "condition": "grayscale",
            "transform": "grayscale",
        }
        matched_profile = resolve_configured_profile(matched)
        self.assertEqual(matched_profile.loader_mode, "condition")
        self.assertFalse(matched_profile.stress_evaluation)

    def test_unmasked_mode_fails_without_dropping_incomplete_rows(self) -> None:
        complete = pd.DataFrame(
            {"genus": ["Lumbricus"], "species": ["L. terrestris"]}
        )
        incomplete = pd.DataFrame(
            {"genus": ["Lumbricus"], "species": [pd.NA]}
        )
        target_cols = {"genus": "genus", "species": "species"}

        require_complete_task_labels({"train": complete}, target_cols)
        with self.assertRaisesRegex(
            ValueError,
            r"use_masked_labels=false.*no rows were dropped.*train.species",
        ):
            require_complete_task_labels(
                {"train": incomplete, "val": complete, "test": complete},
                target_cols,
            )
        self.assertEqual(len(incomplete), 1)

    def test_config_driven_stress_rejects_transformed_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = minimal_config(root / "outputs")
            config["input_condition"] = {
                "enabled": True,
                "condition": "greyscale",
                "transform": "grayscale",
            }
            config["test_cue_suppression"] = {"enabled": True}
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Fixed-RGB stress"):
                resolve_plan(str(config_path), [], [], None)

    def test_legacy_wrappers_select_their_explicit_profiles(self) -> None:
        for module_name, expected_profile in LEGACY_PROFILES.items():
            root_module = importlib.import_module(module_name)
            implementation = importlib.import_module(
                f"scripts.training.{module_name}"
            )
            self.assertEqual(implementation.PROFILE.name, expected_profile)
            with mock.patch.object(
                implementation, "legacy_main", return_value=expected_profile
            ) as canonical:
                self.assertEqual(root_module.main(), expected_profile)
                canonical.assert_called_once_with(expected_profile)

    def test_every_profile_dry_run_resolves_once_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "must-not-be-created"
            config_path = root / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(minimal_config(output_dir)), encoding="utf-8"
            )

            for expected_profile in LEGACY_PROFILES.values():
                args = argparse.Namespace(
                    config=str(config_path),
                    override=[],
                    sweep=[],
                    profile=expected_profile,
                    dry_run=True,
                    print_resolved_config=False,
                    single_run=True,
                )
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(execute(args), [])
                payload = json.loads(stream.getvalue())
                self.assertEqual(
                    payload["plan"]["selected_profile"], expected_profile
                )
                self.assertEqual(
                    payload["plan"]["expected_internal_training_runs"], 1
                )
                self.assertFalse(output_dir.exists())

    def test_one_external_condition_resolves_to_one_training_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = minimal_config(root / "outputs")
            config["input_condition"] = {
                "enabled": True,
                "condition": "gaussian_blur_sigma_2",
                "transform": "gaussian_blur",
                "sigma": 2.0,
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            profile, configs, experiment_types = resolve_plan(
                str(config_path), [], [], "cue_suppression"
            )
            self.assertEqual(profile.name, "cue_suppression")
            self.assertEqual(len(configs), 1)
            self.assertEqual(experiment_types, ["matched_condition"])

    def test_external_condition_rejects_nested_sweep_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = minimal_config(root / "outputs")
            config["input_condition"] = {
                "enabled": True,
                "condition": "original",
                "transform": "original",
            }
            config["sweep"] = {
                "enabled": True,
                "parameters": {"model.name": ["resnet18", "vit_b_16"]},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "internal expanders disabled"):
                resolve_plan(str(config_path), [], [], "cue_suppression")

    def test_matched_transform_and_fixed_rgb_stress_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = minimal_config(root / "outputs")
            config["input_condition"] = {
                "enabled": True,
                "condition": "greyscale",
                "transform": "grayscale",
            }
            config["test_cue_suppression"] = {"enabled": True}
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Fixed-RGB stress"):
                resolve_plan(str(config_path), [], [], "cue_suppression")

    def test_checkpoint_keys_remain_profile_compatible(self) -> None:
        ordinary = {
            "model_state",
            "cfg",
            "label_to_index_by_task",
            "index_to_label_by_task",
            "best_val_score",
            "selection_metric",
            "best_epoch",
        }
        for profile_name in (
            "masked",
            "masked_hloss",
            "masked_hloss_wandb",
        ):
            self.assertEqual(checkpoint_keys(get_profile(profile_name)), ordinary)
        self.assertEqual(
            checkpoint_keys(get_profile("colour_ablation")),
            ordinary | {"colour_retention", "colour_percent"},
        )
        self.assertEqual(
            checkpoint_keys(get_profile("cue_suppression")),
            ordinary
            | {"colour_retention", "colour_percent", "training_condition"},
        )

    def test_disabled_wandb_profile_does_not_initialise_network_client(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            result = initialise_wandb_run(
                {"wandb": {"enabled": False}},
                "test-run",
                Path(temporary_dir),
                get_profile("masked_hloss_wandb"),
            )
        self.assertIsNone(result)

    def test_checkpoint_payload_round_trip_preserves_legacy_schema(self) -> None:
        profile = get_profile("cue_suppression")
        payload = build_checkpoint_payload(
            profile=profile,
            model_state={"weight": torch.tensor([1.0])},
            cfg={"model": {"name": "resnet18"}},
            label_to_index_by_task={"genus": {"Lumbricus": 0}},
            index_to_label_by_task={"genus": {0: "Lumbricus"}},
            best_val_score=0.75,
            selection_metric="mean_macro_f1",
            best_epoch=3,
            colour_retention=1.0,
            colour_percent=100,
            training_condition={"condition": "original"},
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "best_model.pt"
            save_checkpoint(payload, checkpoint_path)
            restored = load_checkpoint(checkpoint_path, map_location="cpu")

        self.assertEqual(set(restored), checkpoint_keys(profile))
        self.assertTrue(
            torch.equal(restored["model_state"]["weight"], torch.tensor([1.0]))
        )
        self.assertEqual(restored["training_condition"], {"condition": "original"})


if __name__ == "__main__":
    unittest.main()
