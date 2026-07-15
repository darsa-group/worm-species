from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.worm_species.config.inspect import main as inspect_main
from src.worm_species.config.inspect import inspection_summary
from src.worm_species.config.loading import ConfigLoadError
from src.worm_species.config.loading import deep_merge
from src.worm_species.config.loading import load_config
from src.worm_species.slurm.config import load_submission_config


ROOT = Path(__file__).resolve().parents[1]
ROOT_CONFIG = ROOT / "config.yaml"
DUAL_CUE = ROOT / "configs" / "experiments" / "dual_cue.yaml"
LOCAL_CLUSTER = ROOT / "configs" / "clusters" / "local.yaml"


class ExtendedConfigurationLoadingContracts(unittest.TestCase):
    def test_root_config_inspection_reports_one_run(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = inspect_main(
                [
                    "--config",
                    str(ROOT_CONFIG),
                    "--workflow",
                    "training",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(status, 0, stderr.getvalue())
        summary = json.loads(stdout.getvalue())["summary"]
        self.assertEqual(summary["expected_model_count"], 1)
        self.assertEqual(summary["expected_sweep_combination_count"], 1)
        self.assertEqual(summary["expected_condition_count"], 1)
        self.assertEqual(summary["expected_total_run_count"], 1)
        self.assertEqual(summary["model"]["configured_name"], "efficientnet_b0")
        self.assertTrue(summary["model"]["pretrained"])
        self.assertFalse(summary["model"]["freeze_backbone"])
        self.assertEqual(summary["model"]["planned_pretrained_values"], [True])
        self.assertEqual(
            summary["model"]["planned_freeze_backbone_values"], [False]
        )
        self.assertEqual(summary["data"]["image_size"], 224)
        self.assertEqual(summary["tasks"]["loss_weights"]["age"], 2.0)
        self.assertEqual(summary["training"]["epochs"], 200)
        self.assertEqual(summary["training"]["batch_size"], 256)
        self.assertEqual(summary["training"]["learning_rate"], 0.0005)
        self.assertEqual(summary["training"]["weight_decay"], 0.0001)
        self.assertTrue(summary["training"]["class_weight"])
        self.assertTrue(summary["training"]["use_amp"])
        self.assertEqual(summary["wandb"]["mode"], "offline")
        self.assertFalse(summary["matched_training"]["enabled"])
        self.assertFalse(summary["fixed_rgb_test"]["enabled"])
        self.assertEqual(summary["fixed_rgb_test"]["effective_condition_names"], [])
        self.assertEqual(summary["expansion"]["owner"], "none")
        self.assertEqual(
            summary["expansion"][
                "expected_internal_training_runs_per_resolved_spec"
            ],
            1,
        )

    def test_relative_inheritance_recursively_merges_without_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parents = root / "parents"
            children = root / "children"
            parents.mkdir()
            children.mkdir()
            (parents / "base.yaml").write_text(
                "first:\n"
                "  retained: base\n"
                "  replaced: base\n"
                "second: base\n",
                encoding="utf-8",
            )
            (parents / "middle.yaml").write_text(
                "extends: base.yaml\n"
                "first:\n"
                "  replaced: middle\n"
                "  middle_only: true\n",
                encoding="utf-8",
            )
            child = children / "child.yaml"
            child.write_text(
                "extends: ../parents/middle.yaml\n"
                "first:\n"
                "  child_only: 3\n"
                "second: child\n"
                "third: child\n",
                encoding="utf-8",
            )

            resolved = load_config(child)

        self.assertEqual(list(resolved), ["first", "second", "third"])
        self.assertEqual(
            list(resolved["first"]),
            ["retained", "replaced", "middle_only", "child_only"],
        )
        self.assertEqual(
            resolved,
            {
                "first": {
                    "retained": "base",
                    "replaced": "middle",
                    "middle_only": True,
                    "child_only": 3,
                },
                "second": "child",
                "third": "child",
            },
        )

    def test_deep_merge_does_not_mutate_inputs(self) -> None:
        base = {"nested": {"kept": [1], "changed": "base"}}
        overlay = {"nested": {"changed": "overlay", "added": [2]}}
        before_base = copy.deepcopy(base)
        before_overlay = copy.deepcopy(overlay)

        merged = deep_merge(base, overlay)
        merged["nested"]["kept"].append(9)
        merged["nested"]["added"].append(8)

        self.assertEqual(base, before_base)
        self.assertEqual(overlay, before_overlay)

    def test_extends_cycle_is_reported_with_the_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.yaml"
            second = root / "second.yaml"
            first.write_text("extends: second.yaml\n", encoding="utf-8")
            second.write_text("extends: first.yaml\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ConfigLoadError, "Configuration extends cycle"
            ) as caught:
                load_config(first)

        message = str(caught.exception)
        self.assertIn(str(first), message)
        self.assertIn(str(second), message)

    def test_missing_parent_and_missing_root_are_clear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child.yaml"
            child.write_text("extends: absent.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigLoadError, "Configuration not found"):
                load_config(child)
            with self.assertRaisesRegex(ConfigLoadError, "Configuration not found"):
                load_config(root / "also-absent.yaml")

    def test_non_mapping_documents_and_invalid_extends_are_rejected(self) -> None:
        cases = {
            "sequence.yaml": "- one\n- two\n",
            "scalar.yaml": "value\n",
            "empty-parent.yaml": "extends: ''\n",
            "mapping-parent.yaml": "extends:\n  file: base.yaml\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, contents in cases.items():
                path = root / filename
                path.write_text(contents, encoding="utf-8")
                with self.subTest(filename=filename):
                    with self.assertRaises(ConfigLoadError):
                        load_config(path)

    def test_experiment_config_loads_and_inspects_through_shared_loader(self) -> None:
        resolved = load_config(DUAL_CUE)
        self.assertNotIn("extends", resolved)
        self.assertEqual(
            resolved["sweep"]["parameters"]["model.name"],
            ["convnext_base", "vit_b_16"],
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = inspect_main(
                [
                    "--config",
                    str(DUAL_CUE),
                    "--workflow",
                    "run_specs",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(status, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        summary = payload["summary"]
        self.assertEqual(summary["expected_model_count"], 2)
        self.assertEqual(summary["expected_condition_count"], 112)
        self.assertEqual(summary["expected_total_run_count"], 224)
        self.assertEqual(summary["expansion"]["owner"], "matched_condition_training")
        self.assertEqual(
            summary["matched_training"]["resolved_condition_names"],
            summary["condition_names"],
        )
        self.assertEqual(summary["wandb"]["project"], "worm-species-cues")
        self.assertEqual(summary["wandb"]["mode"], "online")

    def test_inspection_exposes_resolved_cluster_resources_and_paths(self) -> None:
        config = load_submission_config(DUAL_CUE, LOCAL_CLUSTER)
        summary = inspection_summary(config, "run_specs")
        slurm = summary["slurm"]
        self.assertFalse(slurm["enabled"])
        self.assertEqual(slurm["cluster_profile"], "local")
        self.assertEqual(slurm["resources"]["cpus_per_task"], 1)
        self.assertEqual(slurm["resources"]["gpus_per_task"], 0)
        self.assertEqual(slurm["resources"]["array_max_active"], 1)
        self.assertEqual(slurm["paths"]["results_root"], "outputs_slurm")

    def test_inspection_exposes_one_externally_resolved_training_condition(self) -> None:
        config = load_config(ROOT_CONFIG)
        config["input_condition"] = {
            "enabled": True,
            "condition": "patch_shuffle_grid_4",
            "feature": "shape",
            "transform": "patch_shuffle",
            "strength": 4,
            "grid_size": 4,
            "seed": 2026,
        }
        summary = inspection_summary(config, "training")
        self.assertEqual(summary["condition_names"], ["patch_shuffle_grid_4"])
        self.assertTrue(summary["matched_training"]["enabled"])
        self.assertFalse(summary["matched_training"]["planning_enabled"])
        self.assertTrue(
            summary["matched_training"]["resolved_input_condition_enabled"]
        )
        self.assertEqual(
            summary["matched_training"]["resolved_condition_names"],
            ["patch_shuffle_grid_4"],
        )
        self.assertEqual(summary["expansion"]["owner"], "none")
        self.assertEqual(summary["expected_total_run_count"], 1)


if __name__ == "__main__":
    unittest.main()
