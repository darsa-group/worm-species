from __future__ import annotations

import copy
import hashlib
import unittest
from pathlib import Path

import yaml

from src.worm_species.config.inspect import inspection_summary
from src.worm_species.config.loading import load_config
from src.worm_species.config.validation import validate_config


ROOT = Path(__file__).resolve().parents[1]
ROOT_CONFIG = ROOT / "config.yaml"
BASE_CONFIG = ROOT / "configs" / "defaults" / "base.yaml"
EXPERIMENTS = ROOT / "configs" / "experiments"
MATRIX_EXPERIMENT = "patch_shuffle_matrix.yaml"

# Captured immediately after the condition-matrix feature landed and before the
# hierarchy move. Non-matrix configs are compared after removing the one
# explicitly permitted default-equivalent disabled matrix block.
PRE_HIERARCHY_HASHES = {
    "config.yaml": "f4bb26f6be1fe7bf7baff138dbd75fef9ddd5bc7bf92a65a7ff2c3027e6a7231",
    "colour_ablation.yaml": "50eea5718454d72794c85672abfb64a1b7415e8649dcc8ddbc560ecba1384fdb",
    "dual_cue.yaml": "2c8664c024c0a7e6cf72604665e156c5723b575b5fc58b955370c9fabade355d",
    MATRIX_EXPERIMENT: "f2f70ad06258c7cca8597122df3f7c9db59a8ace01c362c9659350a66dd1705a",
    "persistent_hierarchy.yaml": "d296ef8c9cde69ee6bacc9e364de510dc727fab05493ff8b75e62a2fc57c5375",
    "persistent_hierarchy_wandb.yaml": "18a4c91d7a5526089183a12e39c8f555691d2ad0ac9579319c5dacc6ce141043",
    "standard.yaml": "f779bf5982c8cea557fc9a5fe70828d59f3fe2878cb8a12105d54d13b71955d5",
}


def _canonical_hash(config: dict) -> str:
    payload = yaml.safe_dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SimplifiedConfigurationHierarchyContracts(unittest.TestCase):
    def test_quick_start_is_short_and_exposes_only_common_choices(self) -> None:
        lines = ROOT_CONFIG.read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(len(lines), 55)
        self.assertLessEqual(len(lines), 70)
        raw = yaml.safe_load("\n".join(lines))
        self.assertEqual(raw["extends"], "configs/defaults/base.yaml")
        self.assertEqual(
            set(raw),
            {
                "extends",
                "seed",
                "data",
                "model",
                "training",
                "multi_task",
                "wandb",
                "output",
                "cache",
                "input_condition",
                "test_cue_suppression",
                "condition_matrix_evaluation",
                "matched_condition_training",
                "colour_ablation",
                "sweep",
            },
        )
        self.assertEqual(set(raw["data"]), {"root_dir", "metadata_csv", "image_size"})
        self.assertNotIn("split", raw)
        self.assertNotIn("early_stopping", raw)
        for section in (
            "input_condition",
            "test_cue_suppression",
            "condition_matrix_evaluation",
            "matched_condition_training",
            "colour_ablation",
            "sweep",
        ):
            self.assertEqual(raw[section], {"enabled": False})

    def test_resolved_scientific_configs_match_pre_move_hashes(self) -> None:
        paths = {"config.yaml": ROOT_CONFIG}
        paths.update({name: EXPERIMENTS / name for name in PRE_HIERARCHY_HASHES if name != "config.yaml"})
        for name, expected_hash in PRE_HIERARCHY_HASHES.items():
            with self.subTest(config=name):
                resolved = load_config(paths[name])
                if name != MATRIX_EXPERIMENT:
                    resolved = copy.deepcopy(resolved)
                    self.assertEqual(
                        resolved.pop("condition_matrix_evaluation"),
                        {
                            "enabled": False,
                            "condition_names": ["original"],
                            "write_reports": True,
                        },
                    )
                self.assertEqual(_canonical_hash(resolved), expected_hash)

    def test_experiment_extends_graph_still_flows_through_root(self) -> None:
        expected_parents = {
            "colour_ablation.yaml": "../../config.yaml",
            "dual_cue.yaml": "../../config.yaml",
            MATRIX_EXPERIMENT: "../../config.yaml",
            "persistent_hierarchy.yaml": "../../config.yaml",
            "persistent_hierarchy_wandb.yaml": "persistent_hierarchy.yaml",
            "standard.yaml": "../../config.yaml",
        }
        for name, expected_parent in expected_parents.items():
            path = EXPERIMENTS / name
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            with self.subTest(config=name):
                self.assertEqual(raw["extends"], expected_parent)

    def test_root_and_experiment_count_ladder_is_unchanged(self) -> None:
        expected = {
            "config.yaml": 1,
            "standard.yaml": 2,
            MATRIX_EXPERIMENT: 12,
            "colour_ablation.yaml": 202,
            "dual_cue.yaml": 224,
        }
        for name, count in expected.items():
            path = ROOT_CONFIG if name == "config.yaml" else EXPERIMENTS / name
            with self.subTest(config=name):
                config = load_config(path)
                workflow = "training" if name == "config.yaml" else "run_specs"
                summary = inspection_summary(config, workflow)
                self.assertEqual(summary["expected_total_run_count"], count)
                self.assertEqual(
                    summary["expansion"][
                        "expected_internal_training_runs_per_resolved_spec"
                    ],
                    1,
                )

    def test_base_and_root_are_valid_safe_single_run_configs(self) -> None:
        for path in (BASE_CONFIG, ROOT_CONFIG):
            with self.subTest(config=path.name):
                config = load_config(path)
                validate_config(
                    config,
                    workflow="training",
                    check_paths=False,
                    check_model_registry=False,
                )
                self.assertFalse(config["sweep"]["enabled"])
                self.assertFalse(config["matched_condition_training"]["enabled"])
                self.assertFalse(config["colour_ablation"]["enabled"])
                self.assertFalse(config["test_cue_suppression"]["enabled"])
                self.assertFalse(config["condition_matrix_evaluation"]["enabled"])
                self.assertFalse(config["input_condition"]["enabled"])

    def test_patch_matrix_block_and_order_remain_exact(self) -> None:
        config = load_config(EXPERIMENTS / MATRIX_EXPERIMENT)
        self.assertEqual(
            config["condition_matrix_evaluation"],
            {
                "enabled": True,
                "condition_names": [
                    "original",
                    "patch_shuffle_grid_2",
                    "patch_shuffle_grid_4",
                ],
                "write_reports": True,
            },
        )
        keys = list(config)
        self.assertLess(keys.index("sweep"), keys.index("condition_matrix_evaluation"))
        self.assertLess(keys.index("condition_matrix_evaluation"), keys.index("slurm"))


if __name__ == "__main__":
    unittest.main()
