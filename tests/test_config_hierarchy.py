from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import yaml

from src.worm_species.config.inspect import inspection_summary
from src.worm_species.config.loading import load_config
from src.worm_species.config.normalization import normalize_config
from src.worm_species.config.validation import validate_config


ROOT = Path(__file__).resolve().parents[1]
ROOT_CONFIG = ROOT / "config.yaml"
BASE_CONFIG = ROOT / "configs" / "defaults" / "base.yaml"
EXPERIMENTS = ROOT / "configs" / "experiments"
MATRIX_EXPERIMENT = "patch_shuffle_matrix.yaml"

# Canonical resolved-configuration snapshots. Scientific expansion is protected
# separately by exact run-spec hashes and count contracts.
CANONICAL_CONFIG_HASHES = {
    "config.yaml": "fe4533abc51c8e7645354aa4d0a4c1508144368cf457d8a8763de8991dd51751",
    "colour_ablation.yaml": "d76c960db01430c2c6f081e0199eeaa99a85b7df0ae74991c9f4f9fb36718244",
    "dual_cue.yaml": "eb247799f25baa47ed45062000bf12518bb8a444c7e788cc9b7293f2de376f9c",
    MATRIX_EXPERIMENT: "5495fe07e3210d5e575e38b081215712e4bc4d50334286443fe832bbd797a92e",
    "persistent_hierarchy.yaml": "561e4b945e1214bc65e7d1d4b46fe4602d6788e6f733feee5d3445396bae6798",
    "persistent_hierarchy_wandb.yaml": "9bfa46a89bfa70ba2c9f0fa71717cac234727366f3a88a59ac44ebb0395949a8",
    "standard.yaml": "25a1dbe92c9bb42e0f1f4e69be51574c19d484514b2172823b2b52b011ef23ed",
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
                "preprocessing",
                "augmentation",
                "model",
                "training",
                "multi_task",
                "wandb",
                "output",
                "cache",
                "input_condition",
                "evaluation",
                "sweep",
            },
        )
        self.assertEqual(set(raw["data"]), {"root_dir", "metadata_csv"})
        self.assertEqual(raw["preprocessing"]["image_size"], 224)
        self.assertNotIn("split", raw)
        self.assertNotIn("early_stopping", raw)
        self.assertFalse(raw["input_condition"]["enabled"])
        self.assertFalse(raw["evaluation"]["test_conditions"]["enabled"])
        self.assertFalse(raw["evaluation"]["condition_matrix"]["enabled"])
        self.assertEqual(raw["sweep"], {"enabled": False, "parameters": {}})

    def test_resolved_configs_match_canonical_snapshots(self) -> None:
        paths = {"config.yaml": ROOT_CONFIG}
        paths.update({name: EXPERIMENTS / name for name in CANONICAL_CONFIG_HASHES if name != "config.yaml"})
        for name, expected_hash in CANONICAL_CONFIG_HASHES.items():
            with self.subTest(config=name):
                resolved = load_config(paths[name])
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
        canonical = normalize_config(config)
        self.assertEqual(
            canonical["evaluation"]["condition_matrix"],
            {
                "enabled": True,
                "conditions": [
                    "original",
                    "patch_shuffle_grid_2",
                    "patch_shuffle_grid_4",
                ],
                "write_reports": True,
            },
        )
        raw = yaml.safe_load((EXPERIMENTS / MATRIX_EXPERIMENT).read_text())
        keys = list(raw)
        self.assertLess(keys.index("sweep"), keys.index("evaluation"))
        self.assertLess(keys.index("evaluation"), keys.index("slurm"))


if __name__ == "__main__":
    unittest.main()
