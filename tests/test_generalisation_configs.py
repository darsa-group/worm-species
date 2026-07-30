from __future__ import annotations

import unittest
from pathlib import Path

from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "train" / "generalisation"
CLUSTER = ROOT / "configs" / "clusters" / "genome.yaml"


class GeneralisationConfigTests(unittest.TestCase):
    def test_all_configs_plan_without_hidden_cartesian_expansion(self) -> None:
        expected_runs = {
            "shared_heads.yaml": 15,
            "single_task_genus.yaml": 5,
            "single_task_species.yaml": 5,
            "single_task_age.yaml": 15,
            "split_taxonomy_age.yaml": 15,
            "split_joint_sampler.yaml": 5,
            "split_pcgrad.yaml": 5,
            "split_age_supcon.yaml": 5,
            "split_joint_sampler_pcgrad.yaml": 15,
            "split_full.yaml": 15,
            "split_species_adversary.yaml": 5,
        }
        for filename, count in expected_runs.items():
            with self.subTest(config=filename):
                config = load_submission_config(
                    CONFIG_ROOT / filename,
                    cluster_config=CLUSTER,
                )
                plan = plan_submission(config)
                self.assertEqual(plan.array_size, count)
                prefix = filename.removesuffix(".yaml")
                self.assertTrue(all(
                    spec.run_id.startswith(f"{prefix}_")
                    for spec in plan.run_specs
                ))
                self.assertEqual(
                    {
                        spec.resolved_config["seed"]
                        for spec in plan.run_specs
                    },
                    {40, 41, 42} if count == 15 else {42},
                )
                self.assertEqual(
                    {
                        (
                            spec.resolved_config.get(
                                "data_holdout", {}
                            ) or {}
                        ).get("name")
                        for spec in plan.run_specs
                    },
                    {
                        "original_baseline",
                        "juvenile_aporrectodea_longa",
                        "juvenile_allolobophora_chlorotica",
                        "juvenile_genus_aporrectodea",
                        "unseen_species_aporrectodea_longa_for_genus",
                    },
                )

    def test_full_and_exploratory_configs_remain_separate(self) -> None:
        full = load_submission_config(
            CONFIG_ROOT / "split_full.yaml",
            cluster_config=CLUSTER,
        )
        full_run = plan_submission(full).run_specs[0].resolved_config
        self.assertEqual(
            full_run["model"]["multitask_architecture"],
            "split_taxonomy_age",
        )
        self.assertEqual(
            full_run["data"]["sampler"]["type"],
            "joint_species_stage",
        )
        self.assertEqual(
            full_run["training"]["gradient_strategy"]["type"],
            "pcgrad",
        )
        self.assertTrue(
            full_run["loss"]["age_supervised_contrastive"]["enabled"]
        )
        self.assertFalse(
            full_run["model"]["age_species_adversary"]["enabled"]
        )

        exploratory = load_submission_config(
            CONFIG_ROOT / "split_species_adversary.yaml",
            cluster_config=CLUSTER,
        )
        adversary_run = (
            plan_submission(exploratory).run_specs[0].resolved_config
        )
        self.assertTrue(
            adversary_run["model"]["age_species_adversary"]["enabled"]
        )
        self.assertEqual(
            adversary_run["data"]["sampler"]["type"],
            "default",
        )
        self.assertEqual(
            adversary_run["training"]["gradient_strategy"]["type"],
            "standard",
        )


if __name__ == "__main__":
    unittest.main()
