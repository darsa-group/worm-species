from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from src.worm_species.data.conditions import BinaryForegroundMask
from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission
from src.worm_species.training.metrics import score_for_selection


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "configs" / "clusters" / "genome.yaml"
SEEDS = set(range(40, 2941, 100))


class PublicationDesignTests(unittest.TestCase):
    def test_every_stage_has_the_confirmed_design(self) -> None:
        expected = {
            "genome_publication_30seed_baseline.yaml": (90, {"convnext_base", "vit_b_16", "resnet50"}),
            "genome_publication_30seed_visual.yaml": (690, {"convnext_base"}),
            "genome_publication_30seed_interactions.yaml": (600, {"convnext_base"}),
            "genome_publication_30seed_taxon_baseline.yaml": (30, {"convnext_base"}),
            "genome_publication_30seed_taxon_holdouts.yaml": (330, {"convnext_base"}),
        }
        for filename, (count, models) in expected.items():
            with self.subTest(filename=filename):
                config = load_submission_config(ROOT / "dev" / filename, cluster_config=CLUSTER)
                plan = plan_submission(config)
                self.assertEqual(plan.array_size, count)
                self.assertEqual(set(plan.models), models)
                self.assertEqual({spec.resolved_config["seed"] for spec in plan.run_specs}, SEEDS)
                for spec in plan.run_specs:
                    resolved = spec.resolved_config
                    self.assertEqual(resolved["multi_task"]["loss_weights"], {"genus": 1.0, "species": 0.5, "age": 2.0})
                    self.assertEqual(resolved["multi_task"]["selection_metric"], "loss")
                    self.assertFalse(resolved["multi_task"]["hierarchy_loss"]["enabled"])
                    self.assertEqual(resolved["multi_task"]["hierarchy_loss"]["weight"], 0.0)
                    self.assertFalse(resolved["checkpointing"]["save_last"])
                    self.assertEqual(resolved["early_stopping"]["mode"], "min")
                    self.assertTrue(resolved["wandb"]["focused_metrics"])

    def test_pipeline_total_and_mask_cache_contract(self) -> None:
        pipeline = yaml.safe_load((ROOT / "dev" / "genome_publication_30seed_pipeline.yaml").read_text())
        self.assertEqual(pipeline["required_hierarchy_loss_weights"], [0.0])
        self.assertIn("binary_mask", pipeline["condition_cache"]["transforms"])
        self.assertEqual(sum((90, 690, 600, 30, 330)), 1740)
        self.assertEqual(pipeline["report"]["script"], "../scripts/build_publication_bundle.py")

    def test_binary_mask_is_three_channel_and_binary(self) -> None:
        image = torch.zeros(3, 4, 4)
        image[:, 1:3, 2] = 0.5
        result = BinaryForegroundMask()(image)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(set(result.unique().tolist()), {0.0, 1.0})
        self.assertTrue(torch.equal(result[0], result[1]))
        self.assertTrue(torch.equal(result[1], result[2]))

    def test_validation_loss_selection_is_minimised(self) -> None:
        self.assertEqual(score_for_selection({"loss": 0.25}, "loss", "min"), 0.25)
        self.assertEqual(score_for_selection({}, "loss", "min"), float("inf"))

    def test_generated_array_job_skips_completed_best_checkpoint(self) -> None:
        template = (ROOT / "slurm" / "templates" / "node_shared_cache_array_job.sh.tmpl").read_text()
        self.assertIn("already complete with a best checkpoint; skipping", template)
        self.assertIn("-name best_model.pt", template)


if __name__ == "__main__":
    unittest.main()
