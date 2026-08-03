from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_adult_taxon_ablation_results import (
    build_adult_taxon_report,
    observed_taxon_stage_combinations,
)
from scripts.build_paper_results import (
    _original_test_condition_summary,
    build_report,
)
from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "configs" / "clusters" / "genome.yaml"
EXPECTED_HOLDOUTS = {
    "adult_allolobophora_chlorotica",
    "adult_aporrectodea_caliginosa",
    "adult_aporrectodea_longa",
    "adult_aporrectodea_rosea",
    "adult_aporrectodea_tuberculata",
    "adult_lumbricus_castaneus",
    "adult_lumbricus_festivus",
    "adult_lumbricus_terrestris_herculeus",
    "juvenile_allolobophora_chlorotica",
    "juvenile_aporrectodea_longa",
    "juvenile_aporrectodea_rosea",
}


class AdultTaxonConfigTests(unittest.TestCase):
    def test_plans_cover_every_observed_taxon_stage_combination(self) -> None:
        inventory = observed_taxon_stage_combinations(ROOT)
        self.assertEqual(set(inventory["holdout"]), EXPECTED_HOLDOUTS)
        self.assertEqual(set(inventory["stage"]), {"Adult", "Juvenile"})
        baseline = load_submission_config(
            ROOT / "dev" / "genome_adult_taxon_baseline.yaml",
            cluster_config=CLUSTER,
        )
        holdout = load_submission_config(
            ROOT / "dev" / "genome_adult_taxon_holdouts.yaml",
            cluster_config=CLUSTER,
        )
        baseline_plan = plan_submission(baseline)
        holdout_plan = plan_submission(holdout)
        self.assertEqual(baseline_plan.array_size, 30)
        self.assertEqual(holdout_plan.array_size, 330)
        for plan in (baseline_plan, holdout_plan):
            self.assertEqual(
                {spec.resolved_config["seed"] for spec in plan.run_specs},
                {40, 41, 42},
            )
            self.assertEqual(
                {
                    float(
                        spec.resolved_config["multi_task"]
                        ["hierarchy_loss"]["weight"]
                    )
                    for spec in plan.run_specs
                },
                {0.0, 0.2},
            )
        self.assertEqual(
            {
                spec.resolved_config["data_holdout"]["name"]
                for spec in holdout_plan.run_specs
            },
            EXPECTED_HOLDOUTS,
        )


class AdultTaxonReportTests(unittest.TestCase):
    @staticmethod
    def _write_run(
        root: Path,
        *,
        stage: str,
        seed: int,
        weight: float,
        control: bool,
    ) -> None:
        run = root / "runs" / stage / f"resnet50_seed_{seed}_h_{weight:g}"
        definition = {
            "enabled": True,
            "name": "adult_aporrectodea_longa",
            "where": {
                "genus": "Aporrectodea",
                "species": "Aporrectodea_longa",
                "age": "Adult",
            },
            "evaluation_where": {
                "genus": "Aporrectodea",
                "species": "Aporrectodea_longa",
                "age": "Adult",
            },
            "primary_tasks": ["genus", "species", "age"],
        }
        config = {
            "seed": seed,
            "model": {"name": "resnet50"},
            "multi_task": {
                "hierarchy_loss": {
                    "enabled": weight > 0,
                    "weight": weight,
                }
            },
        }
        evaluation_name = "data_holdout_evaluation"
        if control:
            evaluation_name = "data_holdout_control_evaluation"
            config["evaluation"] = {
                "data_holdout_controls": {"definitions": [definition]}
            }
        else:
            config["data_holdout"] = definition
        run.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run / "label_to_index_by_task.json").write_text(
            json.dumps({
                "genus": {"Aporrectodea": 0, "Lumbricus": 1},
                "species": {
                    "Aporrectodea_longa": 0,
                    "Lumbricus_terrestris": 1,
                },
                "age": {"Adult": 0, "Juvenile": 1},
            }),
            encoding="utf-8",
        )
        score = (0.80 if control else 0.60)
        score += 0.02 * (seed - 40) + 0.05 * (weight > 0)
        rows = [
            {
                "holdout": definition["name"],
                "cohort": cohort,
                "task": task,
                "target_n": 10,
                "target_recall": score,
                "target_recall_image": score,
                "class_supported_by_training_head": True,
            }
            for cohort in ("development_withheld", "independent_test")
            for task in ("genus", "species", "age")
        ]
        evaluation = run / evaluation_name
        evaluation.mkdir()
        pd.DataFrame(rows).to_csv(evaluation / "task_metrics.csv", index=False)

    def test_report_writes_both_hloss_views_and_figure_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper = root / "adult_taxon_ablation_result"
            for seed in (40, 41, 42):
                for weight in (0.0, 0.2):
                    self._write_run(
                        paper,
                        stage="adult_taxon_baseline",
                        seed=seed,
                        weight=weight,
                        control=True,
                    )
                    self._write_run(
                        paper,
                        stage="adult_taxon_holdouts",
                        seed=seed,
                        weight=weight,
                        control=False,
                    )
            splits = root / "split_csv"
            splits.mkdir()
            row = pd.DataFrame([{
                "barcode": "worm-1",
                "genus": "Aporrectodea",
                "species_label": "Aporrectodea_longa",
                "life_stage": "Adult",
            }])
            for filename in (
                "train_split.csv", "val_split.csv", "test_split.csv"
            ):
                row.to_csv(splits / filename, index=False)
            manifest = build_adult_taxon_report(paper, split_root=root)
            self.assertEqual(manifest["completed_metric_runs"], 12)
            self.assertTrue(all(manifest["figures"].values()), manifest)
            for stem in manifest["figures"]:
                for suffix in ("png", "pdf", "svg"):
                    self.assertTrue(
                        (paper / "figures" / f"{stem}.{suffix}").is_file()
                    )
                self.assertTrue(
                    (paper / "figure_sources" / stem / "plot_data.csv").is_file()
                )
            paired = pd.read_csv(
                paper / "tables" / "paired_adult_taxon_ablation_differences.csv"
            )
            self.assertTrue(
                (paired["target_recall_difference"].round(6) == -0.2).all()
            )


class PaperReportRegressionTests(unittest.TestCase):
    def test_incomplete_result_root_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            summary = build_report(
                Path(temporary) / "paper_result",
                split_root=ROOT,
                data_root=Path(temporary) / "missing-images",
            )
        self.assertEqual(summary["completed_runs"], 0)

    def test_original_test_summary_preserves_run_directory(self) -> None:
        cross = pd.DataFrame([
            {
                "run_dir": "/tmp/run-1",
                "run_name": "resolution-run",
                "model": "resnet50",
                "seed": 40,
                "loss_recipe": "genus=1|species=0.5|age=2",
                "hierarchy_loss_weight": 0.0,
                "train_condition": "resolution_loss_050pct",
                "train_transform": "resolution_loss",
                "train_parameters": '{"percent": 50}',
                "test_condition": "original",
                "task": task,
                "macro_f1": 0.7,
                "chance": 0.25,
            }
            for task in ("genus", "species", "age")
        ])
        summary = _original_test_condition_summary(cross)
        self.assertEqual(summary["run_dir"].tolist(), ["/tmp/run-1"])


if __name__ == "__main__":
    unittest.main()
