from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

from scripts.build_paper_results import (
    build_report,
    collect_runs,
    configure_report_style,
)
from src.worm_species.data.conditions import GaussianBlurPercent, ResolutionLoss
from src.worm_species.data.holdouts import apply_data_holdout
from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission


ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "configs" / "clusters" / "genome.yaml"


class VisualAblationTransformTests(unittest.TestCase):
    def test_zero_percent_is_identity(self) -> None:
        image = torch.rand(3, 16, 16)
        self.assertTrue(torch.equal(GaussianBlurPercent(0)(image), image))
        self.assertTrue(torch.equal(ResolutionLoss(0)(image), image))

    def test_full_resolution_loss_reduces_to_one_spatial_value(self) -> None:
        image = torch.rand(3, 16, 16)
        result = ResolutionLoss(100)(image)
        self.assertEqual(result.shape, image.shape)
        expected = result[:, :1, :1].expand_as(result)
        self.assertTrue(torch.allclose(result, expected))


class BiologicalHoldoutTests(unittest.TestCase):
    def test_removes_development_cohort_but_preserves_test(self) -> None:
        train = pd.DataFrame(
            [
                {"id": "a", "species": "A_longa", "age": "Juvenile"},
                {"id": "b", "species": "A_longa", "age": "Adult"},
                {"id": "c", "species": "Other", "age": "Juvenile"},
            ]
        )
        validation = train.copy()
        test = train.copy()
        config = {
            "data_holdout": {
                "enabled": True,
                "name": "juvenile_a_longa",
                "question": "Can the held-out combination still be recognised?",
                "remove_from": ["train", "validation"],
                "where": {"species": "A_longa", "age": "Juvenile"},
                "evaluation_where": {
                    "species": "A_longa",
                    "age": "Juvenile",
                },
                "primary_tasks": ["species", "age"],
            }
        }
        result = apply_data_holdout(
            config=config,
            train=train,
            validation=validation,
            test=test,
            target_cols={"species": "species", "age": "age"},
            group_col="id",
        )
        self.assertEqual(result.train["id"].tolist(), ["b", "c"])
        self.assertEqual(result.validation["id"].tolist(), ["b", "c"])
        pd.testing.assert_frame_equal(result.test, test)
        self.assertEqual(result.evaluation_cohort["id"].tolist(), ["a"])
        self.assertTrue(result.audit["test_unchanged"])


class GenomePaperPlanTests(unittest.TestCase):
    def test_all_three_stages_are_bounded_and_hloss_is_off(self) -> None:
        expected = {
            "genome_ablation_baseline.yaml": (45, "paper-baseline"),
            "genome_visual_ablation.yaml": (
                125,
                "paper-visual-ablation",
            ),
            "genome_data_holdouts.yaml": (20, "paper-data-holdouts"),
        }
        for filename, (run_count, wandb_group) in expected.items():
            with self.subTest(config=filename):
                config = load_submission_config(
                    ROOT / "dev" / filename,
                    cluster_config=CLUSTER,
                )
                plan = plan_submission(config)
                self.assertEqual(plan.array_size, run_count)
                self.assertEqual(plan.array_max_active, 8)
                self.assertEqual(config["wandb"]["project"], "worm-species-paper")
                self.assertEqual(config["wandb"]["group"], wandb_group)
                self.assertTrue(config["wandb"]["compact"])
                self.assertFalse(config["wandb"]["save_code"])
                self.assertFalse(config["wandb"]["log_model"])
                self.assertTrue(
                    all(
                        not spec.resolved_config["multi_task"][
                            "hierarchy_loss"
                        ]["enabled"]
                        for spec in plan.run_specs
                    )
                )

    def test_genome_uses_one_locked_cache_copy_per_node(self) -> None:
        cluster = load_submission_config(
            ROOT / "dev" / "genome_ablation_baseline.yaml",
            cluster_config=CLUSTER,
        )
        scratch = cluster["slurm"]["scratch"]
        self.assertEqual(scratch["mode"], "node_shared_cache")
        template = (
            ROOT / "slurm" / "templates"
            / "node_shared_cache_array_job.sh.tmpl"
        ).read_text(encoding="utf-8")
        self.assertIn('flock -x 200', template)
        self.assertIn('rsync -a "$CACHE_ROOT/" "$partial/"', template)
        self.assertIn("SOURCE_READY.signature", template)


class PaperReportTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_report_builds_tables_and_every_graph_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paper = Path(temporary) / "paper_result"
            weights = {"genus": 1.0, "species": 0.5, "age": 2.0}

            for seed, score in ((2024, 0.78), (2025, 0.80), (2026, 0.82)):
                baseline = (
                    paper / "runs" / "baseline"
                    / f"baseline_resnet50_seed_{seed}"
                )
                self._write_json(
                    baseline / "config.json",
                    {
                        "seed": seed,
                        "model": {
                            "name": "resnet50",
                            "pretrained": True,
                        },
                        "training": {"lr": 0.0005, "batch_size": 64},
                        "multi_task": {
                            "loss_weights": weights,
                            "selection_metric": "mean_macro_f1",
                        },
                        "input_condition": {
                            "name": "original",
                            "transform": "original",
                        },
                    },
                )
                self._write_json(
                    baseline / "run_summary.json",
                    {
                        "best_val_score": score,
                        "test_mean_macro_f1": score,
                        "test_genus_macro_f1": score,
                        "test_species_macro_f1": score,
                        "test_age_macro_f1": score,
                    },
                )
                self._write_json(
                    baseline / "model_parameters.json",
                    {
                        "total_parameters": 1000,
                        "trainable_parameters": 1000,
                    },
                )

            conditions = [
                ("colour", "colour_000pct", "saturation", {"retention": 0}),
                ("patch", "patch_shuffle_2x2", "patch_shuffle", {"grid_size": 2}),
                (
                    "blur",
                    "gaussian_blur_010pct",
                    "gaussian_blur_percent",
                    {"percent": 10},
                ),
                (
                    "resolution",
                    "resolution_loss_010pct",
                    "resolution_loss",
                    {"percent": 10},
                ),
            ]
            for suffix, name, transform, parameters in conditions:
                run = (
                    paper / "runs" / "visual_ablation"
                    / f"{suffix}_resnet50"
                )
                self._write_json(
                    run / "config.json",
                    {
                        "model": {"name": "resnet50"},
                        "multi_task": {"loss_weights": weights},
                        "input_condition": {
                            "name": name,
                            "transform": transform,
                            **parameters,
                        },
                    },
                )
                self._write_json(
                    run / "run_summary.json",
                    {"test_mean_macro_f1": 0.70},
                )
                matrix = run / "condition_matrix_evaluation"
                matrix.mkdir(parents=True)
                pd.DataFrame(
                    [
                        {
                            "run_name": suffix,
                            "model": "resnet50",
                            "train_condition": name,
                            "train_transform": transform,
                            "train_parameters": json.dumps(parameters),
                            "test_condition": "original",
                            "task": task,
                            "macro_f1": score,
                        }
                        for task, score in (
                            ("genus", 0.7),
                            ("species", 0.6),
                            ("age", 0.8),
                        )
                    ]
                ).to_csv(matrix / "task_metrics.csv", index=False)

            holdout = (
                paper / "runs" / "data_holdouts" / "juvenile_resnet50"
            )
            self._write_json(
                holdout / "config.json",
                {
                    "model": {"name": "resnet50"},
                    "multi_task": {"loss_weights": weights},
                    "input_condition": {
                        "name": "original",
                        "transform": "original",
                    },
                    "data_holdout": {
                        "name": "juvenile_a_longa",
                        "question": "Can it recognise the held-out cohort?",
                    },
                },
            )
            self._write_json(
                holdout / "run_summary.json",
                {"test_mean_macro_f1": 0.75},
            )
            self._write_json(
                holdout / "split_summary.json",
                {
                    "data_holdout": {
                        "name": "juvenile_a_longa",
                        "question": "Can it recognise the held-out cohort?",
                        "where": {
                            "species": "A_longa",
                            "age": "Juvenile",
                        },
                        "removed": {
                            "train": {"rows": 10, "individuals": 3},
                            "validation": {"rows": 4, "individuals": 1},
                        },
                        "evaluation_cohort": {
                            "rows": 5,
                            "individuals": 2,
                        },
                    }
                },
            )
            evaluation = holdout / "data_holdout_evaluation"
            evaluation.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "holdout": "juvenile_a_longa",
                        "task": "age",
                        "class_supported_by_training_head": True,
                        "target_recall": 0.65,
                    }
                ]
            ).to_csv(evaluation / "task_metrics.csv", index=False)

            data_root = Path(temporary) / "data"
            split_root = Path(temporary) / "splits"
            split_dir = split_root / "split_csv"
            split_dir.mkdir(parents=True)
            rows = []
            for index, (genus, stage) in enumerate(
                (
                    ("Aporrectodea", "Adult"),
                    ("Aporrectodea", "Juvenile"),
                    ("Lumbricus", "Adult"),
                    ("Lumbricus", "Juvenile"),
                )
            ):
                raw_name = f"image_{index}_raw.jpg"
                seg_name = f"image_{index}_seg.jpg"
                mask_name = f"image_{index}_mask.png"
                data_root.mkdir(exist_ok=True)
                array = np.full((32, 32, 3), 40 + index * 40, dtype=np.uint8)
                Image.fromarray(array).save(data_root / raw_name)
                Image.fromarray(np.flip(array, axis=1).copy()).save(
                    data_root / seg_name
                )
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[4 : 12 + index * 4, 5:27] = 255
                Image.fromarray(mask).save(data_root / mask_name)
                rows.append({
                    "barcode": f"worm_{index}",
                    "genus": genus,
                    "species_label": f"{genus}_species",
                    "taxon_label": f"{genus}_species",
                    "life_stage": stage,
                    "rel_path_raw": raw_name,
                    "rel_path_seg": seg_name,
                    "rel_path_segmask": mask_name,
                })
            for filename in (
                "train_split.csv",
                "val_split.csv",
                "test_split.csv",
            ):
                pd.DataFrame(rows).to_csv(split_dir / filename, index=False)

            summary = build_report(
                paper, split_root=split_root, data_root=data_root
            )
            self.assertEqual(summary["completed_runs"], 8)
            self.assertEqual(len(summary["figures"]), 17)
            self.assertTrue(summary["representative_images_created"])
            self.assertTrue(summary["transformation_examples_created"])
            self.assertTrue(
                summary["all_manuscript_artifacts_ready"],
                summary["manuscript_artifacts"],
            )
            self.assertTrue(all(summary["manuscript_artifacts"].values()))
            self.assertIn(
                "visual_gaussian_blur_by_model.png", summary["figures"]
            )
            self.assertIn(
                "visual_gaussian_blur_original_test_by_model.png",
                summary["figures"],
            )
            self.assertIn(
                "figure_6_matched_vs_original_performance.png",
                summary["figures"],
            )
            self.assertIn(
                "dataset_composition_by_taxon_stage_and_split.csv",
                summary["tables"],
            )
            collected = collect_runs(paper)
            blur = collected[
                collected["transform"].eq("gaussian_blur_percent")
            ].iloc[0]
            self.assertEqual(blur["percent"], 10)

    def test_report_colours_are_editable_without_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            style = Path(temporary) / "style.yaml"
            style.write_text(
                "dpi: 144\n"
                "palette: ['#010203', '#AABBCC']\n"
                "heatmap_colormap: magma\n",
                encoding="utf-8",
            )
            configured = configure_report_style(style)
            self.assertEqual(configured["dpi"], 144)
            self.assertEqual(configured["palette"][0], "#010203")


if __name__ == "__main__":
    unittest.main()
