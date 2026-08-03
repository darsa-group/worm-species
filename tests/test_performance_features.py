from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from src.worm_species.data.datasets import MultiTaskWormImageDataset
from src.worm_species.data.datasets import MultiViewWormImageDataset
from src.worm_species.data.datasets import multiview_collate
from src.worm_species.data.samplers import CrossSpeciesStageContrastiveBatchSampler
from src.worm_species.evaluation.predictions import aggregate_individual_probabilities
from src.worm_species.evaluation.predictions import ensemble_prediction_frames
from src.worm_species.evaluation.predictions import prediction_metrics
from src.worm_species.analysis.performance_report import (
    build_performance_report,
    discover_performance_runs,
)
from src.worm_species.models.multitask import MultiTaskClassifier
from src.worm_species.models.multitask import SingleTaskClassifier
from src.worm_species.models.multitask import SplitTaxonomyAgeClassifier
from src.worm_species.training.losses import genus_supervised_contrastive_loss
from src.worm_species.training.losses import taxonomy_consistency_loss
from src.worm_species.training.optimizers import StagedUnfreezer
from src.worm_species.training.optimizers import build_optimizer
from src.worm_species.training.optimizers import validate_optimizer_coverage
from src.worm_species.training.runner import save_age_embedding_diagnostics
from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission


class ImageBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.fc = nn.Linear(4, 3)

    def forward(self, values):
        values = self.conv(values).mean(dim=(2, 3))
        return self.fc(values)


def _prediction_frame(probabilities):
    rows = []
    for index, probability in enumerate(probabilities):
        rows.append({
            "run_id": "run", "checkpoint": "best", "split": "test",
            "holdout": "", "image_path": f"{index}.png", "barcode": "A",
            "task": "age", "true_label": "Juvenile",
            "predicted_label": ["Juvenile", "Adult"][int(probability[1] > probability[0])],
            "probabilities": json.dumps(probability),
            "_class_names": ["Juvenile", "Adult"],
        })
    return pd.DataFrame(rows)


class PerformanceFeatureTests(unittest.TestCase):
    def test_representation_export_falls_back_to_age_features_with_metadata(self):
        class FeatureModel(nn.Module):
            def forward(self, images):
                features = images.mean(dim=(2, 3))
                return {
                    "age_embedding": None,
                    "age_features": features,
                    "age_logits": features[:, :2],
                }

        loader = [{
            "image": torch.randn(3, 3, 4, 4),
            "label_names": {"age": ["Adult"] * 3},
            "metadata_label_names": {
                "age": ["Adult", "Juvenile", "Adult"],
                "species": ["one", "two", "three"],
            },
            "path": ["a.png", "b.png", "c.png"],
        }]
        with tempfile.TemporaryDirectory() as directory:
            artifacts = save_age_embedding_diagnostics(
                model=FeatureModel(),
                loader=loader,
                device=torch.device("cpu"),
                use_amp=False,
                out_dir=Path(directory),
            )
            self.assertIsNotNone(artifacts)
            manifest = json.loads(
                Path(artifacts["manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["representation_type"], "age_branch_features"
            )
            metadata = pd.read_csv(artifacts["metadata"])
            self.assertEqual(metadata["species"].tolist(), ["one", "two", "three"])

    def test_report_does_not_export_blank_performance_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_performance_report(
                root / "no-runs",
                root / "report",
            )
            self.assertEqual(
                manifest["performance_architectures"], []
            )
            self.assertFalse(
                any(manifest["performance_figure_availability"].values())
            )
            self.assertTrue(
                all(
                    paths == []
                    for paths in manifest["performance_figures"].values()
                )
            )

    def test_report_rejects_non_target_predictions_from_single_task_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "bad_single_task"
            run.mkdir()
            (run / "config.json").write_text(json.dumps({
                "seed": 42,
                "model": {"name": "resnet18", "multitask_architecture": "single_task", "target_task": "species"},
            }))
            bad = _prediction_frame([[0.9, 0.1]])
            bad.to_csv(run / "predictions_best.csv", index=False)
            bad.to_csv(run / "individual_predictions_best.csv", index=False)
            (run / "test_metrics_best.json").write_text("{}")
            (run / "completion_manifest.json").write_text("{}")
            (run / "runtime_provenance.json").write_text(json.dumps({"split_hashes": {}}))
            (run / "model_parameters.json").write_text(json.dumps({"total_parameters": 1}))
            with self.assertRaisesRegex(ValueError, "single-task predictions"):
                discover_performance_runs(Path(directory))

    def test_performance_configs_resolve_three_backbones_and_matched_seeds(self):
        root = Path(__file__).resolve().parents[1]
        config_root = root / "configs/train/performance"
        required = {
            "differential_lr.yaml", "staged_unfreezing.yaml",
            "multiview_inference.yaml", "multiview_training.yaml",
            "cross_species_age_batches.yaml", "genus_supcon.yaml",
            "taxonomy_consistency.yaml", "checkpoint_ensemble.yaml",
            "performance_full.yaml", "shared_heads.yaml",
            "split_taxonomy_age.yaml",
            "single_task_genus.yaml", "single_task_species.yaml",
            "single_task_age.yaml",
        }
        for filename in required:
            plan = plan_submission(load_submission_config(
                config_root / filename,
                cluster_config=root / "configs/clusters/genome.yaml",
            ))
            self.assertEqual(plan.array_size, 45)
            self.assertEqual(set(plan.models), {"convnext_base", "vit_b_16", "resnet18"})
            self.assertEqual({spec.resolved_config["seed"] for spec in plan.run_specs}, {40, 41, 42})

    def test_dataset_returns_barcode_and_multiview_never_mixes_individuals(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(3):
                path = Path(directory) / f"{index}.pt"
                torch.save(torch.full((3, 2, 2), float(index)), path)
                paths.append(path.name)
            frame = pd.DataFrame({
                "image": paths, "barcode": ["A", "A", "B"],
                "age": ["Juvenile", "Juvenile", "Adult"],
            })
            dataset = MultiTaskWormImageDataset(
                frame, directory, "image", target_cols={"age": "age"},
                label_to_index_by_task={"age": {"Juvenile": 0, "Adult": 1}},
                metadata_cols={"age": "age"},
                image_is_tensor=True, crop_to_foreground=False,
            )
            self.assertEqual(dataset[0]["barcode"], "A")
            multiview = MultiViewWormImageDataset(dataset, images_per_individual=3)
            first = multiview[0]
            second = multiview[1]
            self.assertEqual(first["barcode"], "A")
            self.assertEqual(first["image"].shape[0], 2)
            self.assertEqual(second["image"].shape[0], 1)
            batch = multiview_collate([first, second])
            self.assertEqual(tuple(batch["image"].shape[:2]), (2, 2))
            self.assertEqual(batch["view_mask"].sum(dim=1).tolist(), [2, 1])
            self.assertEqual(
                batch["metadata_label_names"]["age"],
                ["Juvenile", "Adult"],
            )

    def test_individual_probability_aggregation_and_metric_levels(self):
        images = _prediction_frame([[0.9, 0.1], [0.2, 0.8]])
        individual = aggregate_individual_probabilities(images)
        self.assertEqual(len(individual), 1)
        self.assertEqual(individual.iloc[0]["predicted_label"], "Juvenile")
        metrics = prediction_metrics(images, individual)
        self.assertIn("age_image_macro_f1", metrics)
        self.assertIn("age_individual_macro_f1", metrics)

    def test_inconsistent_labels_for_one_barcode_fail_closed(self):
        images = _prediction_frame([[0.9, 0.1], [0.8, 0.2]])
        images.loc[1, "true_label"] = "Adult"
        with self.assertRaisesRegex(ValueError, "inconsistent true labels"):
            aggregate_individual_probabilities(images)

    def test_ensemble_averages_probabilities(self):
        first = _prediction_frame([[0.9, 0.1]])
        second = _prediction_frame([[0.1, 0.9]])
        ensemble = ensemble_prediction_frames([first, second])
        self.assertEqual(json.loads(ensemble.iloc[0]["probabilities"]), [0.5, 0.5])

    def test_multiview_forward_is_compatible_with_all_architectures(self):
        classes = {"genus": 2, "species": 3, "age": 2}
        models = [
            MultiTaskClassifier(ImageBackbone(), classes),
            SingleTaskClassifier(ImageBackbone(), classes, "species"),
            SplitTaxonomyAgeClassifier(
                ImageBackbone(), classes, model_name="test", branch_mode="residual_adapter",
                adapter_dim=2, adapter_dropout=0.0, pooling_type="global_average",
                pooling_dropout=0.0, age_projection_enabled=True,
                adversary_enabled=False,
            ),
        ]
        inputs = torch.randn(2, 3, 3, 4, 4)
        mask = torch.tensor([[True, True, True], [True, False, False]])
        for model in models:
            output = model(inputs, view_mask=mask)
            self.assertEqual(next(value for key, value in output.items() if key.endswith("_logits") and value is not None).shape[0], 2)

    def test_optimizer_groups_cover_every_trainable_parameter_once(self):
        model = MultiTaskClassifier(ImageBackbone(), {"genus": 2, "species": 3, "age": 2})
        cfg = {
            "training": {"lr": 1e-3, "weight_decay": 1e-4},
            "optimizer": {"type": "adamw", "weight_decay": 1e-4, "learning_rates": {
                "early_backbone": 1e-5, "final_backbone_stage": 3e-5,
                "task_specific_branches": 1e-4, "classification_heads": 3e-4,
                "projection_heads": 1e-4,
            }},
        }
        optimizer, summary = build_optimizer(model, cfg)
        validate_optimizer_coverage(model, optimizer)
        self.assertTrue(any(row["group_name"] == "classification_heads" for row in summary))
        optimizer.param_groups[0]["params"].append(optimizer.param_groups[0]["params"][0])
        with self.assertRaisesRegex(ValueError, "coverage violation"):
            validate_optimizer_coverage(model, optimizer)

    def test_staged_unfreezing_transitions(self):
        model = MultiTaskClassifier(ImageBackbone(), {"age": 2})
        cfg = {
            "training": {"lr": 1e-3, "weight_decay": 0.0, "staged_unfreezing": {
                "enabled": True, "heads_only_epochs": 5,
                "task_branches_epoch": 5, "full_backbone_epoch": 15,
            }},
            "optimizer": {"type": "adamw"},
        }
        staged = StagedUnfreezer.from_config(cfg)
        initial = staged.initialise(model)
        optimizer, _ = build_optimizer(model, cfg)
        changed, middle = staged.transition(model, optimizer, cfg, 5)
        self.assertFalse(changed)  # shared-head models have no task branch
        changed, full = staged.transition(model, optimizer, cfg, 15)
        self.assertTrue(changed)
        self.assertGreater(full, initial)
        self.assertGreaterEqual(full, middle)

    def test_cross_species_sampler_constructs_valid_stage_batches(self):
        rows = []
        for stage in ("Juvenile", "Adult"):
            for species in ("A one", "B two", "C three"):
                for individual in range(2):
                    rows.append({"stage": stage, "species": species, "barcode": f"{stage}-{species}-{individual}"})
        frame = pd.DataFrame(rows)
        sampler = CrossSpeciesStageContrastiveBatchSampler(
            frame, species_col="species", stage_col="stage", group_col="barcode",
            species_per_stage=3, individuals_per_species_stage=2,
            images_per_individual=1, samples_per_epoch=12,
        )
        batch = next(iter(sampler))
        selected = frame.iloc[batch]
        self.assertEqual(set(selected["stage"]), {"Juvenile", "Adult"})
        self.assertTrue(all(group["species"].nunique() >= 3 for _, group in selected.groupby("stage")))

    def test_genus_supcon_and_taxonomy_consistency(self):
        embeddings = torch.randn(6, 8, requires_grad=True)
        loss, stats = genus_supervised_contrastive_loss(
            embeddings,
            torch.tensor([0, 0, 0, 0, 1, 1]),
            torch.tensor([0, 1, 0, 1, 2, 3]),
            torch.arange(6),
        )
        self.assertIsNotNone(loss)
        self.assertGreater(stats["cross_species_positive_pairs"], 0)
        matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        logits = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
        consistency, agreement = taxonomy_consistency_loss(
            logits, logits, matrix, torch.tensor([True, True]), direction="symmetric"
        )
        self.assertAlmostEqual(float(consistency), 0.0, places=6)
        self.assertEqual(agreement, 1.0)


if __name__ == "__main__":
    unittest.main()
