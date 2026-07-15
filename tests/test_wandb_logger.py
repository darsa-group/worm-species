from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import warnings

from src.worm_species.logging.wandb_logger import CLASSIFICATION_REPORT_COLUMNS
from src.worm_species.logging.wandb_logger import canonical_condition_relation
from src.worm_species.logging.wandb_logger import create_wandb_logger
from src.worm_species.logging.wandb_logger import flatten_slash_config
from src.worm_species.logging.wandb_logger import robustness_ratio


class FakeArtifact:
    def __init__(self, *, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class FakeTable:
    def __init__(self, *, dataframe=None, columns=None, data=None):
        self.dataframe = dataframe
        self.columns = list(columns or [])
        self.data = list(data or [])


class FakePlot:
    def __init__(self):
        self.calls = []

    def confusion_matrix(self, **kwargs):
        self.calls.append(kwargs)
        return {"plot": kwargs}


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.logged = []
        self.metrics = []
        self.artifacts = []
        self.alerts = []
        self.finish_count = 0
        self.fail_log = False
        self.fail_finish = False

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def log(self, payload):
        if self.fail_log:
            raise RuntimeError("synthetic log failure")
        self.logged.append(payload)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def alert(self, **kwargs):
        self.alerts.append(kwargs)

    def finish(self):
        self.finish_count += 1
        if self.fail_finish:
            raise RuntimeError("synthetic finish failure")


class FakeBackend:
    Table = FakeTable
    Artifact = FakeArtifact

    def __init__(self, *, fail_init=False):
        self.plot = FakePlot()
        self.run = FakeRun()
        self.init_calls = []
        self.fail_init = fail_init

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise RuntimeError("synthetic init failure")
        return self.run


def config(*, enabled=True, mode="offline", log_model=False):
    return {
        "seed": 17,
        "model": {"name": "convnext_base", "pretrained": True},
        "preprocessing": {"image_size": 384},
        "data": {"image_size": 224, "colour_retention": 1.0},
        "augmentation": {
            "horizontal_flip": {"enabled": True, "probability": 0.5},
            "vertical_flip": {"enabled": False, "probability": 0.5},
            "rotation": {"enabled": True, "degrees": 90},
        },
        "training": {"lr": 0.001, "batch_size": 16, "epochs": 5},
        "experiment": {"type": "matched_condition"},
        "input_condition": {
            "enabled": True,
            "condition": "gaussian_sigma_2",
            "transform": "gaussian_blur",
            "strength": 2.0,
            "sigma": 2.0,
        },
        "wandb": {
            "enabled": enabled,
            "project": "worm-species-test",
            "entity": None,
            "group": "submission-001",
            "name": None,
            "mode": mode,
            "job_type": "train",
            "tags": ["multitask"],
            "save_code": True,
            "log_model": log_model,
        },
    }


class WandbLoggerInitialisationTests(unittest.TestCase):
    def test_disabled_mode_never_imports_or_initialises_backend(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "src.worm_species.logging.wandb_logger.importlib.import_module"
        ) as importer:
            logger = create_wandb_logger(
                config(enabled=False), "run", directory
            )
        self.assertFalse(logger.enabled)
        self.assertEqual(logger.disabled_reason, "disabled")
        importer.assert_not_called()

        backend = FakeBackend()
        logger = create_wandb_logger(
            config(mode="disabled"), "run", directory, backend=backend
        )
        self.assertFalse(logger.enabled)
        self.assertEqual(backend.init_calls, [])

    def test_legacy_profile_gate_is_preserved(self):
        backend = FakeBackend()
        logger = create_wandb_logger(
            config(),
            "run",
            ".",
            SimpleNamespace(wandb=False),
            backend=backend,
        )
        self.assertFalse(logger.enabled)
        self.assertEqual(backend.init_calls, [])

    def test_unavailable_and_initialisation_failure_degrade_to_null(self):
        with mock.patch(
            "src.worm_species.logging.wandb_logger.importlib.import_module",
            side_effect=ImportError("not installed"),
        ), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            unavailable = create_wandb_logger(config(), "run", ".")
        self.assertFalse(unavailable.enabled)
        self.assertEqual(unavailable.disabled_reason, "unavailable")
        self.assertIn("continuing locally", str(caught[0].message))

        backend = FakeBackend(fail_init=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            failed = create_wandb_logger(
                config(), "run", ".", backend=backend
            )
        self.assertFalse(failed.enabled)
        self.assertEqual(failed.disabled_reason, "initialisation failed")
        self.assertIn("continuing locally", str(caught[0].message))

    def test_offline_init_uses_one_canonical_field_per_setting(self):
        backend = FakeBackend()
        logger = create_wandb_logger(
            config(), "legacy-hashed-run", "/tmp/results", backend=backend
        )
        self.assertTrue(logger.enabled)
        self.assertEqual(len(backend.init_calls), 1)
        call = backend.init_calls[0]
        self.assertEqual(call["mode"], "offline")
        self.assertEqual(call["name"], "legacy-hashed-run")
        self.assertEqual(call["project"], "worm-species-test")
        self.assertEqual(call["group"], "submission-001")
        self.assertEqual(call["job_type"], "train")
        tracking = call["config"]
        self.assertEqual(tracking["model/name"], "convnext_base")
        self.assertNotIn("model__name", tracking)
        self.assertNotIn("data/image_size", tracking)
        self.assertNotIn("seed", tracking)
        self.assertNotIn("input_condition/condition", tracking)
        self.assertEqual(tracking["preprocessing/image_size"], 384)
        self.assertEqual(tracking["augmentation/horizontal_flip"], True)
        self.assertEqual(tracking["augmentation/rotation_degrees"], 90)
        self.assertEqual(tracking["training/seed"], 17)
        self.assertEqual(
            tracking["training_condition/name"], "gaussian_sigma_2"
        )
        self.assertEqual(tracking["experiment/type"], "matched_condition")
        self.assertEqual(
            backend.run.summary["architecture"], "convnext_base"
        )
        self.assertEqual(
            backend.run.summary["training_condition"], "gaussian_sigma_2"
        )
        self.assertEqual(len(tracking), len(set(tracking)))
        self.assertEqual(len(backend.run.metrics), 4)

    def test_slash_flattener_has_one_path_for_each_leaf(self):
        value = {"model.name": {"child.key": 2}, "items": [1, 2]}
        self.assertEqual(
            flatten_slash_config(value),
            {"model.name/child.key": 2, "items": [1, 2]},
        )

    def test_condition_aliases_deduplicate_with_canonical_precedence(self):
        configured = config()
        configured["input_condition"]["name"] = "legacy_name"
        configured["training_condition"] = {
            "name": "canonical_name",
            "transform": "gaussian_blur",
            "strength": 2.0,
        }
        backend = FakeBackend()
        create_wandb_logger(configured, "run", ".", backend=backend)
        tracking = backend.init_calls[0]["config"]
        self.assertEqual(
            tracking["training_condition/name"], "canonical_name"
        )
        self.assertFalse(
            any(key.startswith("input_condition/") for key in tracking)
        )


class WandbLoggerMetricTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.logger = create_wandb_logger(
            config(), "run", ".", backend=self.backend
        )

    def test_epoch_keys_preserve_aliases_and_canonical_names(self):
        payload = self.logger.log_epoch_metrics(
            epoch=2,
            learning_rate=0.0005,
            train_metrics={"loss": 1.2, "genus_loss": 0.4},
            val_metrics={"loss": 1.0, "genus_macro_f1": 0.75},
        )
        self.assertEqual(
            payload,
            {
                "epoch": 2,
                "learning_rate": 0.0005,
                "train/loss": 1.2,
                "train/genus_loss": 0.4,
                "val/loss": 1.0,
                "val/genus_macro_f1": 0.75,
            },
        )

    def test_test_condition_adds_canonical_path_and_original_aliases(self):
        transformed = self.logger.log_test_condition(
            "grayscale",
            {"species_macro_f1": 0.4, "mean_macro_f1": 0.5},
            train_condition="original",
        )
        self.assertEqual(transformed["test/grayscale/species_macro_f1"], 0.4)
        self.assertEqual(transformed["condition_relation"], "rgb_stress")
        self.assertNotIn("test/species_macro_f1", transformed)

        original = self.logger.log_test_condition(
            "original",
            {"genus_macro_f1": 0.8, "mean_macro_f1": 0.75},
            train_condition="original",
        )
        self.assertEqual(original["test/original/genus_macro_f1"], 0.8)
        self.assertEqual(original["test/genus_macro_f1"], 0.8)
        self.assertEqual(
            self.backend.run.summary["test_original_mean_macro_f1"], 0.75
        )
        self.assertEqual(
            self.backend.run.summary["test/genus_macro_f1"], 0.8
        )

    def test_relation_vocabulary_is_unambiguous(self):
        self.assertEqual(
            canonical_condition_relation("original", "original"), "original"
        )
        self.assertEqual(
            canonical_condition_relation("grayscale", "grayscale"), "matched"
        )
        self.assertEqual(
            canonical_condition_relation("original", "grayscale"), "rgb_stress"
        )
        self.assertEqual(
            canonical_condition_relation("grayscale", "original"),
            "cross_condition",
        )

    def test_confusion_matrix_filters_missing_labels_and_uses_unique_keys(self):
        first = self.logger.log_confusion_matrix(
            condition="original",
            task="species",
            y_true=[0, -1, 1, 8, "bad"],
            y_pred=[1, 0, 1, 0, 0],
            class_names=["A", "B"],
        )
        second = self.logger.log_confusion_matrix(
            condition="grayscale",
            task="species",
            y_true=[0, 1],
            y_pred=[0, 1],
            class_names=["A", "B"],
        )
        self.assertEqual(first, "confusion_matrix/original/species")
        self.assertIsNone(second)  # transformed uploads require explicit config
        call = self.backend.plot.calls[0]
        self.assertEqual(call["y_true"], [0, 1])
        self.assertEqual(call["preds"], [1, 1])
        self.assertEqual(call["class_names"], ["A", "B"])
        logged = self.backend.run.logged[-1]
        self.assertIn("confusion_matrix/original/species", logged)
        self.assertIn("confusion_matrix_species", logged)

        self.logger.cfg["wandb"]["confusion_matrices"] = {
            "enabled": True,
            "conditions": ["grayscale"],
            "tasks": ["species"],
        }
        key = self.logger.log_confusion_matrix(
            condition="grayscale",
            task="species",
            y_true=[0, 1],
            y_pred=[0, 1],
            class_names=["A", "B"],
        )
        self.assertEqual(key, "confusion_matrix/grayscale/species")
        self.assertNotIn("confusion_matrix_species", self.backend.run.logged[-1])

    def test_confusion_matrix_with_no_valid_labels_is_a_safe_noop(self):
        key = self.logger.log_confusion_matrix(
            condition="original",
            task="age",
            y_true=[-1, -1],
            y_pred=[0, 1],
            class_names=["adult", "juvenile"],
        )
        self.assertIsNone(key)
        self.assertEqual(self.backend.plot.calls, [])

    def test_classification_report_has_stable_expected_columns(self):
        rows = self.logger.log_classification_report(
            condition="original",
            task="species",
            report={
                "A": {
                    "precision": 0.5,
                    "recall": 1.0,
                    "f1-score": 2 / 3,
                    "support": 2,
                },
                "accuracy": 0.5,
                "macro avg": {
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1-score": 0.5,
                    "support": 2,
                },
            },
            metrics={
                "species_macro_f1": 0.6,
                "species_balanced_accuracy": 0.55,
                "species_accuracy": 0.5,
            },
            train_condition="original",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(tuple(rows[0]), CLASSIFICATION_REPORT_COLUMNS)
        self.assertEqual(rows[0]["class_name"], "A")
        self.assertEqual(rows[0]["f1"], 2 / 3)
        logged = self.backend.run.logged[-1]
        self.assertIn("tables/classification_report_original", logged)
        self.assertIn("tables/classification_report_by_condition", logged)
        for table in logged.values():
            self.assertEqual(len(table.columns), len(set(table.columns)))

    def test_robustness_is_safe_and_keeps_numeric_strength(self):
        self.assertEqual(robustness_ratio(0.5, 1.0), 0.5)
        self.assertTrue(math.isnan(robustness_ratio(0.5, 0.0)))
        self.assertTrue(math.isnan(robustness_ratio(float("nan"), 1.0)))
        rows = [{
            "condition": "saturation_050pct",
            "task": "species",
            "strength": 0.5,
            "macro_f1": 0.4,
            "original_macro_f1": 0.8,
        }]
        payload = self.logger.log_robustness_table(rows)
        self.assertEqual(
            payload["robustness/saturation_050pct/species_ratio"], 0.5
        )
        self.assertEqual(rows[0]["strength"], 0.5)
        logged = self.backend.run.logged[-1]
        self.assertIn("tables/robustness_summary", logged)
        self.assertIn("cue_suppression/macro_f1_ratios", logged)
        table = logged["tables/robustness_summary"]
        self.assertEqual(len(table.columns), len(set(table.columns)))
        self.assertIn("test_condition", table.columns)
        self.assertNotIn("condition", table.columns)

    def test_generic_tables_deduplicate_dataframe_columns(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed")
        frame = pd.DataFrame([["original", "duplicate", 0.8]])
        frame.columns = ["condition", "condition", "mean_macro_f1"]
        self.assertTrue(self.logger.log_test_metrics_table(frame))
        table = self.backend.run.logged[-1][
            "tables/test_metrics_by_condition"
        ]
        self.assertEqual(
            table.columns, ["test_condition", "mean_macro_f1"]
        )

    def test_comparison_uses_canonical_adaptation_gain_key(self):
        self.assertTrue(self.logger.log_comparison_table([{
            "condition": "grayscale",
            "task": "species",
            "adaptation_gain": 0.2,
        }]))
        self.assertEqual(
            self.backend.run.logged[-1][
                "comparison/grayscale/species_adaptation_gain"
            ],
            0.2,
        )


class WandbLoggerFailureAndArtifactTests(unittest.TestCase):
    def test_model_artifact_respects_log_model_and_lightweight_files_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configuration = root / "config.json"
            model = root / "best_model.pt"
            configuration.write_text("{}\n")
            model.write_bytes(b"checkpoint")

            backend = FakeBackend()
            logger = create_wandb_logger(
                config(log_model=False), "run", root, backend=backend
            )
            logged = logger.log_artifacts([configuration, model])
            self.assertEqual(logged, ["run-scientific-record"])
            self.assertEqual(len(backend.run.artifacts), 1)

            enabled_backend = FakeBackend()
            enabled_logger = create_wandb_logger(
                config(log_model=True), "run", root, backend=enabled_backend
            )
            logged = enabled_logger.log_artifacts(
                [configuration, model], model_metadata={"best_epoch": 3}
            )
            self.assertEqual(
                logged, ["run-scientific-record", "run-best-model"]
            )
            model_artifact = enabled_backend.run.artifacts[-1]
            self.assertEqual(model_artifact.type, "model")
            self.assertEqual(model_artifact.metadata["best_epoch"], 3)

    def test_backend_log_and_finish_failures_never_touch_local_record(self):
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "test_metrics.json"
            local.write_bytes(b'{"mean_macro_f1": 0.75}\n')
            before = local.read_bytes()
            backend = FakeBackend()
            logger = create_wandb_logger(
                config(), "run", directory, backend=backend
            )
            backend.run.fail_log = True
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                payload = logger.log_epoch_metrics(
                    epoch=1,
                    learning_rate=0.1,
                    train_metrics={"loss": 1.0},
                    val_metrics={},
                )
                logger.log_test_condition(
                    "original", {"mean_macro_f1": 0.75}
                )
            self.assertEqual(payload["train/loss"], 1.0)
            self.assertTrue(logger.degraded)
            self.assertFalse(logger.enabled)
            self.assertEqual(local.read_bytes(), before)
            self.assertEqual(len(caught), 1)

            backend.run.fail_finish = True
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                logger.finalise_run(status="failed")
                logger.finalise_run(status="failed")
            self.assertEqual(backend.run.finish_count, 1)
            self.assertEqual(len(caught), 1)
            self.assertEqual(local.read_bytes(), before)

    def test_finalise_populates_stable_summary_and_is_idempotent(self):
        backend = FakeBackend()
        logger = create_wandb_logger(config(), "run", ".", backend=backend)
        logger.finalise_run(
            summary={
                "checkpoint_path": "outputs/run/best_model.pt",
                "best_epoch": 4,
            }
        )
        logger.finalise_run()
        self.assertEqual(backend.run.summary["run_status"], "completed")
        self.assertEqual(backend.run.summary["best_epoch"], 4)
        self.assertEqual(backend.run.finish_count, 1)

    def test_multiple_evaluation_calls_reuse_exactly_one_run(self):
        backend = FakeBackend()
        logger = create_wandb_logger(config(), "run", ".", backend=backend)
        logger.log_test_condition("original", {"mean_macro_f1": 0.8})
        logger.log_test_condition("grayscale", {"mean_macro_f1": 0.5})
        logger.log_robustness_table([{
            "condition": "grayscale",
            "task": "species",
            "ratio_to_original": 0.5,
        }])
        self.assertEqual(len(backend.init_calls), 1)
        self.assertEqual(logger.run, backend.run)


if __name__ == "__main__":
    unittest.main()
