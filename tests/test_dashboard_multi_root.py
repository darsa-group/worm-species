from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dashboard.index import (
    SourceSpec,
    load_derived_records,
    load_index,
    parse_source,
    refresh_indexes,
    validate_cache_path,
)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _run(root: Path, experiment: str, name: str, *, multitask: bool) -> Path:
    run = root / experiment / name
    run.mkdir(parents=True)
    config = {
        "seed": 42,
        "model": {"name": "resnet18", "pretrained": True, "freeze_backbone": False},
        "training": {
            "epochs": 12,
            "batch_size": 4,
            "lr": 0.0003,
            "weight_decay": 0.01,
            "class_weight": True,
        },
    }
    if multitask:
        config["data"] = {
            "target_cols": {"genus": "genus", "species": "species_label", "age": "life_stage"}
        }
        config["multi_task"] = {
            "loss_weights": {"genus": 1, "species": 0.5, "age": 2},
            "hierarchy_loss": {"enabled": True, "weight": 0.5},
        }
        metrics = {
            "genus_macro_f1": 0.9,
            "species_macro_f1": 0.7,
            "age_macro_f1": 0.8,
            "mean_macro_f1": 0.8,
        }
        _json(run / "label_to_index_by_task.json", {"genus": {}, "species": {}, "age": {}})
    else:
        config["data"] = {"target_col": "species_label"}
        metrics = {"loss": 1.0, "accuracy": 0.7, "balanced_accuracy": 0.6, "macro_f1": 0.65}
        # Class labels in this legacy map must not be mistaken for task names.
        _json(run / "label_to_index.json", {"Aporrectodea_longa": 0, "Lumbricus_sp": 1})
    _json(run / "config.json", config)
    _json(run / "test_metrics.json", metrics)
    return run


class TestDashboardMultiRoot(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.slurm = self.root / "outputs_slurm"
        self.single = self.root / "single_task" / "outputs"
        _run(self.slurm, "sweep", "multi", multitask=True)
        _run(self.single, "local", "single", multitask=False)
        self.cache = self.root / "cache" / "index.sqlite3"

    def test_two_labelled_roots_share_one_generation(self) -> None:
        summary = refresh_indexes(
            [
                SourceSpec("slurm", self.slurm, "slurm"),
                SourceSpec("single_task", self.single, "single_task"),
            ],
            self.cache,
        )
        index = load_index(self.cache)

        self.assertEqual(summary["runs"], 2)
        self.assertEqual({run["source_label"] for run in index["runs"]}, {"slurm", "single_task"})
        single = next(run for run in index["runs"] if run["source_label"] == "single_task")
        self.assertEqual(single["tasks"], ["species"])
        self.assertEqual(single["schema_version"], "single_task_legacy")
        self.assertEqual(single["effective_macro_f1"], 0.65)
        self.assertEqual(single["effective_macro_f1_label"], "single-task macro-F1")
        self.assertEqual(single["hyperparameters"]["epochs"], 12)
        self.assertEqual(single["hyperparameters"]["learning_rate"], 0.0003)
        multi = next(run for run in index["runs"] if run["source_label"] == "slurm")
        self.assertEqual(multi["hyperparameters"]["age_loss_weight"], 2.0)
        self.assertTrue(multi["hyperparameters"]["hierarchy_loss_enabled"])
        self.assertEqual(multi["effective_macro_f1_label"], "mean macro-F1 across tasks")

    def test_old_index_is_rebuilt_only_during_refresh(self) -> None:
        self.cache.parent.mkdir(parents=True)
        connection = sqlite3.connect(self.cache)
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO metadata VALUES ('schema_version', '1')")
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(ValueError, "outdated"):
            load_index(self.cache)
        refresh_indexes([SourceSpec("slurm", self.slurm, "slurm")], self.cache)
        self.assertEqual(load_index(self.cache)["metadata"]["schema_version"], "2")

    def test_source_and_cache_validation(self) -> None:
        self.assertEqual(parse_source(f"single_task={self.single}").kind, "single_task")
        with self.assertRaisesRegex(ValueError, "duplicate result source label"):
            refresh_indexes(
                [SourceSpec("same", self.slurm, "slurm"), SourceSpec("same", self.single, "single_task")],
                self.cache,
            )
        with self.assertRaisesRegex(ValueError, "outside results root"):
            validate_cache_path(self.slurm / "index.sqlite3", [self.slurm, self.single])

    def test_derived_manifest_is_joined_safely(self) -> None:
        derived = self.root / "derived"
        summary = derived / "single_task" / "abc" / "summary.json"
        _json(summary, {"schema_version": 1, "run_type": "single_task", "metrics": {}})
        _json(
            derived / "manifest.json",
            {
                "schema_version": 1,
                "runs": [
                    {
                        "source_label": "single_task",
                        "run_uid": "abc",
                        "summary": "single_task/abc/summary.json",
                    }
                ],
            },
        )
        records, warnings = load_derived_records(derived)
        self.assertEqual(warnings, [])
        self.assertEqual(records[("single_task", "abc")]["run_type"], "single_task")

        _json(
            derived / "manifest.json",
            {
                "schema_version": 1,
                "runs": [
                    {"source_label": "bad", "run_uid": "bad", "summary": "../outside.json"}
                ],
            },
        )
        records, warnings = load_derived_records(derived)
        self.assertEqual(records, {})
        self.assertTrue(any("escapes cache root" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
