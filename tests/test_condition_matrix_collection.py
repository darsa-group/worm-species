from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.worm_species.evaluation.condition_matrix import evaluation_relation
from src.worm_species.experiments.result_collection import collect_condition_matrix
from src.worm_species.experiments.result_collection import collect_results
from src.worm_species.results.discovery import discover_experiment
from src.worm_species.slurm.collection import DUAL_OUTPUT_NAMES


MODELS = ("resnet18", "resnet50", "efficientnet_b0", "vit_b_16")
CONDITIONS = (
    "original",
    "patch_shuffle_grid_2",
    "patch_shuffle_grid_4",
)
TASKS = ("genus", "species", "age")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _create_run(
    root: Path,
    index: int,
    model: str,
    train_condition: str,
) -> Path:
    run = root / f"run_{index:03d}" / "scientific_run"
    matrix = run / "condition_matrix_evaluation"
    matrix.mkdir(parents=True)
    run_name = f"{model}_{train_condition}"
    _write_json(
        run / "config.json",
        {
            "model": {"name": model},
            "data": {"target_cols": {task: task for task in TASKS}},
            "input_condition": {
                "enabled": True,
                "condition": train_condition,
            },
            "condition_matrix_evaluation": {
                "enabled": True,
                "condition_names": list(CONDITIONS),
            },
        },
    )
    _write_json(run / "test_metrics.json", {"mean_macro_f1": 0.7})
    condition_rows = []
    task_rows = []
    relation_counts = {"matched": 0, "rgb_stress": 0, "cross_condition": 0}
    for test_index, test_condition in enumerate(CONDITIONS):
        relation = evaluation_relation(train_condition, test_condition)
        relation_counts[relation] += 1
        identity = {
            "schema_version": 1,
            "run_name": run_name,
            "model": model,
            "train_condition": train_condition,
            "test_condition": test_condition,
            "evaluation_relation": relation,
        }
        condition_rows.append({**identity, "mean_macro_f1": 0.6 + test_index / 10})
        for task in TASKS:
            task_rows.append({
                **identity,
                "task": task,
                "n": 2,
                "macro_f1": 0.6 + test_index / 10,
            })
    pd.DataFrame(condition_rows).to_csv(matrix / "condition_metrics.csv", index=False)
    pd.DataFrame(task_rows).to_csv(matrix / "task_metrics.csv", index=False)
    _write_json(
        matrix / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "expected_condition_cells": 3,
            "completed_condition_cells": 3,
            "expected_task_rows": 9,
            "completed_task_rows": 9,
            "relation_counts": relation_counts,
        },
    )
    report = matrix / "classification_reports" / "original"
    confusion = matrix / "confusion_matrices" / "original"
    report.mkdir(parents=True)
    confusion.mkdir(parents=True)
    (report / "classification_report_genus.csv").write_text(
        ",precision,recall,f1-score,support\nmacro avg,1,1,1,2\n"
    )
    (confusion / "confusion_matrix_genus.csv").write_text(
        ",a,b\na,1,0\nb,0,1\n"
    )
    return run


def _create_complete_experiment(root: Path) -> list[Path]:
    runs = []
    index = 0
    for model in MODELS:
        for train_condition in CONDITIONS:
            runs.append(_create_run(root, index, model, train_condition))
            index += 1
    return runs


class ConditionMatrixCollectionContracts(unittest.TestCase):
    def test_canonical_collection_writes_exact_complete_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            _create_complete_experiment(root)
            collect_results(root)

            conditions = pd.read_csv(root / "condition_matrix_evaluations.csv")
            tasks = pd.read_csv(root / "condition_matrix_task_metrics.csv")
            summary = json.loads(
                (root / "condition_matrix_collection_summary.json").read_text()
            )
            self.assertEqual(len(conditions), 36)
            self.assertEqual(len(tasks), 108)
            self.assertTrue(set((
                "run_name", "model", "train_condition", "test_condition",
                "evaluation_relation", "source_path",
            )).issubset(conditions.columns))
            self.assertTrue(set(("task", "macro_f1")).issubset(tasks.columns))
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["expected_runs"], 12)
            self.assertEqual(summary["complete_manifests"], 12)
            self.assertEqual(summary["expected_condition_rows"], 36)
            self.assertEqual(summary["collected_condition_rows"], 36)
            self.assertEqual(summary["expected_task_rows"], 108)
            self.assertEqual(summary["collected_task_rows"], 108)
            self.assertEqual(
                summary["relation_counts"],
                {"matched": 12, "rgb_stress": 8, "cross_condition": 16},
            )
            self.assertEqual(summary["warnings"], [])
            self.assertTrue({
                "condition_matrix_evaluations.csv",
                "condition_matrix_task_metrics.csv",
                "condition_matrix_collection_summary.json",
            }.issubset(DUAL_OUTPUT_NAMES))

    def test_missing_malformed_and_duplicate_runs_are_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            runs = [
                _create_run(root, index, "resnet18", condition)
                for index, condition in enumerate(CONDITIONS)
            ]
            first_matrix = runs[0] / "condition_matrix_evaluation"
            for path in first_matrix.iterdir():
                if path.is_file():
                    path.unlink()

            second_matrix = runs[1] / "condition_matrix_evaluation"
            (second_matrix / "manifest.json").write_text("{not-json")

            third_matrix = runs[2] / "condition_matrix_evaluation"
            (third_matrix / "condition_metrics.csv").write_text("wrong,column\n1,2\n")
            task_frame = pd.read_csv(third_matrix / "task_metrics.csv")
            pd.concat([task_frame, task_frame.iloc[[0]]], ignore_index=True).to_csv(
                third_matrix / "task_metrics.csv", index=False
            )

            summary = collect_condition_matrix(root)
            self.assertIsNotNone(summary)
            assert summary is not None
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["expected_runs"], 3)
            self.assertEqual(summary["missing_manifests"], 1)
            self.assertEqual(summary["malformed_manifests"], 1)
            self.assertEqual(summary["incomplete_runs"], 2)
            self.assertEqual(summary["duplicate_task_rows"], 1)
            codes = {warning["code"] for warning in summary["warnings"]}
            self.assertTrue({
                "missing_matrix_manifest",
                "malformed_matrix_manifest",
                "invalid_matrix_schema",
                "duplicate_matrix_rows",
            }.issubset(codes))
            self.assertEqual(summary["warning_counts"]["missing_matrix_manifest"], 1)
            self.assertTrue((root / "condition_matrix_evaluations.csv").is_file())
            self.assertTrue((root / "condition_matrix_task_metrics.csv").is_file())

    def test_discovery_indexes_matrix_artifacts_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "experiment"
            run = _create_run(root, 0, "resnet18", "original")
            collect_condition_matrix(root)
            before = {
                path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in root.rglob("*")
                if path.is_file()
            }

            snapshot = discover_experiment(root, now=10**12)

            after = {
                path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)
            self.assertEqual(len(snapshot.runs), 1)
            kinds = {artifact.kind for artifact in snapshot.runs[0].artifacts}
            self.assertTrue({
                "condition_matrix/manifest.json",
                "condition_matrix/condition_metrics.csv",
                "condition_matrix/task_metrics.csv",
                "condition_matrix/classification_report/original/classification_report_genus.csv",
                "condition_matrix/confusion_matrix/original/confusion_matrix_genus.csv",
            }.issubset(kinds))
            experiment_kinds = {
                artifact.kind for artifact in snapshot.experiment.artifacts
            }
            self.assertTrue({
                "condition_matrix_evaluations.csv",
                "condition_matrix_task_metrics.csv",
                "condition_matrix_collection_summary.json",
            }.issubset(experiment_kinds))
            self.assertEqual(snapshot.runs[0].path, str(run.absolute()))


if __name__ == "__main__":
    unittest.main()
