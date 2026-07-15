from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard.condition_matrix import filter_matrix_rows
from dashboard.condition_matrix import load_indexed_matrix_rows
from dashboard.condition_matrix import load_matrix_completion_summary
from dashboard.condition_matrix import macro_f1_pivot
from dashboard.condition_matrix import matrix_relation_counts
from dashboard.condition_matrix import normalise_matrix_rows
from src.worm_species.evaluation.condition_matrix import evaluation_relation


MODELS = ("resnet18", "resnet50", "efficientnet_b0", "vit_b_16")
CONDITIONS = (
    "original",
    "patch_shuffle_grid_2",
    "patch_shuffle_grid_4",
)
TASKS = ("genus", "species", "age")


def _rows() -> list[dict[str, object]]:
    rows = []
    for model_index, model in enumerate(MODELS):
        for train_index, train_condition in enumerate(CONDITIONS):
            run_name = f"{model}_{train_condition}"
            for test_index, test_condition in enumerate(CONDITIONS):
                relation = evaluation_relation(train_condition, test_condition)
                for task_index, task in enumerate(TASKS):
                    rows.append({
                        "run_name": run_name,
                        "model": model,
                        "train_condition": train_condition,
                        "test_condition": test_condition,
                        "evaluation_relation": relation,
                        "task": task,
                        "macro_f1": 0.5
                        + model_index / 100
                        + train_index / 20
                        + test_index / 40
                        + task_index / 80,
                    })
    return rows


def _artifact(path: Path, kind: str) -> dict[str, object]:
    return {
        "kind": kind,
        "path": str(path),
        "available": True,
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }


class DashboardConditionMatrixContracts(unittest.TestCase):
    def test_pure_filters_counts_and_three_by_three_pivot(self) -> None:
        rows, warnings = normalise_matrix_rows(_rows())
        self.assertEqual(warnings, [])
        self.assertEqual(len(rows), 108)
        original = next(
            row
            for row in rows
            if row["train_condition"] == "original"
            and row["test_condition"] == "original"
        )
        self.assertEqual(original["evaluation_relation"], "matched")
        self.assertEqual(original["condition_relation"], "original")
        self.assertEqual(
            matrix_relation_counts(rows),
            {"matched": 12, "rgb_stress": 8, "cross_condition": 16},
        )
        filtered = filter_matrix_rows(
            rows,
            models={"resnet18"},
            relations={"matched", "cross_condition"},
            tasks={"genus"},
        )
        self.assertEqual(len(filtered), 7)
        pivot = macro_f1_pivot(rows, model="resnet18", task="genus")
        self.assertEqual(tuple(pivot.shape), (3, 3))
        self.assertEqual(set(pivot.index), set(CONDITIONS))
        self.assertEqual(set(pivot.columns), set(CONDITIONS))

    def test_invalid_rows_become_warnings_instead_of_dashboard_failures(self) -> None:
        rows, warnings = normalise_matrix_rows([
            {"model": "missing identities"},
            {
                "run_name": "bad_relation",
                "model": "resnet18",
                "train_condition": "original",
                "test_condition": "original",
                "evaluation_relation": "ambiguous",
                "task": "genus",
                "macro_f1": 0.5,
            },
            {
                "run_name": "bad_score",
                "model": "resnet18",
                "train_condition": "original",
                "test_condition": "original",
                "evaluation_relation": "matched",
                "task": "genus",
                "macro_f1": 2.0,
            },
        ])
        self.assertEqual(rows, [])
        self.assertEqual(len(warnings), 3)

    def test_missing_matrix_relation_is_derived_without_collapsing_logging_relation(self) -> None:
        rows, warnings = normalise_matrix_rows([
            {
                "run_name": "rgb",
                "model": "resnet18",
                "train_condition": "original",
                "test_condition": "original",
                "task": "genus",
                "macro_f1": 0.5,
            },
            {
                "run_name": "stress",
                "model": "resnet18",
                "train_condition": "original",
                "test_condition": "grayscale",
                "condition_relation": "rgb_stress",
                "task": "genus",
                "macro_f1": 0.4,
            },
        ])

        self.assertEqual(warnings, [])
        self.assertEqual(rows[0]["evaluation_relation"], "matched")
        self.assertEqual(rows[0]["condition_relation"], "original")
        self.assertEqual(rows[1]["evaluation_relation"], "rgb_stress")
        self.assertEqual(rows[1]["condition_relation"], "rgb_stress")

    def test_aggregate_is_preferred_over_bounded_per_run_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / "condition_matrix_task_metrics.csv"
            pd.DataFrame(_rows()).to_csv(aggregate, index=False)
            fallback = root / "run_task_metrics.csv"
            pd.DataFrame(_rows()[:9]).to_csv(fallback, index=False)
            experiment = {
                "artifacts": [
                    _artifact(aggregate, "condition_matrix_task_metrics.csv")
                ]
            }
            runs = [{
                "artifacts": [
                    _artifact(fallback, "condition_matrix/task_metrics.csv")
                ]
            }]

            rows, warnings, paths, mode = load_indexed_matrix_rows(
                experiment, runs
            )
            self.assertEqual(mode, "aggregate")
            self.assertEqual(paths, [str(aggregate)])
            self.assertEqual(len(rows), 108)
            self.assertEqual(warnings, [])

            rows, warnings, paths, mode = load_indexed_matrix_rows(
                None, runs, max_rows=5
            )
            self.assertEqual(mode, "per_run")
            self.assertEqual(paths, [str(fallback)])
            self.assertEqual(len(rows), 5)
            self.assertTrue(any("row bound" in warning for warning in warnings))

    def test_completion_summary_is_loaded_only_from_indexed_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "condition_matrix_collection_summary.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "status": "incomplete",
                "expected_condition_rows": 36,
                "collected_condition_rows": 30,
                "warning_counts": {"missing_matrix_manifest": 2},
            }))
            summary, warning = load_matrix_completion_summary({
                "artifacts": [
                    _artifact(path, "condition_matrix_collection_summary.json")
                ]
            })
            self.assertIsNone(warning)
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(summary["collected_condition_rows"], 30)


if __name__ == "__main__":
    unittest.main()
