from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.worm_species.config.inspect import inspection_summary
from src.worm_species.config.validation import ConfigValidationError
from src.worm_species.config.validation import validate_config
from src.worm_species.evaluation.condition_matrix import evaluate_condition_matrix
from src.worm_species.evaluation.condition_matrix import evaluation_relation
from src.worm_species.evaluation.condition_matrix import resolve_condition_matrix_conditions
from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission
from src.worm_species.training.cli import _plan_summary
from src.worm_species.training.modes import resolve_configured_profile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs" / "experiments" / "patch_shuffle_matrix.yaml"
LOCAL_CLUSTER = ROOT / "configs" / "clusters" / "local.yaml"
CONDITIONS = (
    "original",
    "patch_shuffle_grid_2",
    "patch_shuffle_grid_4",
)
TASKS = {"genus": "genus", "species": "species_label", "age": "life_stage"}


def _metrics(score: float) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        "loss": 1.0 - score,
        "mean_macro_f1": score,
        "complete_exact_match_accuracy": score,
        "complete_exact_match_n": 2,
    }
    for task in TASKS:
        values.update({
            f"{task}_n": 2,
            f"{task}_loss": 1.0 - score,
            f"{task}_accuracy": score,
            f"{task}_balanced_accuracy": score,
            f"{task}_macro_f1": score,
        })
    return values


def _labels() -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    true = {task: [0, 1] for task in TASKS}
    pred = {task: [0, 1] for task in TASKS}
    return true, pred


class ConditionMatrixConfigContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_submission_config(EXPERIMENT, LOCAL_CLUSTER)
        cls.plan = plan_submission(cls.config)

    def test_plan_remains_twelve_trainings_but_declares_thirty_six_cells(self) -> None:
        self.assertEqual(self.plan.array_size, 12)
        self.assertEqual(
            [condition["condition"] for condition in resolve_condition_matrix_conditions(self.config)],
            list(CONDITIONS),
        )
        cells = [
            (
                spec.model,
                spec.training_condition,
                test_condition,
                evaluation_relation(spec.training_condition, test_condition),
            )
            for spec in self.plan.run_specs
            for test_condition in CONDITIONS
        ]
        self.assertEqual(len(cells), 36)
        self.assertEqual(sum(cell[3] == "matched" for cell in cells), 12)
        self.assertEqual(sum(cell[3] == "rgb_stress" for cell in cells), 8)
        self.assertEqual(sum(cell[3] == "cross_condition" for cell in cells), 16)
        for spec in self.plan.run_specs:
            self.assertFalse(spec.resolved_config["sweep"]["enabled"])
            self.assertFalse(
                spec.resolved_config["matched_condition_training"]["enabled"]
            )
            self.assertTrue(
                spec.resolved_config["condition_matrix_evaluation"]["enabled"]
            )

    def test_inspection_reports_cells_and_task_rows_without_training_expansion(self) -> None:
        summary = inspection_summary(self.config, "run_specs")
        matrix = summary["condition_matrix_evaluation"]
        self.assertEqual(matrix["condition_names"], list(CONDITIONS))
        self.assertEqual(matrix["expected_condition_cells"], 36)
        self.assertEqual(matrix["expected_task_rows"], 108)
        self.assertFalse(matrix["expands_training_runs"])
        self.assertEqual(summary["expected_total_run_count"], 12)

        resolved = self.plan.run_specs[1].resolved_config
        dry = _plan_summary(
            resolve_configured_profile(resolved),
            [resolved],
            [self.plan.run_specs[1].experiment_type],
        )
        self.assertTrue(dry["post_training_condition_matrix"])
        self.assertEqual(dry["condition_matrix_test_conditions"], list(CONDITIONS))
        self.assertEqual(
            dry["condition_matrix_evaluation_cells_per_training_run"], 3
        )
        self.assertEqual(dry["condition_matrix_task_rows_per_training_run"], 9)
        self.assertEqual(dry["expected_internal_training_runs"], 1)

    def assert_invalid(self, config: dict, expected: str, workflow: str) -> None:
        with self.assertRaises(ConfigValidationError) as caught:
            validate_config(
                config,
                workflow=workflow,
                check_paths=False,
                check_model_registry=False,
            )
        self.assertIn(expected, str(caught.exception))

    def test_unknown_duplicate_and_missing_matched_condition_fail_clearly(self) -> None:
        unknown = copy.deepcopy(self.config)
        unknown["condition_matrix_evaluation"]["condition_names"] = [
            "original",
            "not_a_condition",
        ]
        self.assert_invalid(unknown, "not_a_condition", "run_specs")

        duplicate = copy.deepcopy(self.config)
        duplicate["condition_matrix_evaluation"]["condition_names"] = [
            "original",
            "original",
        ]
        self.assert_invalid(duplicate, "duplicate names", "run_specs")

        resolved = copy.deepcopy(self.plan.run_specs[1].resolved_config)
        resolved["condition_matrix_evaluation"]["condition_names"] = [
            "original",
            "patch_shuffle_grid_4",
        ]
        self.assert_invalid(
            resolved,
            "must include the resolved training condition 'patch_shuffle_grid_2'",
            "training",
        )

    def test_runner_hook_uses_existing_matched_metrics_and_predictions(self) -> None:
        source = (ROOT / "src" / "worm_species" / "training" / "runner.py").read_text()
        cue_position = source.index("stress = evaluate_test_cue_suppression(")
        matrix_position = source.index("condition_matrix = evaluate_condition_matrix(")
        self.assertLess(cue_position, matrix_position)
        hook = source[matrix_position : matrix_position + 1800]
        self.assertIn("baseline_metrics=test_metrics", hook)
        self.assertIn("baseline_true=true", hook)
        self.assertIn("baseline_pred=pred", hook)
        self.assertIn("test_loader_context=bundle.test_loader_context", hook)


class ConditionMatrixEvaluatorContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_submission_config(EXPERIMENT, LOCAL_CLUSTER)
        self.config["sweep"]["enabled"] = False
        self.config["matched_condition_training"]["enabled"] = False
        self.config["input_condition"] = {
            "enabled": True,
            "condition": "patch_shuffle_grid_2",
            "feature": "shape",
            "transform": "patch_shuffle",
            "strength": 2,
            "grid_size": 2,
            "seed": 2026,
        }
        self.training_condition = copy.deepcopy(self.config["input_condition"])
        self.true, self.pred = _labels()
        self.index_to_label = {
            task: {0: f"{task}_zero", 1: f"{task}_one"} for task in TASKS
        }

    def _evaluate(self, out_dir: Path):
        scores = iter((0.4, 0.6))

        def fake_epoch(**kwargs):
            score = next(scores)
            true, pred = _labels()
            return _metrics(score), true, pred

        def fake_loader(_context, condition):
            return condition["condition"]

        with mock.patch(
            "src.worm_species.evaluation.condition_matrix.make_test_condition_loader",
            side_effect=fake_loader,
        ) as loader, mock.patch(
            "src.worm_species.evaluation.condition_matrix.run_epoch",
            side_effect=fake_epoch,
        ) as epoch:
            result = evaluate_condition_matrix(
                cfg=self.config,
                run_name="synthetic_run",
                out_dir=out_dir,
                model=object(),
                training_condition=self.training_condition,
                baseline_metrics=_metrics(0.8),
                baseline_true=self.true,
                baseline_pred=self.pred,
                test_loader_context={"image_size": 16},
                criteria={task: object() for task in TASKS},
                target_cols=TASKS,
                index_to_label_by_task=self.index_to_label,
                device="cpu",
                use_amp=False,
                task_loss_weights={task: 1.0 for task in TASKS},
                normalize_loss_by_active_tasks=True,
                hierarchy_cfg={},
                child_to_parent_matrix=None,
                use_masked_labels=True,
            )
        return result, loader, epoch

    def test_three_cells_nine_task_rows_and_two_extra_inference_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, loader, epoch = self._evaluate(root)
            matrix_dir = root / "condition_matrix_evaluation"
            conditions = pd.read_csv(matrix_dir / "condition_metrics.csv")
            tasks = pd.read_csv(matrix_dir / "task_metrics.csv")
            manifest = json.loads((matrix_dir / "manifest.json").read_text())

            self.assertTrue(result["enabled"])
            self.assertEqual(result["n_conditions"], 3)
            self.assertEqual(result["n_task_rows"], 9)
            self.assertEqual(loader.call_count, 2)
            self.assertEqual(epoch.call_count, 2)
            self.assertEqual(conditions["test_condition"].tolist(), list(CONDITIONS))
            self.assertEqual(
                conditions["evaluation_relation"].tolist(),
                ["cross_condition", "matched", "cross_condition"],
            )
            self.assertEqual(
                conditions["reused_matched_evaluation"].tolist(),
                [False, True, False],
            )
            self.assertEqual(len(tasks), 9)
            self.assertEqual(set(tasks["task"]), set(TASKS))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["relation_counts"]["matched"], 1)
            self.assertEqual(manifest["relation_counts"]["cross_condition"], 2)
            self.assertEqual(manifest["completed_condition_cells"], 3)
            self.assertEqual(manifest["completed_task_rows"], 9)
            self.assertEqual(
                len(list(matrix_dir.rglob("classification_report_*.csv"))), 9
            )
            self.assertEqual(
                len(list(matrix_dir.rglob("confusion_matrix_*.csv"))), 9
            )

    def test_success_manifest_is_absent_when_cross_condition_evaluation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch(
                "src.worm_species.evaluation.condition_matrix.make_test_condition_loader",
                return_value=object(),
            ), mock.patch(
                "src.worm_species.evaluation.condition_matrix.run_epoch",
                side_effect=RuntimeError("synthetic evaluation failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic evaluation failure"):
                    evaluate_condition_matrix(
                        cfg=self.config,
                        run_name="failed_run",
                        out_dir=root,
                        model=object(),
                        training_condition=self.training_condition,
                        baseline_metrics=_metrics(0.8),
                        baseline_true=self.true,
                        baseline_pred=self.pred,
                        test_loader_context={"image_size": 16},
                        criteria={task: object() for task in TASKS},
                        target_cols=TASKS,
                        index_to_label_by_task=self.index_to_label,
                        device="cpu",
                        use_amp=False,
                        task_loss_weights={task: 1.0 for task in TASKS},
                        normalize_loss_by_active_tasks=True,
                        hierarchy_cfg={},
                        child_to_parent_matrix=None,
                        use_masked_labels=True,
                    )
            self.assertFalse(
                (root / "condition_matrix_evaluation" / "manifest.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
