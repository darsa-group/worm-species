from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.worm_species.experiments.result_collection import collect_results
from src.worm_species.slurm.collection import (
    CollectionError,
    DUAL_OUTPUT_NAMES,
    collect_existing_results,
)
from src.worm_species.slurm.status import build_status_report
from src.worm_species.slurm.submission import CommandResult


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_status_experiment(root: Path) -> Path:
    experiment = root / "experiment"
    experiment.mkdir(parents=True)
    (experiment / "sweep_plan.tsv").write_text(
        "run_index\trun_name\tmodel\n"
        "0\trun_000\tresnet18\n"
        "1\trun_001\tvit_b_16\n",
        encoding="utf-8",
    )
    wrapper = experiment / "run_000"
    run = wrapper / "scientific_run"
    run.mkdir(parents=True)
    (wrapper / "run_status.txt").write_text("0\n", encoding="utf-8")
    _write_json(run / "config.json", {"model": {"name": "resnet18"}})
    _write_json(run / "test_metrics.json", {"mean_macro_f1": 0.75})
    return experiment


def _make_collection_tree(root: Path) -> None:
    run = root / "run_000" / "scientific_run"
    cue = run / "cue_suppression"
    cue.mkdir(parents=True)
    _write_json(
        run / "run_summary.json",
        {
            "run_name": "stable_run",
            "model": "resnet18",
            "train_condition": "original",
            "train_feature": "baseline",
            "train_transform": "original",
            "train_strength": 0.0,
            "test_genus_macro_f1": 0.8,
            "best_epoch": 2,
            "best_val_score": 0.7,
            "selection_metric": "mean_macro_f1",
            "out_dir": "outputs/stable_run",
        },
    )
    (root / "run_000" / "run_status.txt").write_text("0\n", encoding="utf-8")
    (cue / "macro_f1_ratios.csv").write_text(
        "model,task,condition,feature,transform,strength,macro_f1,"
        "original_macro_f1,ratio_to_original,relative_drop\n"
        "resnet18,genus,original,baseline,original,0,0.8,0.8,1.0,0.0\n",
        encoding="utf-8",
    )
    (cue / "test_condition_metrics.csv").write_text(
        "model,task,condition,macro_f1\nresnet18,genus,original,0.8\n",
        encoding="utf-8",
    )
    (cue / "transform_summary.csv").write_text(
        "model,condition,macro_f1\nresnet18,original,0.8\n",
        encoding="utf-8",
    )


class FakeSchedulerRunner:
    def __init__(self, responses: dict[str, CommandResult | Exception]):
        self.responses = responses
        self.calls: list[list[str]] = []

    def run(self, argv):
        call = list(argv)
        self.calls.append(call)
        response = self.responses[call[0]]
        if isinstance(response, Exception):
            raise response
        return response


class SlurmStatusContracts(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.experiment = _make_status_experiment(self.root)

    def test_no_ids_skips_scheduler_and_status_is_read_only(self) -> None:
        runner = FakeSchedulerRunner({})
        before = {
            path.relative_to(self.experiment): (
                path.lstat().st_size,
                path.lstat().st_mtime_ns,
            )
            for path in self.experiment.rglob("*")
            if path.is_file()
        }
        with mock.patch(
            "src.worm_species.slurm.status.discover_experiment",
            wraps=__import__(
                "src.worm_species.slurm.status", fromlist=["discover_experiment"]
            ).discover_experiment,
        ) as discover:
            report = build_status_report(self.experiment, runner=runner)
        after = {
            path.relative_to(self.experiment): (
                path.lstat().st_size,
                path.lstat().st_mtime_ns,
            )
            for path in self.experiment.rglob("*")
            if path.is_file()
        }
        self.assertEqual(discover.call_count, 1)
        self.assertEqual(runner.calls, [])
        self.assertEqual(before, after)
        self.assertEqual(report.expected_run_count, 2)
        self.assertEqual(report.materialized_run_count, 1)
        self.assertEqual(report.unmaterialized_run_count, 1)
        self.assertEqual(report.filesystem_counts, {"completed": 1})

    def test_scheduler_array_states_and_child_steps_are_combined(self) -> None:
        submission = self.root / "submission"
        submission.mkdir()
        _write_json(
            submission / "submission_receipt.json",
            {"submitted": {"train_array": "900", "collect": "901"}},
        )
        runner = FakeSchedulerRunner(
            {
                "squeue": CommandResult(
                    0,
                    "900_0|train|RUNNING|None\n900_1|train|PENDING|Resources\n",
                ),
                "sacct": CommandResult(
                    0,
                    "900_0|train|COMPLETED|0:0|None\n"
                    "900_0.batch|batch|FAILED|1:0|None\n"
                    "901|collect|COMPLETED+|0:0|None\n",
                ),
            }
        )
        report = build_status_report(
            self.experiment, submission_root=submission, runner=runner
        )
        self.assertTrue(report.scheduler_available)
        self.assertEqual([call[0] for call in runner.calls], ["squeue", "sacct"])
        run = report.runs[0]
        self.assertEqual(run.scheduler_state, "running")
        self.assertEqual(report.scheduler_counts["running"], 1)
        self.assertEqual(report.scheduler_counts["pending"], 1)
        self.assertEqual(report.scheduler_counts["completed"], 1)
        self.assertFalse(any(job.job_name == "batch" for job in report.jobs))

    def test_legacy_ids_and_missing_slurm_are_tolerated(self) -> None:
        (self.experiment / "submitted_jobs.tsv").write_text(
            "name\tjob_id\ntrain_array\t700\ncollect\tnot-a-job\n",
            encoding="utf-8",
        )
        runner = FakeSchedulerRunner(
            {
                "squeue": FileNotFoundError("squeue missing"),
                "sacct": FileNotFoundError("sacct missing"),
            }
        )
        report = build_status_report(self.experiment, runner=runner)
        self.assertFalse(report.scheduler_available)
        self.assertEqual(report.submitted_jobs, {"train_array": "700"})
        self.assertTrue(
            any(warning.code == "scheduler_unavailable" for warning in report.warnings)
        )
        self.assertTrue(
            any(warning.code == "invalid_job_id" for warning in report.warnings)
        )


class SlurmCollectionContracts(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_adapter_is_byte_identical_to_existing_collector(self) -> None:
        direct = self.root / "direct"
        adapted = self.root / "adapted"
        _make_collection_tree(direct)
        shutil.copytree(direct, adapted)
        collect_results(direct)
        with mock.patch(
            "src.worm_species.slurm.collection.collect_dual_results",
            wraps=collect_results,
        ) as delegated:
            report = collect_existing_results(adapted, kind="dual-cue")
        delegated.assert_called_once_with(adapted.absolute())
        direct_outputs = {
            name: (direct / name).read_bytes()
            for name in DUAL_OUTPUT_NAMES
            if (direct / name).is_file()
        }
        adapted_outputs = {
            name: (adapted / name).read_bytes()
            for name in DUAL_OUTPUT_NAMES
            if (adapted / name).is_file()
        }
        self.assertEqual(adapted_outputs, direct_outputs)
        self.assertEqual(
            {Path(path).name for path in report.output_paths}, set(adapted_outputs)
        )

    def test_auto_marker_delegates_but_unsupported_modes_write_nothing(self) -> None:
        dual = self.root / "dual"
        _make_collection_tree(dual)
        _write_json(dual / "dual_cue_experiment_plan.json", {"n_total_runs": 1})
        report = collect_existing_results(dual)
        self.assertEqual(report.kind, "dual-cue")

        unsupported = self.root / "unsupported"
        unsupported.mkdir()
        before = list(unsupported.iterdir())
        with self.assertRaisesRegex(CollectionError, "unsupported"):
            collect_existing_results(unsupported, kind="colour-ablation")
        self.assertEqual(list(unsupported.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
