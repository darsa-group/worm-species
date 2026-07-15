from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dashboard.data_loader import discover_results
from dashboard.schemas import DiscoveryResult
from src.worm_species.results import (
    DiscoveryLimits,
    FilesystemRunState,
    discover_experiment,
    discover_results_root,
    load_json,
)
from src.worm_species.results.readers import load_text


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_completed_experiment(results_root: Path) -> Path:
    experiment = results_root / "experiment_a"
    experiment.mkdir(parents=True)
    (experiment / "sweep_plan.tsv").write_text(
        "run_index\tarray_name\tmodel\n"
        "0\trun_000\tresnet18\n"
        "1\trun_001\tvit_b_16\n",
        encoding="utf-8",
    )
    wrapper = experiment / "run_000"
    run = wrapper / "scientific_run"
    run.mkdir(parents=True)
    (wrapper / "run_status.txt").write_text("0\n", encoding="utf-8")
    (wrapper / "run_overrides.args").write_text(
        "model.name=resnet18\n", encoding="utf-8"
    )
    _write_json(
        run / "config.json",
        {
            "model": {"name": "resnet18"},
            "data": {
                "target_cols": {
                    "genus": "genus_column",
                    "species": "species_column",
                    "age": "stage_column",
                }
            },
        },
    )
    _write_json(
        run / "run_summary.json",
        {"run_name": "stable_name", "model": "resnet18", "best_epoch": 2},
    )
    _write_json(
        run / "test_metrics.json",
        {"genus_n": 2, "genus_macro_f1": 0.75, "mean_macro_f1": 0.75},
    )
    (run / "history.csv").write_text(
        "epoch,train_loss\n1,1.0\n", encoding="utf-8"
    )
    (run / "best_model.pt").write_bytes(b"checkpoint sentinel")
    return experiment


class TestResultDiscovery(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

    def test_explicit_root_scopes_plan_and_checkpoint_metadata(self) -> None:
        experiment = _make_completed_experiment(self.root / "results")
        original_open = Path.open

        def guarded_open(path: Path, *args, **kwargs):
            if path.suffix == ".pt":
                raise AssertionError("checkpoint body was opened")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            tree = discover_results_root(self.root / "results", now=10**12)
            single = discover_experiment(experiment, now=10**12)

        self.assertEqual(len(tree.experiments), 1)
        self.assertEqual(tree.experiments[0].name, "experiment_a")
        self.assertEqual(len(tree.runs), 1)
        self.assertEqual(single.experiment.name, "experiment_a")
        self.assertEqual(single.experiment.relative_path, ".")
        self.assertEqual(len(single.runs), 1)
        self.assertEqual(single.runs[0].relative_path, "run_000/scientific_run")
        self.assertEqual(single.runs[0].tasks, ["genus", "species", "age"])
        self.assertIs(single.runs[0].status, FilesystemRunState.COMPLETED)
        self.assertEqual(single.runs[0].raw_exit_status, "0")
        self.assertTrue(single.runs[0].terminal_metrics_present)
        self.assertEqual(single.experiment.expected_run_count, 2)
        self.assertEqual(
            [entry.array_name for entry in single.experiment.plan_entries],
            ["run_000", "run_001"],
        )
        self.assertEqual(single.unmaterialized_array_runs, ["run_001"])
        checkpoint = next(
            artifact
            for artifact in single.runs[0].artifacts
            if artifact.kind == "checkpoint"
        )
        self.assertEqual(checkpoint.size, len(b"checkpoint sentinel"))

    def test_filesystem_state_is_deterministic_and_wrappers_not_duplicated(
        self,
    ) -> None:
        experiment = _make_completed_experiment(self.root / "results")
        partial = experiment / "run_001"
        partial.mkdir()
        (partial / "run_overrides.args").write_text(
            "model.name=vit_b_16\n", encoding="utf-8"
        )
        updated_at = (partial / "run_overrides.args").stat().st_mtime
        limits = DiscoveryLimits(active_window_seconds=60)

        recent = discover_experiment(experiment, limits=limits, now=updated_at + 30)
        old = discover_experiment(experiment, limits=limits, now=updated_at + 120)

        self.assertEqual(len(recent.runs), 2)
        self.assertEqual(
            len([run for run in recent.runs if run.array_run == "run_000"]), 1
        )
        self.assertIs(
            next(
                run for run in recent.runs if run.array_run == "run_001"
            ).status,
            FilesystemRunState.POSSIBLY_ACTIVE,
        )
        self.assertIs(
            next(run for run in old.runs if run.array_run == "run_001").status,
            FilesystemRunState.INCOMPLETE,
        )

    def test_failed_wrapper_and_broken_file_symlink_are_tolerated(self) -> None:
        experiment = self.root / "experiment"
        failed = experiment / "run_002"
        failed.mkdir(parents=True)
        (failed / "run_overrides.args").write_text(
            "model.name=resnet50\n", encoding="utf-8"
        )
        (failed / "run_status.txt").write_text("7\n", encoding="utf-8")
        broken_run = experiment / "run_003" / "broken"
        broken_run.mkdir(parents=True)
        _write_json(broken_run / "config.json", {"model": {"name": "resnet18"}})
        (broken_run / "test_metrics.json").symlink_to(self.root / "missing.json")

        snapshot = discover_experiment(experiment, now=10**12)

        self.assertIs(
            next(
                run for run in snapshot.runs if run.array_run == "run_002"
            ).status,
            FilesystemRunState.FAILED,
        )
        broken = next(run for run in snapshot.runs if run.array_run == "run_003")
        self.assertIs(broken.status, FilesystemRunState.INCOMPLETE)
        self.assertTrue(
            any(warning.code == "unavailable_artifact" for warning in broken.warnings)
        )
        artifact = next(
            item for item in broken.artifacts if item.kind == "test_metrics.json"
        )
        self.assertTrue(artifact.is_symlink)
        self.assertFalse(artifact.available)

    def test_directory_symlinks_are_not_followed(self) -> None:
        experiment = self.root / "experiment"
        experiment.mkdir()
        external = self.root / "external" / "run_999" / "escaped"
        external.mkdir(parents=True)
        _write_json(external / "test_metrics.json", {"mean_macro_f1": 1.0})
        (experiment / "linked_results").symlink_to(
            external.parent, target_is_directory=True
        )

        snapshot = discover_experiment(experiment)

        self.assertEqual(snapshot.runs, [])

    def test_dashboard_adapter_and_empty_scopes(self) -> None:
        empty = self.root / "empty"
        empty.mkdir()

        dashboard_result = discover_results(empty)
        single = discover_experiment(empty)

        self.assertIsInstance(dashboard_result, DiscoveryResult)
        self.assertEqual(dashboard_result.experiments, [])
        self.assertEqual(single.experiment.path, str(empty.absolute()))
        self.assertEqual(single.runs, [])

    def test_checkpoint_reader_is_hard_refused(self) -> None:
        checkpoint = self.root / "model.pt"
        checkpoint.write_text('{"not": "metadata"}', encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "checkpoint contents"):
            load_json(checkpoint)

    def test_discovery_does_not_write_and_read_limits_are_enforced(self) -> None:
        experiment = _make_completed_experiment(self.root / "results")
        before = {
            path.relative_to(experiment): (path.lstat().st_size, path.lstat().st_mtime_ns)
            for path in experiment.rglob("*")
            if path.is_file()
        }

        discover_experiment(experiment)

        after = {
            path.relative_to(experiment): (path.lstat().st_size, path.lstat().st_mtime_ns)
            for path in experiment.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)
        with self.assertRaisesRegex(ValueError, "read limit"):
            load_text(experiment / "sweep_plan.tsv", max_bytes=1)

    def test_canonical_and_legacy_condition_facets_are_additive(self) -> None:
        experiment = self.root / "results" / "conditions"
        canonical = experiment / "canonical"
        legacy = experiment / "legacy"
        canonical.mkdir(parents=True)
        legacy.mkdir(parents=True)
        _write_json(
            canonical / "config.json",
            {
                "seed": 1,
                "training": {"seed": 9, "epochs": 2},
                "model": {"name": "resnet18", "pretrained": True},
                "preprocessing": {
                    "image_size": 384,
                    "normalisation": {
                        "enabled": True,
                        "mean": [0.1, 0.2, 0.3],
                        "std": [0.4, 0.5, 0.6],
                    },
                },
                "augmentation": {
                    "enabled": True,
                    "horizontal_flip": {"enabled": True, "probability": 0.25},
                    "vertical_flip": {"enabled": False, "probability": 0.0},
                    "rotation": {"enabled": True, "degrees": 90},
                },
                "experiment": {"type": "matched_condition"},
                "input_condition": {
                    "enabled": True,
                    "name": "gaussian_sigma_2",
                    "transform": "gaussian_blur",
                    "parameters": {"sigma": 2.0},
                },
            },
        )
        _write_json(canonical / "test_metrics.json", {"mean_macro_f1": 0.5})
        _write_json(
            legacy / "config.json",
            {
                "model": {"name": "resnet18"},
                "data": {"image_size": 384},
                "input_condition": {
                    "enabled": True,
                    "condition": "gaussian_sigma_2",
                    "transform": "gaussian_blur",
                    "sigma": 2.0,
                },
            },
        )
        _write_json(legacy / "test_metrics.json", {"mean_macro_f1": 0.5})

        snapshot = discover_experiment(experiment, now=10**12)
        by_name = {Path(run.path).name: run for run in snapshot.runs}
        modern = by_name["canonical"]
        old = by_name["legacy"]

        self.assertEqual(modern.train_condition, "gaussian_sigma_2")
        self.assertEqual(old.train_condition, "gaussian_sigma_2")
        self.assertEqual(modern.train_condition_parameters, {"sigma": 2.0})
        self.assertEqual(old.train_condition_parameters, {"sigma": 2.0})
        self.assertEqual(modern.image_size, 384)
        self.assertEqual(old.image_size, 384)
        self.assertEqual(modern.experiment_type, "matched_condition")
        self.assertEqual(modern.hyperparameters["seed"], 9)
        self.assertEqual(modern.hyperparameters["image_size"], 384)
        self.assertEqual(
            modern.hyperparameters["condition_parameter.sigma"], 2.0
        )
        self.assertEqual(
            modern.hyperparameters["horizontal_flip_probability"], 0.25
        )
        metric = next(
            item for item in modern.metrics if item.metric == "mean_macro_f1"
        )
        self.assertEqual(metric.condition, "gaussian_sigma_2")
        self.assertEqual(metric.condition_relation, "matched")
        self.assertEqual(
            metric.canonical_key,
            "test/gaussian_sigma_2/mean_macro_f1",
        )


if __name__ == "__main__":
    unittest.main()
