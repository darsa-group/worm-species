from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.worm_species.results.derive import derive_results, parse_source


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_matrix(path: Path, labels: list[str], counts: list[list[int]]) -> None:
    lines = ["," + ",".join(labels)]
    lines.extend(
        label + "," + ",".join(str(value) for value in row)
        for label, row in zip(labels, counts)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, macro_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ",precision,recall,f1-score,support\n"
        "class_a,1.0,1.0,1.0,2\n"
        f"macro avg,1.0,1.0,{macro_f1},2\n",
        encoding="utf-8",
    )


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, str | None]]:
    snapshot: dict[str, tuple[int, int, str | None]] = {}
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        stat = path.lstat()
        snapshot[str(path.relative_to(root))] = (
            stat.st_size,
            stat.st_mtime_ns,
            path.readlink().as_posix() if path.is_symlink() else None,
        )
    return snapshot


def _make_multitask_run(root: Path, external: Path) -> Path:
    run = root / "experiment" / "run_000" / "scientific"
    run.mkdir(parents=True)
    _write_json(
        run / "config.json",
        {
            "model": {"name": "resnet18"},
            "data": {
                "target_cols": {
                    "genus": "genus",
                    "species": "species_label",
                    "age": "life_stage",
                }
            },
        },
    )
    _write_json(
        run / "test_metrics.json",
        {
            "genus_macro_f1": 0.9,
            "species_macro_f1": 0.6,
            "age_macro_f1": 0.75,
            "mean_macro_f1": 0.75,
        },
    )
    for task, value in (("genus", 0.9), ("species", 0.6), ("age", 0.75)):
        matrix_target = external / f"confusion_matrix_{task}.csv"
        report_target = external / f"classification_report_{task}.csv"
        _write_matrix(matrix_target, ["a", "b"], [[2, 0], [1, 1]])
        _write_report(report_target, value)
        (run / matrix_target.name).symlink_to(matrix_target)
        (run / report_target.name).symlink_to(report_target)
    (run / "best_model.pt").write_bytes(b"checkpoint must remain opaque")
    return run


class TestResultDerivation(unittest.TestCase):
    def setUp(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

    def test_exact_symlink_matrices_and_stored_mean_are_preserved(self) -> None:
        results = self.root / "outputs_slurm"
        _make_multitask_run(results, self.root / "external")
        before = _tree_snapshot(results)
        cache = self.root / "cache"

        original_open = Path.open

        def guarded_open(path: Path, *args, **kwargs):
            if path.suffix == ".pt":
                raise AssertionError("checkpoint body was opened")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            manifest = derive_results([("slurm", results)], cache, render="none")

        self.assertEqual(_tree_snapshot(results), before)
        self.assertEqual(len(manifest["runs"]), 1)
        summary_path = cache / manifest["runs"][0]["summary"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["run_type"], "multitask")
        self.assertEqual(summary["metrics"]["stored_mean_macro_f1"], 0.75)
        self.assertEqual(summary["metrics"]["effective_mean_macro_f1"], 0.75)
        self.assertEqual(summary["metrics"]["mean_macro_f1_source"], "test_metrics")
        self.assertEqual(
            [item["task"] for item in summary["confusion_matrices"]],
            ["genus", "species", "age"],
        )
        self.assertIsNone(summary["combined_confusion_matrix_image"])

    def test_rendering_is_signature_cached_and_target_change_invalidates(self) -> None:
        results = self.root / "outputs"
        _make_multitask_run(results, self.root / "external")
        cache = self.root / "cache"

        def fake_render(path: Path, matrices, macro_f1) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")

        with mock.patch(
            "src.worm_species.results.derive._render_matrices", side_effect=fake_render
        ) as render:
            first = derive_results([("slurm", results)], cache, render="all")
            summary_path = cache / first["runs"][0]["summary"]
            first_summary_mtime = summary_path.stat().st_mtime_ns
            derive_results([("slurm", results)], cache, render="all")
            second_summary_mtime = summary_path.stat().st_mtime_ns
            target = self.root / "external" / "confusion_matrix_genus.csv"
            _write_matrix(target, ["a", "b"], [[1, 1], [1, 1]])
            third = derive_results([("slurm", results)], cache, render="all")

        self.assertEqual(render.call_count, 2)
        self.assertEqual(second_summary_mtime, first_summary_mtime)
        self.assertEqual(first["rendered_images"], 1)
        self.assertEqual(third["rendered_images"], 1)
        summary = json.loads(
            (cache / third["runs"][0]["summary"]).read_text(encoding="utf-8")
        )
        self.assertTrue(summary["combined_confusion_matrix_image"].endswith(".png"))

    def test_single_task_macro_f1_is_not_presented_as_mean(self) -> None:
        results = self.root / "single_task"
        run = results / "resnet18_species"
        run.mkdir(parents=True)
        _write_json(
            run / "config.json",
            {
                "model": {"name": "resnet18"},
                "data": {"target_col": "species_label"},
            },
        )
        _write_json(run / "run_summary.json", {"run_name": "resnet18_species"})
        _write_matrix(run / "confusion_matrix.csv", ["a", "b"], [[2, 0], [0, 2]])
        _write_report(run / "classification_report.csv", 0.8)

        manifest = derive_results(
            [("single_task", results)], self.root / "cache", render="none"
        )
        summary = json.loads(
            (self.root / "cache" / manifest["runs"][0]["summary"]).read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(summary["run_type"], "single_task")
        self.assertEqual(summary["task"], "species")
        self.assertIsNone(summary["metrics"]["effective_mean_macro_f1"])
        self.assertEqual(summary["metrics"]["effective_task_macro_f1"], 0.8)
        self.assertEqual(
            summary["metrics"]["task_macro_f1_source"], "classification_report"
        )

    def test_missing_and_malformed_artifacts_produce_warnings(self) -> None:
        results = self.root / "outputs"
        run = results / "experiment" / "bad_run"
        run.mkdir(parents=True)
        _write_json(run / "test_metrics.json", {"mean_macro_f1": 0.5})
        (run / "confusion_matrix_genus.csv").write_text(
            ",a,b\na,1,nope\nb,0,1\n", encoding="utf-8"
        )
        (run / "confusion_matrix_species.csv").symlink_to(self.root / "missing.csv")

        manifest = derive_results([("slurm", results)], self.root / "cache", render="none")
        summary = json.loads(
            (self.root / "cache" / manifest["runs"][0]["summary"]).read_text(
                encoding="utf-8"
            )
        )
        codes = {item["code"] for item in summary["warnings"]}

        self.assertIn("malformed_matrix", codes)
        self.assertIn("unavailable_matrix", codes)
        self.assertIn("missing_confusion_matrices", codes)

    def test_multi_root_namespace_and_selected_render(self) -> None:
        slurm = self.root / "slurm"
        single = self.root / "single"
        _make_multitask_run(slurm, self.root / "external_slurm")
        single_run = single / "same_name"
        single_run.mkdir(parents=True)
        _write_json(single_run / "run_summary.json", {"run_name": "scientific"})
        _write_json(single_run / "config.json", {"data": {"target_col": "genus"}})
        _write_matrix(single_run / "confusion_matrix.csv", ["a"], [[1]])
        _write_report(single_run / "classification_report.csv", 1.0)

        cache = self.root / "cache"

        def fake_render(path: Path, matrices, macro_f1) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"png")

        with mock.patch(
            "src.worm_species.results.derive._render_matrices", side_effect=fake_render
        ):
            manifest = derive_results(
                [("slurm", slurm), ("single_task", single)],
                cache,
                render="selected",
                selected_runs=["single_task:scientific"],
            )

        self.assertEqual(len(manifest["runs"]), 2)
        self.assertEqual(manifest["rendered_images"], 1)
        self.assertEqual(manifest["unmatched_selections"], [])
        self.assertEqual(
            {item["source_label"] for item in manifest["runs"]},
            {"slurm", "single_task"},
        )

    def test_cache_validation_source_parsing_and_duplicate_labels(self) -> None:
        results = self.root / "results"
        results.mkdir()
        self.assertEqual(parse_source(f"slurm={results}")[0], "slurm")
        with self.assertRaisesRegex(ValueError, "LABEL=PATH"):
            parse_source(str(results))
        with self.assertRaisesRegex(ValueError, "outside results root"):
            derive_results([("slurm", results)], results / ".cache", render="none")
        with self.assertRaisesRegex(ValueError, "unique"):
            derive_results(
                [("same", results), ("same", results)], self.root / "cache", render="none"
            )
        with self.assertRaisesRegex(ValueError, "requires"):
            derive_results([("slurm", results)], self.root / "cache", render="selected")


if __name__ == "__main__":
    unittest.main()
