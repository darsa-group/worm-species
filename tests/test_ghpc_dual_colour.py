"""Final GHPC migration contracts for dual-cue and colour experiments."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission
from src.worm_species.slurm.rendering import write_artifact_bundle


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs/experiments"
GHPC = ROOT / "configs/clusters/ghpc.yaml"
SNAPSHOT = ROOT / "tests/fixtures/slurm_execution/ghpc_dual_colour.snapshot"
DUAL_ARGS_MANIFEST_SHA256 = (
    "0c9f4238c956d59eb4c46e0a70e7003bb868de994415ccd565bc945af2a53115"
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))[0]


class GhpcDualColourMigrationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data = cls.root / "data"
        (cls.data / "01_Segmented").mkdir(parents=True)
        cls.metadata = cls.data / "01_Segmented/global_metadata.csv"
        cls.metadata.write_text("image\n", encoding="utf-8")
        cls.conda_sh = cls.root / "conda.sh"
        cls.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        cls.nodes = ["gpu01", "gpu02"]
        cls.configs = {}
        cls.plans = {}
        cls.manifests = {}
        cls.bundles = {}
        cls.results = {}
        definitions = (
            ("dual", "ghpc_dual_cue.yaml"),
            ("colour", "ghpc_colour_ablation.yaml"),
        )
        for offset, (name, filename) in enumerate(definitions):
            results = cls.root / f"results-{name}"
            config = load_submission_config(
                EXPERIMENTS / filename,
                GHPC,
                [
                    f"slurm.scratch.nodes={','.join(cls.nodes)}",
                    f"slurm.scratch.root={cls.root / ('scratch-' + name)}",
                    f"slurm.paths.project_root={ROOT}",
                    f"slurm.paths.data_root={cls.data}",
                    f"slurm.paths.metadata_csv={cls.metadata}",
                    f"slurm.paths.results_root={results}",
                    f"slurm.environment.conda_sh={cls.conda_sh}",
                ],
                environment={"HOME": str(cls.root), "USER": "alice"},
                cwd=cls.root,
                submission_stamp="20260102_030405",
                process_id=5000 + offset,
            )
            plan = plan_submission(config)
            bundle = cls.root / f"bundle-{name}"
            manifest = write_artifact_bundle(plan, config, bundle)
            cls.configs[name] = config
            cls.plans[name] = plan
            cls.manifests[name] = manifest
            cls.bundles[name] = bundle
            cls.results[name] = results

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _projection(self) -> dict[str, Any]:
        projection = {}
        for name in ("dual", "colour"):
            config = self.configs[name]
            plan = self.plans[name]
            manifest = self.manifests[name]
            train = next(
                job for job in manifest["jobs"] if job["role"] == "train_array"
            )
            collect = next(job for job in manifest["jobs"] if job["role"] == "collect")
            setup_jobs = [job for job in manifest["jobs"] if job["role"] == "setup"]
            cleanup_jobs = [
                job for job in manifest["jobs"] if job["role"] == "cleanup"
            ]
            scratch = config["slurm"]["scratch"]
            projection[name] = {
                "array": train["array"],
                "array_size": plan.array_size,
                "collector": manifest["metadata"]["collector"],
                "collector_dependencies": collect["dependencies"],
                "condition_count": len(plan.conditions),
                "generated_files": sorted(
                    path.name
                    for path in (self.bundles[name] / "generated_slurm").glob("*.sh")
                ),
                "gate": {
                    "cleanup": [
                        [job["nodelist"], job["dependencies"]]
                        for job in cleanup_jobs
                    ],
                    "nodes": scratch["nodes"],
                    "root_contains_submission_id": (
                        scratch["submission_id"] in scratch["root"]
                    ),
                    "setup_nodes": [job["nodelist"] for job in setup_jobs],
                    "train_dependencies": train["dependencies"],
                    "unique_per_submission": scratch["unique_per_submission"],
                },
                "models": list(plan.models),
                "resources": {
                    "cpus_per_task": train["cpus_per_task"],
                    "memory_mib": train["memory_mib"],
                    "time_limit": train["time_limit"],
                    "max_active": plan.array_max_active,
                },
                "wandb": {
                    "enabled": train["exports"]["WANDB_ENABLED"],
                    "group": train["exports"]["WANDB_RUN_GROUP"],
                    "mode": train["exports"]["WANDB_MODE"],
                    "project": train["exports"]["WANDB_PROJECT"],
                },
            }
        return projection

    def test_migration_snapshot(self) -> None:
        self.assertEqual(
            self._projection(), json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        )

    def test_dual_cue_exact_specs_and_scientific_separation(self) -> None:
        plan = self.plans["dual"]
        self.assertEqual(plan.array_size, 224)
        self.assertEqual(len(plan.models), 2)
        self.assertEqual(len(plan.conditions), 112)
        manifest = b"".join(
            hashlib.sha256(spec.args_text.encode()).hexdigest().encode("ascii")
            + b"  "
            + f"{spec.run_id}.args".encode()
            + b"\n"
            for spec in plan.run_specs
        )
        self.assertEqual(hashlib.sha256(manifest).hexdigest(), DUAL_ARGS_MANIFEST_SHA256)
        original = [
            spec for spec in plan.run_specs if spec.training_transform == "original"
        ]
        transformed = [
            spec for spec in plan.run_specs if spec.training_transform != "original"
        ]
        self.assertEqual(len(original), 2)
        self.assertEqual(len(transformed), 222)
        self.assertTrue(
            all(spec.experiment_type == "matched_and_rgb_stress" for spec in original)
        )
        self.assertTrue(
            all(spec.experiment_type == "matched_condition" for spec in transformed)
        )
        self.assertTrue(
            all(spec.resolved_config["test_cue_suppression"]["enabled"] for spec in original)
        )
        self.assertTrue(
            all(
                not spec.resolved_config["test_cue_suppression"]["enabled"]
                for spec in transformed
            )
        )
        self.assertTrue(
            all(
                not spec.resolved_config["condition_matrix_evaluation"]["enabled"]
                for spec in plan.run_specs
            )
        )

    def test_colour_exact_endpoints_and_no_nested_expansion(self) -> None:
        plan = self.plans["colour"]
        self.assertEqual(plan.array_size, 202)
        self.assertEqual(len(plan.models), 2)
        self.assertEqual(len(plan.conditions), 101)
        self.assertEqual(plan.run_specs[0].run_id, "run_000_colour_100pct")
        self.assertEqual(plan.run_specs[-1].run_id, "run_201_colour_000pct")
        self.assertEqual(plan.run_specs[0].training_condition, "colour_100pct")
        self.assertEqual(plan.run_specs[-1].training_condition, "colour_000pct")
        for spec in plan.run_specs:
            self.assertEqual(spec.experiment_type, "matched_condition")
            self.assertFalse(spec.resolved_config["sweep"]["enabled"])
            self.assertFalse(spec.resolved_config["colour_ablation"]["enabled"])
            self.assertFalse(
                spec.resolved_config["matched_condition_training"]["enabled"]
            )
            self.assertFalse(spec.resolved_config["test_cue_suppression"]["enabled"])
            self.assertFalse(
                spec.resolved_config["condition_matrix_evaluation"]["enabled"]
            )

    def test_shared_gate_one_trainer_output_layout_and_bash_syntax(self) -> None:
        filenames = {
            "dual": "node_local_cue_array_job.sh",
            "colour": "node_local_colour_array_job.sh",
        }
        for name, array_name in filenames.items():
            generated = self.bundles[name] / "generated_slurm"
            array = (generated / array_name).read_text(encoding="utf-8")
            self.assertEqual(array.count("python -m worm_species.training"), 1)
            for fragment in (
                'RUN_SCRATCH_OUT="${RUN_OUTPUT_ROOT}/${RUN_ID}"',
                'RUN_BACK_OUT="${RESULTS_ROOT}/${RUN_ID}"',
                "run_overrides.args",
                "run_status.txt",
                "IMAGE_CACHE.lock",
                "IMAGE_CACHE_READY",
                '"$PROFILE_DIR/gpu_usage.csv"',
                '"$PROFILE_DIR/cpu_usage.txt"',
                "if ((copy_status != 0 && status == 0)); then",
                "status=90",
            ):
                self.assertIn(fragment, array)
            for script in generated.glob("*.sh"):
                subprocess.run(["bash", "-n", str(script)], check=True)
            self.assertEqual(
                self.plans[name].expected_internal_training_runs_per_task,
                1,
            )

    def test_standard_collector_also_uses_the_canonical_command(self) -> None:
        config = copy.deepcopy(self.configs["dual"])
        config["slurm"]["collection"]["kind"] = "standard"
        plan = plan_submission(config)
        bundle = self.root / "standard-collector"
        write_artifact_bundle(plan, config, bundle)
        source = bundle.joinpath(
            "generated_slurm/result_collector_job.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("python -m worm_species.slurm collect", source)
        self.assertIn("--kind standard", source)
        self.assertNotIn("python -c", source)

    def test_dual_collector_exact_outputs_and_failure_schema(self) -> None:
        root = self.results["dual"]
        cue = root / "run_000/scientific/cue_suppression"
        cue.mkdir(parents=True)
        _write_json(
            cue.parent / "run_summary.json",
            {
                "run_name": "dual-complete",
                "model": "convnext_base",
                "train_condition": "original",
                "train_feature": "baseline",
                "train_transform": "original",
                "train_strength": 0.0,
                "test_genus_macro_f1": 0.8,
                "best_epoch": 3,
                "best_val_score": 0.7,
                "selection_metric": "mean_macro_f1",
                "out_dir": "run_000/scientific",
            },
        )
        (root / "run_000/run_status.txt").write_text("0\n", encoding="utf-8")
        (root / "run_001").mkdir()
        (root / "run_001/run_status.txt").write_text("7\n", encoding="utf-8")
        (cue / "macro_f1_ratios.csv").write_text(
            "model,task,condition,feature,transform,strength,macro_f1,"
            "original_macro_f1,ratio_to_original,relative_drop\n"
            "convnext_base,genus,original,baseline,original,0,0.8,0.8,1,0\n",
            encoding="utf-8",
        )
        (cue / "test_condition_metrics.csv").write_text(
            "model,task,condition,macro_f1\nconvnext_base,genus,original,0.8\n",
            encoding="utf-8",
        )
        (cue / "transform_summary.csv").write_text(
            "model,condition,macro_f1\nconvnext_base,original,0.8\n",
            encoding="utf-8",
        )
        script = self.bundles["dual"] / "generated_slurm/result_collector_job.sh"
        completed = subprocess.run(
            ["bash", str(script)],
            check=False,
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            _csv_rows(root / "failed_runs.csv"),
            [{"array_run": "run_001", "status": "7"}],
        )
        self.assertEqual(
            _csv_header(root / "matched_condition_results.csv"),
            [
                "run_name",
                "model",
                "train_condition",
                "train_feature",
                "train_transform",
                "train_strength",
                "test_genus_macro_f1",
                "best_epoch",
                "best_val_score",
                "selection_metric",
                "out_dir",
                "summary_path",
            ],
        )
        self.assertEqual(
            _csv_header(root / "matched_condition_macro_f1_long.csv"),
            [
                "run_name",
                "model",
                "task",
                "train_condition",
                "train_feature",
                "train_transform",
                "train_strength",
                "matched_test_macro_f1",
                "best_epoch",
                "best_val_score",
                "selection_metric",
                "out_dir",
                "summary_path",
            ],
        )
        self.assertEqual(_csv_header(root / "failed_runs.csv"), ["array_run", "status"])
        self.assertEqual(len(_csv_rows(root / "matched_condition_results.csv")), 1)
        self.assertEqual(
            len(_csv_rows(root / "matched_condition_macro_f1_long.csv")), 1
        )
        self.assertEqual(len(_csv_rows(root / "matched_vs_rgb_stress_test.csv")), 1)
        for name in (
            "condition_matrix_evaluations.csv",
            "condition_matrix_task_metrics.csv",
            "condition_matrix_collection_summary.json",
        ):
            self.assertFalse((root / name).exists())
        self.assertEqual(
            {path.name for path in root.glob("*.csv")},
            {
                "failed_runs.csv",
                "matched_condition_macro_f1_long.csv",
                "matched_condition_results.csv",
                "matched_vs_rgb_stress_test.csv",
                "rgb_model_cue_suppression_macro_f1_ratios.csv",
                "rgb_model_cue_suppression_test_metrics.csv",
                "rgb_model_cue_suppression_transform_summary.csv",
            },
        )

    def test_colour_collector_exact_outputs_order_and_failure_schema(self) -> None:
        root = self.results["colour"]
        for run_id, percent, status in (
            ("run_000_colour_100pct", 100, "0"),
            ("run_201_colour_000pct", 0, "5"),
        ):
            wrapper = root / run_id
            _write_json(
                wrapper / "scientific/run_summary.json",
                {
                    "run_name": run_id,
                    "model": "convnext_base",
                    "colour_percent": percent,
                    "test_mean_macro_f1": percent / 100.0,
                },
            )
            (wrapper / "run_status.txt").write_text(status + "\n", encoding="utf-8")
        script = self.bundles["colour"] / "generated_slurm/result_collector_job.sh"
        completed = subprocess.run(
            ["bash", str(script)], check=False, text=True, capture_output=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rows = _csv_rows(root / "colour_ablation_results.csv")
        self.assertEqual([row["colour_percent"] for row in rows], ["100", "0"])
        self.assertEqual(
            _csv_header(root / "colour_ablation_results.csv"),
            [
                "run_name",
                "model",
                "colour_percent",
                "test_mean_macro_f1",
                "summary_path",
            ],
        )
        self.assertEqual(
            _csv_rows(root / "failed_runs.csv"),
            [{"run_name": "run_201_colour_000pct", "status": "5"}],
        )
        self.assertEqual(_csv_header(root / "failed_runs.csv"), ["run_name", "status"])
        self.assertEqual(
            {path.name for path in root.glob("*.csv")},
            {"colour_ablation_results.csv", "failed_runs.csv"},
        )


if __name__ == "__main__":
    unittest.main()
