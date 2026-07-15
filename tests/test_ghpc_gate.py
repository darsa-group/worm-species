"""Operational gate contracts shared by every future GHPC launcher."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.worm_species.slurm.config import SlurmConfigError
from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission
from src.worm_species.slurm.rendering import render_template
from src.worm_species.slurm.rendering import shell_quote
from src.worm_species.slurm.rendering import write_artifact_bundle


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests/fixtures/slurm_execution/ghpc_gate.snapshot"


def _executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


class GhpcOperationalGateContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.project = cls.root / "project"
        cls.data = cls.root / "data"
        cls.results = cls.root / "results"
        cls.project.mkdir()
        (cls.project / "src").mkdir()
        (cls.project / "keep.py").write_text("pass\n", encoding="utf-8")
        (cls.project / "outputs").mkdir()
        (cls.project / "outputs_slurm").mkdir()
        (cls.project / "__pycache__").mkdir()
        segmented = cls.data / "01_Segmented"
        segmented.mkdir(parents=True)
        cls.metadata = segmented / "global_metadata.csv"
        cls.metadata.write_text("image\n", encoding="utf-8")
        (segmented / "one_seg.jpg").write_bytes(b"jpg")
        (segmented / "ignored.raw").write_bytes(b"raw")
        cls.conda_sh = cls.root / "conda.sh"
        cls.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")

        cls.config = load_submission_config(
            ROOT / "configs/experiments/standard.yaml",
            ROOT / "configs/clusters/ghpc.yaml",
            [
                "sweep.enabled=false",
                "slurm.scratch.nodes=gpu01,gpu02",
                f"slurm.scratch.root={cls.root / 'scratch'}",
                f"slurm.paths.project_root={cls.project}",
                f"slurm.paths.data_root={cls.data}",
                f"slurm.paths.metadata_csv={cls.metadata}",
                f"slurm.paths.results_root={cls.results}",
                f"slurm.environment.conda_sh={cls.conda_sh}",
            ],
            environment={"HOME": str(cls.root), "USER": "alice"},
            cwd=cls.root,
            submission_stamp="20260102_030405",
            process_id=4321,
        )
        cls.plan = plan_submission(cls.config)
        cls.bundle = cls.root / "bundle"
        cls.manifest = write_artifact_bundle(cls.plan, cls.config, cls.bundle)
        cls.generated = cls.bundle / "generated_slurm"
        cls.setup_path = cls.generated / "node_local_setup_job.sh"
        cls.array_path = cls.generated / "node_local_training_array_job.sh"
        cls.cleanup_path = cls.generated / "node_local_cleanup_job.sh"
        cls.setup = cls.setup_path.read_text(encoding="utf-8")
        cls.array = cls.array_path.read_text(encoding="utf-8")
        cls.cleanup = cls.cleanup_path.read_text(encoding="utf-8")
        cls.scratch = Path(cls.config["slurm"]["scratch"]["root"])
        cls.submission_id = cls.config["slurm"]["scratch"]["submission_id"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _projection(self) -> dict:
        jobs = {job["name"]: job for job in self.manifest["jobs"]}
        setup_jobs = [job for job in self.manifest["jobs"] if job["role"] == "setup"]
        cleanup_jobs = [
            job for job in self.manifest["jobs"] if job["role"] == "cleanup"
        ]
        return {
            "array": {
                "cache_lock": "IMAGE_CACHE.lock" in self.array,
                "cache_parent": 'cache.root_dir_cache="$CACHE_PARENT"' in self.array,
                "image_ready_marker": "IMAGE_CACHE_READY" in self.array,
                "monitoring": "gpu_usage.csv" in self.array
                and "cpu_usage.txt" in self.array,
                "one_trainer": self.array.count("python -m worm_species.training") == 1,
                "run_overrides": "run_overrides.args" in self.array,
                "run_status": "run_status.txt" in self.array,
                "setup_ready_marker": str(self.scratch / "READY") in self.array,
                "split_root": 'split.predefined_split_dir="$PROJECT_ROOT"' in self.array,
            },
            "cleanup": {
                "afterany": all(
                    job["dependencies"]
                    == [{"job": "train_array", "kind": "afterany"}]
                    for job in cleanup_jobs
                ),
                "contains_submission_guard": (
                    '"$SCRATCH_ROOT" == *"$SUBMISSION_ID"*' in self.cleanup
                ),
                "per_node": len(cleanup_jobs),
            },
            "collector": self.manifest["metadata"]["collector"],
            "setup": {
                "data_filters": "--include=global_metadata.csv" in self.setup
                and "--include=*_seg.jpg" in self.setup,
                "nodes": [job["nodelist"] for job in setup_jobs],
                "project_filters": all(
                    item in self.setup
                    for item in (
                        "--exclude .git",
                        "--exclude __pycache__",
                        "--exclude .ipynb_checkpoints",
                        "--exclude outputs",
                        "--exclude outputs_slurm",
                    )
                ),
                "ready_marker": 'touch "${SCRATCH_ROOT}/${READY_MARKER}"' in self.setup,
            },
            "submission": {
                "array_depends_afterok": jobs["train_array"]["dependencies"]
                == [
                    {"job": "setup:gpu01", "kind": "afterok"},
                    {"job": "setup:gpu02", "kind": "afterok"},
                ],
                "scratch_contains_id": self.submission_id in str(self.scratch),
                "submission_id": self.submission_id,
            },
        }

    def test_gate_snapshot_and_shell_syntax(self) -> None:
        self.assertEqual(
            self._projection(), json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        )
        for script in self.generated.glob("*.sh"):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_unique_root_is_deterministic_per_plan_and_changes_with_pid(self) -> None:
        same = load_submission_config(
            ROOT / "configs/experiments/standard.yaml",
            ROOT / "configs/clusters/ghpc.yaml",
            ["slurm.scratch.nodes=gpu01"],
            environment={"HOME": str(self.root), "USER": "alice"},
            submission_stamp="20260102_030405",
            process_id=4321,
        )
        different = load_submission_config(
            ROOT / "configs/experiments/standard.yaml",
            ROOT / "configs/clusters/ghpc.yaml",
            ["slurm.scratch.nodes=gpu01"],
            environment={"HOME": str(self.root), "USER": "alice"},
            submission_stamp="20260102_030405",
            process_id=9876,
        )
        self.assertTrue(same["slurm"]["scratch"]["root"].endswith(self.submission_id))
        self.assertNotEqual(
            same["slurm"]["scratch"]["root"],
            different["slurm"]["scratch"]["root"],
        )

    def test_static_or_node_less_scratch_fails_before_rendering(self) -> None:
        with self.assertRaisesRegex(SlurmConfigError, "scratch.nodes"):
            load_submission_config(
                ROOT / "configs/experiments/standard.yaml",
                ROOT / "configs/clusters/ghpc.yaml",
                environment={"HOME": str(self.root), "USER": "alice"},
                submission_stamp="20260102_030405",
                process_id=4321,
            )
        with self.assertRaisesRegex(SlurmConfigError, "unique_per_submission"):
            load_submission_config(
                ROOT / "configs/experiments/standard.yaml",
                ROOT / "configs/clusters/ghpc.yaml",
                [
                    "slurm.scratch.nodes=gpu01",
                    "slurm.scratch.unique_per_submission=false",
                ],
                environment={"HOME": str(self.root), "USER": "alice"},
                submission_stamp="20260102_030405",
                process_id=4321,
            )

    def test_setup_filters_and_ready_marker_in_temporary_tree(self) -> None:
        bin_dir = self.root / "setup-bin"
        bin_dir.mkdir()
        calls = self.root / "setup-rsync.calls"
        calls.write_text("", encoding="utf-8")
        _executable(
            bin_dir / "rsync",
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"${RSYNC_CALLS:?}\"\n",
        )
        _executable(bin_dir / "lscpu", "#!/usr/bin/env bash\necho mocked-cpu\n")
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{bin_dir}:{environment['PATH']}",
                "RSYNC_CALLS": str(calls),
            }
        )
        completed = subprocess.run(
            ["bash", str(self.setup_path)],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((self.scratch / "READY").is_file())
        invocations = calls.read_text(encoding="utf-8")
        self.assertIn("--exclude outputs_slurm", invocations)
        self.assertIn("--include=global_metadata.csv", invocations)
        self.assertIn("--include=*_seg.jpg", invocations)

    def _run_array(self, trainer_status: int, copyback_status: int):
        self.scratch.mkdir(parents=True, exist_ok=True)
        (self.scratch / "project").mkdir(exist_ok=True)
        (self.scratch / "data").mkdir(exist_ok=True)
        (self.scratch / "READY").touch()
        (self.scratch / "IMAGE_CACHE_READY").touch()
        runtime = Path(tempfile.mkdtemp(prefix="array-", dir=self.root))
        bin_dir = runtime / "bin"
        bin_dir.mkdir()
        trainer_calls = runtime / "trainer.calls"
        trainer_calls.write_text("", encoding="utf-8")
        _executable(bin_dir / "python", "#!/usr/bin/env bash\nexit 0\n")
        _executable(
            bin_dir / "srun",
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"${TRAINER_CALLS:?}\"\nexit \"${TRAINER_STATUS:-0}\"\n",
        )
        _executable(
            bin_dir / "nvidia-smi",
            "#!/usr/bin/env bash\necho mocked-gpu\n",
        )
        _executable(
            bin_dir / "rsync",
            "#!/usr/bin/env bash\nexit \"${COPYBACK_STATUS:-0}\"\n",
        )
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{bin_dir}:{environment['PATH']}",
                "SLURM_ARRAY_TASK_ID": "0",
                "SLURM_JOB_ID": "123",
                "TRAINER_CALLS": str(trainer_calls),
                "TRAINER_STATUS": str(trainer_status),
                "COPYBACK_STATUS": str(copyback_status),
                "USER": "alice",
            }
        )
        completed = subprocess.run(
            ["bash", str(self.array_path)],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        status_path = self.scratch / "outputs/run_000/run_status.txt"
        return completed, status_path, trainer_calls

    def test_array_profiling_overrides_and_exit_table(self) -> None:
        success, status, calls = self._run_array(0, 0)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(status.read_text(encoding="utf-8"), "0\n")
        invocation = calls.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(invocation), 1)
        self.assertIn(f"cache.root_dir_cache={self.scratch}", invocation[0])
        self.assertIn(
            f"split.predefined_split_dir={self.scratch / 'project'}", invocation[0]
        )
        run_root = self.scratch / "outputs/run_000"
        self.assertTrue((run_root / "run_overrides.args").is_file())
        self.assertTrue((run_root / "profiling/gpu_usage.csv").is_file())
        self.assertTrue((run_root / "profiling/cpu_usage.txt").is_file())
        self.assertTrue((self.scratch / "IMAGE_CACHE.lock").is_file())

        trained, status, _ = self._run_array(7, 0)
        self.assertEqual(trained.returncode, 7)
        self.assertEqual(status.read_text(encoding="utf-8"), "7\n")
        copy_failed, status, _ = self._run_array(0, 23)
        self.assertEqual(copy_failed.returncode, 90)
        self.assertEqual(status.read_text(encoding="utf-8"), "0\n")
        both_failed, status, _ = self._run_array(7, 23)
        self.assertEqual(both_failed.returncode, 7)
        self.assertEqual(status.read_text(encoding="utf-8"), "7\n")

    def test_cleanup_refuses_static_root_and_removes_only_unique_root(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        unsafe_script = self.root / "unsafe-cleanup.sh"
        unsafe_script.write_text(
            render_template(
                "node_local_cleanup_job.sh.tmpl",
                {
                    "SCRATCH_ROOT": shell_quote(outside),
                    "SUBMISSION_ID": shell_quote(self.submission_id),
                },
            ),
            encoding="utf-8",
        )
        refused = subprocess.run(["bash", str(unsafe_script)], check=False)
        self.assertEqual(refused.returncode, 2)
        self.assertTrue(outside.is_dir())

        neighbor = self.scratch.parent / "neighbor-must-remain"
        neighbor.mkdir(exist_ok=True)
        self.scratch.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(["bash", str(self.cleanup_path)], check=False)
        self.assertEqual(completed.returncode, 0)
        self.assertFalse(self.scratch.exists())
        self.assertTrue(neighbor.is_dir())


if __name__ == "__main__":
    unittest.main()
