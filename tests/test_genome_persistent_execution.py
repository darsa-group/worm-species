"""Genome persistent-cache hierarchy and W&B execution contracts."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.environment import (
    ResolutionContext,
    resolve_submission_environment,
)
from src.worm_species.slurm.planning import plan_submission
from src.worm_species.slurm.rendering import write_artifact_bundle
from src.worm_species.slurm.submission import build_submission_commands


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    ROOT / "tests/fixtures/slurm_execution/genome_persistent.snapshot"
)


def _executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


class GenomePersistentExecutionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.project = cls.root / "project"
        cls.data = cls.root / "data"
        cls.cache = cls.data / "image_cache"
        cls.project.mkdir()
        (cls.project / "src").mkdir()
        (cls.data / "01_Segmented").mkdir(parents=True)
        cls.metadata = cls.data / "01_Segmented/global_metadata.csv"
        cls.metadata.write_text("image\n", encoding="utf-8")
        cls.cache.mkdir()
        (cls.cache / "CACHE_READY").write_text("ready\n", encoding="utf-8")
        cls.conda_sh = cls.root / "conda.sh"
        cls.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")

        cls.hierarchy_root, cls.hierarchy_config, cls.hierarchy_manifest = cls._render(
            "persistent_hierarchy", cls.root / "results"
        )
        cls.wandb_root, cls.wandb_config, cls.wandb_manifest = cls._render(
            "persistent_hierarchy_wandb", cls.root / "wandb-results"
        )
        cls.array_path = (
            cls.hierarchy_root / "generated_slurm/persistent_cache_array_job.sh"
        )
        cls.array_script = cls.array_path.read_text(encoding="utf-8")
        cls.cache_job_path = cls.hierarchy_root / "generated_slurm/cache_build_job.sh"
        cls.cache_job = cls.cache_job_path.read_text(encoding="utf-8")

    @classmethod
    def _render(cls, experiment: str, results: Path):
        config = load_submission_config(
            ROOT / "configs/experiments" / f"{experiment}.yaml",
            ROOT / "configs/clusters/genome_persistent.yaml",
            [
                f"slurm.paths.project_root={cls.project}",
                f"slurm.paths.data_root={cls.data}",
                f"slurm.paths.metadata_csv={cls.metadata}",
                f"slurm.paths.cache_root={cls.cache}",
                f"slurm.paths.results_root={results}",
                f"slurm.environment.conda_sh={cls.conda_sh}",
            ],
        )
        plan = plan_submission(config)
        bundle = cls.root / f"{experiment}-bundle"
        manifest = write_artifact_bundle(plan, config, bundle)
        return bundle, config, manifest

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _projection(self) -> dict:
        hierarchy_job = self.hierarchy_manifest["jobs"][0]
        wandb_job = self.wandb_manifest["jobs"][0]
        hierarchy_resolved = self.hierarchy_config
        wandb_array = (
            self.wandb_root / "generated_slurm/persistent_cache_array_job.sh"
        )
        return {
            "cache_job": {
                "build_command": "python -m worm_species.cache build" in self.cache_job,
                "cpus": next(
                    line.rsplit("=", 1)[1]
                    for line in self.cache_job.splitlines()
                    if line.startswith("#SBATCH --cpus-per-task=")
                ),
                "force_rebuild": "--force-rebuild" in self.cache_job,
                "memory": next(
                    line.rsplit("=", 1)[1]
                    for line in self.cache_job.splitlines()
                    if line.startswith("#SBATCH --mem=")
                ),
                "time": next(
                    line.rsplit("=", 1)[1]
                    for line in self.cache_job.splitlines()
                    if line.startswith("#SBATCH --time=")
                ),
                "verify_command": "python -m worm_species.cache verify" in self.cache_job,
            },
            "hierarchy": {
                "array_size": self.hierarchy_manifest["array_size"],
                "dependencies": hierarchy_job["dependencies"],
                "exclude_nodes": hierarchy_job["exclude_nodes"],
                "hierarchy_enabled": hierarchy_resolved["multi_task"][
                    "hierarchy_loss"
                ]["enabled"],
                "jobs": [job["name"] for job in self.hierarchy_manifest["jobs"]],
                "partition": hierarchy_job["partition"],
                "trainer_cpus": hierarchy_job["cpus_per_task"],
                "wandb_enabled": bool(
                    hierarchy_resolved.get("wandb", {}).get("enabled", False)
                ),
            },
            "script_contract": {
                "cache_parent_override": (
                    'cache.root_dir_cache="$CACHE_PARENT"' in self.array_script
                ),
                "cache_modes": "0|1|auto" in self.array_script,
                "cache_ready_copy_check": "local_ready_marker" in self.array_script,
                "cache_reserve": (
                    "TMP_RESERVE_GB * 1024 * 1024 * 1024" in self.array_script
                ),
                "cuda_preflight": "torch.cuda.is_available()" in self.array_script,
                "one_canonical_trainer": (
                    self.array_script.count("python -m worm_species.training") == 1
                ),
                "run_overrides_copy": "run_overrides.args" in self.array_script,
                "split_root_override": (
                    'split.predefined_split_dir="$PROJECT_ROOT"'
                    in self.array_script
                ),
                "uses_srun": (
                    "srun python -m worm_species.training" in self.array_script
                ),
            },
            "wandb": {
                "array_size": self.wandb_manifest["array_size"],
                "group": wandb_job["exports"]["WANDB_RUN_GROUP"],
                "mode": wandb_job["exports"]["WANDB_MODE"],
                "project": wandb_job["exports"]["WANDB_PROJECT"],
                "uses_shared_template": wandb_array.is_file()
                and not (
                    self.wandb_root
                    / "generated_slurm/persistent_cache_wandb_array_job.sh"
                ).exists(),
            },
        }

    def test_snapshot_and_rendered_shell_syntax(self) -> None:
        self.assertEqual(
            self._projection(), json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        )
        for bundle in (self.hierarchy_root, self.wandb_root):
            for script in (bundle / "generated_slurm").glob("*.sh"):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_excluded_node_is_emitted_and_no_collector_is_planned(self) -> None:
        for manifest in (self.hierarchy_manifest, self.wandb_manifest):
            self.assertEqual([job["name"] for job in manifest["jobs"]], ["train_array"])
            commands = build_submission_commands(manifest)
            self.assertEqual(len(commands), 1)
            self.assertIn("--exclude=gn-1002", commands[0])

    def test_exact_two_model_specs_and_one_resolved_run_per_task(self) -> None:
        expected = {
            "run_000.args": b"model.name=convnext_base\n",
            "run_001.args": b"model.name=vit_b_16\n",
        }
        actual = {
            path.name: path.read_bytes()
            for path in sorted((self.hierarchy_root / "run_specs").glob("*.args"))
        }
        self.assertEqual(actual, expected)
        for spec in plan_submission(self.hierarchy_config).run_specs:
            self.assertFalse(spec.resolved_config["sweep"]["enabled"])
            self.assertFalse(
                spec.resolved_config["matched_condition_training"]["enabled"]
            )
            self.assertTrue(
                spec.resolved_config["multi_task"]["hierarchy_loss"]["enabled"]
            )
            self.assertEqual(spec.trainer_command.count("--single-run"), 1)

    def test_cache_maintenance_job_invokes_build_then_verify(self) -> None:
        runtime = self.root / "cache-job-runtime"
        runtime.mkdir()
        bin_dir = runtime / "bin"
        bin_dir.mkdir()
        calls = runtime / "python.calls"
        calls.write_text("", encoding="utf-8")
        _executable(
            bin_dir / "python",
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"${PYTHON_CALLS:?}\"\n",
        )
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{bin_dir}:{environment['PATH']}",
                "PYTHON_CALLS": str(calls),
                "FORCE_REBUILD": "1",
                "IMAGE_COL": "rel_path_seg",
            }
        )
        completed = subprocess.run(
            ["bash", str(self.cache_job_path)],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        invocations = calls.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(invocations), 2)
        self.assertTrue(invocations[0].startswith("-m worm_species.cache build "))
        self.assertIn("--force-rebuild", invocations[0])
        self.assertEqual(
            invocations[1], f"-m worm_species.cache verify --cache-dir {self.cache}"
        )

    def test_array_task_uses_local_cache_parent_and_invokes_trainer_once(self) -> None:
        runtime = self.root / "array-runtime"
        runtime.mkdir()
        tmpdir = runtime / "tmp"
        tmpdir.mkdir()
        bin_dir = runtime / "bin"
        bin_dir.mkdir()
        calls = runtime / "trainer.calls"
        calls.write_text("", encoding="utf-8")
        _executable(bin_dir / "python", "#!/usr/bin/env bash\nexit 0\n")
        _executable(
            bin_dir / "srun",
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"${TRAIN_CALLS:?}\"\n",
        )
        _executable(
            bin_dir / "rsync",
            "#!/usr/bin/env bash\n"
            "source_path=${@: -2:1}\n"
            "destination=${@: -1}\n"
            "if [[ $source_path == */image_cache/ ]]; then\n"
            "  mkdir -p \"$destination\"\n"
            "  cp \"${source_path}CACHE_READY\" \"${destination}CACHE_READY\"\n"
            "fi\n",
        )
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{bin_dir}:{environment['PATH']}",
                "SLURM_ARRAY_TASK_ID": "0",
                "SLURM_JOB_ID": "123",
                "TMPDIR": str(tmpdir),
                "TRAIN_CALLS": str(calls),
            }
        )
        completed = subprocess.run(
            ["bash", str(self.array_path)],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        invocation = calls.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(invocation), 1)
        local_root = tmpdir / "worm_species"
        self.assertIn(f"cache.dir={local_root / 'image_cache'}", invocation[0])
        self.assertIn(f"cache.root_dir_cache={local_root}", invocation[0])
        self.assertIn(f"split.predefined_split_dir={self.project}", invocation[0])
        status = local_root / "outputs/run_000/run_status.txt"
        self.assertEqual(status.read_text(encoding="utf-8"), "0\n")

    def test_legacy_env_default_home_is_symlink_and_timestamp_aware(self) -> None:
        legacy_home = self.root / "legacy-home"
        legacy_home.mkdir()
        target = self.root / "persistent-project"
        target.mkdir()
        (legacy_home / "worm-species").symlink_to(target, target_is_directory=True)
        resolved = resolve_submission_environment(
            self.hierarchy_config,
            ResolutionContext(
                cwd=self.root,
                environ={"HOME": str(legacy_home), "USER": "alice"},
                submission_stamp="20260102_030405",
                process_id=123,
            ),
            import_legacy=True,
        )
        paths = resolved.config["slurm"]["paths"]
        self.assertEqual(paths["project_root"], str(target / "source"))
        self.assertEqual(paths["data_root"], str(target / "data"))
        self.assertEqual(
            paths["results_root"],
            str(
                target
                / "source/outputs_slurm/persistent_cache_sweep_20260102_030405"
            ),
        )


if __name__ == "__main__":
    unittest.main()
