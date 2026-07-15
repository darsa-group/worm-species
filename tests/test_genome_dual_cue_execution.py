"""Focused operational contracts for the canonical Genome dual-cue stage."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission
from src.worm_species.slurm.rendering import write_artifact_bundle
from src.worm_species.slurm.submission import build_submission_commands


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    ROOT
    / "tests"
    / "fixtures"
    / "slurm_execution"
    / "genome_dual_cue.snapshot"
)


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


class GenomeDualCueExecutionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.project = cls.root / "project"
        cls.data = cls.root / "data"
        cls.cache = cls.data / "image_cache"
        cls.results = cls.root / "results"
        cls.scratch = cls.root / "configured-scratch"
        cls.project.mkdir()
        (cls.project / "src").mkdir()
        (cls.data / "01_Segmented").mkdir(parents=True)
        (cls.data / "01_Segmented" / "global_metadata.csv").write_text(
            "image\n", encoding="utf-8"
        )
        cls.cache.mkdir()
        (cls.cache / "CACHE_READY").write_text("ready\n", encoding="utf-8")
        cls.conda_sh = cls.root / "conda.sh"
        cls.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")

        overrides = [
            f"slurm.paths.project_root={cls.project}",
            f"slurm.paths.data_root={cls.data}",
            (
                "slurm.paths.metadata_csv="
                f"{cls.data / '01_Segmented' / 'global_metadata.csv'}"
            ),
            f"slurm.paths.cache_root={cls.cache}",
            f"slurm.paths.results_root={cls.results}",
            f"slurm.scratch.root={cls.scratch}",
            "slurm.scratch.copy_cache_to_tmp=0",
            f"slurm.environment.conda_sh={cls.conda_sh}",
        ]
        cls.config = load_submission_config(
            ROOT / "configs" / "experiments" / "dual_cue.yaml",
            ROOT / "configs" / "clusters" / "genome.yaml",
            overrides,
        )
        cls.plan = plan_submission(cls.config)
        cls.bundle = cls.root / "bundle"
        cls.manifest = write_artifact_bundle(cls.plan, cls.config, cls.bundle)
        cls.script_path = (
            cls.bundle / "generated_slurm" / "job_local_cue_array_job.sh"
        )
        cls.script = cls.script_path.read_text(encoding="utf-8")

        cls.bin_dir = cls.root / "bin"
        cls.bin_dir.mkdir()
        _write_executable(
            cls.bin_dir / "python",
            "#!/usr/bin/env bash\n# CUDA/W&B preflight double.\nexit \"${PREFLIGHT_STATUS:-0}\"\n",
        )
        _write_executable(
            cls.bin_dir / "srun",
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"$*\" >> \"${TRAIN_CALLS:?}\"\n"
            "exit \"${TRAIN_STATUS:-0}\"\n",
        )
        _write_executable(
            cls.bin_dir / "rsync",
            "#!/usr/bin/env bash\n"
            "source_path=${@: -2:1}\n"
            "destination=${@: -1}\n"
            "if [[ $source_path == */image_cache/ ]]; then\n"
            "  printf 'cache-copy\\n' >> \"${RSYNC_CALLS:?}\"\n"
            "  mkdir -p \"$destination\"\n"
            "  if [[ ${DROP_CACHE_MARKER:-0} != 1 ]]; then\n"
            "    cp \"${source_path}CACHE_READY\" \"${destination}CACHE_READY\"\n"
            "  fi\n"
            "  exit \"${CACHE_COPY_STATUS:-0}\"\n"
            "fi\n"
            "printf 'copyback\\n' >> \"${RSYNC_CALLS:?}\"\n"
            "exit \"${COPYBACK_STATUS:-0}\"\n",
        )
        _write_executable(
            cls.bin_dir / "df",
            "#!/usr/bin/env bash\n"
            "if [[ ${FORCE_SMALL_TMP:-0} == 1 ]]; then\n"
            "  printf 'Filesystem 1-blocks Used Available Capacity Mounted\\n'\n"
            "  printf 'fake 1 1 0 100%% /tmp\\n'\n"
            "else\n"
            "  exec /usr/bin/df \"$@\"\n"
            "fi\n",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _snapshot_projection(self) -> dict:
        jobs = {job["name"]: job for job in self.manifest["jobs"]}
        train = jobs["train_array"]
        collect = jobs["collect"]
        return {
            "array_size": self.manifest["array_size"],
            "collection": {
                "dependencies": collect["dependencies"],
                "partition": collect["partition"],
            },
            "script_contract": {
                "cache_parent_override": (
                    'cache.root_dir_cache="$CACHE_PARENT"' in self.script
                ),
                "cache_ready_copy_check": "local_ready_marker" in self.script,
                "copyback_status_90": "status=90" in self.script,
                "cuda_launch_blocking": "CUDA_LAUNCH_BLOCKING" in self.script,
                "cuda_preflight": "torch.cuda.is_available()" in self.script,
                "one_canonical_trainer": (
                    self.script.count("python -m worm_species.training") == 1
                ),
                "run_overrides_copy": "run_overrides.args" in self.script,
                "split_root_override": (
                    'split.predefined_split_dir="$PROJECT_ROOT"' in self.script
                ),
                "uses_srun": "srun python -m worm_species.training" in self.script,
            },
            "training": {
                "dependencies": train["dependencies"],
                "partition": train["partition"],
                "wandb_group": train["exports"]["WANDB_RUN_GROUP"],
                "wandb_mode": train["exports"]["WANDB_MODE"],
                "wandb_project": train["exports"]["WANDB_PROJECT"],
            },
        }

    def test_rendered_contract_snapshot_and_shell_syntax(self) -> None:
        expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        self.assertEqual(self._snapshot_projection(), expected)
        subprocess.run(["bash", "-n", str(self.script_path)], check=True)

    def test_collection_uses_afterany_and_omits_partition_argument(self) -> None:
        commands = build_submission_commands(self.manifest)
        collect = commands[1]
        self.assertIn("--dependency=afterany:@train_array", collect)
        self.assertFalse(any(item.startswith("--partition=") for item in collect))

    def _run_script(
        self,
        *,
        copy_mode: str = "0",
        train_status: int = 0,
        copyback_status: int = 0,
        drop_cache_marker: bool = False,
        force_small_tmp: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
        runtime_root = self.root / "runtime"
        runtime_root.mkdir(parents=True, exist_ok=True)
        run_root = Path(
            tempfile.mkdtemp(
                prefix=f"{self.id().rsplit('.', 1)[-1]}-", dir=runtime_root
            )
        )
        script = run_root / "array.sh"
        script.write_text(
            self.script.replace(
                "COPY_CACHE_TO_TMP=0", f"COPY_CACHE_TO_TMP={copy_mode}", 1
            ),
            encoding="utf-8",
        )
        script.chmod(0o755)
        train_calls = run_root / "trainer.calls"
        rsync_calls = run_root / "rsync.calls"
        train_calls.write_text("", encoding="utf-8")
        rsync_calls.write_text("", encoding="utf-8")
        environment = dict(os.environ)
        environment.update(
            {
                "PATH": f"{self.bin_dir}:{environment['PATH']}",
                "SLURM_ARRAY_TASK_ID": "0",
                "SLURM_JOB_ID": "123",
                "TMPDIR": str(run_root / "tmp"),
                "TRAIN_CALLS": str(train_calls),
                "RSYNC_CALLS": str(rsync_calls),
                "TRAIN_STATUS": str(train_status),
                "COPYBACK_STATUS": str(copyback_status),
                "DROP_CACHE_MARKER": "1" if drop_cache_marker else "0",
                "FORCE_SMALL_TMP": "1" if force_small_tmp else "0",
                "WANDB_ENABLED": "true",
                "WANDB_PROJECT": "worm-species-cues",
                "WANDB_MODE": "online",
                "WANDB_RUN_GROUP": "results",
            }
        )
        Path(environment["TMPDIR"]).mkdir()
        completed = subprocess.run(
            ["bash", str(script)],
            check=False,
            text=True,
            capture_output=True,
            env=environment,
        )
        status_file = (
            Path(environment["TMPDIR"])
            / "worm_species"
            / "outputs"
            / "run_000"
            / "run_status.txt"
        )
        return completed, status_file, train_calls, rsync_calls

    def test_one_trainer_call_split_root_cache_parent_and_status(self) -> None:
        completed, status_file, train_calls, _ = self._run_script()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(status_file.read_text(encoding="utf-8"), "0\n")
        self.assertTrue((status_file.parent / "run_overrides.args").is_file())
        calls = train_calls.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 1)
        self.assertIn(f"split.predefined_split_dir={self.project}", calls[0])
        self.assertIn(f"cache.root_dir_cache={self.data}", calls[0])

    def test_required_cache_copy_uses_local_dir_and_its_parent(self) -> None:
        completed, status_file, train_calls, rsync_calls = self._run_script(
            copy_mode="1"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        local_root = status_file.parents[2]
        local_cache = local_root / "image_cache"
        calls = train_calls.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 1)
        self.assertIn(f"cache.dir={local_cache}", calls[0])
        self.assertIn(f"cache.root_dir_cache={local_root}", calls[0])
        self.assertEqual(
            rsync_calls.read_text(encoding="utf-8").splitlines(),
            ["cache-copy", "copyback"],
        )

    def test_trainer_and_copyback_exit_contract(self) -> None:
        trained, status_file, _, _ = self._run_script(train_status=7)
        self.assertEqual(trained.returncode, 7)
        self.assertEqual(status_file.read_text(encoding="utf-8"), "7\n")

        copy_failed, status_file, _, _ = self._run_script(copyback_status=23)
        self.assertEqual(copy_failed.returncode, 90)
        self.assertEqual(status_file.read_text(encoding="utf-8"), "0\n")

    def test_required_cache_copy_verifies_ready_marker(self) -> None:
        completed, status_file, train_calls, rsync_calls = self._run_script(
            copy_mode="1", drop_cache_marker=True
        )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(status_file.read_text(encoding="utf-8"), "1\n")
        self.assertEqual(train_calls.read_text(encoding="utf-8"), "")
        self.assertEqual(
            rsync_calls.read_text(encoding="utf-8").splitlines(),
            ["cache-copy", "copyback"],
        )

    def test_auto_cache_mode_falls_back_to_persistent_cache(self) -> None:
        completed, _, train_calls, rsync_calls = self._run_script(
            copy_mode="auto", force_small_tmp=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(train_calls.read_text(encoding="utf-8").splitlines()), 1)
        self.assertEqual(
            rsync_calls.read_text(encoding="utf-8").splitlines(), ["copyback"]
        )


if __name__ == "__main__":
    unittest.main()
