from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.worm_species.slurm.cli import build_parser
from src.worm_species.slurm.config import SlurmConfigError, load_submission_config
from src.worm_species.slurm.environment import (
    EnvironmentResolutionError,
    ResolutionContext,
    resolve_submission_environment,
)


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "tests" / "fixtures" / "slurm_environment"


def _assert_snapshot(test: unittest.TestCase, name: str, value: dict) -> None:
    expected = json.loads(
        (SNAPSHOTS / f"{name}.snapshot").read_text(encoding="utf-8")
    )
    test.assertEqual(value, expected)


class SlurmEnvironmentResolutionTests(unittest.TestCase):
    def test_genome_resolution_snapshot(self) -> None:
        config = {
            "wandb": {"enabled": False, "project": None},
            "slurm": {
                "cluster_profile": "genome",
                "account": "default",
                "partition": "default",
                "cpus_per_task": 1,
                "memory": "1G",
                "time_limit": "00:10:00",
                "array": {"max_active": 1},
                "scratch": {
                    "mode": "job_local_cache",
                    "root": "/tmp/${USER}/worm_species",
                    "nodes": [],
                    "copy_cache_to_tmp": "auto",
                    "tmp_reserve_gb": 0,
                },
                "setup": {"enabled": False},
                "collection": {"enabled": True},
                "cleanup": {"enabled": False},
                "environment": {
                    "conda_sh": "${HOME}/miniforge3/etc/profile.d/conda.sh",
                    "conda_env": "wormspecies",
                },
                "paths": {
                    "project_root": "${HOME}/worm-species/source",
                    "data_root": "${HOME}/worm-species/data",
                    "metadata_csv": "${HOME}/worm-species/data/metadata.csv",
                    "results_root": "outputs_slurm",
                    "cache_root": "${HOME}/worm-species/data/image_cache",
                },
                "logging": {"directory": "logs/slurm"},
                "submission": {"extra_sbatch_args": []},
            },
        }
        context = ResolutionContext(
            cwd=Path("/work/login"),
            environ={
                "HOME": "/home/alice",
                "USER": "alice",
                "PROJECT_ROOT": "/srv/worm-species",
                "GPU_ACCOUNT": "worm-species",
                "GPU_PARTITION": "gpu-short,gpu-h200,gpu-l40s",
                "GPU_CPUS_PER_TASK": "12",
                "GPU_MEM": "12384",
                "GPU_TIME": "01:30:00",
                "MAX_ACTIVE": "12",
                "COPY_CACHE_TO_TMP": "1",
                "TMP_RESERVE_GB": "5",
                "WANDB_ENABLED": "true",
                "WANDB_PROJECT": "worm-species-cues",
                "WANDB_ENTITY": "",
                "WANDB_MODE": "online",
                "WANDB_RUN_GROUP": "dual-cue-20260102",
                "TRAIN_SCRIPT": "train_multitask_cue_suppression.py",
            },
            submission_stamp="20260102_030405",
            process_id=1234,
        )
        resolved = resolve_submission_environment(config, context, import_legacy=True)
        _assert_snapshot(self, "genome_dual_cue", resolved.snapshot())

    def test_ghpc_resolution_snapshot(self) -> None:
        config = {
            "wandb": {"enabled": False, "project": None},
            "slurm": {
                "cluster_profile": "ghpc",
                "account": "worm-species",
                "partition": "ghpc_gpu",
                "cpus_per_task": 8,
                "memory": 16384,
                "time_limit": "04:00:00",
                "array": {"max_active": 10},
                "scratch": {
                    "mode": "node_local",
                    "root": "/scratch/${USER}/worm_species",
                    "nodes": [],
                    "copy_cache_to_tmp": 0,
                    "tmp_reserve_gb": 0,
                },
                "setup": {"enabled": True, "cpus_per_task": 4},
                "collection": {"enabled": True, "cpus_per_task": 1},
                "cleanup": {"enabled": True, "cpus_per_task": 1},
                "environment": {
                    "conda_sh": "/usr/home/qgg/mehrot/miniconda3/etc/profile.d/conda.sh",
                    "conda_env": "wormspecies",
                },
                "paths": {
                    "project_root": ".",
                    "data_root": "/shared/data",
                    "metadata_csv": "/shared/data/metadata.csv",
                    "results_root": "outputs_slurm",
                    "cache_root": "cache/images",
                },
                "logging": {"directory": "logs/slurm"},
                "submission": {"extra_sbatch_args": []},
            },
        }
        context = ResolutionContext(
            cwd=Path("/work/worm-species"),
            environ={
                "HOME": "/home/alice",
                "USER": "alice",
                "PROJECT_SRC": "/work/worm-species",
                "DATA_SRC": "/data/earthworms",
                "GPU_NODES": "gpu001 gpu002",
                "GPU_CPUS_PER_TASK": "16",
                "MAX_ACTIVE": "2",
                "SETUP_MEM": "8192",
                "SETUP_TIME": "01:00:00",
                "CLEANUP_PARTITION": "ghpc_gpu",
                "CLEANUP_MEM": "2048",
                "CLEANUP_TIME": "00:30:00",
                "COLLECT_MEM": "4096",
                "COLLECT_TIME": "00:20:00",
                "WANDB_ENABLED": "true",
                "WANDB_PROJECT": "worm-species-color-ablation",
                "WANDB_MODE": "online",
                "GPU_EXTRA_SBATCH_ARGS": "--constraint=a100 --qos=normal",
                "RESULT_COLLECTOR": "collect_dual_cue_results.py",
            },
            submission_stamp="20260102_030405",
            process_id=4321,
        )
        resolved = resolve_submission_environment(config, context, import_legacy=True)
        _assert_snapshot(self, "ghpc_colour", resolved.snapshot())

    def test_ambient_legacy_variables_are_ignored_without_opt_in(self) -> None:
        config = {
            "slurm": {
                "paths": {"project_root": "configured"},
                "scratch": {"root": "/tmp/${USER}/configured"},
            }
        }
        context = ResolutionContext(
            cwd=Path("/work"),
            environ={"USER": "alice", "PROJECT_SRC": "/ambient/project"},
        )
        resolved = resolve_submission_environment(config, context)
        self.assertEqual(resolved.config["slurm"]["paths"]["project_root"], "configured")
        self.assertEqual(resolved.config["slurm"]["scratch"]["root"], "/tmp/alice/configured")
        self.assertEqual(resolved.imported_variables, ())

    def test_explicit_override_wins_over_imported_environment(self) -> None:
        config = load_submission_config(
            ROOT / "configs" / "experiments" / "standard.yaml",
            ROOT / "configs" / "clusters" / "local.yaml",
            ["sweep.enabled=false", "slurm.paths.project_root=/cli/project"],
            import_legacy_environment=True,
            environment={
                "HOME": "/home/alice",
                "USER": "alice",
                "PROJECT_SRC": "/environment/project",
            },
            cwd="/work",
        )
        self.assertEqual(config["slurm"]["paths"]["project_root"], "/cli/project")

    def test_conflicting_aliases_fail_clearly(self) -> None:
        context = ResolutionContext(
            cwd=Path("/work"),
            environ={"SOURCE_ROOT": "/one", "PROJECT_SRC": "/two"},
        )
        with self.assertRaisesRegex(EnvironmentResolutionError, "Conflicting legacy aliases"):
            resolve_submission_environment({"slurm": {}}, context, import_legacy=True)

    def test_unresolved_path_variable_fails_clearly(self) -> None:
        context = ResolutionContext(cwd=Path("/work"), environ={})
        with self.assertRaisesRegex(
            EnvironmentResolutionError, "unavailable environment variable HOME"
        ):
            resolve_submission_environment(
                {"slurm": {"paths": {"project_root": "${HOME}/project"}}},
                context,
            )

    def test_symlinked_genome_project_entry_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.mkdir()
            entry = root / "entry"
            entry.symlink_to(target, target_is_directory=True)
            context = ResolutionContext(
                cwd=root,
                environ={"HOME": str(root), "USER": "alice", "PROJECT_ROOT": str(entry)},
                submission_stamp="20260102_030405",
            )
            resolved = resolve_submission_environment(
                {"slurm": {"paths": {}, "scratch": {}}},
                context,
                import_legacy=True,
            )
            self.assertEqual(
                resolved.config["slurm"]["paths"]["project_root"],
                str(target / "source"),
            )

    def test_cli_requires_explicit_legacy_env_switch(self) -> None:
        parser = build_parser()
        normal = parser.parse_args(["render", "--artifacts-dir", "/tmp/plan"])
        imported = parser.parse_args(
            ["render", "--artifacts-dir", "/tmp/plan", "--legacy-env"]
        )
        self.assertFalse(normal.legacy_env)
        self.assertTrue(imported.legacy_env)

    def test_managed_sbatch_option_from_legacy_env_is_rejected(self) -> None:
        with self.assertRaisesRegex(SlurmConfigError, "--array is managed"):
            load_submission_config(
                ROOT / "configs" / "experiments" / "standard.yaml",
                ROOT / "configs" / "clusters" / "local.yaml",
                ["sweep.enabled=false"],
                import_legacy_environment=True,
                environment={
                    "HOME": "/home/alice",
                    "USER": "alice",
                    "GPU_EXTRA_SBATCH_ARGS": "--array=0-3",
                },
            )


if __name__ == "__main__":
    unittest.main()
