"""Focused CPU-only contracts for SLURM rendering and mocked submission."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.worm_species.slurm.cli import build_parser
from src.worm_species.slurm.cli import execute
from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission
from src.worm_species.slurm.rendering import RenderError
from src.worm_species.slurm.rendering import verify_artifact_bundle
from src.worm_species.slurm.rendering import write_artifact_bundle
from src.worm_species.slurm.submission import RecordingSbatchClient
from src.worm_species.slurm.submission import parse_job_id
from src.worm_species.slurm.submission import submit_manifest


ROOT = Path(__file__).resolve().parents[1]


class SlurmExecutionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.temp_root = Path(cls.temporary.name)
        cls.local_root = cls._render(
            "configs/experiments/standard.yaml",
            "configs/clusters/local.yaml",
            "local",
        )
        cls.genome_root = cls._render(
            "configs/experiments/dual_cue.yaml",
            "configs/clusters/genome.yaml",
            "genome",
        )
        cls.ghpc_root = cls._render(
            "configs/experiments/colour_ablation.yaml",
            "configs/clusters/ghpc.yaml",
            "ghpc",
            overrides=["slurm.scratch.nodes=[gpu01,gpu02]"],
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _render(
        cls,
        experiment: str,
        cluster: str,
        name: str,
        overrides: list[str] | None = None,
    ) -> Path:
        config = load_submission_config(
            ROOT / experiment,
            ROOT / cluster,
            overrides=overrides or [],
        )
        plan = plan_submission(config)
        path = cls.temp_root / name
        write_artifact_bundle(plan, config, path)
        return path

    def test_required_artifacts_and_224_run_plan(self):
        required = {
            "artifact_checksums.json",
            "condition_manifest.json",
            "dry_run.json",
            "launch_plan.json",
            "launcher_settings.txt",
            "resolved_config.yaml",
            "resolved_submission_config.yaml",
            "submission_manifest.json",
            "submission_plan.json",
            "sweep_plan.tsv",
        }
        self.assertTrue(required.issubset({path.name for path in self.genome_root.iterdir()}))
        manifest = json.loads(
            (self.genome_root / "submission_manifest.json").read_text()
        )
        self.assertEqual(manifest["array_size"], 224)
        self.assertEqual(manifest["metadata"]["counts"]["runs"], 224)
        self.assertEqual(manifest["metadata"]["trainer_selection"], "configuration")
        self.assertNotIn("training_profile", manifest["metadata"])
        self.assertIn("commit", manifest["metadata"]["git"])
        self.assertIn("warning", manifest["metadata"]["git"])

    def test_rendered_shell_and_operational_invariants(self):
        for root in (self.local_root, self.genome_root, self.ghpc_root):
            for script in (root / "generated_slurm").glob("*.sh"):
                subprocess.run(["bash", "-n", str(script)], check=True)

        setup = (self.ghpc_root / "generated_slurm/node_local_setup_job.sh").read_text()
        colour = (
            self.ghpc_root / "generated_slurm/node_local_colour_array_job.sh"
        ).read_text()
        collector = (
            self.ghpc_root / "generated_slurm/result_collector_job.sh"
        ).read_text()
        genome = (
            self.genome_root / "generated_slurm/job_local_cue_array_job.sh"
        ).read_text()
        self.assertIn("--include=global_metadata.csv", setup)
        self.assertIn("--include=*_seg.jpg", setup)
        self.assertIn("IMAGE_CACHE.lock", colour)
        self.assertIn("IMAGE_CACHE_READY", colour)
        self.assertIn("WANDB_PROJECT", colour)
        self.assertIn("CACHE_READY", genome)
        self.assertIn("status=90", genome)
        self.assertEqual(genome.count("python -m worm_species.training"), 1)
        self.assertNotIn("--profile", genome)
        self.assertIn('export PYTHONPATH="$PROJECT_ROOT/src', genome)
        self.assertIn("python -m worm_species.slurm collect", collector)
        self.assertIn("--kind colour-ablation", collector)
        self.assertIn('export PYTHONPATH="$PROJECT_ROOT/src', collector)

    def test_node_local_templates_use_canonical_metadata_import(self):
        for name in (
            "node_local_colour_array_job.sh.tmpl",
            "node_local_cue_array_job.sh.tmpl",
            "node_local_training_array_job.sh.tmpl",
        ):
            with self.subTest(template=name):
                source = (ROOT / "slurm" / "templates" / name).read_text()
                self.assertIn(
                    "from worm_species.data.metadata import prepare_metadata",
                    source,
                )
                self.assertNotIn("from src.dataset_multitask", source)

    def test_exact_dependency_dags_and_non_secret_wandb_exports(self):
        ghpc = json.loads((self.ghpc_root / "submission_manifest.json").read_text())
        jobs = {job["name"]: job for job in ghpc["jobs"]}
        self.assertEqual(
            jobs["train_array"]["dependencies"],
            [
                {"job": "setup:gpu01", "kind": "afterok"},
                {"job": "setup:gpu02", "kind": "afterok"},
            ],
        )
        self.assertEqual(
            jobs["collect"]["dependencies"],
            [{"job": "train_array", "kind": "afterany"}],
        )
        self.assertEqual(
            jobs["cleanup:gpu01"]["dependencies"],
            [{"job": "train_array", "kind": "afterany"}],
        )
        self.assertIn("WANDB_PROJECT", jobs["train_array"]["exports"])
        self.assertNotIn("WANDB_API_KEY", jobs["train_array"]["exports"])

    def test_default_launch_is_dry_and_never_constructs_scheduler(self):
        artifact_dir = self.temp_root / "cli-dry"
        args = build_parser().parse_args(
            [
                "launch",
                "--config",
                str(ROOT / "configs/experiments/standard.yaml"),
                "--cluster-config",
                str(ROOT / "configs/clusters/local.yaml"),
                "--artifacts-dir",
                str(artifact_dir),
            ]
        )
        with patch(
            "src.worm_species.slurm.submission.SubprocessSbatchClient",
            side_effect=AssertionError("scheduler client constructed during dry-run"),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(execute(args), 0)
        state = json.loads((artifact_dir / "dry_run.json").read_text())
        self.assertFalse(state["submitted"])
        self.assertEqual(state["scheduler_calls"], 0)

    def test_validate_and_inspect_are_read_only_public_commands(self):
        common = [
            "--config",
            str(ROOT / "configs/experiments/patch_shuffle_matrix.yaml"),
            "--cluster-config",
            str(ROOT / "configs/clusters/local.yaml"),
        ]
        with patch(
            "src.worm_species.slurm.cli.write_artifact_bundle",
            side_effect=AssertionError("read-only command rendered artifacts"),
        ), patch(
            "src.worm_species.slurm.cli.submit_manifest",
            side_effect=AssertionError("read-only command submitted jobs"),
        ):
            validate_output = io.StringIO()
            validate = build_parser().parse_args(["validate", *common])
            with contextlib.redirect_stdout(validate_output):
                self.assertEqual(execute(validate), 0)
            self.assertIn("12 task(s)", validate_output.getvalue())

            inspect_output = io.StringIO()
            inspect = build_parser().parse_args(
                ["inspect", *common, "--format", "json"]
            )
            with contextlib.redirect_stdout(inspect_output):
                self.assertEqual(execute(inspect), 0)
            payload = json.loads(inspect_output.getvalue())
            self.assertEqual(payload["plan"]["total_run_count"], 12)
            self.assertEqual(payload["plan"]["internal_runs_per_task"], 1)
            self.assertEqual(
                payload["resolved_config"]["slurm"]["cluster_profile"],
                "local",
            )

    def test_checksum_tamper_is_rejected(self):
        script = next((self.local_root / "generated_slurm").glob("*.sh"))
        original = script.read_text()
        script.write_text(original + "\n# tampered\n")
        with self.assertRaises(RenderError):
            verify_artifact_bundle(self.local_root / "submission_manifest.json")
        script.write_text(original)

    def test_mocked_submission_dependency_argv_and_job_id_parsing(self):
        client = RecordingSbatchClient(["11", "12", "13;cluster", "14", "15", "16"])
        submitted = submit_manifest(
            self.ghpc_root / "submission_manifest.json",
            client=client,
        )
        self.assertEqual(submitted["train_array"], "13")
        self.assertIn("--dependency=afterok:11:12", client.calls[2])
        self.assertIn("--dependency=afterany:13", client.calls[3])
        self.assertEqual(parse_job_id("987;genome\n"), "987")


if __name__ == "__main__":
    unittest.main()
