from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MakefileContracts(unittest.TestCase):
    def _make(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            ["make", *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_help_and_validate_use_configuration_driven_training(self) -> None:
        help_result = self._make("help")
        self.assertEqual(help_result.returncode, 0, help_result.stdout)
        self.assertIn("make dashboard-prepare", help_result.stdout)
        self.assertNotIn("PROFILE", help_result.stdout)

        validate = self._make(
            "validate",
            "CONFIG=configs/experiments/standard.yaml",
            "CLUSTER=configs/clusters/local.yaml",
        )
        self.assertEqual(validate.returncode, 0, validate.stdout)
        self.assertIn("2 task(s)", validate.stdout)
        self.assertIn("modes=multitask", validate.stdout)
        self.assertNotIn("profile=", validate.stdout)

    def test_dry_run_renders_module_trainer_without_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            artifacts = root / "plan"
            result = self._make(
                "dry-run",
                "CONFIG=configs/experiments/standard.yaml",
                "CLUSTER=configs/clusters/local.yaml",
                f"ARTIFACTS_DIR={artifacts}",
                f"RESULTS_ROOT={root / 'results'}",
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            plan = json.loads((artifacts / "submission_plan.json").read_text())["plan"]
            command = plan["canonical_trainer_command"]
            self.assertEqual(command[:3], ["python", "-m", "worm_species.training"])
            self.assertNotIn("--profile", command)
            self.assertNotIn("training_profile", plan)
            self.assertEqual(plan["expected_internal_training_runs_per_task"], 1)


if __name__ == "__main__":
    unittest.main()
