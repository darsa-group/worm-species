from __future__ import annotations

import importlib
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_GENERATOR = ROOT / "src" / "generate_sweep_run_specs.py"
ARCHIVED_GENERATOR = (
    ROOT / "legacy" / "python" / "experiments" / "generate_sweep_run_specs.py"
)
ARCHIVED_GENERATOR_SHA256 = (
    "9ad6899591897f561d78cddaea10e1ad494671114e23f0687dbd43e04c2f7b1d"
)


def run_generator(script: Path, config: Path, output_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--out-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def files_below(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class LegacyPythonLayoutContracts(unittest.TestCase):
    def test_archived_generator_body_is_byte_identical_to_history(self) -> None:
        self.assertEqual(
            hashlib.sha256(ARCHIVED_GENERATOR.read_bytes()).hexdigest(),
            ARCHIVED_GENERATOR_SHA256,
        )

    def test_old_import_path_reexports_archived_symbols(self) -> None:
        public = importlib.import_module("src.generate_sweep_run_specs")
        archived = importlib.import_module(
            "legacy.python.experiments.generate_sweep_run_specs"
        )

        self.assertIs(public.main, archived.main)
        self.assertIs(public.format_value, archived.format_value)
        self.assertEqual(public.format_value(True), "true")
        self.assertEqual(public.format_value(None), "null")

    def test_old_cli_help_is_byte_identical_to_archived_command(self) -> None:
        commands = (
            [sys.executable, str(PUBLIC_GENERATOR), "--help"],
            [sys.executable, str(ARCHIVED_GENERATOR), "--help"],
        )
        results = [
            subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            for command in commands
        ]

        self.assertEqual(results[0].returncode, 0)
        self.assertEqual(results[0].stdout, results[1].stdout)
        self.assertEqual(results[0].stderr, results[1].stderr)

    def test_old_cli_generates_byte_identical_sweep_artifacts(self) -> None:
        config = {
            "sweep": {
                "enabled": True,
                "parameters": {
                    "model.name": ["resnet18", "vit_b_16"],
                    "model.pretrained": [True, False],
                    "training.limit": [None],
                },
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            public_root = root / "public"
            archived_root = root / "archived"

            public_result = run_generator(
                PUBLIC_GENERATOR, config_path, public_root / "run_specs"
            )
            archived_result = run_generator(
                ARCHIVED_GENERATOR, config_path, archived_root / "run_specs"
            )

            self.assertEqual(public_result.stdout, archived_result.stdout)
            self.assertEqual(public_result.stderr, archived_result.stderr)
            self.assertEqual(files_below(public_root), files_below(archived_root))
            self.assertEqual(public_result.stdout, "4\n")


if __name__ == "__main__":
    unittest.main()
