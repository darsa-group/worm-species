from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "legacy" / "compatibility.map"
RESTORE = ROOT / "legacy" / "restore_compatibility.sh"


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class LegacyArchiveContracts(unittest.TestCase):
    def test_manifest_is_complete_and_active_paths_are_absent(self) -> None:
        rows = manifest_rows()
        self.assertEqual(len(rows), 37)
        active_paths = [row["active_path"] for row in rows]
        self.assertEqual(len(active_paths), len(set(active_paths)))
        for row in rows:
            with self.subTest(active=row["active_path"]):
                active = ROOT / row["active_path"]
                self.assertFalse(active.exists() or active.is_symlink())

    def test_every_archive_source_matches_manifest_hash(self) -> None:
        for row in manifest_rows():
            with self.subTest(archive=row["archive_path"]):
                archive = ROOT / row["archive_path"]
                self.assertTrue(archive.is_file())
                self.assertEqual(
                    hashlib.sha256(archive.read_bytes()).hexdigest(),
                    row["sha256"],
                )

    def test_archived_shell_and_restore_scripts_parse(self) -> None:
        scripts = sorted((ROOT / "legacy" / "slurm").glob("*.sh"))
        scripts.append(RESTORE)
        for script in scripts:
            with self.subTest(script=script.relative_to(ROOT)):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_dry_run_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            result = subprocess.run(
                [str(RESTORE), "--root", str(target), "--dry-run"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.count("WOULD RESTORE:"), 37)
            self.assertEqual(list(target.iterdir()), [])

    def test_restore_recreates_regular_files_and_relative_links(self) -> None:
        rows = manifest_rows()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            shutil.copytree(ROOT / "legacy" / "slurm", target / "legacy" / "slurm")
            shutil.copytree(ROOT / "legacy" / "configs", target / "legacy" / "configs")

            first = subprocess.run(
                [str(RESTORE), "--root", str(target)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first.stdout.count("RESTORED:"), 37)

            for row in rows:
                with self.subTest(active=row["active_path"]):
                    restored = target / row["active_path"]
                    archive = ROOT / row["archive_path"]
                    if row["restore_kind"] == "copy":
                        self.assertTrue(restored.is_file())
                        self.assertFalse(restored.is_symlink())
                        self.assertEqual(restored.read_bytes(), archive.read_bytes())
                        self.assertEqual(
                            oct(restored.stat().st_mode & 0o777)[2:],
                            row["mode"],
                        )
                    else:
                        self.assertTrue(restored.is_symlink())
                        self.assertEqual(os.readlink(restored), row["link_target"])
                        self.assertEqual(restored.resolve().read_bytes(), archive.read_bytes())

            second = subprocess.run(
                [str(RESTORE), "--root", str(target)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(second.stdout.count("ALREADY RESTORED:"), 37)
            self.assertNotIn("RESTORED:", second.stdout.replace("ALREADY RESTORED:", ""))

    def test_conflict_fails_before_any_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            conflict = target / "train_multitask_colour_ablation.py"
            conflict.write_text("user-owned\n", encoding="utf-8")

            result = subprocess.run(
                [str(RESTORE), "--root", str(target)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("refusing to overwrite", result.stderr)
            self.assertIn("no compatibility paths were restored", result.stderr)
            self.assertEqual(conflict.read_text(encoding="utf-8"), "user-owned\n")
            self.assertFalse((target / "train_multitask_masked.py").exists())
            self.assertFalse((target / "scripts").exists())

    def test_symlinked_parent_is_rejected_before_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            outside = Path(directory) / "outside"
            target.mkdir()
            outside.mkdir()
            (target / "scripts").symlink_to(outside, target_is_directory=True)

            result = subprocess.run(
                [str(RESTORE), "--root", str(target)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("destination parent is a symlink", result.stderr)
            self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
