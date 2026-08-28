from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.upload_cvat_curated_archives import append_manifest
from scripts.upload_cvat_curated_archives import discover_archives
from scripts.upload_cvat_curated_archives import next_numbered_name
from scripts.upload_cvat_curated_archives import read_successful_archives
from scripts.upload_cvat_curated_archives import stage_archive


class CVATCuratedArchiveUploadTests(unittest.TestCase):
    def _archive(self, root: Path, index: int, names: list[str]) -> Path:
        path = root / f"curated_images_{index}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                archive.writestr(name, b"image bytes")
        return path

    def test_archive_discovery_is_numeric_and_ignores_corrupt_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._archive(root, 1, ["b.png"])
            self._archive(root, 0, ["a.jpg"])
            (root / "curated_images_1.corrupt.zip").touch()

            archives = discover_archives(root, expected_count=2)

            self.assertEqual([archive.index for archive in archives], [0, 1])
            self.assertEqual([archive.image_count for archive in archives], [1, 1])

    def test_archive_discovery_rejects_cross_archive_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._archive(root, 0, ["same.jpg"])
            self._archive(root, 1, ["same.jpg"])

            with self.assertRaisesRegex(RuntimeError, "more than one archive"):
                discover_archives(root, expected_count=2)

    def test_numbered_task_names_start_at_two(self) -> None:
        names = {"curated_images_0", "curated_images_0_2", "curated_images_0_3"}
        self.assertEqual(next_numbered_name("curated_images_0", names), "curated_images_0_4")
        self.assertEqual(next_numbered_name("curated_images_1", names), "curated_images_1")

    def test_archive_is_staged_as_individual_image_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging = root / "staging"
            self._archive(root, 0, ["a.jpg", "b.png"])
            archive = discover_archives(root, expected_count=1)[0]

            temporary, resources = stage_archive(archive, staging, reserve_gb=0)
            try:
                self.assertEqual([path.name for path in resources], ["a.jpg", "b.png"])
                self.assertTrue(all(path.is_file() for path in resources))
            finally:
                temporary.cleanup()

            self.assertFalse(Path(temporary.name).exists())

    def test_manifest_tracks_only_completed_archive_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.csv"
            base = {
                "timestamp_utc": "2026-08-27T00:00:00+00:00",
                "archive_path": "/data/archive.zip",
                "archive_bytes": 10,
                "image_count": 2,
                "project_id": 1,
                "project_name": "project",
                "task_id": 2,
                "task_name": "task",
                "task_url": "https://app.cvat.ai/tasks/2",
                "attempt": 1,
                "error": "",
            }
            append_manifest(path, {**base, "status": "created"})
            append_manifest(path, {**base, "status": "complete"})

            successful = read_successful_archives(path)

            self.assertEqual(list(successful), [str(Path("/data/archive.zip").resolve())])
            with path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)


if __name__ == "__main__":
    unittest.main()
