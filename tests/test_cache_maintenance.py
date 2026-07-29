from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.worm_species.cache.maintenance import CacheMaintenanceError
from src.worm_species.cache.maintenance import build_persistent_cache
from src.worm_species.cache.maintenance import verify_persistent_cache


class PersistentCacheContracts(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.metadata = self.data / "metadata.csv"
        self.metadata.write_text("rel_path_seg\nimage.jpg\n", encoding="utf-8")
        self.config = self.root / "config.yaml"
        self.config.write_text(
            "data:\n  image_size: 224\ncache:\n  num_workers: 1\n",
            encoding="utf-8",
        )
        self.cache = self.data / "image_cache"

    @staticmethod
    def _metadata(_config):
        return pd.DataFrame({"rel_path_seg": ["one.jpg", "two.jpg"]})

    @staticmethod
    def _complete(config, frame):
        cache = Path(config["cache"]["dir"])
        return frame.assign(
            _cached_image_path=[str(cache / "one.png"), str(cache / "two.png")]
        )

    def test_success_writes_ordered_manifest_then_ready_marker(self) -> None:
        seen = {}

        def builder(config, frame):
            seen["config"] = config
            self.assertFalse((self.cache / "CACHE_READY").exists())
            return self._complete(config, frame)

        result = build_persistent_cache(
            self.config,
            data_root=self.data,
            metadata_csv=self.metadata,
            cache_dir=self.cache,
            prepare=self._metadata,
            builder=builder,
            now=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            hostname=lambda: "test-host",
        )
        self.assertEqual(result.status, "built")
        self.assertTrue((self.cache / "CACHE_READY").is_file())
        lines = (self.cache / "cache_manifest.txt").read_text().splitlines()
        self.assertEqual(lines[0], "created_utc=2026-01-02T03:04:05+00:00")
        self.assertEqual(lines[1], "host=test-host")
        self.assertEqual(lines[-2:], ["rows=2", "cached_rows=2"])
        runtime = seen["config"]
        for key in ("cache_dir", "dir", "root_dir", "root_dir_cache"):
            self.assertEqual(runtime["cache"][key], str(self.cache.resolve()))
        verified = verify_persistent_cache(self.cache)
        self.assertEqual((verified.rows, verified.cached_rows), (2, 2))

    def test_ready_fast_path_does_not_call_builders(self) -> None:
        build_persistent_cache(
            self.config,
            data_root=self.data,
            metadata_csv=self.metadata,
            cache_dir=self.cache,
            prepare=self._metadata,
            builder=self._complete,
        )
        result = build_persistent_cache(
            self.config,
            data_root=self.data,
            metadata_csv=self.metadata,
            cache_dir=self.cache,
            prepare=lambda _config: self.fail("metadata should not be prepared"),
            builder=lambda _config, _frame: self.fail("cache should not build"),
        )
        self.assertEqual(result.status, "already_ready")

    def test_force_removes_stale_payload_but_preserves_lock(self) -> None:
        self.cache.mkdir()
        (self.cache / "CACHE_READY").touch()
        (self.cache / "stale.bin").write_bytes(b"stale")
        (self.cache / "nested").mkdir()
        build_persistent_cache(
            self.config,
            data_root=self.data,
            metadata_csv=self.metadata,
            cache_dir=self.cache,
            force=True,
            prepare=self._metadata,
            builder=self._complete,
        )
        self.assertFalse((self.cache / "stale.bin").exists())
        self.assertFalse((self.cache / "nested").exists())
        self.assertTrue((self.cache / "CACHE_BUILD.lock").is_file())

    def test_incomplete_or_failed_build_never_marks_ready(self) -> None:
        def raises(_config, _frame):
            raise RuntimeError("synthetic builder failure")

        cases = (
            lambda _config, frame: frame,
            lambda _config, frame: frame.assign(_cached_image_path=["one", None]),
            lambda _config, _frame: None,
            raises,
        )
        for builder in cases:
            with self.subTest(builder=builder):
                if self.cache.exists():
                    for child in self.cache.iterdir():
                        if child.name != "CACHE_BUILD.lock":
                            child.unlink()
                with self.assertRaises(CacheMaintenanceError):
                    build_persistent_cache(
                        self.config,
                        data_root=self.data,
                        metadata_csv=self.metadata,
                        cache_dir=self.cache,
                        prepare=self._metadata,
                        builder=builder,
                    )
                self.assertFalse((self.cache / "CACHE_READY").exists())

    def test_force_refuses_data_root_and_symlink_cache(self) -> None:
        with self.assertRaisesRegex(ValueError, "dedicated child"):
            build_persistent_cache(
                self.config,
                data_root=self.data,
                metadata_csv=self.metadata,
                cache_dir=self.data,
                force=True,
            )
        real = self.root / "real_cache"
        real.mkdir()
        link = self.data / "linked_cache"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            build_persistent_cache(
                self.config,
                data_root=self.data,
                metadata_csv=self.metadata,
                cache_dir=link,
                force=True,
            )

    def test_verify_rejects_missing_or_incomplete_manifest(self) -> None:
        self.cache.mkdir()
        (self.cache / "CACHE_READY").touch()
        (self.cache / "cache_manifest.txt").write_text(
            "rows=2\ncached_rows=1\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(CacheMaintenanceError, "incomplete"):
            verify_persistent_cache(self.cache)


if __name__ == "__main__":
    unittest.main()
