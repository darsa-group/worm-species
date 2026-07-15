from __future__ import annotations

import hashlib
import os
import subprocess
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

SLURM_HASHES = {
    "01_build_persistent_cache_resolved.sh": (
        "d4bd31dca0490458d273358de4719eaed3260589e4b725be620c555ae0d152ec",
        True,
    ),
    "02_submit_sweep_cache_to_tmp_resolved.sh": (
        "6f1e551a47f3d106e869550386803a652ff7a910a0c4c319306058a60945c67e",
        True,
    ),
    "run_persistent_cache_sweep_wandb.sh": (
        "6ac4cc85b90c3205650723325641036bce15e51bb34e82a09c54bf817986c729",
        False,
    ),
    "submit_colour_ablation_sweep.sh": (
        "808e0823340182f34970739f8ae0a2fdf8c8e0b122d212cf58afd8e7ff9289af",
        False,
    ),
    "submit_dual_cue_experiment.sh": (
        "dda1a3c3c51448a742472f44a96156a4ae472c50d609361f289cb596190a735f",
        False,
    ),
    "submit_dual_cue_experiment_genome.sh": (
        "7d626bb136b381370a8702f15c8ee1b7d1713454781d2a2ba3fde27a33ac89f3",
        False,
    ),
    "submit_worm_node_local_scratch_sweep.sh": (
        "1083f1bc033af02246a473caee98bdb09a772626a4ecbdd03ba91e2360a7daf0",
        True,
    ),
    "submit_worm_node_local_scratch_sweep_hloss.sh": (
        "545d6a250517d67c6eb0b70bffeea28680f43d40f6ee46fb680ede99f59f7170",
        True,
    ),
}

CONFIG_OLD_HASH = (
    "2db60ec34d8b4bdb7c6d8247917c5aba7e189c0cc47107c8eb9d622d57ab4bc5"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LegacyShellLayoutContracts(unittest.TestCase):
    def test_historical_slurm_bodies_and_modes_are_preserved(self) -> None:
        for name, (expected_hash, executable) in SLURM_HASHES.items():
            with self.subTest(script=name):
                archived = ROOT / "legacy" / "slurm" / name
                scripts_path = ROOT / "scripts" / "slurm" / name
                root_path = ROOT / name

                self.assertTrue(archived.is_file())
                self.assertTrue(scripts_path.is_symlink())
                self.assertEqual(
                    os.readlink(scripts_path), f"../../legacy/slurm/{name}"
                )
                self.assertTrue(root_path.is_symlink())
                self.assertEqual(os.readlink(root_path), f"scripts/slurm/{name}")
                self.assertEqual(scripts_path.resolve(), archived.resolve())
                self.assertEqual(root_path.resolve(), archived.resolve())
                self.assertEqual(sha256(archived), expected_hash)
                self.assertEqual(sha256(scripts_path), expected_hash)
                self.assertEqual(sha256(root_path), expected_hash)
                self.assertEqual(bool(archived.stat().st_mode & 0o111), executable)

    def test_every_legacy_and_compatibility_shell_path_parses(self) -> None:
        for name in SLURM_HASHES:
            for path in (
                ROOT / "legacy" / "slurm" / name,
                ROOT / "scripts" / "slurm" / name,
                ROOT / name,
            ):
                with self.subTest(path=path.relative_to(ROOT)):
                    subprocess.run(["bash", "-n", str(path)], cwd=ROOT, check=True)

    def test_old_configuration_body_and_both_links_are_preserved(self) -> None:
        archived = ROOT / "legacy" / "configs" / "config_old.yaml"
        configs_path = ROOT / "configs" / "config_old.yaml"
        root_path = ROOT / "config_old.yaml"

        self.assertTrue(configs_path.is_symlink())
        self.assertEqual(os.readlink(configs_path), "../legacy/configs/config_old.yaml")
        self.assertTrue(root_path.is_symlink())
        self.assertEqual(os.readlink(root_path), "configs/config_old.yaml")
        self.assertEqual(configs_path.resolve(), archived.resolve())
        self.assertEqual(root_path.resolve(), archived.resolve())
        self.assertEqual(sha256(archived), CONFIG_OLD_HASH)
        self.assertEqual(sha256(configs_path), CONFIG_OLD_HASH)
        self.assertEqual(sha256(root_path), CONFIG_OLD_HASH)
        self.assertEqual(yaml.safe_load(root_path.read_text()), yaml.safe_load(archived.read_text()))


if __name__ == "__main__":
    unittest.main()
