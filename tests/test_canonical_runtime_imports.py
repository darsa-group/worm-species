from __future__ import annotations

import ast
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src import utils as legacy_utils
from src.worm_species.results.writing import save_json
from src.worm_species.training.naming import make_run_name
from src.worm_species.training.naming import short_hash
from src.worm_species.training.reproducibility import set_seed


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "worm_species"


class CanonicalRuntimeContracts(unittest.TestCase):
    def test_hash_and_run_name_snapshots(self) -> None:
        config = {
            "model": {"name": "efficientnet_b0"},
            "data": {"image_col": "rel_path_seg", "target_col": "genus"},
            "training": {"lr": 0.0005},
        }
        self.assertEqual(short_hash({"a": 1}), "42b7b4f2")
        self.assertEqual(short_hash(config), "f72a53f2")
        self.assertEqual(short_hash(config, 12), "f72a53f28446")
        self.assertEqual(short_hash(config), legacy_utils.short_hash(config))
        self.assertEqual(
            make_run_name(config),
            "efficientnet_b0__rel_path_seg__genus__f72a53f2",
        )
        self.assertEqual(make_run_name(config), legacy_utils.make_run_name(config))
        with self.assertRaisesRegex(ValueError, "positive"):
            short_hash(config, 0)

    def test_json_writer_preserves_exact_bytes_and_creates_parents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            legacy_output = Path(directory) / "legacy" / "result.json"
            value = {"name": "worm", "metrics": {"f1": 0.75}}
            save_json(value, output)
            legacy_utils.save_json(value, legacy_output)
            self.assertEqual(
                output.read_bytes(),
                b'{\n  "name": "worm",\n  "metrics": {\n'
                b'    "f1": 0.75\n  }\n}\n',
            )
            self.assertEqual(output.read_bytes(), legacy_output.read_bytes())
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["name"],
                "worm",
            )

    def test_seed_reproduces_python_numpy_and_torch_sequences(self) -> None:
        set_seed(2026)
        first = (
            [random.random() for _ in range(3)],
            np.random.random(3),
            torch.rand(3),
        )
        set_seed(2026)
        second = (
            [random.random() for _ in range(3)],
            np.random.random(3),
            torch.rand(3),
        )
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

        legacy_utils.set_seed(2026)
        legacy = (
            [random.random() for _ in range(3)],
            np.random.random(3),
            torch.rand(3),
        )
        self.assertEqual(first[0], legacy[0])
        np.testing.assert_array_equal(first[1], legacy[1])
        torch.testing.assert_close(first[2], legacy[2], rtol=0, atol=0)

    def test_canonical_package_has_no_src_utils_import(self) -> None:
        offenders = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "src.utils":
                    offenders.append(str(path.relative_to(ROOT)))
                if isinstance(node, ast.Import):
                    if any(alias.name == "src.utils" for alias in node.names):
                        offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
