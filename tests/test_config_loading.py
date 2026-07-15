from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.worm_species.config.inspect import main as inspect_main
from src.worm_species.config.loading import ConfigLoadError
from src.worm_species.config.loading import deep_merge
from src.worm_species.config.loading import load_config


ROOT = Path(__file__).resolve().parents[1]
ROOT_CONFIG = ROOT / "config.yaml"
DUAL_CUE = ROOT / "configs" / "experiments" / "dual_cue.yaml"


class ExtendedConfigurationLoadingContracts(unittest.TestCase):
    def test_root_config_inspection_reports_one_run(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = inspect_main(
                [
                    "--config",
                    str(ROOT_CONFIG),
                    "--workflow",
                    "training",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(status, 0, stderr.getvalue())
        summary = json.loads(stdout.getvalue())["summary"]
        self.assertEqual(summary["expected_model_count"], 1)
        self.assertEqual(summary["expected_sweep_combination_count"], 1)
        self.assertEqual(summary["expected_condition_count"], 1)
        self.assertEqual(summary["expected_total_run_count"], 1)

    def test_relative_inheritance_recursively_merges_without_reordering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parents = root / "parents"
            children = root / "children"
            parents.mkdir()
            children.mkdir()
            (parents / "base.yaml").write_text(
                "first:\n"
                "  retained: base\n"
                "  replaced: base\n"
                "second: base\n",
                encoding="utf-8",
            )
            (parents / "middle.yaml").write_text(
                "extends: base.yaml\n"
                "first:\n"
                "  replaced: middle\n"
                "  middle_only: true\n",
                encoding="utf-8",
            )
            child = children / "child.yaml"
            child.write_text(
                "extends: ../parents/middle.yaml\n"
                "first:\n"
                "  child_only: 3\n"
                "second: child\n"
                "third: child\n",
                encoding="utf-8",
            )

            resolved = load_config(child)

        self.assertEqual(list(resolved), ["first", "second", "third"])
        self.assertEqual(
            list(resolved["first"]),
            ["retained", "replaced", "middle_only", "child_only"],
        )
        self.assertEqual(
            resolved,
            {
                "first": {
                    "retained": "base",
                    "replaced": "middle",
                    "middle_only": True,
                    "child_only": 3,
                },
                "second": "child",
                "third": "child",
            },
        )

    def test_deep_merge_does_not_mutate_inputs(self) -> None:
        base = {"nested": {"kept": [1], "changed": "base"}}
        overlay = {"nested": {"changed": "overlay", "added": [2]}}
        before_base = copy.deepcopy(base)
        before_overlay = copy.deepcopy(overlay)

        merged = deep_merge(base, overlay)
        merged["nested"]["kept"].append(9)
        merged["nested"]["added"].append(8)

        self.assertEqual(base, before_base)
        self.assertEqual(overlay, before_overlay)

    def test_extends_cycle_is_reported_with_the_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.yaml"
            second = root / "second.yaml"
            first.write_text("extends: second.yaml\n", encoding="utf-8")
            second.write_text("extends: first.yaml\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ConfigLoadError, "Configuration extends cycle"
            ) as caught:
                load_config(first)

        message = str(caught.exception)
        self.assertIn(str(first), message)
        self.assertIn(str(second), message)

    def test_missing_parent_and_missing_root_are_clear(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child.yaml"
            child.write_text("extends: absent.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(ConfigLoadError, "Configuration not found"):
                load_config(child)
            with self.assertRaisesRegex(ConfigLoadError, "Configuration not found"):
                load_config(root / "also-absent.yaml")

    def test_non_mapping_documents_and_invalid_extends_are_rejected(self) -> None:
        cases = {
            "sequence.yaml": "- one\n- two\n",
            "scalar.yaml": "value\n",
            "empty-parent.yaml": "extends: ''\n",
            "mapping-parent.yaml": "extends:\n  file: base.yaml\n",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, contents in cases.items():
                path = root / filename
                path.write_text(contents, encoding="utf-8")
                with self.subTest(filename=filename):
                    with self.assertRaises(ConfigLoadError):
                        load_config(path)

    def test_experiment_config_loads_and_inspects_through_shared_loader(self) -> None:
        resolved = load_config(DUAL_CUE)
        self.assertNotIn("extends", resolved)
        self.assertEqual(
            resolved["sweep"]["parameters"]["model.name"],
            ["convnext_base", "vit_b_16"],
        )

        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = inspect_main(
                [
                    "--config",
                    str(DUAL_CUE),
                    "--workflow",
                    "run_specs",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(status, 0, stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["summary"]["expected_model_count"], 2)
        self.assertEqual(payload["summary"]["expected_condition_count"], 112)
        self.assertEqual(payload["summary"]["expected_total_run_count"], 224)


if __name__ == "__main__":
    unittest.main()
