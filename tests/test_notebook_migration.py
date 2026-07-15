from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TRACKED_NOTEBOOKS = {
    "notebooks/analysis/earthworm_colour_robustness_analysis.ipynb",
    "notebooks/analysis/earthworm_cue_suppression_analysis_v2.ipynb",
    "notebooks/analysis/single_task_analyze_species_outputs_v2.ipynb",
    "notebooks/analysis/worm_multitask_results_comparison.ipynb",
    "notebooks/analysis/worm_species_sweep_analysis.ipynb",
    "notebooks/data/check_splits_copy.ipynb",
    "notebooks/data/data_analysis.ipynb",
    "notebooks/data/data_explore.ipynb",
    "notebooks/data/dataset_tes.ipynb",
    "notebooks/diagnostics/data_leakage_check.ipynb",
    "notebooks/diagnostics/worm_advanced_test_diagnostics_top_models.ipynb",
    "notebooks/diagnostics/worm_same_individual_predictions_top_models.ipynb",
    "notebooks/interpretability/cam_mutlitask.ipynb",
    "notebooks/interpretability/gradcam_multitask_all_tasks.ipynb",
    "notebooks/interpretability/single_task_cam.ipynb",
    "notebooks/interpretability/single_task_umap.ipynb",
    "notebooks/interpretability/single_task_umap_species_embeddings.ipynb",
    "notebooks/interpretability/worm_gradcam_top_model_families.ipynb",
    "notebooks/interpretability/worm_umap_top_model_families.ipynb",
}

MIGRATED_NOTEBOOKS = {
    "notebooks/analysis/earthworm_colour_robustness_analysis.ipynb",
    "notebooks/analysis/earthworm_cue_suppression_analysis_v2.ipynb",
    "notebooks/analysis/single_task_analyze_species_outputs_v2.ipynb",
    "notebooks/analysis/worm_multitask_results_comparison.ipynb",
    "notebooks/analysis/worm_species_sweep_analysis.ipynb",
    "notebooks/data/dataset_tes.ipynb",
    "notebooks/diagnostics/data_leakage_check.ipynb",
    "notebooks/diagnostics/worm_advanced_test_diagnostics_top_models.ipynb",
    "notebooks/diagnostics/worm_same_individual_predictions_top_models.ipynb",
    "notebooks/interpretability/gradcam_multitask_all_tasks.ipynb",
    "notebooks/interpretability/single_task_umap_species_embeddings.ipynb",
    "notebooks/interpretability/worm_gradcam_top_model_families.ipynb",
    "notebooks/interpretability/worm_umap_top_model_families.ipynb",
}

CANONICAL_IMPORT_NOTEBOOKS = {
    "notebooks/data/dataset_tes.ipynb",
    "notebooks/diagnostics/worm_same_individual_predictions_top_models.ipynb",
    "notebooks/interpretability/gradcam_multitask_all_tasks.ipynb",
    "notebooks/interpretability/worm_gradcam_top_model_families.ipynb",
    "notebooks/interpretability/worm_umap_top_model_families.ipynb",
}

ROUTING_MARKERS = {
    "notebooks/analysis/earthworm_colour_robustness_analysis.ipynb": (
        'PROJECT_ROOT / "figures/colour_robustness"',
        'PROJECT_ROOT / "tables/colour_robustness"',
    ),
    "notebooks/analysis/earthworm_cue_suppression_analysis_v2.ipynb": (
        'PROJECT_ROOT / "figures/cue_suppression"',
        'PROJECT_ROOT / "tables/cue_suppression"',
    ),
    "notebooks/analysis/single_task_analyze_species_outputs_v2.ipynb": (
        'PROJECT_ROOT / "tables/single_task_analysis"',
    ),
    "notebooks/analysis/worm_multitask_results_comparison.ipynb": (
        'PROJECT_ROOT / "figures/multitask_results_comparison"',
        'PROJECT_ROOT / "tables/multitask_results_comparison"',
    ),
    "notebooks/analysis/worm_species_sweep_analysis.ipynb": (
        'PROJECT_ROOT / "figures/sweep_analysis"',
        'PROJECT_ROOT / "tables/sweep_analysis"',
    ),
    "notebooks/diagnostics/data_leakage_check.ipynb": (
        'PROJECT_ROOT / "figures/sweep_analysis"',
        'PROJECT_ROOT / "tables/sweep_analysis"',
    ),
    "notebooks/diagnostics/worm_advanced_test_diagnostics_top_models.ipynb": (
        'PROJECT_ROOT / "figures/advanced_test_diagnostics"',
        'PROJECT_ROOT / "tables/advanced_test_diagnostics"',
    ),
    "notebooks/diagnostics/worm_same_individual_predictions_top_models.ipynb": (
        'PROJECT_ROOT / "figures" / "same_individual_predictions"',
        'PROJECT_ROOT / "tables" / "same_individual_predictions"',
    ),
    "notebooks/interpretability/gradcam_multitask_all_tasks.ipynb": (
        'PROJECT_ROOT / "figures" / "gradcam_multitask_all_tasks"',
    ),
    "notebooks/interpretability/single_task_umap_species_embeddings.ipynb": (
        'PROJECT_ROOT / "tables" / "umap_species_embeddings"',
    ),
    "notebooks/interpretability/worm_gradcam_top_model_families.ipynb": (
        'PROJECT_ROOT / "figures" / "gradcam_top_model_families"',
        'PROJECT_ROOT / "tables" / "gradcam_top_model_families"',
    ),
    "notebooks/interpretability/worm_umap_top_model_families.ipynb": (
        'PROJECT_ROOT / "figures" / "umap_top_model_families"',
        'PROJECT_ROOT / "tables" / "umap_top_model_families"',
    ),
}

UNCHANGED_SPECIAL_CASES = {
    "notebooks/interpretability/cam_mutlitask.ipynb": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "notebooks/interpretability/single_task_umap.ipynb": (
        119,
        "9309e50d839e2ef0462015687d5d4b549e0d574d60fc36b5f7df1d7ebdaf36c1",
    ),
}


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=ROOT)


def load_notebook_bytes(raw: bytes) -> dict:
    return json.loads(raw.decode("utf-8"))


def code_source(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def filename_literals(source: str) -> Counter[str]:
    extensions = (".args", ".csv", ".err", ".json", ".npy", ".out", ".png", ".tsv", ".txt")
    values: Counter[str] = Counter()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.lower().endswith(extensions):
                values[Path(node.value).name] += 1
    return values


def dpi_values(source: str) -> Counter[int | float]:
    values: Counter[int | float] = Counter()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg == "dpi" and isinstance(keyword.value, ast.Constant):
                values[keyword.value.value] += 1
    return values


class NotebookMigrationContracts(unittest.TestCase):
    maxDiff = None

    def test_tracked_notebook_inventory_is_explicit(self) -> None:
        output = subprocess.check_output(
            ["git", "ls-files", "notebooks/*.ipynb", "notebooks/**/*.ipynb"],
            cwd=ROOT,
            text=True,
        )
        self.assertEqual(set(output.splitlines()), TRACKED_NOTEBOOKS)

    def test_valid_notebooks_are_json_and_code_cells_parse(self) -> None:
        for relative_path in sorted(TRACKED_NOTEBOOKS - {"notebooks/interpretability/cam_mutlitask.ipynb"}):
            with self.subTest(notebook=relative_path):
                notebook = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
                for index, cell in enumerate(notebook.get("cells", [])):
                    if cell.get("cell_type") == "code":
                        ast.parse("".join(cell.get("source", [])), filename=f"{relative_path}:cell-{index}")

    def test_invalid_and_empty_notebooks_remain_byte_for_byte_unchanged(self) -> None:
        for relative_path, (expected_size, expected_hash) in UNCHANGED_SPECIAL_CASES.items():
            with self.subTest(notebook=relative_path):
                raw = (ROOT / relative_path).read_bytes()
                self.assertEqual(len(raw), expected_size)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected_hash)
                self.assertEqual(raw, git_bytes(relative_path))

    def test_migration_changes_only_code_cell_source(self) -> None:
        for relative_path in sorted(MIGRATED_NOTEBOOKS):
            with self.subTest(notebook=relative_path):
                before = load_notebook_bytes(git_bytes(relative_path))
                after = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
                self.assertEqual(
                    {key: value for key, value in before.items() if key != "cells"},
                    {key: value for key, value in after.items() if key != "cells"},
                )
                self.assertEqual(len(before["cells"]), len(after["cells"]))
                for old_cell, new_cell in zip(before["cells"], after["cells"], strict=True):
                    if old_cell.get("cell_type") == "code":
                        self.assertEqual(
                            {key: value for key, value in old_cell.items() if key != "source"},
                            {key: value for key, value in new_cell.items() if key != "source"},
                        )
                    else:
                        self.assertEqual(old_cell, new_cell)

    def test_scientific_input_ids_filenames_and_dpi_are_preserved(self) -> None:
        run_id_pattern = re.compile(r"(?:node_local_sweep_\d{8}_\d{6}|run_\d+)")
        for relative_path in sorted(MIGRATED_NOTEBOOKS):
            with self.subTest(notebook=relative_path):
                old_source = code_source(load_notebook_bytes(git_bytes(relative_path)))
                new_source = code_source(json.loads((ROOT / relative_path).read_text(encoding="utf-8")))
                self.assertEqual(Counter(run_id_pattern.findall(old_source)), Counter(run_id_pattern.findall(new_source)))
                self.assertEqual(filename_literals(old_source), filename_literals(new_source))
                self.assertEqual(dpi_values(old_source), dpi_values(new_source))

    def test_canonical_imports_replace_legacy_multitask_imports(self) -> None:
        for relative_path in sorted(CANONICAL_IMPORT_NOTEBOOKS):
            with self.subTest(notebook=relative_path):
                source = code_source(json.loads((ROOT / relative_path).read_text(encoding="utf-8")))
                self.assertNotIn("from src.dataset_multitask", source)
                self.assertNotIn("from src.models", source)
                self.assertIn("src.worm_species", source)

    def test_future_generated_outputs_are_routed_outside_run_directories(self) -> None:
        for relative_path, markers in ROUTING_MARKERS.items():
            with self.subTest(notebook=relative_path):
                source = code_source(json.loads((ROOT / relative_path).read_text(encoding="utf-8")))
                for marker in markers:
                    self.assertIn(marker, source)

        all_migrated_source = "\n".join(
            code_source(json.loads((ROOT / path).read_text(encoding="utf-8")))
            for path in sorted(MIGRATED_NOTEBOOKS)
        )
        forbidden = (
            'PROJECT_ROOT = Path("~/worm-species")',
            'ANALYSIS_DIR = OUTPUTS_ROOT / "combined_sweep_analysis"',
            'OUT_DIR = RUN_DIR / "gradcam_all_tasks"',
            'cache_dir = RUN_DIR / "_embedding_cache"',
        )
        for marker in forbidden:
            self.assertNotIn(marker, all_migrated_source)


if __name__ == "__main__":
    unittest.main()
