from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
import nbformat
import pandas as pd
import yaml

from scripts.build_gbif_combined_results_notebook import build_notebook


matplotlib.use("Agg")


class GBIFCombinedResultsNotebookTests(unittest.TestCase):
    def test_baseline_inference_cells_report_row_level_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "outputs"
            inference_root = output_root / "inference" / "baseline" / "convnext_base"
            inference_root.mkdir(parents=True)
            predictions = pd.DataFrame({
                "image_id": ["image-1", "image-2", "image-3", "image-4"],
                "gbif_id": ["gbif-1", "gbif-2", "gbif-3", "gbif-4"],
                "genus": ["A", "B", "C", "D"],
                "species_label": ["A_one", "B_two", "C_three", "D_four"],
                "predicted_genus": ["A", "A", "A", "D"],
                "predicted_genus_confidence": [0.9, 0.6, 0.4, 0.8],
                "checkpoint_genus_scope": ["known", "known", "unknown", "unknown"],
                "genus_label_agreement": ["True", "False", "", ""],
                "predicted_species": ["A_one", "A_one", "A_one", "A_one"],
                "predicted_species_confidence": [0.8, 0.5, 0.7, 0.3],
                "checkpoint_species_scope": ["known", "unknown", "known", "unknown"],
                "species_label_agreement": ["True", "", "False", ""],
                "predicted_age": ["Adult", "Adult", "Juvenile", "Adult"],
                "predicted_age_confidence": [0.7, 0.8, 0.6, 0.9],
            })
            predictions_path = inference_root / "predictions.csv"
            predictions.to_csv(predictions_path, index=False)
            predictions_path.with_suffix(".summary.json").write_text(json.dumps({
                "rows": 4,
                "shard_count": 12,
                "coverage_validated": True,
            }))
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump({
                "paths": {"output_root": str(output_root)},
                "models": {"primary": ["convnext_base"]},
                "reporting": {"formats": ["png"]},
            }))
            notebook_path = root / "report.ipynb"
            build_notebook(str(config_path), notebook_path)

            notebook = nbformat.read(notebook_path, as_version=4)
            namespace = {"display": lambda _value: None}
            markers = (
                "from pathlib import Path",
                "inference_frames =",
                "coverage_rows =",
                "prediction_distribution_rows =",
                "if not inference.empty:",
            )
            for marker in markers:
                source = next(
                    cell.source for cell in notebook.cells
                    if cell.cell_type == "code" and cell.source.startswith(marker)
                )
                exec(compile(source, f"<notebook:{marker}>", "exec"), namespace)

            metrics = pd.read_csv(output_root / "combined_report" / "baseline_inference_metrics.csv")
            self.assertEqual(metrics["task"].tolist(), ["genus", "species"])
            self.assertEqual(metrics["known_coverage"].tolist(), [0.5, 0.5])
            self.assertEqual(metrics["known_label_agreement"].tolist(), [0.5, 0.5])
            self.assertEqual(metrics["majority_accuracy"].tolist(), [0.5, 0.5])
            self.assertEqual(metrics["balanced_accuracy"].tolist(), [0.5, 0.5])
            self.assertEqual(metrics["balanced_chance_1_over_k"].tolist(), [0.5, 0.5])
            self.assertEqual(metrics["balanced_accuracy_minus_chance"].tolist(), [0.0, 0.0])
            self.assertTrue(
                (output_root / "combined_report" / "baseline_known_class_recall.csv").is_file()
            )
            self.assertTrue(
                (output_root / "combined_report" / "baseline_prediction_distribution.csv").is_file()
            )
            self.assertTrue(
                (output_root / "combined_report" / "baseline_confidence_summary.csv").is_file()
            )
            self.assertTrue(
                (output_root / "combined_report" / "baseline_representative_predictions.csv").is_file()
            )
            self.assertTrue(
                (output_root / "combined_report" / "figures" / "baseline_inference_confidence.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
