from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import nbformat
import pandas as pd
import yaml

from scripts.build_gbif_full_taxonomy_status_notebook import build_notebook
from scripts.gbif_full_taxonomy_pipeline import build_specs


matplotlib.use("Agg")


class FullTaxonomyStatusNotebookTests(unittest.TestCase):
    def test_notebook_reports_complete_resumable_and_pending_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_root = root / "gbif_full_taxonomy_results"
            config_root = root / "configs"
            config_root.mkdir()
            config_path = config_root / "full.yaml"
            config = {
                "paths": {"experiment_root": str(root / "configured-results")},
                "models": {
                    "backbones": ["convnext_base", "vit_b_16", "resnet50"],
                    "seeds": [40, 140, 240],
                },
                "training": {
                    "petri_steps": 1000,
                    "gbif_steps": 2000,
                    "revised_hierarchy": {"weight": 0.5},
                },
            }
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            configured = dict(config)
            configured["paths"] = {"experiment_root": str(result_root)}
            specs = build_specs(configured)

            def complete(spec: dict, species_f1: float) -> None:
                output = Path(spec["output_dir"])
                output.mkdir(parents=True, exist_ok=True)
                (output / "best_model.pt").write_bytes(b"checkpoint")
                (output / "run_status.json").write_text(json.dumps({
                    "status": "complete",
                    "best_validation_species_macro_f1": species_f1,
                    "best_validation_genus_macro_f1": species_f1 + 0.1,
                }), encoding="utf-8")
                (output / "test_metrics.json").write_text(json.dumps({
                    "loss": 0.5,
                    "genus": {
                        "n": 10, "accuracy": 0.8,
                        "balanced_accuracy": 0.7, "macro_f1": 0.72,
                    },
                    "species": {
                        "n": 10, "accuracy": 0.6,
                        "balanced_accuracy": 0.5, "macro_f1": species_f1,
                    },
                }), encoding="utf-8")

            complete(specs["petri"][0], 0.55)
            complete(specs["primary"][0], 0.65)

            resumable = Path(specs["primary"][1]["output_dir"])
            resumable.mkdir(parents=True)
            (resumable / "last_model.pt").write_bytes(b"last")
            (resumable / "history.jsonl").write_text(
                json.dumps({"step": 400, "train_loss": 1.2}) + "\n",
                encoding="utf-8",
            )
            started = Path(specs["hierarchy"][2]["output_dir"])
            started.mkdir(parents=True)
            (started / "history.jsonl").write_text(
                json.dumps({"step": 20, "train_loss": 2.0}) + "\n",
                encoding="utf-8",
            )

            inference_output = (
                result_root / "inference" / f"{specs['primary'][0]['run_id']}.csv.gz"
            )
            inference_output.parent.mkdir(parents=True)
            inference_output.write_bytes(b"predictions")
            inference_output.with_suffix("").with_suffix(".summary.json").write_text(
                json.dumps({"status": "complete", "rows": 25}), encoding="utf-8"
            )
            (result_root / "final_report.md").write_text("# Results\n", encoding="utf-8")
            (result_root / "final_manifest.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            generated = result_root / "generated"
            generated.mkdir()
            (generated / "submission_receipt.json").write_text(
                json.dumps({"hierarchy_job_id": "123"}), encoding="utf-8"
            )

            notebook_path = root / "status.ipynb"
            status_output = root / "status-output"
            build_notebook(str(config_path), notebook_path)
            notebook = nbformat.read(notebook_path, as_version=4)
            namespace = {"display": lambda _value: None}
            with mock.patch.dict("os.environ", {
                "GBIF_FULL_TAXONOMY_ROOT": str(result_root),
                "GBIF_FULL_TAXONOMY_STATUS_OUTPUT": str(status_output),
            }):
                for cell in notebook.cells:
                    if cell.cell_type != "code":
                        continue
                    exec(compile(cell.source, "<full-taxonomy-status>", "exec"), namespace)
                    namespace["plt"].close("all")

            training = pd.read_csv(status_output / "training_run_status.csv")
            self.assertEqual(len(training), 45)
            self.assertEqual(training["artifact_status"].value_counts()["complete"], 2)
            self.assertEqual(training["artifact_status"].value_counts()["resumable"], 1)
            self.assertEqual(training["artifact_status"].value_counts()["started"], 1)
            inference = pd.read_csv(status_output / "inference_status.csv")
            self.assertEqual(len(inference), 36)
            self.assertEqual(inference["status"].value_counts()["complete"], 1)
            summary = json.loads((status_output / "status_summary.json").read_text())
            self.assertEqual(summary["training_complete"], 2)
            self.assertEqual(summary["training_resumable"], 1)
            self.assertEqual(summary["inference_complete"], 1)
            self.assertTrue(summary["final_report_complete"])
            self.assertTrue((status_output / "training_completion.png").is_file())


if __name__ == "__main__":
    unittest.main()
