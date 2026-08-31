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
from PIL import Image

from scripts.build_gbif_earthworm_dataset_notebook import build_notebook


matplotlib.use("Agg")


class GBIFDatasetOverviewNotebookTests(unittest.TestCase):
    def test_notebook_builds_distributions_and_deterministic_mosaic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_root = root / "images"
            image_root.mkdir()
            rows = []
            counter = 0
            for species, count in (
                ("Aporrectodea_longa", 4),
                ("Lumbricus_terrestris", 3),
                ("Eisenia_fetida", 2),
                ("Dendrobaena_rubida", 1),
            ):
                genus = species.split("_", 1)[0]
                for index in range(count):
                    path = image_root / f"image-{counter}.png"
                    Image.new("RGB", (32, 24), (counter * 15, 50, 90)).save(path)
                    rows.append({
                        "image_id": f"image-{counter}",
                        "gbif_id": f"occurrence-{counter // 2}",
                        "genus": genus,
                        "species_label": species,
                        "local_path": str(path),
                        "download_status": "downloaded",
                        "curation_label": "keep",
                        "country": "Denmark" if index % 2 else "Sweden",
                        "year": 2020 + index,
                        "decimal_latitude": 55 + index / 10,
                        "decimal_longitude": 10 + index / 10,
                        "split": ("train", "validation", "test")[counter % 3],
                    })
                    counter += 1
            manifest = root / "curated_manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            config_root = root / "configs"
            config_root.mkdir()
            config_path = config_root / "gbif.yaml"
            config_path.write_text(yaml.safe_dump({
                "workspace": {
                    "curated_manifest": str(manifest),
                    "downloaded_manifest": str(root / "missing-downloaded.csv"),
                    "manifest": str(root / "missing-raw.csv"),
                }
            }), encoding="utf-8")
            notebook_path = root / "overview.ipynb"
            output_root = root / "overview-output"
            build_notebook(str(config_path), notebook_path)

            notebook = nbformat.read(notebook_path, as_version=4)
            self.assertIn("dataset overview", notebook.cells[0].source.lower())
            self.assertNotIn("dataset audit", notebook.cells[0].source.lower())
            namespace = {"display": lambda _value: None}
            with mock.patch.dict(
                "os.environ", {"GBIF_DATASET_OVERVIEW_OUTPUT": str(output_root)}
            ):
                for cell in notebook.cells:
                    if cell.cell_type != "code":
                        continue
                    exec(compile(cell.source, "<gbif-dataset-overview>", "exec"), namespace)
                    namespace["plt"].close("all")

            facts = json.loads((output_root / "dataset_facts.json").read_text())
            self.assertEqual(facts["images"], 10)
            self.assertEqual(facts["species"], 4)
            self.assertEqual(facts["species_with_one_image"], 1)
            distribution = pd.read_csv(output_root / "species_distribution.csv")
            self.assertEqual(distribution["images"].tolist(), [4, 3, 2, 1])
            mosaic = pd.read_csv(output_root / "sample_mosaic_manifest.csv")
            self.assertEqual(len(mosaic), 4)
            self.assertEqual(mosaic["species_rank"].tolist(), [1, 2, 3, 4])
            self.assertTrue((output_root / "sample_mosaic.png").is_file())
            self.assertTrue((output_root / "species_distribution.png").is_file())


if __name__ == "__main__":
    unittest.main()
