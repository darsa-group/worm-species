"""Execute generated notebook cells against incomplete dataset snapshots."""
from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.build_publication_dataset_notebook import build_notebook


class DatasetNotebookTests(unittest.TestCase):
    def snapshot_rows(self):
        rows = []
        for index, (year, camera, specimen, kind) in enumerate([
            ("2025", "original_camera", "Adult_1", "worm"),
            ("2025", "original_camera", "Adult_1", "worm"),
            ("2026", "gphoto2", "Adult_2", "worm"),
            ("2026", "webcam", "Adult_2", "worm"),
            ("2026", "gphoto2", "Color_Test", "calibration"),
        ]):
            barcode = "Aporrectodea_rosea_" + specimen
            rows.append({
                "image_id": str(index), "individual_id": year + ":" + barcode,
                "barcode": barcode, "year": year, "camera": camera, "kind": kind,
                "taxon": "Aporrectodea_rosea", "life_stage": "Adult",
                "location_code": "04" if year == "2026" else "", "capture_id": "session",
                "raw_path": f"{year}/{camera}/{index}.jpg", "segmented_path": f"seg/{index}.jpg",
                "segmentation_status": "pending" if kind == "worm" else "excluded_calibration",
                "original_paper_split": "train" if year == "2025" else "",
                "dataset_role": "paper_train" if year == "2025" else "external_test",
                "timestamp": year + "-09-03T12:00:00+00:00", "weight_g": "0.5",
            })
        return rows

    def run_snapshot(self, inventory_only):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata"
            metadata.mkdir()
            rows = self.snapshot_rows()
            if inventory_only:
                (metadata / "inventory.json").write_text(json.dumps({"schema": 1, "rows": rows}))
            else:
                with (metadata / "images.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            before = {p.name: p.read_bytes() for p in metadata.iterdir()}
            scope = {}
            with patch.dict(os.environ, {"PUBLICATION_DATASET_ROOT": str(root)}), \
                    patch("IPython.display.display"), \
                    patch.object(plt, "show", side_effect=lambda: plt.close("all")), \
                    redirect_stdout(io.StringIO()):
                for index, cell in enumerate(build_notebook()["cells"]):
                    if cell["cell_type"] != "code":
                        continue
                    source = cell["source"].replace("RUN_IMAGE_CHECKS = True", "RUN_IMAGE_CHECKS = False")
                    source = source.replace("SHOW_GALLERIES = True", "SHOW_GALLERIES = False")
                    exec(compile(source, f"notebook-cell-{index}", "exec"), scope)
            self.assertEqual(before, {p.name: p.read_bytes() for p in metadata.iterdir()})
            self.assertEqual(scope["snapshot"]["worm_images"], 4)
            self.assertEqual(scope["snapshot"]["unique_worms"], 2)
            self.assertEqual(scope["by_year"].loc["2026", "individuals"], 1)
            self.assertEqual(scope["coverage"]["Both cameras"], 1)
            self.assertEqual(set(scope["known_locations"].location_code), {"04"})
            self.assertTrue(scope["failed"].empty)
            self.assertEqual(scope["progress"].attempted_new.sum(), 0)
            self.assertTrue(scope["progress"].attempt_success_pct.isna().all())
            self.assertEqual(scope["checks"]["individuals_in_multiple_original_paper_splits"], 0)
            return scope

    def test_pending_snapshot_is_not_failure_and_camera_worms_are_deduplicated(self):
        scope = self.run_snapshot(inventory_only=False)
        self.assertEqual(scope["status"]["pending"].sum(), 4)

    def test_inventory_fallback_works_before_first_csv_is_published(self):
        scope = self.run_snapshot(inventory_only=True)
        self.assertEqual(scope["status"]["not_yet_published"].sum(), 4)


if __name__ == "__main__":
    unittest.main()
