from __future__ import annotations

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from scripts import build_publication_dataset as dataset
from segment.inference_coco import process_file_two_stage_rotated_square_mask_vis_largest_cc


def table(path, rows):
    dataset.write_csv(path, rows)


def picture(path, size=(48, 32)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (100, 130, 160)).save(path)


class FakeYOLO:
    """Deterministic two-stage detections to test masks and output bookkeeping."""
    def __init__(self):
        self.calls = 0

    def to(self, device):
        return self

    def predict(self, image, **kwargs):
        self.calls += 1
        width, height = image.size
        polygon = np.array([[width*.3, height*.3], [width*.7, height*.3],
                            [width*.7, height*.7], [width*.3, height*.7]], dtype=np.float32)
        mask = torch.zeros((1, 32, 32))
        mask[:, 8:24, 8:24] = 1

        class Boxes:
            xyxy = torch.tensor([[width*.3, height*.3, width*.7, height*.7]])
            cls = torch.tensor([0])

            def __len__(self):
                return 1

        class Masks:
            xy = [polygon]
            data = mask

            def __len__(self):
                return 1

        return [SimpleNamespace(boxes=Boxes(), masks=Masks())]


class PublicationDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old = self.root / "old"
        self.new = self.root / "new"
        self.splits = self.root / "splits"
        self.output = self.root / "combined"
        old_name = "Aporrectodea_rosea_Adult_1"
        raw = self.old / "00_RawData" / old_name / "image.jpg"
        segmented = self.old / "01_Segmented" / old_name / "image_seg.jpg"
        mask = segmented.with_suffix(".png")
        picture(raw)
        picture(segmented, (20, 20))
        picture(mask)
        old_meta = {"barcode": old_name, "filename": raw.name, "timestamp": "2025-09-01",
                    "rel_path_seg": str(segmented.relative_to(self.old)),
                    "rel_path_segmask": str(mask.relative_to(self.old))}
        table(self.old / "01_Segmented/global_metadata.csv", [old_meta])
        for split in ("train", "val", "test"):
            dataset.write_csv(self.splits / f"{split}_split.csv", [old_meta] if split == "train" else [], list(old_meta))
        captures = []
        for camera, age, barcode in [
            ("gphoto2", "Adult", "Aporrectodea_rosea_Adult_2"),
            ("webcam", "Adult", "Aporrectodea_rosea_Adult_2"),
            ("gphoto2", "Test", "Color_Test_01"),
        ]:
            path = self.new / camera / barcode / "session" / "image.jpg"
            picture(path)
            captures.append({"barcode": barcode, "image_path": str(path.relative_to(self.new)),
                             "storage_folder": camera, "capture_id": "session", "timestamp": "2026-09-03",
                             "location_barcode": "04", "age": age, "sha256": dataset.digest(path)})
        table(self.new / "captures.csv", captures)

    def rows(self):
        return dataset.inventory(self.old, self.new, self.splits)

    def prepare(self):
        rows = self.rows()
        dataset.prepare(self.output, rows, self.old, self.new, self.splits)
        return rows

    def test_inventory_and_per_folder_csv_preserve_identity_and_locations(self):
        rows = self.prepare()
        self.assertEqual(len(rows), 4)
        newer = [r for r in rows if r["kind"] == "worm" and r["year"] == "2026"]
        self.assertEqual(len({r["individual_id"] for r in newer}), 1)
        self.assertEqual(len({r["image_id"] for r in newer}), 2)
        for camera in ("gphoto2", "webcam"):
            folder = self.output / "2026" / camera / "00_RawData"
            raw_rows = dataset.read_csv(folder / "metadata.csv")
            self.assertEqual(raw_rows[0]["location_code"], "04")
            self.assertTrue((folder / raw_rows[0]["image_path"]).is_file())
            self.assertEqual(dataset.read_csv(folder.parent / "01_Segmented/metadata.csv"), [])
        old = dataset.read_csv(self.output / "2025/original_camera/01_Segmented/metadata.csv")[0]
        self.assertEqual(old["original_paper_split"], "train")
        self.assertTrue(old["rel_path_rawmask"])
        self.assertFalse(old["rel_path_segmask"])
        for row in rows:
            self.assertEqual(dataset.digest(row["source_raw"]), dataset.digest(self.output / row["raw_path"]))
        self.assertEqual(dataset.read_csv(self.output / "metadata/original_paper_splits/train_split.csv"),
                         dataset.read_csv(self.splits / "train_split.csv"))

    def test_duplicate_and_escaping_paths_fail_before_copy(self):
        captures = dataset.read_csv(self.new / "captures.csv")
        table(self.new / "captures.csv", captures + captures[:1])
        with self.assertRaisesRegex(ValueError, "Duplicate capture"):
            self.rows()
        captures[0]["image_path"] = "../elsewhere.jpg"
        table(self.new / "captures.csv", captures)
        with self.assertRaisesRegex(ValueError, "escapes source"):
            self.rows()
        self.assertFalse(self.output.exists())

    def test_prepare_resumes_and_refuses_overwriting_changed_files(self):
        rows = self.prepare()
        dataset.prepare(self.output, rows, self.old, self.new, self.splits)
        destination = self.output / rows[0]["raw_path"]
        destination.write_bytes(b"user change")
        with self.assertRaisesRegex(ValueError, "differs from source"):
            dataset.prepare(self.output, rows, self.old, self.new, self.splits)
        self.assertEqual(destination.read_bytes(), b"user change")
        self.assertFalse((self.output / "metadata/prepared.json").exists())

    def test_segmentation_has_aligned_masks_and_resumes_without_inference(self):
        rows = self.prepare()
        weights = self.root / "weights.pt"
        weights.write_bytes(b"fixture weights")
        fake = FakeYOLO()
        with patch("ultralytics.YOLO", return_value=fake):
            self.assertEqual(dataset.segment(self.output, rows, weights, "cpu", 1), 0)
            self.assertEqual(fake.calls, 4)
            self.assertEqual(dataset.segment(self.output, rows, weights, "cpu", 1), 0)
            self.assertEqual(fake.calls, 4)
        for row in rows:
            if row["year"] != "2026" or row["kind"] != "worm":
                continue
            with Image.open(self.output / row["segmented_path"]) as rgb, \
                    Image.open(self.output / row["crop_mask_path"]) as mask:
                self.assertEqual(rgb.size, mask.size)
            csv_rows = dataset.read_csv(self.output / row["year"] / row["camera"] / "01_Segmented/metadata.csv")
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["segmentation_status"], "segmented")

    def test_no_detection_is_reported_and_excluded_from_segmented_csv(self):
        rows = self.prepare()
        weights = self.root / "weights.pt"
        weights.write_bytes(b"fixture weights")
        fake = FakeYOLO()
        fake.predict = lambda *a, **kw: [SimpleNamespace(boxes=None)]
        with patch("ultralytics.YOLO", return_value=fake):
            self.assertEqual(dataset.segment(self.output, rows, weights, "cpu", 1), 2)
        status = dataset.read_csv(self.output / "metadata/segmentation_status.csv")
        self.assertEqual(sum(r["segmentation_status"] == "failed" for r in status), 2)
        self.assertEqual(dataset.read_csv(self.output / "2026/gphoto2/01_Segmented/metadata.csv"), [])

    def test_square_margin_is_respected_and_masks_keep_raw_coordinates(self):
        source = self.new / "gphoto2/Aporrectodea_rosea_Adult_2/session/image.jpg"
        ok, info = process_file_two_stage_rotated_square_mask_vis_largest_cc(
            str(source), str(self.root / "sample.jpg"), FakeYOLO().predict,
            pad_px=0, sq_margin=10)
        self.assertTrue(ok)
        self.assertEqual(info["mask_on_orig"].shape, (32, 48))
        self.assertLess(info["mask_on_crop"].shape[0], 100)


if __name__ == "__main__":
    unittest.main()
