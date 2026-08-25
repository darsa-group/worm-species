from __future__ import annotations

import json
import hashlib
import io
import argparse
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image

from src.worm_species.gbif.pipeline import build_download_request
from src.worm_species.gbif.pipeline import build_media_manifest
from src.worm_species.gbif.pipeline import download_one_image
from src.worm_species.gbif.pipeline import resolve_manifest_image_path
from src.worm_species.gbif.pipeline import label_overlap_audit
from src.worm_species.gbif.pipeline import load_pipeline_config
from src.worm_species.gbif.pipeline import prune_missing_image_rows
from src.worm_species.gbif.pipeline import filter_active_manifest_by_dataset
from src.worm_species.gbif.embedding import cluster_embeddings
from src.worm_species.gbif.inference import infer_existing_checkpoint
from src.worm_species.gbif.transfer import prepare_transfer_bundle
from src.worm_species.gbif.transfer import validate_transfer_bundle
from scripts.gbif_oligochaeta_pipeline import command_images


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gbif_oligochaeta.yaml"


class GBIFOligochaetaTests(unittest.TestCase):
    def test_download_request_is_order_scoped_and_requires_genus(self) -> None:
        config = load_pipeline_config(CONFIG)
        self.assertTrue(all(
            set(order) == {"key", "name", "reason"}
            for order in config["gbif"]["explicitly_excluded_orders"]
        ))
        request = build_download_request(config, email="researcher@example.org")
        predicates = request["predicate"]["predicates"]
        self.assertEqual(request["format"], "DWCA")
        self.assertEqual(predicates[0]["key"], "ORDER_KEY")
        self.assertNotIn("255", predicates[0]["values"])
        self.assertEqual(set(predicates[0]["values"]), {"5958860", "11229348"})
        self.assertIn(
            {"type": "equals", "key": "MEDIA_TYPE", "value": "StillImage"},
            predicates,
        )
        self.assertIn(
            {"type": "isNotNull", "parameter": "GENUS_KEY"}, predicates
        )
        self.assertIn(
            {
                "type": "equals",
                "key": "DATASET_KEY",
                "value": "50c9509d-22c7-4a22-a47d-8c48425ef4a7",
            },
            predicates,
        )

    def test_manifest_joins_media_and_rechecks_genus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            occurrence = pd.DataFrame([
                {
                    "gbifID": "1", "occurrenceID": "occ-1", "datasetKey": "d1",
                    "basisOfRecord": "HUMAN_OBSERVATION", "scientificName": "Lumbricus terrestris",
                    "taxonKey": "100", "taxonRank": "SPECIES", "order": "Crassiclitellata",
                    "orderKey": "5958860", "family": "Lumbricidae", "familyKey": "10",
                    "genus": "Lumbricus", "genusKey": "20", "species": "Lumbricus terrestris",
                    "speciesKey": "100", "license": "CC_BY_4_0",
                },
                {
                    "gbifID": "2", "scientificName": "Unknown worm",
                    "genus": "", "genusKey": "",
                },
            ])
            media = pd.DataFrame([
                {"coreid": "1", "identifier": "https://example.org/a.jpg", "type": "StillImage", "format": "image/jpeg", "creator": "A"},
                {"coreid": "1", "identifier": "https://example.org/a-detail.jpg", "type": "StillImage", "format": "image/jpeg", "creator": "A"},
                {"coreid": "1", "identifier": "https://example.org/audio.mp3", "type": "Sound", "format": "audio/mpeg"},
                {"coreid": "2", "identifier": "https://example.org/b.jpg", "type": "StillImage", "format": "image/jpeg"},
            ])
            occurrence_path = root / "occurrence.txt"
            media_path = root / "multimedia.txt"
            occurrence.to_csv(occurrence_path, sep="\t", index=False)
            media.to_csv(media_path, sep="\t", index=False)
            archive_path = root / "fixture.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.write(occurrence_path, occurrence_path.name)
                archive.write(media_path, media_path.name)
            output = root / "manifest.csv"
            summary = build_media_manifest(archive_path, output)
            manifest = pd.read_csv(output, dtype=str, keep_default_na=False)
            self.assertEqual(len(manifest), 2)
            self.assertEqual(manifest.iloc[0]["genus"], "Lumbricus")
            self.assertEqual(manifest.iloc[0]["species_label"], "Lumbricus_terrestris")
            self.assertEqual(summary["skipped"]["missing_genus"], 1)
            self.assertEqual(summary["skipped"]["not_still_image"], 1)
            self.assertEqual(summary["manifest_rows"], 2)

    def test_overlap_audit_keeps_unknown_taxa_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.csv"
            pd.DataFrame([
                {"image_id": "1", "genus": "Lumbricus", "species_label": "Lumbricus_terrestris"},
                {"image_id": "2", "genus": "Lumbricus", "species_label": "Lumbricus_rubellus"},
                {"image_id": "3", "genus": "Eisenia", "species_label": "Eisenia_fetida"},
            ]).to_csv(manifest, index=False)
            labels = root / "labels.json"
            labels.write_text(json.dumps({
                "genus": {"Lumbricus": 0},
                "species": {"Lumbricus_terrestris": 0},
            }))
            summary = label_overlap_audit(manifest, labels)
            self.assertEqual(summary["scope_counts"], {
                "known_genus_only": 1,
                "known_species": 1,
                "unknown_genus": 1,
            })

    def test_cluster_artifact_keeps_embedding_row_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            embeddings = root / "embeddings.npy"
            points = [[float(index), 0.0, 1.0] for index in range(8)]
            import numpy as np
            np.save(embeddings, np.asarray(points, dtype=np.float32))
            index_path = root / "index.csv"
            pd.DataFrame({
                "image_id": [f"image-{index}" for index in range(8)],
                "embedding_row": list(range(8)),
            }).to_csv(index_path, index=False)
            output = root / "clusters.csv"
            summary = cluster_embeddings(
                embeddings,
                index_path,
                output,
                seed=42,
                pca_dimensions=3,
                projection="pca",
                min_cluster_size=3,
                min_samples=2,
            )
            result = pd.read_csv(output, dtype=str, keep_default_na=False)
            self.assertEqual(result["image_id"].tolist(), [f"image-{index}" for index in range(8)])
            self.assertEqual(summary["rows"], 8)

    def test_existing_inference_reports_agreement_not_accuracy(self) -> None:
        class TinyModel(torch.nn.Module):
            def forward(self, images):
                rows = len(images)
                return {
                    "genus": torch.tensor([[0.0, 2.0]]).repeat(rows, 1),
                    "species": torch.tensor([[0.0, 2.0]]).repeat(rows, 1),
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "image.png"
            Image.new("RGB", (16, 16), "brown").save(image)
            manifest = root / "manifest.csv"
            pd.DataFrame([{
                "image_id": "image-1",
                "local_path": str(image),
                "download_status": "downloaded",
                "curation_label": "keep",
                "genus": "Lumbricus",
                "species_label": "Lumbricus_terrestris",
            }]).to_csv(manifest, index=False)
            checkpoint = root / "checkpoint.pt"
            torch.save({
                "cfg": {
                    "model": {"name": "tiny", "pretrained": True},
                    "preprocessing": {"image_size": 16, "normalisation": {"enabled": False}},
                    "augmentation": {"enabled": False},
                },
                "model_state": {},
                "label_to_index_by_task": {
                    "genus": {"Eisenia": 0, "Lumbricus": 1},
                    "species": {"Eisenia_fetida": 0, "Lumbricus_terrestris": 1},
                },
                "best_epoch": 3,
                "selection_metric": "loss",
            }, checkpoint)
            output = root / "predictions.csv"
            with patch(
                "src.worm_species.gbif.inference.build_multitask_model",
                return_value=TinyModel(),
            ):
                summary = infer_existing_checkpoint(
                    manifest, checkpoint, output, batch_size=1, num_workers=0, device_name="cpu"
                )
            self.assertEqual(summary["genus_label_agreement"], 1.0)
            self.assertEqual(summary["species_label_agreement"], 1.0)
            self.assertIn("not an independently verified accuracy", summary["interpretation"])

    def test_repeated_source_url_is_downloaded_once_but_keeps_all_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "media.csv"
            output = root / "downloaded.csv"
            image_dir = root / "images"
            pd.DataFrame([
                {"image_id": "image-1", "gbif_id": "occ-1", "source_url": "https://example.org/shared.jpg"},
                {"image_id": "image-2", "gbif_id": "occ-2", "source_url": "https://example.org/shared.jpg"},
            ]).to_csv(manifest, index=False)
            calls = []

            def fake_download(row, destination, **kwargs):
                calls.append(row["source_url"])
                path = Path(destination) / "shared.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), "brown").save(path)
                return {
                    **row,
                    "download_status": "downloaded",
                    "local_path": str(path),
                    "sha256": "a" * 64,
                    "dhash": "0" * 16,
                    "width": "8",
                    "height": "8",
                    "content_type": "image/jpeg",
                    "bytes": str(path.stat().st_size),
                    "error": "",
                }

            config = {
                "images": {
                    "workers": 2,
                    "require_complete": True,
                    "attempts": 1,
                    "retry_backoff_seconds": 0,
                    "max_bytes": 100000,
                    "connect_timeout_seconds": 1,
                    "read_timeout_seconds": 1,
                    "user_agent": "test",
                }
            }
            args = argparse.Namespace(
                manifest=str(manifest), output=str(output),
                image_dir=str(image_dir), workers=2,
            )
            with patch(
                "scripts.gbif_oligochaeta_pipeline.download_one_image",
                side_effect=fake_download,
            ):
                command_images(config, args)
            result = pd.read_csv(output, dtype=str, keep_default_na=False)
            self.assertEqual(calls, ["https://example.org/shared.jpg"])
            self.assertEqual(result["image_id"].tolist(), ["image-1", "image-2"])
            self.assertEqual(result["local_path"].nunique(), 1)

    def test_image_download_retries_http_429(self) -> None:
        class FakeResponse:
            def __init__(self, status, content=b"", headers=None):
                self.status_code = status
                self.content = content
                self.headers = headers or {}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def raise_for_status(self):
                if self.status_code >= 400:
                    import requests
                    raise requests.HTTPError(f"status {self.status_code}")

            def iter_content(self, chunk_size):
                yield self.content

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "brown").save(buffer, format="JPEG")
        responses = [
            FakeResponse(429, headers={"Retry-After": "0"}),
            FakeResponse(200, buffer.getvalue(), {"Content-Type": "image/jpeg"}),
        ]
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "src.worm_species.gbif.pipeline.requests.get", side_effect=responses
        ) as get, patch("src.worm_species.gbif.pipeline.time.sleep"):
            result = download_one_image(
                {"image_id": "image-1", "source_url": "https://example.org/a.jpg"},
                temp_dir,
                max_bytes=100000,
                connect_timeout=1,
                read_timeout=1,
                user_agent="test",
                attempts=2,
                retry_backoff_seconds=0,
            )
        self.assertEqual(get.call_count, 2)
        self.assertEqual(result["download_status"], "downloaded")

    def test_transfer_bundle_is_portable_and_checksum_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "gbif_oligochaeta"
            manifests = bundle / "manifests"
            images = bundle / "images"
            download = bundle / "download"
            manifests.mkdir(parents=True)
            images.mkdir()
            download.mkdir()
            image = images / "shared.jpg"
            Image.new("RGB", (8, 8), "brown").save(image)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            relative = "gbif_oligochaeta/images/shared.jpg"
            media = pd.DataFrame([
                {"image_id": "one", "gbif_id": "occ-1", "source_url": "https://example.org/shared.jpg", "genus": "Lumbricus", "genus_key": "1", "dataset_key": "inat"},
                {"image_id": "two", "gbif_id": "occ-2", "source_url": "https://example.org/shared.jpg", "genus": "Lumbricus", "genus_key": "1", "dataset_key": "inat"},
            ])
            downloaded = media.assign(
                download_status="downloaded", local_path=relative, sha256=digest
            )
            media.to_csv(manifests / "media_manifest.csv", index=False)
            downloaded.to_csv(manifests / "downloaded_manifest.csv", index=False)
            (download / "occurrence.zip").write_bytes(b"fixture")

            resolved = resolve_manifest_image_path(
                manifests / "downloaded_manifest.csv", relative
            )
            self.assertEqual(resolved, image)
            checked = validate_transfer_bundle(bundle, verify_image_hashes=True)
            self.assertEqual(checked["media_rows"], 2)
            self.assertEqual(checked["unique_image_files"], 1)
            prepared = prepare_transfer_bundle(bundle)
            self.assertTrue(prepared["image_hashes_verified"])
            sums = (bundle / "transfer" / "SHA256SUMS").read_text()
            self.assertIn("images/shared.jpg", sums)

    def test_transfer_bundle_rejects_partial_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir)
            manifests = bundle / "manifests"
            manifests.mkdir()
            media = pd.DataFrame([{
                "image_id": "one", "gbif_id": "occ-1",
                "source_url": "https://example.org/a.jpg",
                "genus": "Lumbricus", "genus_key": "1", "dataset_key": "inat",
            }])
            media.to_csv(manifests / "media_manifest.csv", index=False)
            media.head(0).assign(
                download_status=pd.Series(dtype=str),
                local_path=pd.Series(dtype=str),
                sha256=pd.Series(dtype=str),
            ).to_csv(manifests / "downloaded_manifest.csv", index=False)
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                validate_transfer_bundle(bundle)

    def test_prune_missing_images_retains_exclusion_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "gbif_oligochaeta"
            manifests = bundle / "manifests"
            images = bundle / "images"
            manifests.mkdir(parents=True)
            images.mkdir()
            image = images / "one.jpg"
            Image.new("RGB", (8, 8), "brown").save(image)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            media = pd.DataFrame([
                {"image_id": "one", "gbif_id": "occ-1", "source_url": "https://example.org/one.jpg", "genus": "Lumbricus", "genus_key": "1", "dataset_key": "inat"},
                {"image_id": "two", "gbif_id": "occ-2", "source_url": "https://example.org/two.jpg", "genus": "Lumbricus", "genus_key": "1", "dataset_key": "inat"},
            ])
            downloaded = media.copy()
            downloaded["download_status"] = ["downloaded", "failed"]
            downloaded["local_path"] = ["gbif_oligochaeta/images/one.jpg", ""]
            downloaded["sha256"] = [digest, ""]
            downloaded["error"] = ["", "HTTPError: 404"]
            media_path = manifests / "media_manifest.csv"
            active_path = manifests / "downloaded_manifest.csv"
            excluded_path = manifests / "excluded_missing_images.csv"
            media.to_csv(media_path, index=False)
            downloaded.to_csv(active_path, index=False)

            preview = prune_missing_image_rows(active_path, excluded_path)
            self.assertFalse(preview["apply"])
            self.assertEqual(preview["retained_rows"], 1)
            self.assertEqual(preview["excluded_rows"], 1)
            self.assertEqual(len(pd.read_csv(active_path)), 2)

            applied = prune_missing_image_rows(
                active_path, excluded_path, apply=True
            )
            self.assertTrue(applied["apply"])
            self.assertEqual(len(pd.read_csv(active_path)), 1)
            excluded = pd.read_csv(excluded_path, dtype=str, keep_default_na=False)
            self.assertEqual(excluded["image_id"].tolist(), ["two"])
            self.assertEqual(excluded["exclusion_reason"].tolist(), ["download_failed"])
            validated = validate_transfer_bundle(bundle)
            self.assertEqual(validated["active_media_rows"], 1)
            self.assertEqual(validated["excluded_media_rows"], 1)

    def test_dataset_filter_keeps_only_selected_rows_and_audits_others(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "downloaded_manifest.csv"
            excluded = root / "excluded_outside_dataset.csv"
            pd.DataFrame([
                {"image_id": "one", "gbif_id": "occ-1", "dataset_key": "inat", "local_path": "images/one.jpg"},
                {"image_id": "two", "gbif_id": "occ-2", "dataset_key": "other", "local_path": "images/two.jpg"},
            ]).to_csv(manifest, index=False)
            preview = filter_active_manifest_by_dataset(
                manifest, "inat", "iNaturalist", excluded
            )
            self.assertEqual(preview["retained_rows"], 1)
            self.assertEqual(preview["excluded_rows"], 1)
            filter_active_manifest_by_dataset(
                manifest, "inat", "iNaturalist", excluded, apply=True
            )
            self.assertEqual(pd.read_csv(manifest)["image_id"].tolist(), ["one"])
            rejected = pd.read_csv(excluded)
            self.assertEqual(rejected["image_id"].tolist(), ["two"])


if __name__ == "__main__":
    unittest.main()
