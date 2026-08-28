from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image

from scripts.gbif_full_taxonomy_pipeline import build_specs, render_pipeline
from src.worm_species.gbif.full_taxonomy import (
    canonical_taxonomy, coverage_aware_split, load_full_taxonomy_config,
    run_full_taxonomy_audit,
)
from src.worm_species.training.losses import (
    build_child_to_parent_matrix, ground_truth_taxonomic_mass_loss,
    hierarchy_consistency_loss,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "gbif_full_taxonomy.yaml"


class FullTaxonomyTests(unittest.TestCase):
    def test_canonical_taxonomy_never_maps_into_petri_vocabulary(self) -> None:
        self.assertEqual(
            canonical_taxonomy("Lumbricus", "Lumbricus terrestris"),
            ("Lumbricus", "Lumbricus_terrestris", "valid"),
        )
        self.assertEqual(
            canonical_taxonomy("Aporrectodea", "Nicodrilus_gognus")[2],
            "species_genus_mismatch",
        )
        self.assertEqual(canonical_taxonomy("Eisenia", "")[2], "missing_species_label")

    def test_coverage_split_obeys_three_two_one_group_contract(self) -> None:
        rows = []
        for species, count in (("G_a", 3), ("G_b", 2), ("G_c", 1)):
            for index in range(count):
                rows.append({"group_id": f"{species}-{index}", "genus": "G", "species": species})
        frame = pd.DataFrame(rows)
        config = load_full_taxonomy_config(BASE_CONFIG)
        split = coverage_aware_split(frame, config)
        frame["split"] = split
        observed = {
            species: set(group["split"])
            for species, group in frame.groupby("species")
        }
        self.assertEqual(observed["G_a"], {"train", "validation", "test"})
        self.assertEqual(observed["G_b"], {"train", "test"})
        self.assertEqual(observed["G_c"], {"train"})

    def test_current_symmetric_loss_can_pull_genus_towards_wrong_species(self) -> None:
        maps = {"genus": {"G": 0, "H": 1}, "species": {"G_a": 0, "H_b": 1}}
        matrix = build_child_to_parent_matrix(maps, "genus", "species", torch.device("cpu"))
        parent = torch.tensor([[5.0, -5.0]], requires_grad=True)
        child = torch.tensor([[-5.0, 5.0]], requires_grad=True)
        loss = hierarchy_consistency_loss(parent, child, matrix, torch.tensor([True]))
        loss.backward()
        self.assertGreater(parent.grad[0, 0].item(), 0.0)
        self.assertLess(parent.grad[0, 1].item(), 0.0)
        self.assertTrue(torch.isfinite(parent.grad).all())
        self.assertTrue(torch.isfinite(child.grad).all())

    def test_ground_truth_loss_updates_species_mass_not_genus_head(self) -> None:
        maps = {"genus": {"G": 0, "H": 1}, "species": {"G_a": 0, "H_b": 1}}
        matrix = build_child_to_parent_matrix(maps, "genus", "species", torch.device("cpu"))
        child = torch.tensor([[-5.0, 5.0]], dtype=torch.float16, requires_grad=True)
        loss = ground_truth_taxonomic_mass_loss(
            child, torch.tensor([0]), matrix, torch.tensor([True])
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertLess(child.grad[0, 0].item(), 0.0)
        self.assertGreater(child.grad[0, 1].item(), 0.0)
        self.assertTrue(torch.isfinite(child.grad).all())

    def test_rendered_pipeline_has_required_stage_counts_and_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_full_taxonomy_config(BASE_CONFIG)
            config["paths"]["experiment_root"] = str(Path(temp_dir) / "gbif_full_taxonomy_test")
            specs = build_specs(config)
            manifest = render_pipeline(config, BASE_CONFIG.resolve())
            self.assertEqual(len(specs["petri"]), 9)
            self.assertEqual(len(specs["primary"]), 18)
            self.assertEqual(len(specs["hierarchy"]), 18)
            self.assertEqual(manifest["inference_task_count"], 36)
            audit = Path(manifest["scripts"]["audit"]).read_text()
            primary = Path(manifest["scripts"]["primary"]).read_text()
            report = Path(manifest["scripts"]["report"]).read_text()
            self.assertIn("#SBATCH --cpus-per-task=128", audit)
            self.assertIn("#SBATCH --array=0-17%12", primary)
            self.assertIn("#SBATCH --cpus-per-task=128", report)
            self.assertIn("miniforge3", report)

    def test_synthetic_audit_reports_trainable_and_evaluable_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            images.mkdir()
            manifest_rows = []
            counter = 0
            for species, groups in (("G_a", 3), ("G_b", 2), ("G_c", 1)):
                for group in range(groups):
                    path = images / f"{counter}.png"
                    Image.new("RGB", (8, 8), (counter, 20, 30)).save(path)
                    manifest_rows.append({
                        "image_id": f"image-{counter}", "gbif_id": f"gbif-{counter}",
                        "occurrence_id": f"occ-{counter}", "local_path": str(path),
                        "genus": "G", "species_label": species,
                        "download_status": "downloaded", "curation_label": "keep",
                        "sha256": f"sha-{counter}", "dhash": f"dhash-{counter}",
                        "source_url": f"https://example/{counter}",
                        "media_reference": f"https://example/media/{counter}",
                    })
                    counter += 1
            for species in ("G_a", "G_b"):
                path = images / f"{counter}.png"
                Image.new("RGB", (8, 8), (counter, 20, 30)).save(path)
                manifest_rows.append({
                    "image_id": f"conflict-image-{counter}",
                    "gbif_id": f"conflict-gbif-{counter}",
                    "occurrence_id": f"conflict-occ-{counter}",
                    "local_path": str(path), "genus": "G", "species_label": species,
                    "download_status": "downloaded", "curation_label": "keep",
                    "sha256": "shared-contradictory-sha",
                    "dhash": "shared-contradictory-dhash",
                    "source_url": f"https://example/conflict/{counter}",
                    "media_reference": f"https://example/media/conflict/{counter}",
                })
                counter += 1
            manifest = root / "curated_manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
            petri = root / "petri"
            petri.mkdir()
            for filename in ("train_split.csv", "val_split.csv", "test_split.csv"):
                pd.DataFrame([{
                    "barcode": "p1", "rel_path_seg": str(images / "0.png"),
                    "genus": "G", "species_label": "G_a", "life_stage": "Adult",
                }]).to_csv(petri / filename, index=False)
            payload = yaml.safe_load(BASE_CONFIG.read_text())
            payload["paths"].update({
                "project_root": str(ROOT), "gbif_manifest": str(manifest),
                "petri_split_dir": str(petri), "petri_data_root": "/",
                "experiment_root": str(root / "gbif_full_taxonomy_audit_test"),
            })
            payload["data"]["expected_raw_species_minimum"] = 4
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            config = load_full_taxonomy_config(config_path)
            result = run_full_taxonomy_audit(config, config_path)
            self.assertEqual(result["raw_species_count"], 3)
            self.assertEqual(result["canonical_valid_species_count"], 3)
            self.assertEqual(result["trainable_species_count"], 3)
            self.assertEqual(result["test_evaluable_species_count"], 2)
            self.assertEqual(result["validation_evaluable_species_count"], 1)
            self.assertEqual(result["fatal_leakage_rows"], 0)
            self.assertEqual(result["quarantined_conflicting_groups"], 1)
            self.assertEqual(result["quarantined_conflicting_images"], 2)
            audit_root = Path(config["paths"]["experiment_root"]) / "audit"
            self.assertTrue((audit_root / "species_count_discrepancy.json").is_file())
            self.assertTrue((audit_root / "semantic_messiness_annotation_manifest.csv").is_file())
            conflicts = pd.read_csv(audit_root / "connected_group_taxonomy_conflicts.csv")
            self.assertEqual(conflicts.iloc[0]["resolution"], "quarantine_entire_connected_group")


if __name__ == "__main__":
    unittest.main()
