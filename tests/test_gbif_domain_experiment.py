from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import torch
from PIL import Image
from unittest.mock import patch

from src.worm_species.gbif.domain_data import load_domain_config
from src.worm_species.gbif.domain_data import prepare_domain_manifests
from src.worm_species.gbif.domain_cache import build_domain_cache
from src.worm_species.gbif.domain_cache import domain_cache_directory
from src.worm_species.gbif.domain_cache import load_cached_domain_frames
from src.worm_species.gbif.domain_training import _wandb_run
from src.worm_species.gbif.domain_orchestration import render_training
from src.worm_species.gbif.domain_orchestration import submit_training
from src.worm_species.gbif.inference import merge_inference_shards
from src.worm_species.gbif.inference import infer_existing_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gbif_training.yaml"


class GBIFDomainExperimentTests(unittest.TestCase):
    def test_wandb_run_uses_stable_hashed_run_id(self) -> None:
        wandb = SimpleNamespace(init=MagicMock(return_value=object()))
        config = {
            "wandb": {
                "enabled": True,
                "project": "worms",
                "group": "domain-training",
            }
        }
        spec = {
            "run_id": "primary-wave1-resnet50-seed40",
            "stage": "wave1",
            "phase": "primary",
            "regime": "pooled",
            "model": "resnet50",
        }

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            "sys.modules", {"wandb": wandb}
        ):
            run = _wandb_run(config, spec, Path(temp_dir))

        self.assertIsNotNone(run)
        self.assertEqual(wandb.init.call_args.kwargs["id"], "d3737dbdf81a3160")

    def test_approved_genome_resources_and_models(self) -> None:
        config = load_domain_config(CONFIG)
        self.assertEqual(config["models"]["primary"], ["convnext_base"])
        self.assertEqual(config["models"]["primary_seeds"], [40, 140, 240])
        self.assertEqual(config["models"]["dino"], ["dinov3_vitb16"])
        self.assertEqual(config["models"]["dino_seeds"], [40, 140, 240])
        self.assertEqual(config["slurm"]["partition"], "gpu-short,gpu-l40s,gpu-h200")
        self.assertEqual(config["slurm"]["memory"], "20G")
        self.assertEqual(config["slurm"]["cpus_per_task"], 16)
        self.assertEqual(config["slurm"]["array_max_active"], 12)
        self.assertEqual(config["training"]["num_workers"], 12)
        self.assertEqual(config["training"]["hierarchy_loss"]["weights"], [0.0, 0.5])
        self.assertEqual(config["inference"]["shards"], 12)
        self.assertTrue(config["preprocessed_cache"]["enabled"])
        self.assertEqual(config["preprocessed_cache"]["format"], "png")
        self.assertTrue(config["preprocessed_cache"]["node_root"].startswith("/tmp/"))

    def test_training_plan_has_expected_arrays_and_final_models(self) -> None:
        config = load_domain_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            config["paths"]["output_root"] = temp_dir
            primary = render_training(config, CONFIG, "primary", prepare=False)
            dino = render_training(config, CONFIG, "dino", prepare=False)
            self.assertEqual(primary["wave1"]["count"], 24)
            self.assertEqual(primary["wave2"]["count"], 6)
            self.assertEqual(primary["final_model_count"], 24)
            self.assertEqual(dino["wave1"]["count"], 24)
            self.assertEqual(dino["wave2"]["count"], 6)
            self.assertEqual(dino["final_model_count"], 24)
            script = Path(primary["wave1"]["script"]).read_text()
            self.assertIn("#SBATCH --array=0-23%12", script)
            self.assertIn("#SBATCH --mem=20G", script)
            self.assertIn("#SBATCH --cpus-per-task=16", script)
            self.assertIn('-v array_id="$SLURM_ARRAY_TASK_ID"', script)
            self.assertNotIn('-v index=', script)
            self.assertIn('flock -x 200', script)
            self.assertIn('rsync -a "$PERSISTENT_CACHE/" "$partial/"', script)
            self.assertIn('cache-status --cache-root "$partial" --verify-files', script)
            self.assertIn('copied_image_count=$(find "$partial/images"', script)
            self.assertIn('export WORM_GBIF_NODE_CACHE="$NODE_CACHE"', script)
            preprocessing = Path(primary["preprocessed_cache"]["script"])
            self.assertTrue(preprocessing.is_file())
            self.assertIn("build-cache", preprocessing.read_text())
            self.assertEqual(primary["preprocessed_cache"]["dependency"], "before:wave1")

    def test_training_submission_builds_cache_before_wave1_when_missing(self) -> None:
        config = load_domain_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config["paths"]["output_root"] = str(root)
            config["paths"]["peti_checkpoint"] = str(root / "peti.pt")
            Path(config["paths"]["peti_checkpoint"]).touch()
            inference = root / "inference" / "baseline" / "predictions.csv"
            inference.parent.mkdir(parents=True)
            inference.touch()
            manifest = {
                "preprocessed_cache": {"script": "/jobs/preprocess.sbatch"},
                "wave1": {"script": "/jobs/wave1.sbatch"},
                "wave2": {"script": "/jobs/wave2.sbatch"},
            }
            with (
                patch("src.worm_species.gbif.domain_orchestration._require_training_runtime"),
                patch(
                    "src.worm_species.gbif.domain_orchestration.render_training",
                    return_value=manifest,
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration.domain_cache_status",
                    return_value={"ready": False},
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration._sbatch",
                    side_effect=["100", "101", "102"],
                ) as submit,
            ):
                receipt = submit_training(config, CONFIG, "primary")

            self.assertEqual(receipt["preprocessing_job_id"], "100")
            self.assertEqual(receipt["wave1_job_id"], "101")
            self.assertEqual(receipt["wave2_job_id"], "102")
            self.assertEqual(submit.call_args_list[0].args, ("/jobs/preprocess.sbatch",))
            self.assertEqual(
                submit.call_args_list[1].args,
                ("/jobs/wave1.sbatch", "afterok:100"),
            )
            self.assertEqual(
                submit.call_args_list[2].args,
                ("/jobs/wave2.sbatch", "afterok:101"),
            )

    def test_preparation_preserves_petri_splits_and_groups_gbif(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "curated.csv"
            rows = []
            for species, genus in (("Alpha_one", "Alpha"), ("Beta_two", "Beta")):
                for index in range(3):
                    image = root / f"{species}-{index}.jpg"
                    Image.new("RGB", (20, 12), "brown").save(image)
                    rows.append({
                        "image_id": f"{species}-{index}", "gbif_id": f"{species}-occ-{index}",
                        "local_path": str(image), "genus": genus,
                        "species_label": species, "curation_label": "keep",
                        "download_status": "downloaded", "sha256": f"{species}-{index}",
                    })
            pd.DataFrame(rows).to_csv(manifest, index=False)
            split_root = root / "splits"
            split_root.mkdir()
            (root / "images").mkdir()
            for split, filename in (
                ("train", "train_split.csv"),
                ("validation", "val_split.csv"),
                ("test", "test_split.csv"),
            ):
                petri = []
                for index in range(3):
                    path = f"images/petri-{split}-{index}.jpg"
                    Image.new("RGB", (12, 20), "pink").save(root / path)
                    petri.append({
                        "barcode": f"petri-{split}-{index}", "rel_path_seg": path,
                        "genus": "PetriOnly", "species_label": "PetriOnly_species",
                        "life_stage": "Adult" if index % 2 == 0 else "Juvenile",
                    })
                pd.DataFrame(petri).to_csv(split_root / filename, index=False)
            config = load_domain_config(CONFIG)
            config["paths"].update({
                "gbif_manifest": str(manifest), "petri_split_dir": str(split_root),
                "petri_data_root": str(root), "output_root": str(root / "output"),
            })
            config["preprocessed_cache"].update({
                "root": str(root / "cache"),
                "workers": 1,
                "progress_interval_images": 100,
            })
            summary = prepare_domain_manifests(config)
            prepared = root / "output" / "prepared"
            labels = json.loads((prepared / "label_maps.json").read_text())
            self.assertIn("PetriOnly", labels["genus"])
            self.assertIn("PetriOnly_species", labels["species"])
            self.assertEqual(summary["rows"]["petri"], {
                "train": 3, "validation": 3, "test": 3,
            })
            self.assertEqual(summary["source_inventory"]["rows"], 15)
            gbif_frames = {
                split: pd.read_csv(prepared / f"gbif_{split}.csv")
                for split in ("train", "validation", "test")
            }
            group_sets = {split: set(frame["group_id"]) for split, frame in gbif_frames.items()}
            self.assertTrue(group_sets["train"].isdisjoint(group_sets["validation"]))
            self.assertTrue(group_sets["train"].isdisjoint(group_sets["test"]))
            self.assertTrue(group_sets["validation"].isdisjoint(group_sets["test"]))

            built = build_domain_cache(config)
            self.assertEqual(built["status"], "built")
            self.assertEqual(built["copied_this_call"], 15)
            cache_root = domain_cache_directory(config)
            cached = load_cached_domain_frames(config, cache_root)
            cached_image = Path(cached["gbif"]["train"].iloc[0]["image_path"])
            self.assertTrue(cached_image.is_file())
            with Image.open(cached_image) as image:
                self.assertEqual(image.size, (224, 224))
            reused = build_domain_cache(config)
            self.assertEqual(reused["status"], "reused")
            self.assertEqual(reused["copied_this_call"], 0)

    def test_merge_inference_requires_exact_nonoverlapping_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = []
            manifest_rows = []
            for index in range(4):
                image = root / f"image-{index}.jpg"
                image.touch()
                images.append(image)
                manifest_rows.append({
                    "image_id": f"id-{index}", "local_path": str(image),
                    "download_status": "downloaded", "curation_label": "keep",
                })
            manifest = root / "manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest, index=False)
            shards = root / "shards"
            shards.mkdir()
            for index, ids in enumerate((("id-0", "id-2"), ("id-1", "id-3"))):
                path = shards / f"shard-{index:03d}.csv"
                pd.DataFrame({"image_id": ids, "checkpoint_genus_scope": ["known"] * 2,
                              "checkpoint_species_scope": ["known"] * 2}).to_csv(path, index=False)
                path.with_suffix(".summary.json").write_text(json.dumps({
                    "shard_index": index, "shard_count": 2,
                    "checkpoint_sha256": "abc", "checkpoint": "/model.pt",
                    "checkpoint_model": "resnet50",
                }))
            output = root / "predictions.csv"
            summary = merge_inference_shards(manifest, shards, output, shard_count=2)
            self.assertEqual(summary["rows"], 4)
            self.assertTrue(summary["coverage_validated"])
            self.assertEqual(pd.read_csv(output)["image_id"].tolist(), [
                "id-0", "id-1", "id-2", "id-3",
            ])

    def test_inference_hash_shards_merge_to_exact_manifest(self) -> None:
        class TinyModel(torch.nn.Module):
            def forward(self, images):
                return {"genus": torch.tensor([[0.0, 1.0]]).repeat(len(images), 1)}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for index in range(12):
                image = root / f"image-{index}.png"
                Image.new("RGB", (8, 8), "brown").save(image)
                rows.append({
                    "image_id": f"image-{index}", "local_path": str(image),
                    "download_status": "downloaded", "curation_label": "keep",
                    "genus": "Lumbricus", "species_label": "",
                })
            manifest = root / "manifest.csv"
            pd.DataFrame(rows).to_csv(manifest, index=False)
            checkpoint = root / "checkpoint.pt"
            torch.save({
                "cfg": {
                    "model": {"name": "tiny", "pretrained": False},
                    "preprocessing": {"image_size": 8, "normalisation": {"enabled": False}},
                    "augmentation": {"enabled": False},
                },
                "model_state": {},
                "label_to_index_by_task": {"genus": {"Eisenia": 0, "Lumbricus": 1}},
                "best_epoch": 1, "selection_metric": "macro_f1",
            }, checkpoint)
            shard_dir = root / "shards"
            shard_dir.mkdir()
            with patch(
                "src.worm_species.gbif.inference.build_multitask_model",
                return_value=TinyModel(),
            ):
                for shard_index in range(2):
                    infer_existing_checkpoint(
                        manifest, checkpoint, shard_dir / f"shard-{shard_index:03d}.csv",
                        batch_size=4, num_workers=0, device_name="cpu",
                        shard_index=shard_index, shard_count=2,
                    )
            output = root / "merged.csv"
            summary = merge_inference_shards(
                manifest, shard_dir, output, shard_count=2,
            )
            self.assertEqual(summary["rows"], 12)
            self.assertEqual(set(pd.read_csv(output)["image_id"]), {
                f"image-{index}" for index in range(12)
            })


if __name__ == "__main__":
    unittest.main()
