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
from src.worm_species.gbif.domain_training import _domain_selection_score
from src.worm_species.gbif.domain_training import mixed_batch_per_domain
from src.worm_species.gbif.domain_orchestration import discover_publication_checkpoints
from src.worm_species.gbif.domain_orchestration import _training_specs
from src.worm_species.gbif.domain_orchestration import render_primary_inference
from src.worm_species.gbif.domain_orchestration import render_training
from src.worm_species.gbif.domain_orchestration import submit_primary_pipeline
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
        self.assertEqual(
            config["models"]["primary"],
            ["convnext_base", "vit_b_16", "resnet50"],
        )
        self.assertEqual(config["models"]["primary_seeds"], [40, 140, 240])
        self.assertEqual(config["models"]["dino"], ["dinov3_vitb16"])
        self.assertEqual(config["models"]["dino_seeds"], [40, 140, 240])
        self.assertEqual(config["slurm"]["partition"], "gpu-short,gpu-l40s,gpu-h200")
        self.assertEqual(config["slurm"]["array_max_active"], 12)
        self.assertEqual(
            config["slurm"]["training"],
            {"cpus_per_task": 16, "memory": "20G", "time_limit": "04:00:00"},
        )
        self.assertEqual(
            config["slurm"]["inference"],
            {"cpus_per_task": 12, "memory": "16G", "time_limit": "02:00:00"},
        )
        self.assertEqual(
            config["slurm"]["analysis"],
            {
                "partition": None,
                "cpus_per_task": 128,
                "memory": "64G",
                "time_limit": "02:00:00",
            },
        )
        self.assertEqual(config["analysis"]["quality_workers"], 128)
        self.assertEqual(config["training"]["batch_size"], 256)
        self.assertNotIn("mixed_batch_per_domain", config["training"])
        self.assertEqual(mixed_batch_per_domain(config), 128)
        config["training"]["batch_size"] = 64
        self.assertEqual(mixed_batch_per_domain(config), 32)
        config["training"]["batch_size"] = 63
        with self.assertRaisesRegex(ValueError, "positive even"):
            mixed_batch_per_domain(config)
        self.assertEqual(config["training"]["num_workers"], 12)
        self.assertEqual(config["training"]["hierarchy_loss"]["weights"], [0.0, 0.5])
        self.assertEqual(config["inference"]["shards"], 12)
        self.assertTrue(config["preprocessed_cache"]["enabled"])
        self.assertTrue(config["training"]["fixed_budget"])
        self.assertEqual(config["training"]["checkpoint_selection_min_delta"], 0.0)
        self.assertEqual(config["preprocessed_cache"]["format"], "png")
        self.assertTrue(config["preprocessed_cache"]["node_root"].startswith("/tmp/"))

    def test_training_plan_has_expected_arrays_and_final_models(self) -> None:
        config = load_domain_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            config["paths"]["output_root"] = temp_dir
            primary = render_training(config, CONFIG, "primary", prepare=False)
            dino = render_training(config, CONFIG, "dino", prepare=False)
            self.assertEqual(primary["wave1"]["count"], 72)
            self.assertEqual(primary["wave2"]["count"], 36)
            self.assertEqual(primary["final_model_count"], 72)
            self.assertEqual(primary["stage_job_count"], 108)
            self.assertEqual(dino["wave1"]["count"], 24)
            self.assertEqual(dino["wave2"]["count"], 12)
            self.assertEqual(dino["final_model_count"], 24)
            script = Path(primary["wave1"]["script"]).read_text()
            self.assertIn("#SBATCH --array=0-71%12", script)
            self.assertIn("#SBATCH --mem=20G", script)
            self.assertIn("#SBATCH --cpus-per-task=16", script)
            self.assertIn("#SBATCH --time=04:00:00", script)
            self.assertIn('-v array_id="$SLURM_ARRAY_TASK_ID"', script)
            self.assertNotIn('-v index=', script)
            self.assertIn('flock -x 200', script)
            self.assertIn('rsync -a "$PERSISTENT_CACHE/" "$partial/"', script)
            self.assertIn('cache-status --cache-root "$partial" --verify-files', script)
            self.assertIn('copied_image_count=$(find "$partial/images"', script)
            self.assertIn('export WORM_GBIF_NODE_CACHE="$NODE_CACHE"', script)
            self.assertLess(
                script.index("stage-complete --spec"),
                script.index("PERSISTENT_CACHE=$("),
            )
            preprocessing = Path(primary["preprocessed_cache"]["script"])
            self.assertTrue(preprocessing.is_file())
            self.assertIn("build-cache", preprocessing.read_text())
            self.assertEqual(primary["preprocessed_cache"]["dependency"], "before:wave1")
            plan = json.loads(Path(primary["plan"]).read_text())
            peti_to_gbif = [
                row for row in plan["runs"] if row["strategy"] == "peti_to_gbif"
            ]
            self.assertEqual({row["stage"] for row in peti_to_gbif}, {"stage1", "stage2"})
            specs = [
                json.loads(path.read_text())
                for path in (Path(temp_dir) / "generated" / "primary" / "specs").glob("*.json")
            ]
            expected_selection = {
                ("gbif_only", "stage1"): ["gbif"],
                ("peti_to_gbif", "stage1"): ["petri"],
                ("peti_to_gbif", "stage2"): ["gbif"],
                ("gbif_to_peti", "stage1"): ["gbif"],
                ("gbif_to_peti", "stage2"): ["petri"],
                ("mixed", "stage1"): ["gbif", "petri"],
            }
            for spec in specs:
                self.assertEqual(
                    spec["selection_domains"],
                    expected_selection[(spec["strategy"], spec["stage"])],
                )
                if spec["stage"] == "stage1":
                    self.assertIsNone(spec["initial_checkpoint"])
            stage2_spec = next(
                spec for spec in specs
                if spec["strategy"] == "gbif_to_peti" and spec["stage"] == "stage2"
            )
            self.assertTrue(
                stage2_spec["initial_checkpoint"].endswith("/stage1/best_model.pt")
            )

    def test_training_submission_builds_cache_before_wave1_when_missing(self) -> None:
        config = load_domain_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config["paths"]["output_root"] = str(root)
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
                    "src.worm_species.gbif.domain_orchestration.discover_publication_checkpoints",
                    return_value={
                        "selected": {
                            model: {"checkpoint": str(root / f"{model}.pt")}
                            for model in config["models"]["primary"]
                        }
                    },
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration._primary_inference_ready",
                    return_value=True,
                ),
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
            self.assertEqual(submit.call_args_list[1].kwargs, {"array": "0-71%12"})
            self.assertEqual(
                submit.call_args_list[2].args,
                ("/jobs/wave2.sbatch", "afterok:101"),
            )
            self.assertEqual(submit.call_args_list[2].kwargs, {"array": "0-35%12"})

    def test_resume_submits_only_incomplete_stage_indices(self) -> None:
        config = load_domain_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config["paths"]["output_root"] = str(root)
            wave1, _wave2 = _training_specs(config, "primary")
            missing_wave1 = {3, 15, 39}
            for index, spec in enumerate(wave1):
                if index in missing_wave1:
                    continue
                output = Path(spec["output_dir"])
                output.mkdir(parents=True)
                (output / "last_model.pt").touch()
                (output / "run_status.json").write_text(
                    json.dumps({"status": "complete"}), encoding="utf-8"
                )
            manifest = {
                "preprocessed_cache": {"script": "/jobs/preprocess.sbatch"},
                "wave1": {"script": "/jobs/wave1.sbatch"},
                "wave2": {"script": "/jobs/wave2.sbatch"},
            }
            with (
                patch("src.worm_species.gbif.domain_orchestration._require_training_runtime"),
                patch(
                    "src.worm_species.gbif.domain_orchestration.discover_publication_checkpoints",
                    return_value={
                        "selected": {
                            model: {"checkpoint": str(root / f"{model}.pt")}
                            for model in config["models"]["primary"]
                        }
                    },
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration._primary_inference_ready",
                    return_value=True,
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration.render_training",
                    return_value=manifest,
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration.domain_cache_status",
                    return_value={"ready": True},
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration._sbatch",
                    side_effect=["200", "201"],
                ) as submit,
            ):
                receipt = submit_training(config, CONFIG, "primary")

            self.assertEqual(receipt["wave1_task_count"], 3)
            self.assertEqual(receipt["wave1_array_indices"], [3, 15, 39])
            self.assertEqual(receipt["wave2_task_count"], 36)
            self.assertEqual(
                submit.call_args_list[0].args,
                ("/jobs/wave1.sbatch", None),
            )
            self.assertEqual(
                submit.call_args_list[0].kwargs,
                {"array": "3,15,39%12"},
            )
            self.assertEqual(
                submit.call_args_list[1].args,
                ("/jobs/wave2.sbatch", "afterok:200"),
            )
            self.assertEqual(
                submit.call_args_list[1].kwargs,
                {"array": "0-35%12"},
            )

    def test_resume_can_submit_wave2_without_relaunching_complete_wave1(self) -> None:
        config = load_domain_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config["paths"]["output_root"] = str(root)
            wave1 = [{"run_id": "wave1", "output_dir": str(root / "wave1")}]
            wave2 = [{"run_id": "wave2", "output_dir": str(root / "wave2")}]
            wave1_output = Path(wave1[0]["output_dir"])
            wave1_output.mkdir(parents=True)
            (wave1_output / "last_model.pt").touch()
            (wave1_output / "run_status.json").write_text(
                json.dumps({"status": "complete"}), encoding="utf-8"
            )
            manifest = {
                "preprocessed_cache": {"script": "/jobs/preprocess.sbatch"},
                "wave1": {"script": "/jobs/wave1.sbatch"},
                "wave2": {"script": "/jobs/wave2.sbatch"},
            }
            with (
                patch("src.worm_species.gbif.domain_orchestration._require_training_runtime"),
                patch(
                    "src.worm_species.gbif.domain_orchestration.discover_publication_checkpoints",
                    return_value={
                        "selected": {
                            model: {"checkpoint": str(root / f"{model}.pt")}
                            for model in config["models"]["primary"]
                        }
                    },
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration._primary_inference_ready",
                    return_value=True,
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration.render_training",
                    return_value=manifest,
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration._training_specs",
                    return_value=(wave1, wave2),
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration.domain_cache_status",
                    return_value={"ready": False},
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration._sbatch",
                    side_effect=["300", "301"],
                ) as submit,
            ):
                receipt = submit_training(config, CONFIG, "primary")

            self.assertIsNone(receipt["wave1_job_id"])
            self.assertEqual(receipt["wave1_task_count"], 0)
            self.assertEqual(receipt["wave2_job_id"], "301")
            self.assertEqual(receipt["wave2_task_count"], 1)
            self.assertEqual(submit.call_args_list[0].args, ("/jobs/preprocess.sbatch",))
            self.assertEqual(
                submit.call_args_list[1].args,
                ("/jobs/wave2.sbatch", "afterok:300"),
            )
            self.assertEqual(submit.call_args_list[1].kwargs, {"array": "0%12"})

    def test_domain_balanced_selection_excludes_missing_tasks_and_balances_domains(self) -> None:
        metrics = {
            "gbif": {
                "genus_macro_f1": 0.8,
                "species_macro_f1": 0.6,
                "age_macro_f1": None,
            },
            "petri": {
                "genus_macro_f1": 0.9,
                "species_macro_f1": 0.6,
                "age_macro_f1": 0.3,
            },
        }
        domain_scores, gbif_score = _domain_selection_score(metrics, ("gbif",))
        self.assertAlmostEqual(domain_scores["gbif"], 0.7)
        self.assertAlmostEqual(domain_scores["petri"], 0.6)
        self.assertAlmostEqual(gbif_score, 0.7)
        _, joint_score = _domain_selection_score(metrics, ("gbif", "petri"))
        self.assertAlmostEqual(joint_score, 0.65)

    def test_publication_checkpoint_selection_uses_validation_loss_per_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_domain_config(CONFIG)
            config["paths"]["publication_baseline_root"] = str(root / "baseline")
            config["paths"]["output_root"] = str(root / "output")
            baseline = Path(config["paths"]["publication_baseline_root"])
            for model in config["models"]["primary"]:
                for seed, score in ((40, 0.4), (140, 0.2)):
                    run = baseline / f"run_{model}_{seed}" / "fit"
                    run.mkdir(parents=True)
                    (run.parent / "run_status.txt").write_text("0\n")
                    (run / "best_model.pt").touch()
                    (run / "run_summary.json").write_text(json.dumps({
                        "model": model,
                        "train_condition": "original",
                        "selection_metric": "loss",
                        "best_val_score": score,
                    }))
                    (run / "config.json").write_text(json.dumps({
                        "model": {"name": model}, "seed": seed,
                    }))
            selected = discover_publication_checkpoints(config)
            self.assertEqual(
                {model: row["seed"] for model, row in selected["selected"].items()},
                {model: 140 for model in config["models"]["primary"]},
            )
            self.assertTrue(Path(selected["manifest"]).is_file())

    def test_unified_submission_runs_inference_and_cache_in_parallel(self) -> None:
        config = load_domain_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            config["paths"]["output_root"] = temp_dir
            manifest = {
                "manifest": "/jobs/pipeline.json",
                "checkpoint_manifest": "/jobs/checkpoints.json",
                "cache_ready": False,
                "inference": {
                    "task_count": 36,
                    "array_script": "/jobs/inference.sbatch",
                    "merge_script": "/jobs/merge.sbatch",
                },
                "training": {
                    "preprocessed_cache": {"script": "/jobs/cache.sbatch"},
                    "wave1": {"script": "/jobs/wave1.sbatch"},
                    "wave2": {"script": "/jobs/wave2.sbatch"},
                },
            }
            with (
                patch("src.worm_species.gbif.domain_orchestration._require_training_runtime"),
                patch(
                    "src.worm_species.gbif.domain_orchestration.render_primary_pipeline",
                    return_value=manifest,
                ),
                patch(
                    "src.worm_species.gbif.domain_orchestration._sbatch",
                    side_effect=["100", "101", "102", "103", "104"],
                ) as submit,
            ):
                receipt = submit_primary_pipeline(config, CONFIG)
            self.assertEqual(receipt["inference_array_job_id"], "100")
            self.assertEqual(receipt["preprocessing_job_id"], "102")
            self.assertEqual(
                submit.call_args_list[3].args,
                ("/jobs/wave1.sbatch", "afterok:101:102"),
            )
            self.assertEqual(
                submit.call_args_list[4].args,
                ("/jobs/wave2.sbatch", "afterok:103"),
            )

    def test_three_model_inference_is_one_globally_capped_array(self) -> None:
        config = load_domain_config(CONFIG)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config["paths"]["output_root"] = str(root / "output")
            checkpoints = {}
            for model in config["models"]["primary"]:
                checkpoint = root / f"{model}.pt"
                checkpoint.touch()
                checkpoints[model] = str(checkpoint)
            manifest = render_primary_inference(config, checkpoints)
            self.assertEqual(manifest["task_count"], 36)
            self.assertEqual(manifest["pending_models"], config["models"]["primary"])
            script = Path(manifest["array_script"]).read_text()
            self.assertIn("#SBATCH --array=0-35%12", script)
            self.assertIn("#SBATCH --cpus-per-task=12", script)
            self.assertIn("#SBATCH --mem=16G", script)
            self.assertIn("#SBATCH --time=02:00:00", script)
            tasks = pd.read_csv(manifest["index"], sep="\t")
            self.assertEqual(tasks.groupby("model").size().to_dict(), {
                model: 12 for model in config["models"]["primary"]
            })

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
