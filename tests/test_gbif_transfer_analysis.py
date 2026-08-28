from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from scripts.analyse_gbif_transfer import (
    _checkpoint_specs,
    _hierarchy_effects,
    _interaction_effects,
    _quality_record,
    _rarity_label,
    _render_inference_script,
    _render_report_script,
    _run_metrics,
    _prediction_path,
    build_report,
    build_species_metadata,
)
from src.worm_species.gbif.domain_data import load_domain_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gbif_training.yaml"


class GBIFTransferAnalysisTests(unittest.TestCase):
    def test_plan_uses_36_final_checkpoints_and_18_audit_pairs(self) -> None:
        config = load_domain_config(CONFIG)
        selected, audit = _checkpoint_specs(config)
        self.assertEqual(len(selected), 36)
        self.assertEqual(len(audit), 18)
        self.assertEqual({row["strategy"] for row in selected}, {"gbif_only", "peti_to_gbif"})
        for base, transfer in audit:
            self.assertEqual(base["strategy"], "gbif_only")
            self.assertEqual(transfer["strategy"], "gbif_to_peti")
            self.assertEqual(base["model"], transfer["model"])
            self.assertEqual(base["seed"], transfer["seed"])
            self.assertEqual(base["hierarchy_loss_weight"], transfer["hierarchy_loss_weight"])

    def test_rendered_resources_are_two_hours_and_128_report_cores(self) -> None:
        config = load_domain_config(CONFIG)
        report = _render_report_script(config, CONFIG.resolve())
        inference = _render_inference_script(
            config, CONFIG.resolve(), Path("/tmp/pending.tsv"), 36
        )
        self.assertIn("#SBATCH --cpus-per-task=128", report)
        self.assertIn("#SBATCH --time=02:00:00", report)
        self.assertNotIn("#SBATCH --gres=gpu", report)
        self.assertIn("#SBATCH --array=0-35%12", inference)
        self.assertIn("#SBATCH --cpus-per-task=12", inference)
        self.assertIn("#SBATCH --time=02:00:00", inference)

    def test_species_metadata_retains_zero_and_rare_gbif_species(self) -> None:
        columns = [
            "sample_id", "group_id", "true_genus", "true_species", "genus", "species"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = Path(temp_dir)
            pd.DataFrame([
                ["a1", "ga1", "G", "G_a", "G", "G_a"],
                ["a2", "ga2", "G", "G_a", "G", "G_a"],
            ], columns=columns).to_csv(prepared / "gbif_train.csv", index=False)
            pd.DataFrame([
                ["a3", "ga3", "G", "G_a", "G", "G_a"],
                ["b1", "gb1", "H", "H_b", "H", ""],
            ], columns=columns).to_csv(prepared / "gbif_test.csv", index=False)
            pd.DataFrame([
                ["p1", "gp1", "G", "G_a", "G", "G_a"],
            ], columns=columns).to_csv(prepared / "petri_train.csv", index=False)
            config = load_domain_config(CONFIG)
            metadata = build_species_metadata(config, prepared).set_index("true_species")
        self.assertEqual(metadata.loc["G_a", "gbif_train_images"], 2)
        self.assertEqual(metadata.loc["G_a", "rarity_band"], "1-10")
        self.assertTrue(metadata.loc["G_a", "petri_seen_species"])
        self.assertEqual(metadata.loc["H_b", "gbif_train_images"], 0)
        self.assertEqual(metadata.loc["H_b", "rarity_band"], "0")
        self.assertFalse(metadata.loc["H_b", "petri_seen_species"])
        self.assertFalse(metadata.loc["H_b", "species_evaluable"])

    def test_rarity_bands_have_requested_boundaries(self) -> None:
        bands = load_domain_config(CONFIG)["analysis"]["rarity_bands"]
        expected = {0: "0", 1: "1-10", 10: "1-10", 11: "11-25", 25: "11-25", 26: "26-100", 100: "26-100", 101: ">100"}
        self.assertEqual({value: _rarity_label(value, bands) for value in expected}, expected)

    def test_taxonomic_metrics_distinguish_within_and_between_genus_errors(self) -> None:
        frame = pd.DataFrame({
            "run_id": ["r"] * 3,
            "model": ["m"] * 3,
            "seed": [40] * 3,
            "hierarchy_loss_weight": [0.0] * 3,
            "strategy": ["gbif_only"] * 3,
            "genus_evaluable": [True] * 3,
            "species_evaluable": [True] * 3,
            "mapped_true_genus": ["G", "G", "H"],
            "mapped_true_species": ["G_a", "G_b", "H_c"],
            "predicted_genus": ["G", "G", "H"],
            "predicted_species": ["G_b", "G_b", "G_a"],
            "genus_top3_correct": [True] * 3,
            "genus_top5_correct": [True] * 3,
            "species_top3_correct": [True, True, False],
            "species_top5_correct": [True, True, False],
        })
        frame["genus_top1_correct"] = frame["mapped_true_genus"].eq(frame["predicted_genus"])
        frame["species_top1_correct"] = frame["mapped_true_species"].eq(frame["predicted_species"])
        metrics = _run_metrics(frame, {"G_a": "G", "G_b": "G", "H_c": "H"})
        values = metrics.set_index("metric")["value"]
        genus_chance = metrics.loc[
            metrics["task"].eq("genus")
            & metrics["metric"].eq("balanced_chance_1_over_k"),
            "value",
        ].iloc[0]
        self.assertAlmostEqual(genus_chance, 0.5)
        self.assertAlmostEqual(values["within_genus_error_fraction"], 0.5)
        self.assertAlmostEqual(values["between_genus_error_fraction"], 0.5)
        self.assertAlmostEqual(values["taxonomic_error_severity_0_1_2"], 1.0)
        self.assertAlmostEqual(values["genus_species_consistency"], 2 / 3)

    def test_hierarchy_and_petri_interaction_are_matched_differences(self) -> None:
        rows = []
        for strategy, base in (("gbif_only", 0.60), ("peti_to_gbif", 0.65)):
            for hierarchy, gain in ((0.0, 0.0), (0.5, 0.02 if strategy == "gbif_only" else 0.05)):
                rows.append({
                    "model": "m", "seed": 40, "strategy": strategy,
                    "hierarchy_loss_weight": hierarchy, "task": "species",
                    "metric": "balanced_accuracy", "value": base + gain,
                })
        metrics = pd.DataFrame(rows)
        hierarchy = _hierarchy_effects(metrics)
        observed = hierarchy.set_index("strategy")["h0_5_minus_h0"]
        self.assertAlmostEqual(observed["gbif_only"], 0.02)
        self.assertAlmostEqual(observed["peti_to_gbif"], 0.05)
        paired = metrics.pivot_table(
            index=["model", "seed", "hierarchy_loss_weight", "task", "metric"],
            columns="strategy", values="value",
        ).reset_index()
        paired["petri_minus_gbif"] = paired["peti_to_gbif"] - paired["gbif_only"]
        interaction = _interaction_effects(paired)
        self.assertAlmostEqual(interaction.iloc[0]["petri_by_hierarchy_interaction"], 0.03)

    def test_quality_record_extracts_objective_measures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "gradient.png"
            image = Image.new("RGB", (32, 16))
            image.putdata([(index % 256, index % 256, index % 256) for index in range(32 * 16)])
            image.save(image_path)
            record = _quality_record(("sample", str(image_path)))
        self.assertEqual(record["quality_status"], "ok")
        self.assertEqual(record["width_px"], 32)
        self.assertEqual(record["height_px"], 16)
        self.assertIn("laplacian_variance", record)
        self.assertIn("grayscale_entropy_bits", record)

    def test_synthetic_report_writes_seed_rarity_quality_and_audit_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_domain_config(CONFIG)
            config["paths"]["output_root"] = str(root)
            prepared = root / "prepared"
            prepared.mkdir()
            columns = [
                "sample_id", "group_id", "image_path", "true_genus",
                "true_species", "genus", "species",
            ]
            gbif_train = pd.DataFrame([
                ["train-a", "train-ga", "/tmp/a", "G", "G_a", "G", "G_a"],
            ], columns=columns)
            gbif_test = pd.DataFrame([
                ["test-a", "test-ga", "/tmp/a", "G", "G_a", "G", "G_a"],
                ["test-b", "test-gb", "/tmp/b", "H", "H_b", "H", ""],
            ], columns=columns)
            petri = pd.DataFrame([
                ["petri-a", "petri-ga", "/tmp/p", "G", "G_a", "G", "G_a"],
                ["petri-h", "petri-gh", "/tmp/h", "H", "H_c", "H", "H_c"],
            ], columns=columns)
            for split in ("train", "validation", "test"):
                gbif_train.to_csv(prepared / f"gbif_{split}.csv", index=False)
                petri.to_csv(prepared / f"petri_{split}.csv", index=False)
            gbif_test.to_csv(prepared / "gbif_test.csv", index=False)
            (prepared / "label_maps.json").write_text(
                '{"genus":{"G":0,"H":1},"species":{"G_a":0,"H_c":1},"age":{}}'
            )
            specs = []
            for hierarchy in (0.0, 0.5):
                for strategy in ("gbif_only", "peti_to_gbif"):
                    run_id = f"{strategy}-h{hierarchy}"
                    spec = {
                        "run_id": run_id, "model": "resnet50", "seed": 40,
                        "hierarchy_loss_weight": hierarchy, "strategy": strategy,
                        "stage": "stage1" if strategy == "gbif_only" else "stage2",
                        "output_dir": str(root / "runs" / run_id), "final_model": True,
                    }
                    specs.append(spec)
                    correct = strategy == "peti_to_gbif"
                    prediction = pd.DataFrame({
                        "sample_id": ["test-a", "test-b"],
                        "group_id": ["test-ga", "test-gb"],
                        "gbif_id": ["1", "2"],
                        "image_path": ["/tmp/a", "/tmp/b"],
                        "raw_true_genus": ["G", "H"],
                        "raw_true_species": ["G_a", "H_b"],
                        "mapped_true_genus": ["G", "H"],
                        "mapped_true_species": ["G_a", ""],
                        "model": ["resnet50"] * 2,
                        "seed": [40] * 2,
                        "hierarchy_loss_weight": [hierarchy] * 2,
                        "strategy": [strategy] * 2,
                        "stage": [spec["stage"]] * 2,
                        "run_id": [run_id] * 2,
                        "genus_evaluable": [True, True],
                        "species_evaluable": [True, False],
                        "predicted_genus": ["G" if correct else "H", "H" if correct else "G"],
                        "predicted_species": ["G_a" if correct else "H_c", "H_c"],
                        "true_genus_probability": [0.8 if correct else 0.2] * 2,
                        "true_species_probability": [0.8 if correct else 0.2, float("nan")],
                        "genus_top1_correct": [correct, correct],
                        "genus_top3_correct": [True, True],
                        "genus_top5_correct": [True, True],
                        "species_top1_correct": [correct, pd.NA],
                        "species_top3_correct": [True, pd.NA],
                        "species_top5_correct": [True, pd.NA],
                    })
                    path = _prediction_path(config, spec)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    prediction.to_csv(path, index=False, compression="gzip")
            quality = pd.DataFrame({
                "sample_id": ["test-a", "test-b"],
                "image_path": ["/tmp/a", "/tmp/b"],
                "quality_status": ["ok", "ok"],
                "quality_challenge_index": [0.1, 0.9],
                "quality_quartile": ["Q1_cleaner", "Q4_messier"],
            })
            with (
                patch("scripts.analyse_gbif_transfer._checkpoint_specs", return_value=(specs, [])),
                patch("scripts.analyse_gbif_transfer._validate_completed_checkpoint", return_value=root / "fake.pt"),
                patch("scripts.analyse_gbif_transfer._prediction_complete", return_value=True),
                patch("scripts.analyse_gbif_transfer.extract_image_quality", return_value=quality),
                patch("scripts.analyse_gbif_transfer.audit_checkpoints", return_value=pd.DataFrame([{"reuse_status": "synthetic"}])),
                patch("scripts.analyse_gbif_transfer._plots"),
            ):
                manifest = build_report(config)
            tables = root / "transfer_analysis" / "tables"
            self.assertEqual(manifest["checkpoint_count"], 4)
            for name in (
                "per_image_predictions.csv.gz", "per_seed_effects.csv",
                "per_species_petri_effects.csv", "rarity_band_effects_per_seed.csv",
                "petri_seen_unseen_contrast_per_seed.csv",
                "quality_messier_vs_cleaner_contrast_per_seed.csv",
                "gbif_only_vs_gbif_to_peti_stage1_checkpoint_audit.csv",
                "probability_label_order.json", "statistical_tests.csv",
            ):
                self.assertTrue((tables / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
