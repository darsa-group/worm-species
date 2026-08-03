from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.worm_species.analysis.generalisation_report import (
    HOLDOUT_LABELS,
    architecture_name,
    build_generalisation_report,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mechanism_config(architecture: str) -> tuple[dict, dict, dict, dict]:
    model = {
        "name": "convnext_tiny",
        "multitask_architecture": architecture,
    }
    data = {"sampler": {"type": "default"}}
    training = {"gradient_strategy": {"type": "standard"}}
    loss = {
        "age_supervised_contrastive": {"enabled": False}
    }
    if architecture == "single_task_age":
        model.update(
            multitask_architecture="single_task",
            target_task="age",
        )
    elif architecture.startswith("split"):
        model["multitask_architecture"] = "split_taxonomy_age"
        if architecture in {
            "split_joint_sampler",
            "split_joint_sampler_pcgrad",
            "split_full",
        }:
            data["sampler"]["type"] = "joint_species_stage"
        if architecture in {
            "split_pcgrad",
            "split_joint_sampler_pcgrad",
            "split_full",
        }:
            training["gradient_strategy"]["type"] = "pcgrad"
        if architecture in {"split_age_supcon", "split_full"}:
            loss["age_supervised_contrastive"]["enabled"] = True
    return model, data, training, loss


class GeneralisationReportTests(unittest.TestCase):
    def test_performance_full_is_not_mislabelled_as_age_supcon(self) -> None:
        config = {
            "model": {"multitask_architecture": "split_taxonomy_age"},
            "data": {
                "multiview": {"enabled": True},
                "sampler": {"type": "cross_species_stage_contrastive"},
            },
            "optimizer": {"learning_rates": {"early_backbone": 1e-5}},
            "training": {"staged_unfreezing": {"enabled": True}},
            "loss": {
                "age_supervised_contrastive": {"enabled": True},
                "genus_supervised_contrastive": {"enabled": True},
                "taxonomy_consistency": {"enabled": True},
            },
            "evaluation": {"checkpoint_ensemble": {"enabled": True}},
        }
        self.assertEqual(architecture_name(config), "performance_full")

    def _write_run(
        self,
        root: Path,
        *,
        architecture: str,
        seed: int,
        holdout: str = "original_baseline",
    ) -> Path:
        run_dir = root / architecture / str(seed) / holdout
        model, data, training, loss = _mechanism_config(architecture)
        config = {
            "seed": seed,
            "model": model,
            "data": data,
            "training": training,
            "loss": loss,
            "data_holdout": {
                "enabled": holdout != "original_baseline",
                "name": holdout,
            },
        }
        _write_json(run_dir / "config.json", config)
        architecture_gain = {
            "shared_heads": 0.0,
            "single_task_age": 0.015,
            "split_taxonomy_age": 0.03,
            "split_joint_sampler": 0.04,
            "split_pcgrad": 0.05,
            "split_full": 0.08,
        }[architecture]
        seed_offset = (seed - 41) * 0.006
        metrics = {
            "mean_macro_f1": 0.60 + architecture_gain + seed_offset,
            "genus_macro_f1": 0.70 + architecture_gain / 2 + seed_offset,
            "species_macro_f1": 0.62 + architecture_gain / 2 + seed_offset,
            "age_macro_f1": 0.48 + architecture_gain + seed_offset,
        }
        if architecture == "single_task_age":
            metrics["mean_macro_f1"] = metrics["age_macro_f1"]
            metrics["genus_macro_f1"] = None
            metrics["species_macro_f1"] = None
        _write_json(run_dir / "test_metrics_best.json", metrics)
        if holdout != "original_baseline":
            rows = []
            for (candidate_holdout, task), _ in HOLDOUT_LABELS.items():
                if candidate_holdout != holdout:
                    continue
                if architecture == "single_task_age" and task != "age":
                    continue
                rows.append({
                    "holdout": holdout,
                    "cohort": "independent_test",
                    "task": task,
                    "target_label": "juvenile" if task == "age" else "target",
                    "target_n": 12,
                    "target_recall": (
                        0.35 + architecture_gain + seed_offset
                    ),
                })
            metrics_dir = run_dir / "data_holdout_evaluation"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(
                metrics_dir / "task_metrics.csv",
                index=False,
            )
        if (
            architecture in {
                "shared_heads", "split_taxonomy_age", "split_pcgrad",
            }
            and holdout == "original_baseline"
        ):
            pd.DataFrame({
                "epoch": [1, 1, 2],
                "step": [100, 200, 300],
                "genus_gradient_norm": [1.0, 0.9, 0.8],
                "species_gradient_norm": [1.1, 1.0, 0.9],
                "age_gradient_norm": [0.8, 0.75, 0.7],
                "genus_species_cosine": [0.2, 0.1, 0.15],
                "genus_age_cosine": [-0.2, -0.1, -0.05],
                "species_age_cosine": [-0.1, 0.0, 0.05],
            }).to_csv(run_dir / "gradient_diagnostics.csv", index=False)
        if (
            architecture in {"shared_heads", "single_task_age", "split_full"}
            and seed == 40
            and holdout == "original_baseline"
        ):
            rng = np.random.default_rng(2026)
            np.savez_compressed(
                run_dir / "age_embeddings_best.npz",
                embeddings=rng.normal(size=(12, 8)),
                representation_type=np.asarray(
                    "age_projection"
                    if architecture == "split_full"
                    else "age_branch_features"
                ),
            )
            pd.DataFrame({
                "developmental_stage": ["adult", "juvenile"] * 6,
                "species": ["species_a"] * 4
                + ["species_b"] * 4
                + ["species_c"] * 4,
            }).to_csv(
                run_dir / "age_embeddings_best_metadata.csv",
                index=False,
            )
        return run_dir

    def test_completed_run_report_generates_all_tables_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            output = root / "report"
            architectures = (
                "shared_heads",
                "single_task_age",
                "split_taxonomy_age",
                "split_joint_sampler",
                "split_pcgrad",
                "split_full",
            )
            holdouts = (
                "original_baseline",
                "juvenile_aporrectodea_longa",
                "juvenile_allolobophora_chlorotica",
                "juvenile_genus_aporrectodea",
                "unseen_species_aporrectodea_longa_for_genus",
            )
            for architecture in architectures:
                for seed in (40, 41, 42):
                    for holdout in holdouts:
                        self._write_run(
                            results,
                            architecture=architecture,
                            seed=seed,
                            holdout=holdout,
                        )
            _write_json(
                results / "incomplete" / "config.json",
                {"model": {"multitask_architecture": "shared_heads"}},
            )

            manifest = build_generalisation_report(results, output)

            self.assertEqual(
                manifest["completed_runs"],
                len(architectures) * 3 * len(holdouts),
            )
            self.assertTrue(all(manifest["figures"].values()))
            for filename in (
                "all_generalisation_runs.csv",
                "architecture_summary.csv",
                "holdout_target_recall.csv",
                "task_performance_summary.csv",
                "gradient_conflict_summary.csv",
                "results_summary.md",
                "latex/results_summary.tex",
            ):
                self.assertTrue((output / filename).is_file(), filename)
            for figure in (
                "figure_a_architecture_comparison",
                "figure_b_modelling_change_effects",
                "figure_c_age_specific",
                "figure_d_gradient_interactions",
                "figure_e_embedding_diagnostics",
            ):
                for extension in ("svg", "pdf", "png"):
                    self.assertTrue(
                        (output / "figures" / f"{figure}.{extension}").is_file()
                    )
            embedding_source = pd.read_csv(
                output / "figures" / "figure_e_embedding_diagnostics_source.csv"
            )
            self.assertEqual(
                set(embedding_source["architecture"]),
                {"shared_heads", "single_task_age", "split_full"},
            )
            summary = (output / "results_summary.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("absolute change", summary)
            self.assertIn("across all 3 seeds", summary)

    def test_missing_optional_metrics_and_incomplete_runs_do_not_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            output = root / "report"
            run_dir = self._write_run(
                results,
                architecture="shared_heads",
                seed=40,
            )
            (run_dir / "gradient_diagnostics.csv").unlink()
            _write_json(
                results / "incomplete" / "config.json",
                {"model": {"multitask_architecture": "split_taxonomy_age"}},
            )
            manifest = build_generalisation_report(results, output)
            self.assertEqual(manifest["completed_runs"], 1)
            self.assertTrue(
                manifest["figures"]["figure_a_architecture_comparison"]
            )
            self.assertFalse(
                manifest["figures"]["figure_d_gradient_interactions"]
            )
            self.assertTrue(
                (output / "gradient_conflict_summary.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
