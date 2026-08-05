from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.build_holdout_visual_notebook import (
    _model_only,
    build_holdout_visual_notebook_figures,
    pair_taxon_metrics,
    prepare_baseline_frame,
    prepare_taxon_stage_holdout_frame,
    shared_variance_effect_summary,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class HoldoutVisualNotebookTests(unittest.TestCase):
    def test_figures_three_through_six_model_filter_keeps_only_convnext_base(self) -> None:
        mixed = pd.DataFrame({
            "model": ["convnext_base", "resnet18", "dinov3_vitb16"],
            "value": [1, 2, 3],
        })
        selected = _model_only(
            mixed, "convnext_base", context="Figures 3 through 6"
        )
        self.assertEqual(selected["model"].tolist(), ["convnext_base"])
        self.assertEqual(selected["value"].tolist(), [1])

    def test_taxon_stage_metrics_are_included_as_individual_holdouts(self) -> None:
        metrics = pd.DataFrame({
            "holdout": ["adult_species_a", "adult_species_a"],
            "training_regime": ["adult_combo_withheld", "full_data_control"],
            "target_recall_image": [0.4, 0.6],
            "hierarchy_loss_weight": [0.0, 0.0],
            "model": ["resnet18", "resnet18"],
            "cohort": ["independent_test", "independent_test"],
            "task": ["species", "species"],
            "class_supported_by_training_head": [True, True],
        })
        prepared = prepare_taxon_stage_holdout_frame(metrics)
        self.assertEqual(
            prepared["system"].tolist(),
            ["Ablated training", "Full-data baseline"],
        )
        self.assertEqual(prepared["target_recall"].tolist(), [0.4, 0.6])

    def test_shared_variance_effects_are_additive(self) -> None:
        paired = pd.DataFrame({
            "seed": [40, 41, 42],
            "stage": ["Adult"] * 3,
            "task": ["species"] * 3,
            "chance": [0.5] * 3,
            "baseline_target_recall": [0.8, 0.9, 1.0],
            "ablated_target_recall": [0.6, 0.7, 0.8],
        })
        summary = shared_variance_effect_summary(
            paired, identity_columns=("stage",)
        )
        row = summary.iloc[0]
        self.assertEqual(row["n_seeds"], 3)
        self.assertAlmostEqual(row["shared_normal_sd"], 0.1)
        self.assertAlmostEqual(row["d_total"], 4.0)
        self.assertAlmostEqual(row["d_ablation"], 2.0)
        self.assertAlmostEqual(row["d_retained"], 2.0)
        self.assertAlmostEqual(row["m_total"], 0.4)
        self.assertAlmostEqual(row["m_lost"], 0.2)
        self.assertAlmostEqual(row["m_retained"], 0.2)
        self.assertEqual(row["number_of_classes"], 2)
        self.assertEqual(
            row["chance_method"],
            "uniform random prediction across K classes: 1/K",
        )
        self.assertAlmostEqual(
            row["d_total"], row["d_ablation"] + row["d_retained"]
        )

    def test_taxon_stage_design_metadata_is_recovered_from_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            _write_json(run_dir / "config.json", self._base_config(41))
            metrics = pd.DataFrame({
                "holdout": ["adult_aporrectodea_longa"],
                "training_regime": ["adult_combo_withheld"],
                "target_recall_image": [0.4], "macro_f1": [0.5],
                "model": ["convnext_base"],
                "cohort": ["independent_test"], "task": ["species"],
                "class_supported_by_training_head": [True],
                "run_dir": [str(run_dir)],
            })
            prepared = prepare_taxon_stage_holdout_frame(metrics)
        self.assertEqual(prepared["seed"].tolist(), [41])
        self.assertEqual(prepared["hierarchy_loss_weight"].tolist(), [0.0])
        self.assertEqual(
            prepared["loss_recipe"].tolist(),
            ["genus-1_species-0.5_age-2"],
        )

    def _base_config(
        self,
        seed: int,
        *,
        hierarchy_weight: float = 0.0,
        model: str = "resnet18",
    ) -> dict:
        return {
            "seed": seed,
            "model": {"name": model},
            "multi_task": {
                "loss_weights": {"genus": 1.0, "species": 0.5, "age": 2.0},
                "hierarchy_loss": {
                    "enabled": hierarchy_weight > 0,
                    "weight": hierarchy_weight,
                },
            },
        }

    def _write_baseline_runs(self, paper_root: Path) -> None:
        for seed in (40, 140, 240):
            baseline_dir = paper_root / "runs" / "baseline" / f"baseline_{seed}"
            _write_json(baseline_dir / "config.json", self._base_config(seed))
            score = 0.72 + (seed - 40) * 0.0001
            _write_json(baseline_dir / "run_summary.json", {
                "test_mean_macro_f1": score,
                "test_genus_macro_f1": score + 0.10,
                "test_species_macro_f1": score - 0.05,
                "test_age_macro_f1": score - 0.02,
            })
            _write_json(
                baseline_dir / "label_to_index_by_task.json",
                {"genus": {"a": 0, "b": 1}, "species": {"a": 0, "b": 1}, "age": {"Adult": 0, "Juvenile": 1}},
            )

    def _write_visual_runs(self, paper_root: Path) -> None:
        interactions = []
        for gaussian in (25, 50, 75, 100):
            interactions.extend((
                (f"colour_{gaussian}", [
                    {"transform": "gaussian_blur_percent", "parameters": {"percent": gaussian, "max_sigma": 64}},
                    {"transform": "saturation", "parameters": {"retention": 0.0}},
                ]),
                (f"patch8_{gaussian}", [
                    {"transform": "gaussian_blur_percent", "parameters": {"percent": gaussian, "max_sigma": 64}},
                    {"transform": "patch_shuffle", "parameters": {"grid_size": 8}},
                ]),
                (f"patch16_{gaussian}", [
                    {"transform": "gaussian_blur_percent", "parameters": {"percent": gaussian, "max_sigma": 64}},
                    {"transform": "patch_shuffle", "parameters": {"grid_size": 16}},
                ]),
            ))
        for seed in (40, 140, 240):
            simple_conditions = (
                ("original", "resolution_loss", {"percent": 0}),
                ("colour", "saturation", {"retention": 0.0}),
                ("patch8", "patch_shuffle", {"grid_size": 8}),
                ("patch16", "patch_shuffle", {"grid_size": 16}),
                ("blur25", "gaussian_blur_percent", {"percent": 25, "max_sigma": 64}),
                ("blur50", "gaussian_blur_percent", {"percent": 50, "max_sigma": 64}),
                ("blur75", "gaussian_blur_percent", {"percent": 75, "max_sigma": 64}),
                ("blur100", "gaussian_blur_percent", {"percent": 100, "max_sigma": 64}),
                ("resolution50", "resolution_loss", {"percent": 50}),
                ("resolution100", "resolution_loss", {"percent": 100}),
            )
            for name, transform, parameters in simple_conditions:
                run_dir = paper_root / "runs" / "visual_ablation" / f"{name}_{seed}"
                config = self._base_config(seed)
                config["input_condition"] = {
                    "name": name, "transform": transform, "parameters": parameters,
                }
                _write_json(run_dir / "config.json", config)
                _write_json(run_dir / "run_summary.json", {"test_mean_macro_f1": 0.75 - seed * 0.0005})
            for name, operations in interactions:
                run_dir = paper_root / "runs" / "visual_interactions" / f"{name}_{seed}"
                config = self._base_config(seed)
                config["input_condition"] = {
                    "name": f"blur025_{name}",
                    "transform": "composed",
                    "parameters": {"operations": operations},
                }
                _write_json(run_dir / "config.json", config)
                _write_json(
                    run_dir / "run_summary.json",
                    {"test_mean_macro_f1": 0.6 + seed * 0.001},
                )
                _write_json(
                    run_dir / "label_to_index_by_task.json",
                    {
                        "genus": {"a": 0, "b": 1},
                        "species": {"a": 0, "b": 1},
                        "age": {"Adult": 0, "Juvenile": 1},
                    },
                )

    def _write_taxon_runs(self, taxon_root: Path) -> None:
        definitions = [
            {
                "name": f"{stage.lower()}_aporrectodea_longa",
                "where": {
                    "genus": "Aporrectodea",
                    "species": "Aporrectodea_longa",
                    "age": stage,
                },
                "evaluation_where": {
                    "genus": "Aporrectodea",
                    "species": "Aporrectodea_longa",
                    "age": stage,
                },
            }
            for stage in ("Adult", "Juvenile")
        ]
        for seed in (40, 140, 240):
            baseline_dir = taxon_root / "runs" / "adult_taxon_baseline" / f"run_{seed}"
            config = self._base_config(seed, model="convnext_base")
            config["evaluation"] = {
                "data_holdout_controls": {"definitions": definitions}
            }
            _write_json(baseline_dir / "config.json", config)
            _write_json(baseline_dir / "label_to_index_by_task.json", {
                "genus": {"a": 0, "b": 1},
                "species": {"a": 0, "b": 1},
                "age": {"Adult": 0, "Juvenile": 1},
            })
            baseline_rows = []
            for definition_index, definition in enumerate(definitions):
                base = 0.72 + (seed - 40) * 0.0001 - definition_index * 0.04
                baseline_rows.extend({
                    "holdout": definition["name"],
                    "cohort": "independent_test",
                    "task": task,
                    "macro_f1": base - task_index * 0.02,
                    "target_recall": base - task_index * 0.03,
                    "target_recall_image": base - task_index * 0.03,
                    "class_supported_by_training_head": True,
                } for task_index, task in enumerate(("genus", "species", "age")))
            metrics_dir = baseline_dir / "data_holdout_control_evaluation"
            metrics_dir.mkdir(parents=True)
            pd.DataFrame(baseline_rows).to_csv(
                metrics_dir / "task_metrics.csv", index=False
            )

            for definition_index, definition in enumerate(definitions):
                run_dir = (
                    taxon_root / "runs" / "adult_taxon_holdouts"
                    / f"{definition['name']}_{seed}"
                )
                config = self._base_config(seed, model="convnext_base")
                config["data_holdout"] = definition
                _write_json(run_dir / "config.json", config)
                _write_json(run_dir / "label_to_index_by_task.json", {
                    "genus": {"a": 0, "b": 1}, "species": {"a": 0, "b": 1}, "age": {"Adult": 0, "Juvenile": 1},
                })
                base = 0.72 + (seed - 40) * 0.0001 - definition_index * 0.04
                shift = -0.08 - definition_index * 0.02
                rows = [{
                    "holdout": definition["name"],
                    "cohort": "independent_test",
                    "task": task,
                    "macro_f1": base + shift - index * 0.02,
                    "target_recall": base + shift - index * 0.03,
                    "target_recall_image": base + shift - index * 0.03,
                    "class_supported_by_training_head": True,
                } for index, task in enumerate(("genus", "species", "age"))]
                metrics_dir = run_dir / "data_holdout_evaluation"
                metrics_dir.mkdir(parents=True)
                pd.DataFrame(rows).to_csv(metrics_dir / "task_metrics.csv", index=False)

    def test_pairing_keeps_f1_recall_and_paired_shift(self) -> None:
        metrics = pd.DataFrame({
            "training_regime": ["full_data_control", "adult_combo_withheld"],
            "model": ["convnext_base", "convnext_base"], "seed": [40, 40],
            "holdout": ["adult_aporrectodea_longa"] * 2,
            "cohort": ["independent_test"] * 2, "task": ["species"] * 2,
            "species": ["Aporrectodea_longa"] * 2, "stage": ["Adult"] * 2,
            "macro_f1": [0.8, 0.7], "target_recall_image": [0.75, 0.60],
            "class_supported_by_training_head": [True, True],
        })
        paired = pair_taxon_metrics(prepare_taxon_stage_holdout_frame(metrics))
        self.assertAlmostEqual(paired.iloc[0]["delta_macro_f1"], -0.1)
        self.assertAlmostEqual(paired.iloc[0]["delta_target_recall"], -0.15)

    def test_pairing_excludes_non_test_cohorts(self) -> None:
        rows = []
        for cohort, baseline, ablated in (
            ("independent_test", 0.8, 0.7),
            ("development_withheld", 0.3, 0.95),
        ):
            for regime, score in (
                ("full_data_control", baseline),
                ("adult_combo_withheld", ablated),
            ):
                rows.append({
                    "training_regime": regime,
                    "model": "convnext_base", "seed": 40,
                    "holdout": "adult_aporrectodea_longa",
                    "cohort": cohort, "task": "species",
                    "species": "Aporrectodea_longa", "stage": "Adult",
                    "macro_f1": score, "target_recall_image": score,
                    "class_supported_by_training_head": True,
                })
        paired = pair_taxon_metrics(
            prepare_taxon_stage_holdout_frame(pd.DataFrame(rows))
        )
        self.assertEqual(paired["cohort"].unique().tolist(), ["independent_test"])
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired.iloc[0]["delta_macro_f1"], -0.1)

    def test_baseline_rejects_other_seeds_hierarchy_and_weight_recipes(self) -> None:
        correct_recipe = "genus-1_species-0.5_age-2"
        rows = []
        for seed, hierarchy, recipe in (
            (40, 0.0, correct_recipe),
            (99, 0.0, correct_recipe),
            (41, 0.2, correct_recipe),
            (42, 0.0, "genus-1_species-1_age-1"),
        ):
            rows.append({
                "stage": "baseline", "model": "convnext_base",
                "seed": seed, "hierarchy_loss_weight": hierarchy,
                "loss_recipe": recipe, "run_dir": f"/tmp/{seed}-{hierarchy}",
                "test_mean_macro_f1": 0.8,
                "test_genus_macro_f1": 0.8,
                "test_species_macro_f1": 0.8,
                "test_age_macro_f1": 0.8,
            })
        prepared = prepare_baseline_frame(pd.DataFrame(rows))
        self.assertEqual(prepared["seed"].unique().tolist(), [40])
        self.assertEqual(prepared["loss_recipe"].unique().tolist(), [correct_recipe])

    def test_builds_six_metric_figures_and_tracks_figure_seven(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paper_root = root / "paper_result"
            taxon_root = root / "adult_taxon_ablation_result"
            output_dir = root / "figures"
            self._write_baseline_runs(paper_root)
            self._write_visual_runs(paper_root)
            self._write_taxon_runs(taxon_root)

            manifest = build_holdout_visual_notebook_figures(
                paper_root, output_dir, taxon_root, visual_model="resnet18"
            )
            self.assertIn("representative_transformations", manifest["figures"])

            expected = (
                "figure_01_all_models_all_tasks",
                "figure_02_convnext_visual_ablation",
                "figure_03_species_ablation",
                "figure_04_all_data_ablations",
                "figure_05_species_ablation_raw_margins",
                "figure_06_all_data_ablations_raw_margins",
            )
            for stem in expected:
                for extension in ("png", "pdf", "svg"):
                    self.assertTrue(
                        (output_dir / f"{stem}.{extension}").is_file(),
                        f"{stem}.{extension}",
                    )
            summary = pd.read_csv(
                output_dir / "figure_sources" / "figure_02_convnext_visual_ablation"
                / "seed_summary.csv"
            )
            self.assertEqual(set(summary["number_of_seeds"]), {3})
            effect = pd.read_csv(
                output_dir / "figure_sources" / "figure_04_all_data_ablations"
                / "shared_variance_effects.csv"
            )
            taxon_plot_data = pd.read_csv(
                output_dir / "figure_sources" / "figure_04_all_data_ablations"
                / "recall_seed_data.csv"
            )
            self.assertEqual(set(taxon_plot_data["model"]), {"convnext_base"})
            self.assertEqual(manifest["taxon_model"], "convnext_base")
            self.assertEqual(set(effect["task"]), {"genus", "species", "age"})
            self.assertEqual(set(effect["stage"]), {"Adult", "Juvenile"})
            self.assertEqual(set(effect["n_seeds"]), {3})
            self.assertNotIn("macro_f1", taxon_plot_data)
            self.assertNotIn("mean_skill_retained", effect)
            self.assertTrue(effect[[
                "d_total", "d_ablation", "d_retained"
            ]].notna().all().all())
            additive_error = (
                effect["d_total"]
                - effect["d_ablation"]
                - effect["d_retained"]
            ).abs()
            self.assertTrue((additive_error < 1e-10).all())
            figure_three_effect = pd.read_csv(
                output_dir / "figure_sources" / "figure_03_species_ablation"
                / "shared_variance_effects.csv"
            )
            self.assertEqual(
                set(figure_three_effect["stage"]), {"Adult", "Juvenile"}
            )
            figure_five_margin = pd.read_csv(
                output_dir / "figure_sources"
                / "figure_05_species_ablation_raw_margins"
                / "raw_margin_decomposition.csv"
            )
            self.assertEqual(set(figure_five_margin["stage"]), {"Adult", "Juvenile"})
            self.assertTrue((
                figure_five_margin["m_total"]
                - figure_five_margin["m_lost"]
                - figure_five_margin["m_retained"]
            ).abs().lt(1e-10).all())
            figure_six_manifest = json.loads((
                output_dir / "figure_sources"
                / "figure_06_all_data_ablations_raw_margins"
                / "manifest.json"
            ).read_text(encoding="utf-8"))
            self.assertEqual(figure_six_manifest["chance_formula"], "1/K")
            self.assertIn("label_to_index_by_task.json", figure_six_manifest["chance_source"])

    def test_notebook_is_valid_json_with_well_formed_code_cells(self) -> None:
        notebook = (
            Path(__file__).resolve().parents[1]
            / "notebooks" / "holdouts_and_visual_combinations.ipynb"
        )
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        code_cells = [
            cell for cell in payload["cells"] if cell["cell_type"] == "code"
        ]
        self.assertTrue(code_cells)
        self.assertTrue(all(isinstance(cell.get("outputs"), list) for cell in code_cells))
        self.assertTrue(all("source" in cell for cell in code_cells))


if __name__ == "__main__":
    unittest.main()
