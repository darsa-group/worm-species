from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

import collect_dual_cue_results
import generate_dual_cue_run_specs
import train_multitask_masked_hloss as hierarchy_trainer
from src.dataset_multitask import (
    MISSING_LABEL,
    ColourRetention,
    build_condition_transform,
    build_test_condition_transform,
    is_missing_label,
)
from src.utils import apply_overrides, make_run_name, parse_scalar, short_hash
from src.worm_species.models.multitask import MultiTaskClassifier
from src.worm_species.data.labels import read_csvs_from_dir


ROOT = Path(__file__).resolve().parents[1]
TRAINERS = [
    "train_multitask_masked.py",
    "train_multitask_masked_hloss.py",
    "train_multitask_masked_hloss_wandb.py",
    "train_multitask_colour_ablation.py",
    "train_multitask_cue_suppression.py",
]
ORDINARY_CHECKPOINT_KEYS = {
    "model_state",
    "cfg",
    "label_to_index_by_task",
    "index_to_label_by_task",
    "best_val_score",
    "selection_metric",
    "best_epoch",
}


def torch_save_dict_keys(path: Path) -> list[set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    schemas: list[set[str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if ast.unparse(node.func) != "torch.save" or not isinstance(node.args[0], ast.Dict):
            continue
        keys = {
            str(key.value)
            for key in node.args[0].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        schemas.append(keys)
    return schemas


class FixedDataset(Dataset):
    def __init__(self) -> None:
        self.labels = {
            "genus": torch.tensor([0, 1, 0, 1]),
            "species": torch.tensor([0, 1, MISSING_LABEL, 1]),
        }

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.zeros(1),
            "labels": {task: values[index] for task, values in self.labels.items()},
        }


class FixedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "genus_logits",
            torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]),
        )
        self.register_buffer(
            "species_logits",
            torch.tensor([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [0.0, 4.0]]),
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        size = inputs.shape[0]
        return {
            "genus": self.genus_logits[:size],
            "species": self.species_logits[:size],
        }


class ConfigAndSweepContracts(unittest.TestCase):
    def test_scalar_and_override_contract(self) -> None:
        self.assertIs(parse_scalar("null"), None)
        self.assertIs(parse_scalar("true"), True)
        self.assertEqual(parse_scalar("12"), 12)
        self.assertEqual(parse_scalar("0.25"), 0.25)
        self.assertEqual(parse_scalar("value"), "value")

        original = {"model": {"name": "a"}, "training": {"lr": 0.1}}
        changed = apply_overrides(original, ["model.name=b", "training.lr=0.01"])
        self.assertEqual(original["model"]["name"], "a")
        self.assertEqual(changed["model"]["name"], "b")
        self.assertEqual(changed["training"]["lr"], 0.01)

    def test_sweep_product_and_run_name_contract(self) -> None:
        config = {
            "seed": 3,
            "model": {"name": "resnet18"},
            "data": {"image_col": "rel_path_seg", "target_col": "genus"},
            "sweep": {
                "enabled": True,
                "parameters": {"model.name": ["a", "b"], "training.lr": [0.1, 0.2]},
            },
        }
        expanded = hierarchy_trainer.generate_sweep_configs(config)
        self.assertEqual(len(expanded), 4)
        self.assertEqual(
            [(item["model"]["name"], item["training"]["lr"]) for item in expanded],
            [("a", 0.1), ("a", 0.2), ("b", 0.1), ("b", 0.2)],
        )
        self.assertEqual(short_hash({"a": 1}), "42b7b4f2")
        self.assertEqual(
            make_run_name(config),
            f"resnet18__rel_path_seg__genus__{short_hash(config)}",
        )


class ConditionContracts(unittest.TestCase):
    def minimal_config(self, models: list[str]) -> dict:
        return {
            "seed": 42,
            "model": {"name": models[0]},
            "sweep": {"enabled": True, "parameters": {"model.name": models}},
            "matched_condition_training": {
                "enabled": True,
                "include_original": True,
                "deduplicate_equivalent_conditions": True,
                "evaluate_original_model_on_all_test_conditions": True,
            },
            "test_cue_suppression": {
                "saturation": {"enabled": True, "values": [1.0, 0.5, 0.0]},
                "grayscale": {"enabled": True},
                "channel_shuffle": {"enabled": False},
                "bilateral_filter": {"enabled": False},
                "gaussian_blur": {"enabled": False},
                "patch_shuffle": {"enabled": False},
            },
        }

    def test_condition_endpoint_deduplication(self) -> None:
        conditions = generate_dual_cue_run_specs.generate_conditions(
            self.minimal_config(["resnet18"])
        )
        self.assertEqual(
            [item["condition"] for item in conditions],
            ["original", "saturation_050pct", "grayscale"],
        )
        self.assertEqual(generate_dual_cue_run_specs.inclusive_sequence(1, 0, 0.5), [1.0, 0.5, 0.0])

    def test_one_and_two_model_run_specs_disable_nested_expansion(self) -> None:
        for models in (["resnet18"], ["resnet18", "vit_b_16"]):
            with self.subTest(models=models), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config_path = root / "config.yaml"
                specs = root / "specs"
                plan = root / "sweep_plan.tsv"
                config_path.write_text(yaml.safe_dump(self.minimal_config(models)))
                result = subprocess.run(
                    [sys.executable, str(ROOT / "generate_dual_cue_run_specs.py"), str(config_path), str(specs), str(plan)],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                expected = len(models) * 3
                self.assertEqual(int(result.stdout.strip()), expected)
                self.assertEqual(len(list(specs.glob("run_*.args"))), expected)
                for spec in specs.glob("run_*.args"):
                    text = spec.read_text()
                    self.assertIn("matched_condition_training.enabled=false", text)
                self.assertEqual(len(plan.read_text().splitlines()), expected + 1)

        for launcher in ("submit_dual_cue_experiment.sh", "submit_dual_cue_experiment_genome.sh"):
            self.assertIn("sweep.enabled=false", (ROOT / launcher).read_text())


class TransformAndLabelContracts(unittest.TestCase):
    def test_predefined_split_path_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            split_dir = Path(directory) / "split_csv"
            split_dir.mkdir()
            for name, value in (("train", 1), ("val", 2), ("test", 3)):
                pd.DataFrame({"value": [value]}).to_csv(
                    split_dir / f"{name}_split.csv", index=False
                )
            train, validation, test = read_csvs_from_dir(directory)
            self.assertEqual(train["value"].tolist(), [1])
            self.assertEqual(validation["value"].tolist(), [2])
            self.assertEqual(test["value"].tolist(), [3])

    def test_transform_order_and_deterministic_test_transform(self) -> None:
        condition = {
            "transform": "channel_shuffle",
            "order": [2, 0, 1],
        }
        evaluation = build_test_condition_transform(16, condition)
        self.assertEqual(
            [type(operation).__name__ for operation in evaluation.transforms],
            ["Resize", "ToTensor", "ColourRetention", "ChannelShuffle", "Normalize"],
        )
        training = build_condition_transform(16, True, condition)
        self.assertEqual(
            [type(operation).__name__ for operation in training.transforms],
            [
                "Resize",
                "RandomHorizontalFlip",
                "RandomVerticalFlip",
                "RandomRotation",
                "ToTensor",
                "ColourRetention",
                "ChannelShuffle",
                "Normalize",
            ],
        )
        image = Image.fromarray(np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3))
        self.assertTrue(torch.equal(evaluation(image), evaluation(image)))
        self.assertIsInstance(evaluation.transforms[2], ColourRetention)

    def test_missing_label_and_class_map_contract(self) -> None:
        for value in (None, np.nan, "", "NA", "unknown", "unidentified"):
            self.assertTrue(is_missing_label(value))
        frame = pd.DataFrame(
            {
                "genus": ["Lumbricus", "Aporrectodea", "Lumbricus"],
                "species": ["Lumbricus_rubellus", None, "Lumbricus_terrestris"],
            }
        )
        label_maps, inverse_maps = hierarchy_trainer.build_label_maps(
            frame, {"genus": "genus", "species": "species"}
        )
        self.assertEqual(label_maps["genus"], {"Aporrectodea": 0, "Lumbricus": 1})
        self.assertEqual(
            label_maps["species"],
            {"Lumbricus_rubellus": 0, "Lumbricus_terrestris": 1},
        )
        self.assertEqual(inverse_maps["genus"], {0: "Aporrectodea", 1: "Lumbricus"})


class LossMetricAndCheckpointContracts(unittest.TestCase):
    def test_multitask_head_names_and_shapes(self) -> None:
        class Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Linear(3, 4)
                self.fc = nn.Linear(4, 2)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.fc(self.projection(inputs))

        model = MultiTaskClassifier(Backbone(), {"genus": 2, "species": 3})
        outputs = model(torch.zeros(2, 3))
        self.assertEqual(outputs["genus"].shape, (2, 2))
        self.assertEqual(outputs["species"].shape, (2, 3))
        self.assertEqual(
            list(model.state_dict()),
            [
                "backbone.projection.weight",
                "backbone.projection.bias",
                "heads.genus.weight",
                "heads.genus.bias",
                "heads.species.weight",
                "heads.species.bias",
            ],
        )

    def test_hierarchy_mapping_and_zero_consistency_loss(self) -> None:
        mapping = hierarchy_trainer.build_child_to_parent_matrix(
            {
                "genus": {"Aporrectodea": 0, "Lumbricus": 1},
                "species": {"Aporrectodea_rosea": 0, "Lumbricus_rubellus": 1},
            },
            parent_task="genus",
            child_task="species",
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.equal(mapping, torch.eye(2)))
        logits = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
        loss = hierarchy_trainer.hierarchy_consistency_loss(
            logits, logits, mapping, torch.tensor([True, True])
        )
        self.assertIsNotNone(loss)
        self.assertTrue(math.isclose(float(loss), 0.0, abs_tol=1e-7))

    def test_metric_keys_and_macro_f1(self) -> None:
        metrics, true, predicted = hierarchy_trainer.run_epoch(
            model=FixedModel(),
            loader=DataLoader(FixedDataset(), batch_size=4),
            criteria={"genus": nn.CrossEntropyLoss(), "species": nn.CrossEntropyLoss()},
            optimizer=None,
            device=torch.device("cpu"),
            train=False,
            use_amp=False,
        )
        expected = {
            "loss",
            "genus_loss", "genus_n", "genus_accuracy", "genus_balanced_accuracy", "genus_macro_f1",
            "species_loss", "species_n", "species_accuracy", "species_balanced_accuracy", "species_macro_f1",
            "mean_macro_f1", "complete_exact_match_accuracy", "complete_exact_match_n",
        }
        self.assertEqual(set(metrics), expected)
        self.assertEqual(metrics["genus_macro_f1"], 1.0)
        self.assertEqual(metrics["species_macro_f1"], 1.0)
        self.assertEqual(metrics["mean_macro_f1"], 1.0)
        self.assertEqual(true, predicted)

    def test_checkpoint_top_level_schemas(self) -> None:
        trainer_root = ROOT / "scripts" / "training"
        ordinary = [
            trainer_root / "train_multitask_masked.py",
            trainer_root / "train_multitask_masked_hloss.py",
            trainer_root / "train_multitask_masked_hloss_wandb.py",
        ]
        for path in ordinary:
            self.assertIn(ORDINARY_CHECKPOINT_KEYS, torch_save_dict_keys(path))
        self.assertIn(
            ORDINARY_CHECKPOINT_KEYS | {"colour_retention", "colour_percent"},
            torch_save_dict_keys(trainer_root / "train_multitask_colour_ablation.py"),
        )
        self.assertIn(
            ORDINARY_CHECKPOINT_KEYS | {"colour_retention", "colour_percent", "training_condition"},
            torch_save_dict_keys(trainer_root / "train_multitask_cue_suppression.py"),
        )


class CollectionAndInterfaceContracts(unittest.TestCase):
    def test_matched_stress_join_contract(self) -> None:
        matched = pd.DataFrame(
            [{
                "run_name": "r", "model": "m", "train_condition": "original",
                "train_feature": "baseline", "train_transform": "original",
                "train_strength": 0.0, "test_genus_macro_f1": 0.8,
            }]
        )
        long = collect_dual_cue_results.matched_results_long(matched)
        cue = pd.DataFrame(
            [{
                "model": "m", "task": "genus", "condition": "original",
                "feature": "baseline", "transform": "original", "strength": 0.0,
                "macro_f1": 0.7, "original_macro_f1": 0.7,
                "ratio_to_original": 1.0, "relative_drop": 0.0,
            }]
        )
        comparison = collect_dual_cue_results.build_comparison(long, cue)
        self.assertEqual(len(comparison), 1)
        self.assertAlmostEqual(comparison.loc[0, "adaptation_gain_macro_f1"], 0.1)
        self.assertEqual(comparison.loc[0, "train_condition"], "original")

    def test_legacy_cli_help_flags(self) -> None:
        for script in TRAINERS:
            result = subprocess.run(
                [sys.executable, str(ROOT / script), "--help"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                env={"PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertIn("--config CONFIG", result.stdout)
            self.assertIn("--override [OVERRIDE ...]", result.stdout)
            self.assertIn("--sweep [SWEEP ...]", result.stdout)

    def test_shell_syntax(self) -> None:
        scripts = [
            path for path in ROOT.glob("*.sh")
            if path.is_file()
        ]
        self.assertTrue(scripts)
        for script in scripts:
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True, cwd=ROOT)

    def test_notebook_parse_and_code_cell_contract(self) -> None:
        invalid: list[str] = []
        for notebook_path in ROOT.rglob("*.ipynb"):
            relative = notebook_path.relative_to(ROOT)
            if any(part in {"outputs", "outputs_slurm", ".ipynb_checkpoints"} for part in relative.parts):
                continue
            try:
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                invalid.append(relative.as_posix())
                continue
            for index, cell in enumerate(notebook.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                source = cell.get("source", [])
                text = "".join(source) if isinstance(source, list) else str(source)
                sanitised = "\n".join(
                    line for line in text.splitlines()
                    if not line.lstrip().startswith(("%", "!"))
                )
                compile(sanitised, f"{relative}:cell{index}", "exec")
        self.assertEqual(invalid, ["notebooks/interpretability/cam_mutlitask.ipynb"])


if __name__ == "__main__":
    unittest.main()
