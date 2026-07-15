from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from src.worm_species.experiments.run_specs import write_run_specs
from src.worm_species.slurm.config import (
    SlurmConfigError,
    load_submission_config,
    parse_memory,
    parse_time_limit,
    validate_slurm_config,
)
from src.worm_species.slurm.planning import plan_submission
from src.worm_species.training.modes import infer_experiment_type
from src.worm_species.training.modes import resolve_configured_profile


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiments"
CLUSTERS = ROOT / "configs" / "clusters"
PRE_REFACTOR_ARGS_MANIFEST_SHA256 = (
    "0c9f4238c956d59eb4c46e0a70e7003bb868de994415ccd565bc945af2a53115"
)
PRE_REFACTOR_SWEEP_PLAN_SHA256 = (
    "8340964b84ae0677304324384a3e81703e3f5b9829497ee08ecc7849f920954c"
)


class SlurmPlanningContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dual_local_config = load_submission_config(
            EXPERIMENTS / "dual_cue.yaml", CLUSTERS / "local.yaml"
        )
        cls.dual_local_plan = plan_submission(cls.dual_local_config)

    def test_cluster_merge_and_resource_parsing(self) -> None:
        config = self.dual_local_config
        self.assertEqual(config["slurm"]["cluster_profile"], "local")
        self.assertEqual(config["slurm"]["cpus_per_task"], 1)
        self.assertEqual(
            config["slurm"]["planning"]["external_expansion"], "dual_cue"
        )
        self.assertEqual(parse_memory("16G"), 16384)
        self.assertEqual(parse_memory(12384), 12384)
        self.assertEqual(parse_time_limit("01:30:00"), 5400)

    def test_dual_cue_has_224_byte_compatible_specs(self) -> None:
        plan = self.dual_local_plan
        self.assertEqual(plan.array_size, 224)
        self.assertEqual(len(plan.models), 2)
        self.assertEqual(len(plan.conditions), 112)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs = root / "specs"
            count = write_run_specs(
                ROOT / "config.yaml", specs, root / "sweep_plan.tsv"
            )
            self.assertEqual(count, 224)
            manifest = b"".join(
                hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii")
                + b"  "
                + path.name.encode("utf-8")
                + b"\n"
                for path in sorted(specs.glob("*.args"))
            )
            self.assertEqual(
                hashlib.sha256(manifest).hexdigest(),
                PRE_REFACTOR_ARGS_MANIFEST_SHA256,
            )
            self.assertEqual(
                hashlib.sha256((root / "sweep_plan.tsv").read_bytes()).hexdigest(),
                PRE_REFACTOR_SWEEP_PLAN_SHA256,
            )
            for run_spec in plan.run_specs:
                self.assertEqual(
                    run_spec.args_text.encode(),
                    (specs / f"{run_spec.run_id}.args").read_bytes(),
                )
                self.assertNotIn("profile", run_spec.resolved_config["training"])
                self.assertFalse(run_spec.resolved_config["sweep"]["enabled"])
                self.assertFalse(
                    run_spec.resolved_config["colour_ablation"]["enabled"]
                )
                self.assertFalse(
                    run_spec.resolved_config["matched_condition_training"]["enabled"]
                )

        original = [
            spec for spec in plan.run_specs if spec.training_transform == "original"
        ]
        transformed = [
            spec for spec in plan.run_specs if spec.training_transform != "original"
        ]
        self.assertEqual(len(original), 2)
        self.assertEqual(len(transformed), 222)
        self.assertTrue(
            all(spec.experiment_type == "matched_and_rgb_stress" for spec in original)
        )
        self.assertTrue(
            all(spec.experiment_type == "matched_condition" for spec in transformed)
        )
        self.assertTrue(
            all(
                spec.resolved_config["test_cue_suppression"]["enabled"]
                for spec in original
            )
        )
        self.assertTrue(
            all(
                not spec.resolved_config["test_cue_suppression"]["enabled"]
                for spec in transformed
            )
        )

    def test_colour_ablation_has_202_runs_and_both_endpoints(self) -> None:
        config = load_submission_config(
            EXPERIMENTS / "colour_ablation.yaml", CLUSTERS / "local.yaml"
        )
        plan = plan_submission(config)
        self.assertEqual(plan.array_size, 202)
        self.assertEqual(len(plan.conditions), 101)
        self.assertEqual(plan.run_specs[0].run_id, "run_000_colour_100pct")
        self.assertEqual(plan.run_specs[-1].run_id, "run_201_colour_000pct")
        for spec in (plan.run_specs[0], plan.run_specs[-1]):
            self.assertEqual(spec.experiment_type, "matched_condition")
            self.assertEqual(
                resolve_configured_profile(spec.resolved_config).loader_mode,
                "colour",
            )

    def test_ghpc_requires_explicit_nodes_and_preserves_dependency_dag(self) -> None:
        with self.assertRaisesRegex(SlurmConfigError, "scratch.nodes"):
            load_submission_config(
                EXPERIMENTS / "dual_cue.yaml", CLUSTERS / "ghpc.yaml"
            )
        config = load_submission_config(
            EXPERIMENTS / "dual_cue.yaml",
            CLUSTERS / "ghpc.yaml",
            ["slurm.scratch.nodes=gpu001,gpu002"],
        )
        plan = plan_submission(config)
        dependencies = {
            (item.upstream, item.downstream, item.kind)
            for item in plan.dependencies
        }
        self.assertEqual(
            dependencies,
            {
                ("setup", "train_array", "afterok"),
                ("train_array", "collect", "afterany"),
                ("train_array", "cleanup", "afterany"),
            },
        )

    def test_scientific_hashes_do_not_depend_on_cluster(self) -> None:
        ghpc = load_submission_config(
            EXPERIMENTS / "dual_cue.yaml",
            CLUSTERS / "ghpc.yaml",
            ["slurm.scratch.nodes=gpu001,gpu002"],
        )
        genome = load_submission_config(
            EXPERIMENTS / "dual_cue.yaml", CLUSTERS / "genome.yaml"
        )
        ghpc_plan = plan_submission(ghpc)
        genome_plan = plan_submission(genome)
        self.assertEqual(
            [spec.config_sha256 for spec in ghpc_plan.run_specs],
            [spec.config_sha256 for spec in genome_plan.run_specs],
        )
        self.assertTrue(
            all("slurm" not in spec.resolved_config for spec in ghpc_plan.run_specs)
        )

    def test_managed_sbatch_arguments_are_rejected(self) -> None:
        config = copy.deepcopy(self.dual_local_config)
        config["slurm"]["submission"]["extra_sbatch_args"] = ["--array=0-3"]
        with self.assertRaisesRegex(SlurmConfigError, "managed"):
            validate_slurm_config(config)

    def test_single_spec_maps_to_one_configuration_driven_trainer(self) -> None:
        config = load_submission_config(
            EXPERIMENTS / "standard.yaml",
            CLUSTERS / "local.yaml",
            ["sweep.enabled=false"],
        )
        plan = plan_submission(config)
        self.assertEqual(plan.array_size, 1)
        self.assertEqual(plan.expected_internal_training_runs_per_task, 1)
        resolved = plan.run_specs[0].resolved_config
        self.assertTrue(resolved["training"]["use_masked_labels"])
        self.assertFalse(resolved["multi_task"]["hierarchy_loss"]["enabled"])
        self.assertFalse(resolved["wandb"]["enabled"])
        self.assertEqual(infer_experiment_type(resolved), "standard")
        self.assertEqual(resolve_configured_profile(resolved).loader_mode, "standard")
        self.assertEqual(
            plan.run_specs[0].trainer_command,
            (
                "python",
                "-m",
                "worm_species.training",
                "--config",
                "resolved_run_config.yaml",
                "--single-run",
            ),
        )
        self.assertNotIn("--profile", plan.canonical_trainer_command)
        self.assertTrue(
            all(
                not spec.resolved_config.get("sweep", {}).get("enabled", False)
                and not spec.resolved_config.get("colour_ablation", {}).get(
                    "enabled", False
                )
                and not spec.resolved_config.get(
                    "matched_condition_training", {}
                ).get("enabled", False)
                for spec in plan.run_specs
            )
        )

    def test_named_profile_controls_are_rejected_clearly(self) -> None:
        slurm_profile = copy.deepcopy(self.dual_local_config)
        slurm_profile["slurm"]["planning"]["training_profile"] = "cue_suppression"
        with self.assertRaisesRegex(SlurmConfigError, "no longer supported"):
            validate_slurm_config(slurm_profile)

        trainer_profile = copy.deepcopy(self.dual_local_config)
        trainer_profile.setdefault("training", {})["profile"] = "cue_suppression"
        with self.assertRaisesRegex(SlurmConfigError, "training.profile"):
            plan_submission(trainer_profile)


if __name__ == "__main__":
    unittest.main()
