from __future__ import annotations

import copy
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


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiments"
CLUSTERS = ROOT / "configs" / "clusters"


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
            for run_spec in plan.run_specs:
                self.assertEqual(
                    run_spec.args_text.encode(),
                    (specs / f"{run_spec.run_id}.args").read_bytes(),
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

    def test_single_spec_maps_to_one_profiled_trainer_command(self) -> None:
        config = load_submission_config(
            EXPERIMENTS / "standard.yaml",
            CLUSTERS / "local.yaml",
            ["sweep.enabled=false"],
        )
        plan = plan_submission(config)
        self.assertEqual(plan.array_size, 1)
        self.assertEqual(plan.expected_internal_training_runs_per_task, 1)
        self.assertEqual(plan.training_profile, "masked")
        self.assertEqual(
            plan.run_specs[0].trainer_command,
            (
                "python",
                "train.py",
                "--config",
                "resolved_run_config.yaml",
                "--profile",
                "masked",
                "--single-run",
            ),
        )


if __name__ == "__main__":
    unittest.main()
