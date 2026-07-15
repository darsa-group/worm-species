"""Migration contracts for the two ordinary GHPC training launchers."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.worm_species.slurm.config import SlurmConfigError
from src.worm_species.slurm.config import load_submission_config
from src.worm_species.slurm.planning import plan_submission
from src.worm_species.slurm.rendering import write_artifact_bundle


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs/experiments"
GHPC = ROOT / "configs/clusters/ghpc.yaml"
SNAPSHOT = ROOT / "tests/fixtures/slurm_execution/ghpc_standard_hierarchy.snapshot"


def _different_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        paths: set[str] = set()
        for key in left.keys() | right.keys():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths.update(_different_paths(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = set()
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            paths.update(
                _different_paths(left_item, right_item, f"{prefix}[{index}]")
            )
        if len(left) != len(right):
            paths.add(f"{prefix}.length")
        return paths
    return set() if left == right else {prefix}


class GhpcStandardHierarchyMigrationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.project = cls.root / "project"
        cls.data = cls.root / "data"
        cls.project.mkdir()
        (cls.project / "src").mkdir()
        (cls.data / "01_Segmented").mkdir(parents=True)
        cls.metadata = cls.data / "01_Segmented/global_metadata.csv"
        cls.metadata.write_text("image\n", encoding="utf-8")
        cls.conda_sh = cls.root / "conda.sh"
        cls.conda_sh.write_text("conda() { :; }\n", encoding="utf-8")
        cls.nodes = ["gpu01", "gpu02"]
        cls.configs = {}
        cls.plans = {}
        cls.manifests = {}
        cls.generated = {}
        for offset, (name, filename) in enumerate(
            (("standard", "standard.yaml"), ("hierarchy", "hierarchy.yaml"))
        ):
            results = cls.root / f"results-{name}"
            config = load_submission_config(
                EXPERIMENTS / filename,
                GHPC,
                [
                    f"slurm.scratch.nodes={','.join(cls.nodes)}",
                    f"slurm.scratch.root={cls.root / ('scratch-' + name)}",
                    f"slurm.paths.project_root={cls.project}",
                    f"slurm.paths.data_root={cls.data}",
                    f"slurm.paths.metadata_csv={cls.metadata}",
                    f"slurm.paths.results_root={results}",
                    f"slurm.environment.conda_sh={cls.conda_sh}",
                ],
                environment={"HOME": str(cls.root), "USER": "alice"},
                cwd=cls.root,
                submission_stamp="20260102_030405",
                process_id=4321 + offset,
            )
            plan = plan_submission(config)
            bundle = cls.root / f"bundle-{name}"
            manifest = write_artifact_bundle(plan, config, bundle)
            cls.configs[name] = config
            cls.plans[name] = plan
            cls.manifests[name] = manifest
            cls.generated[name] = bundle / "generated_slurm"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _projection(self) -> dict[str, Any]:
        profiles = {}
        for name in ("standard", "hierarchy"):
            config = self.configs[name]
            plan = self.plans[name]
            manifest = self.manifests[name]
            train_job = next(
                job for job in manifest["jobs"] if job["role"] == "train_array"
            )
            profiles[name] = {
                "array": train_job["array"],
                "array_size": plan.array_size,
                "collector_enabled": config["slurm"]["collection"]["enabled"],
                "dependencies": [
                    [item.upstream, item.downstream, item.kind]
                    for item in plan.dependencies
                ],
                "generated_files": sorted(
                    path.name for path in self.generated[name].glob("*.sh")
                ),
                "hierarchy_enabled": [
                    spec.resolved_config["multi_task"]["hierarchy_loss"]["enabled"]
                    for spec in plan.run_specs
                ],
                "models": list(plan.models),
                "nodelists": [
                    job["nodelist"]
                    for job in manifest["jobs"]
                    if job["role"] in {"setup", "cleanup"}
                ],
                "output_relpaths": [spec.output_relpath for spec in plan.run_specs],
                "plan_label": plan.experiment_type,
                "resources": {
                    "cpus_per_task": train_job["cpus_per_task"],
                    "memory_mib": train_job["memory_mib"],
                    "time_limit": train_job["time_limit"],
                    "max_active": plan.array_max_active,
                },
                "run_ids": [spec.run_id for spec in plan.run_specs],
                "trainer_commands_per_script": (
                    self.generated[name]
                    .joinpath("node_local_training_array_job.sh")
                    .read_text(encoding="utf-8")
                    .count("python -m worm_species.training")
                ),
                "wandb_enabled": config["wandb"]["enabled"],
            }
        profiles["scientific_delta"] = sorted(
            _different_paths(
                self.plans["standard"].run_specs[0].resolved_config,
                self.plans["hierarchy"].run_specs[0].resolved_config,
            )
        )
        return profiles

    def test_migration_snapshot(self) -> None:
        self.assertEqual(
            self._projection(), json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        )

    def test_scientific_delta_is_only_hierarchy_enablement(self) -> None:
        for standard_spec, hierarchy_spec in zip(
            self.plans["standard"].run_specs,
            self.plans["hierarchy"].run_specs,
            strict=True,
        ):
            self.assertEqual(
                _different_paths(
                    standard_spec.resolved_config,
                    hierarchy_spec.resolved_config,
                ),
                {"multi_task.hierarchy_loss.enabled"},
            )
            self.assertFalse(standard_spec.resolved_config["sweep"]["enabled"])
            self.assertFalse(hierarchy_spec.resolved_config["sweep"]["enabled"])
            self.assertEqual(standard_spec.experiment_type, "standard")
            self.assertEqual(hierarchy_spec.experiment_type, "standard")

    def test_two_external_tasks_and_one_trainer_per_task(self) -> None:
        for name in ("standard", "hierarchy"):
            plan = self.plans[name]
            self.assertEqual(plan.array_size, 2)
            self.assertEqual(plan.array_size, len(plan.run_specs))
            self.assertEqual(
                plan.models,
                ("convnext_base", "vit_b_16"),
            )
            self.assertEqual(plan.expected_internal_training_runs_per_task, 1)
            self.assertEqual(
                [spec.args_text for spec in plan.run_specs],
                ["model.name=convnext_base\n", "model.name=vit_b_16\n"],
            )
            script = self.generated[name] / "node_local_training_array_job.sh"
            self.assertEqual(
                script.read_text(encoding="utf-8").count(
                    "python -m worm_species.training"
                ),
                1,
            )

    def test_shared_gate_paths_exit_contract_and_shell_syntax(self) -> None:
        required_fragments = (
            'RUN_SCRATCH_OUT="${RUN_OUTPUT_ROOT}/${RUN_ID}"',
            'RUN_BACK_OUT="${RESULTS_ROOT}/${RUN_ID}"',
            "run_overrides.args",
            "run_status.txt",
            '"$PROFILE_DIR/gpu_usage.csv"',
            '"$PROFILE_DIR/cpu_usage.txt"',
            "IMAGE_CACHE.lock",
            "IMAGE_CACHE_READY",
            'cache.root_dir_cache="$CACHE_PARENT"',
            'split.predefined_split_dir="$PROJECT_ROOT"',
            "if ((copy_status != 0 && status == 0)); then",
            "status=90",
            'exit "$status"',
        )
        for name in ("standard", "hierarchy"):
            generated = self.generated[name]
            self.assertEqual(
                sorted(path.name for path in generated.glob("*.sh")),
                [
                    "node_local_cleanup_job.sh",
                    "node_local_setup_job.sh",
                    "node_local_training_array_job.sh",
                ],
            )
            array_source = generated.joinpath(
                "node_local_training_array_job.sh"
            ).read_text(encoding="utf-8")
            for fragment in required_fragments:
                self.assertIn(fragment, array_source)
            self.assertIn(
                str(Path(self.configs[name]["slurm"]["paths"]["results_root"])),
                array_source,
            )
            for script in generated.glob("*.sh"):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_explicit_nodes_are_required(self) -> None:
        for filename in ("standard.yaml", "hierarchy.yaml"):
            with self.subTest(config=filename):
                with self.assertRaisesRegex(SlurmConfigError, "scratch.nodes"):
                    load_submission_config(
                        EXPERIMENTS / filename,
                        GHPC,
                        environment={"HOME": str(self.root), "USER": "alice"},
                        submission_stamp="20260102_030405",
                        process_id=4321,
                    )

    def test_opted_in_historical_result_and_scratch_paths_are_preserved(self) -> None:
        project = self.root / "legacy-project"
        data = self.root / "legacy-data"
        environment = {
            "HOME": str(self.root),
            "USER": "alice",
            "PROJECT_SRC": str(project),
            "DATA_SRC": str(data),
            "GPU_NODES": "gpu01 gpu02",
        }
        for filename in ("standard.yaml", "hierarchy.yaml"):
            with self.subTest(config=filename):
                config = load_submission_config(
                    EXPERIMENTS / filename,
                    GHPC,
                    import_legacy_environment=True,
                    environment=environment,
                    cwd=self.root,
                    submission_stamp="20260102_030405",
                    process_id=4321,
                )
                self.assertEqual(
                    config["slurm"]["paths"]["results_root"],
                    str(
                        project
                        / "outputs_slurm"
                        / "node_local_sweep_20260102_030405"
                    ),
                )
                self.assertEqual(
                    config["slurm"]["scratch"]["root"],
                    "/scratch/alice/worm_node_local_sweep_20260102_030405_4321",
                )
                self.assertEqual(config["slurm"]["scratch"]["nodes"], self.nodes)


if __name__ == "__main__":
    unittest.main()
