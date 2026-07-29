#!/usr/bin/env python3
"""Render or submit the complete Genome paper-ablation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from worm_species.slurm.config import load_submission_config, parse_memory
from worm_species.slurm.planning import plan_submission
from worm_species.slurm.rendering import write_artifact_bundle
from worm_species.slurm.submission import (
    SubprocessSbatchClient,
    build_submission_commands,
    parse_job_id,
    submit_manifest,
)
from worm_species.cache.condition_variants import (
    cacheable_conditions,
    condition_cache_directory,
    condition_cache_settings,
)
from worm_species.config.normalization import normalize_config
from worm_species.training.loaders import get_input_condition


def _load_pipeline(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Pipeline config must be a mapping.")
    if not isinstance(raw.get("stages"), list) or not raw["stages"]:
        raise ValueError("Pipeline config requires a non-empty stages list.")
    names = [stage.get("name") for stage in raw["stages"]]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Every pipeline stage requires a clear name.")
    if len(names) != len(set(names)):
        raise ValueError("Pipeline stage names must be unique.")
    if raw.get("dependency", "afterok") != "afterok":
        raise ValueError("This scientific pipeline requires dependency: afterok.")
    return raw


def _write_report_job(
    *,
    path: Path,
    project_root: Path,
    paper_root: Path,
    report_script: Path,
    data_root: Path,
    style_path: Path,
    conda_sh: str,
    conda_env: str,
) -> None:
    source = f"""#!/usr/bin/env bash
set -euo pipefail
source {shlex.quote(conda_sh)}
conda activate {shlex.quote(conda_env)}
cd {shlex.quote(str(project_root))}
python {shlex.quote(str(report_script))} \
  --paper-result {shlex.quote(str(paper_root))} \
  --split-root {shlex.quote(str(project_root))} \
  --data-root {shlex.quote(str(data_root))} \
  --style {shlex.quote(str(style_path))}
"""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _write_condition_cache_job(
    *,
    path: Path,
    project_root: Path,
    config_path: Path,
    data_root: Path,
    metadata_csv: Path,
    base_cache_root: Path,
    condition_cache_root: Path,
    transforms: list[str],
    conda_sh: str,
    conda_env: str,
    num_workers: int,
) -> None:
    transform_args = " ".join(shlex.quote(item) for item in transforms)
    source = f"""#!/usr/bin/env bash
set -euo pipefail
source {shlex.quote(conda_sh)}
conda activate {shlex.quote(conda_env)}
cd {shlex.quote(str(project_root))}
export PYTHONPATH={shlex.quote(str(project_root / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}
python -m worm_species.cache build-conditions \
  --config {shlex.quote(str(config_path))} \
  --data-root {shlex.quote(str(data_root))} \
  --metadata-csv {shlex.quote(str(metadata_csv))} \
  --base-cache-dir {shlex.quote(str(base_cache_root))} \
  --condition-cache-dir {shlex.quote(str(condition_cache_root))} \
  --condition-index "${{SLURM_ARRAY_TASK_ID:?}}" \
  --num-workers {int(num_workers)} \
  --transforms {transform_args}
"""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _write_base_cache_job(
    *,
    path: Path,
    project_root: Path,
    config_path: Path,
    data_root: Path,
    metadata_csv: Path,
    cache_root: Path,
    conda_sh: str,
    conda_env: str,
) -> None:
    source = f"""#!/usr/bin/env bash
set -euo pipefail
source {shlex.quote(conda_sh)}
conda activate {shlex.quote(conda_env)}
cd {shlex.quote(str(project_root))}
export PYTHONPATH={shlex.quote(str(project_root / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}
python -m worm_species.cache build \
  --config {shlex.quote(str(config_path))} \
  --data-root {shlex.quote(str(data_root))} \
  --metadata-csv {shlex.quote(str(metadata_csv))} \
  --cache-dir {shlex.quote(str(cache_root))}
python -m worm_species.cache verify \
  --cache-dir {shlex.quote(str(cache_root))}
"""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def run_pipeline(pipeline_path: Path, mode: str) -> dict:
    pipeline_path = pipeline_path.resolve()
    pipeline = _load_pipeline(pipeline_path)
    pipeline_dir = pipeline_path.parent
    project_root = pipeline_dir.parent
    cluster_path = (pipeline_dir / pipeline["cluster_config"]).resolve()
    paper_root = (pipeline_dir / pipeline["paper_result_dir"]).resolve()
    stamp = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{os.getpid()}"
    )
    artifact_root = paper_root / "artifacts" / stamp
    (paper_root / "runs").mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=False)

    scheduler = SubprocessSbatchClient()
    paper_base_cache_root: str | None = None
    paper_condition_cache_root: str | None = None
    base_cache_record = {"enabled": False}
    base_cache_job_id: str | None = None
    base_cache = pipeline.get("base_cache", {}) or {}
    if bool(base_cache.get("enabled", False)):
        source_stage_name = str(base_cache["source_stage"])
        source_stage = next(
            (
                stage
                for stage in pipeline["stages"]
                if stage["name"] == source_stage_name
            ),
            None,
        )
        if source_stage is None:
            raise ValueError(
                f"base_cache.source_stage {source_stage_name!r} "
                "is not a pipeline stage"
            )
        source_config_path = (
            pipeline_dir / source_stage["config"]
        ).resolve()
        source_config = load_submission_config(
            source_config_path,
            cluster_config=cluster_path,
        )
        slurm = source_config["slurm"]
        paths = slurm["paths"]
        generic_cache_root = Path(paths["cache_root"])
        paper_base_cache_root = str(
            generic_cache_root.parent
            / str(
                base_cache.get(
                    "directory_name", "image_cache_paper_v1"
                )
            )
        )
        environment = slurm["environment"]
        cluster_project_root = Path(paths["project_root"])
        cache_job = artifact_root / "base_cache_job.sh"
        _write_base_cache_job(
            path=cache_job,
            project_root=cluster_project_root,
            config_path=cluster_project_root
            / "dev"
            / source_config_path.name,
            data_root=Path(paths["data_root"]),
            metadata_csv=Path(paths["metadata_csv"]),
            cache_root=Path(paper_base_cache_root),
            conda_sh=environment["conda_sh"],
            conda_env=environment["conda_env"],
        )
        logs = paper_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        cache_argv = [
            "sbatch",
            "--parsable",
            f"--account={slurm['account']}",
            "--nodes=1",
            "--ntasks=1",
            (
                "--cpus-per-task="
                f"{int(base_cache.get('cpus_per_task', 8))}"
            ),
            f"--mem={parse_memory(base_cache.get('memory', '16G'))}",
            f"--time={base_cache.get('time_limit', '04:00:00')}",
            "--job-name=worm_base_cache",
            f"--output={logs / 'base_cache_%j.out'}",
            f"--error={logs / 'base_cache_%j.err'}",
            "--export=ALL",
            str(cache_job),
        ]
        base_cache_record = {
            "enabled": True,
            "source_stage": source_stage_name,
            "cache_root": paper_base_cache_root,
            "script": str(cache_job),
            "command": cache_argv,
            "submitted_job_id": None,
        }
        if mode == "submit":
            completed = scheduler.run(cache_argv)
            if completed.returncode != 0:
                raise RuntimeError(
                    "Could not submit base cache job: "
                    f"{completed.stderr.strip()}"
                )
            base_cache_job_id = parse_job_id(completed.stdout)
            base_cache_record["submitted_job_id"] = base_cache_job_id

    condition_cache_record = {"enabled": False}
    condition_cache_job_id: str | None = None
    condition_cache = pipeline.get("condition_cache", {}) or {}
    if bool(condition_cache.get("enabled", False)):
        source_stage_name = str(condition_cache["source_stage"])
        source_stage = next(
            (
                stage
                for stage in pipeline["stages"]
                if stage["name"] == source_stage_name
            ),
            None,
        )
        if source_stage is None:
            raise ValueError(
                f"condition_cache.source_stage {source_stage_name!r} "
                "is not a pipeline stage"
            )
        source_config_path = (
            pipeline_dir / source_stage["config"]
        ).resolve()
        preliminary_config = load_submission_config(
            source_config_path, cluster_config=cluster_path
        )
        preliminary_paths = preliminary_config["slurm"]["paths"]
        if paper_base_cache_root is None:
            generic_cache_root = Path(preliminary_paths["cache_root"])
            paper_base_cache_root = str(
                generic_cache_root.parent / "image_cache_paper_v1"
            )
        paper_condition_cache_root = str(
            Path(paper_base_cache_root).parent
            / str(
                condition_cache.get(
                    "directory_name",
                    "image_condition_cache_paper_v1",
                )
            )
        )
        source_config = load_submission_config(
            source_config_path,
            cluster_config=cluster_path,
            overrides=[
                f"slurm.paths.cache_root={paper_base_cache_root}",
                (
                    "slurm.paths.condition_cache_root="
                    f"{paper_condition_cache_root}"
                ),
            ],
        )
        transforms = [
            str(item) for item in condition_cache.get("transforms", [])
        ]
        if not transforms:
            raise ValueError("condition_cache.transforms must not be empty")
        cache_conditions = cacheable_conditions(source_config, transforms)
        if not cache_conditions:
            raise ValueError("condition cache selected no sweep conditions")
        slurm = source_config["slurm"]
        paths = slurm["paths"]
        environment = slurm["environment"]
        condition_cache_root = paths.get("condition_cache_root")
        if not isinstance(condition_cache_root, str) or not condition_cache_root:
            raise ValueError(
                "slurm.paths.condition_cache_root is required"
            )
        protocol_version = int(
            condition_cache_settings(source_config)["protocol_version"]
        )
        builder_directories = {
            condition_cache_directory(
                condition_cache_root,
                condition,
                protocol_version=protocol_version,
            )
            for condition in cache_conditions
        }
        source_plan = plan_submission(source_config)
        checked_training_specs = 0
        mismatched_training_specs = []
        for spec in source_plan.run_specs:
            if spec.training_transform not in transforms:
                continue
            checked_training_specs += 1
            training_condition = get_input_condition(
                normalize_config(spec.resolved_config)
            )
            expected_directory = condition_cache_directory(
                condition_cache_root,
                training_condition,
                protocol_version=protocol_version,
            )
            if expected_directory not in builder_directories:
                mismatched_training_specs.append(
                    {
                        "run_id": spec.run_id,
                        "condition": spec.training_condition,
                        "directory": str(expected_directory),
                    }
                )
        if mismatched_training_specs:
            example = mismatched_training_specs[0]
            raise ValueError(
                "condition-cache identity mismatch between builder and "
                f"training specifications; {len(mismatched_training_specs)} "
                f"runs are affected; first={example}"
            )

        resolved_cache_config = (
            artifact_root / "condition_cache_source_config.yaml"
        )
        resolved_cache_config.write_text(
            yaml.safe_dump(source_config, sort_keys=True),
            encoding="utf-8",
        )
        cache_job = artifact_root / "condition_cache_job.sh"
        cluster_project_root = Path(paths["project_root"])
        _write_condition_cache_job(
            path=cache_job,
            project_root=cluster_project_root,
            config_path=resolved_cache_config,
            data_root=Path(paths["data_root"]),
            metadata_csv=Path(paths["metadata_csv"]),
            base_cache_root=Path(paths["cache_root"]),
            condition_cache_root=Path(condition_cache_root),
            transforms=transforms,
            conda_sh=environment["conda_sh"],
            conda_env=environment["conda_env"],
            num_workers=int(condition_cache.get("cpus_per_task", 8)),
        )
        logs = paper_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        max_active = int(condition_cache.get("max_active", 8))
        if max_active < 1 or max_active > 12:
            raise ValueError(
                "condition_cache.max_active must be between 1 and 12"
            )
        cache_argv = [
            "sbatch",
            "--parsable",
            f"--account={slurm['account']}",
            "--nodes=1",
            "--ntasks=1",
            (
                "--cpus-per-task="
                f"{int(condition_cache.get('cpus_per_task', 8))}"
            ),
            f"--mem={parse_memory(condition_cache.get('memory', '16G'))}",
            f"--time={condition_cache.get('time_limit', '04:00:00')}",
            (
                "--array=0-"
                f"{len(cache_conditions) - 1}%{max_active}"
            ),
            "--job-name=worm_condition_cache",
            f"--output={logs / 'condition_cache_%A_%a.out'}",
            f"--error={logs / 'condition_cache_%A_%a.err'}",
            "--export=ALL",
        ]
        if mode == "submit" and base_cache_job_id:
            cache_argv.append(
                f"--dependency=afterok:{base_cache_job_id}"
            )
        elif mode == "dry-run" and base_cache_record["enabled"]:
            cache_argv.append("--dependency=afterok:@base_cache")
        cache_argv.append(str(cache_job))
        condition_cache_record = {
            "enabled": True,
            "source_stage": source_stage_name,
            "condition_count": len(cache_conditions),
            "training_spec_paths_checked": checked_training_specs,
            "conditions": [
                condition["name"] for condition in cache_conditions
            ],
            "transforms": transforms,
            "cache_root": condition_cache_root,
            "resolved_source_config": str(resolved_cache_config),
            "script": str(cache_job),
            "command": cache_argv,
            "submitted_job_id": None,
        }
        if mode == "submit":
            completed = scheduler.run(cache_argv)
            if completed.returncode != 0:
                raise RuntimeError(
                    "Could not submit condition cache job: "
                    f"{completed.stderr.strip()}"
                )
            condition_cache_job_id = parse_job_id(completed.stdout)
            condition_cache_record["submitted_job_id"] = (
                condition_cache_job_id
            )

    previous_train_id: str | None = None
    stage_records = []
    last_config = None
    for stage in pipeline["stages"]:
        name = stage["name"]
        config_path = (pipeline_dir / stage["config"]).resolve()
        results_root = paper_root / "runs" / name
        config = load_submission_config(
            config_path,
            cluster_config=cluster_path,
            overrides=[
                f"slurm.paths.results_root={results_root}",
                *(
                    [f"slurm.paths.cache_root={paper_base_cache_root}"]
                    if paper_base_cache_root is not None
                    else []
                ),
                *(
                    [
                        "slurm.paths.condition_cache_root="
                        f"{paper_condition_cache_root}"
                    ]
                    if paper_condition_cache_root is not None
                    else []
                ),
                (
                    "slurm.logging.directory="
                    f"{paper_root / 'logs' / 'slurm' / name}"
                ),
            ],
        )
        plan = plan_submission(config)
        if plan.array_max_active > 12:
            raise ValueError(
                f"Stage {name!r} must cap active training tasks at 12, "
                f"got {plan.array_max_active}."
            )
        hierarchy_runs = [
            spec.run_id
            for spec in plan.run_specs
            if bool(
                spec.resolved_config.get("multi_task", {})
                .get("hierarchy_loss", {})
                .get("enabled", False)
            )
        ]
        if hierarchy_runs:
            raise ValueError(
                f"Stage {name!r} unexpectedly enables hierarchy loss."
            )
        stage_artifacts = artifact_root / name
        manifest = write_artifact_bundle(plan, config, stage_artifacts)
        dry_run_commands = build_submission_commands(manifest)
        if mode == "dry-run":
            symbolic_dependencies = []
            if stage_records:
                symbolic_dependencies.append(
                    f"@{stage_records[-1]['name']}_train"
                )
            if (
                base_cache_record["enabled"]
                and name == base_cache_record["source_stage"]
            ):
                symbolic_dependencies.append("@base_cache")
            if (
                condition_cache_record["enabled"]
                and name == condition_cache_record["source_stage"]
            ):
                symbolic_dependencies.append("@condition_cache")
            if symbolic_dependencies:
                dry_run_commands[0].insert(
                    -1,
                    "--dependency=afterok:"
                    + ":".join(symbolic_dependencies),
                )
        record = {
            "name": name,
            "config": str(config_path),
            "results_root": str(results_root),
            "artifact_root": str(stage_artifacts),
            "run_count": plan.array_size,
            "max_active": plan.array_max_active,
            "models": list(plan.models),
            "submitted": {},
            "dry_run_commands": dry_run_commands,
        }
        if mode == "submit":
            dependencies = []
            if previous_train_id:
                dependencies.append(
                    {"kind": "afterok", "job_id": previous_train_id}
                )
            if (
                base_cache_job_id
                and name == base_cache_record["source_stage"]
            ):
                dependencies.append(
                    {"kind": "afterok", "job_id": base_cache_job_id}
                )
            if (
                condition_cache_job_id
                and name == condition_cache_record["source_stage"]
            ):
                dependencies.append(
                    {
                        "kind": "afterok",
                        "job_id": condition_cache_job_id,
                    }
                )
            submitted = submit_manifest(
                stage_artifacts / "submission_manifest.json",
                client=scheduler,
                train_dependencies=dependencies,
            )
            previous_train_id = submitted["train_array"]
            record["submitted"] = submitted
        else:
            record["depends_on_previous_train"] = previous_train_id is not None
            previous_train_id = f"dry-run-{name}"
        stage_records.append(record)
        last_config = config

    report_record = {"enabled": False}
    report = pipeline.get("report", {}) or {}
    if bool(report.get("enabled", False)):
        if last_config is None:
            raise ValueError("Cannot create report job without a training stage.")
        report_script = (pipeline_dir / report["script"]).resolve()
        style_path = (
            pipeline_dir
            / report.get("style", "paper_report_style.yaml")
        ).resolve()
        report_job = artifact_root / "paper_report_job.sh"
        environment = last_config["slurm"]["environment"]
        _write_report_job(
            path=report_job,
            project_root=project_root,
            paper_root=paper_root,
            report_script=report_script,
            data_root=Path(last_config["slurm"]["paths"]["data_root"]),
            style_path=style_path,
            conda_sh=environment["conda_sh"],
            conda_env=environment["conda_env"],
        )
        logs = paper_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        report_argv = [
            "sbatch",
            "--parsable",
            f"--account={last_config['slurm']['account']}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={int(report.get('cpus_per_task', 4))}",
            f"--mem={parse_memory(report.get('memory', '16G'))}",
            f"--time={report.get('time_limit', '01:00:00')}",
            "--job-name=worm_paper_report",
            f"--output={logs / 'paper_report_%j.out'}",
            f"--error={logs / 'paper_report_%j.err'}",
            "--export=ALL",
        ]
        if mode == "submit":
            report_argv.append(f"--dependency=afterok:{previous_train_id}")
        else:
            report_argv.append("--dependency=afterok:@data_holdouts_train")
        report_argv.append(str(report_job))
        report_record = {
            "enabled": True,
            "script": str(report_job),
            "command": report_argv,
            "submitted_job_id": None,
        }
        if mode == "submit":
            result = scheduler.run(report_argv)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Could not submit paper report job: {result.stderr.strip()}"
                )
            report_record["submitted_job_id"] = parse_job_id(result.stdout)

    result = {
        "schema_version": 1,
        "pipeline": pipeline["name"],
        "mode": mode,
        "paper_result": str(paper_root),
        "artifact_root": str(artifact_root),
        "total_model_fits": sum(stage["run_count"] for stage in stage_records),
        "base_cache": base_cache_record,
        "condition_cache": condition_cache_record,
        "stages": stage_records,
        "report": report_record,
    }
    manifest_path = artifact_root / "pipeline_manifest.json"
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline", default="dev/genome_ablation_pipeline.yaml"
    )
    parser.add_argument(
        "--mode", choices=("dry-run", "submit"), default="dry-run"
    )
    args = parser.parse_args()
    result = run_pipeline(Path(args.pipeline), args.mode)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
