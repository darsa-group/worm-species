#!/usr/bin/env python3
"""Render or submit the complete Genome paper-ablation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path

import yaml

from worm_species.slurm.config import load_submission_config, parse_memory
from worm_species.slurm.planning import plan_submission
from worm_species.slurm.rendering import write_artifact_bundle
from worm_species.slurm.submission import (
    SubprocessSbatchClient,
    build_submission_commands,
    parse_job_id,
    submit_manifest,
)


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
        if mode == "dry-run" and stage_records:
            dry_run_commands[0].insert(
                -1,
                (
                    "--dependency=afterok:"
                    f"@{stage_records[-1]['name']}_train"
                ),
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
            dependencies = (
                [{"kind": "afterok", "job_id": previous_train_id}]
                if previous_train_id
                else []
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
