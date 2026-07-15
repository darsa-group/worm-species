"""Safe-by-default command line interface for canonical SLURM execution."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .collection import CollectionError
from .collection import collect_existing_results
from .config import SlurmConfigError
from .config import load_submission_config
from .planning import plan_submission
from .rendering import RenderError
from .rendering import write_artifact_bundle
from .submission import SubmissionError
from .submission import build_submission_commands
from .submission import submit_manifest
from .status import build_status_report


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--cluster-config")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Scheduler/config override key=value; repeat for multiple values.",
    )
    parser.add_argument(
        "--legacy-env",
        action="store_true",
        help=(
            "Import the allow-listed environment variables used by historical "
            "launchers. Ambient launcher variables are ignored by default."
        ),
    )
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--json", action="store_true", dest="json_output")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m worm_species.slurm",
        description=(
            "Render SLURM artifacts by default; scheduler submission always "
            "requires an explicit submit command or --submit."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="write artifacts; never submit")
    _add_plan_arguments(render)

    launch = subparsers.add_parser(
        "launch", help="render, then optionally submit with explicit --submit"
    )
    _add_plan_arguments(launch)
    mode = launch.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")

    submit = subparsers.add_parser(
        "submit", help="submit an existing, checksum-verified artifact manifest"
    )
    submit.add_argument("--manifest", required=True)
    submit.add_argument("--json", action="store_true", dest="json_output")

    status = subparsers.add_parser(
        "status", help="summarise filesystem and optional scheduler state"
    )
    status.add_argument("--results-root", required=True)
    status.add_argument("--submission-root")
    status.add_argument("--no-scheduler", action="store_true")
    status.add_argument("--json", action="store_true", dest="json_output")

    collect = subparsers.add_parser(
        "collect", help="aggregate existing results without training"
    )
    collect.add_argument("--results-root", required=True)
    collect.add_argument("--kind", default="auto")
    collect.add_argument("--json", action="store_true", dest="json_output")
    return parser


def _render(args: argparse.Namespace) -> tuple[dict, list[list[str]]]:
    config = load_submission_config(
        args.config,
        cluster_config=args.cluster_config,
        overrides=args.override,
        import_legacy_environment=args.legacy_env,
    )
    plan = plan_submission(config)
    manifest = write_artifact_bundle(plan, config, args.artifacts_dir)
    commands = build_submission_commands(manifest)
    return manifest, commands


def _print_render_summary(
    manifest: dict,
    commands: list[list[str]],
    *,
    json_output: bool,
) -> None:
    summary = {
        "submitted": False,
        "scheduler_calls": 0,
        "artifact_root": manifest["artifact_root"],
        "array_size": manifest["array_size"],
        "jobs": [job["name"] for job in manifest["jobs"]],
        "sbatch_commands": commands,
    }
    if json_output:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"Rendered SLURM artifacts: {summary['artifact_root']}")
    print(f"Array tasks: {summary['array_size']}")
    print("Dry run: no scheduler commands were executed.")
    for command in commands:
        print("  " + shlex.join(command))


def execute(args: argparse.Namespace) -> int:
    if args.command == "status":
        report = build_status_report(
            args.results_root,
            submission_root=args.submission_root,
            query_scheduler=not args.no_scheduler,
        )
        data = report.as_dict()
        if args.json_output:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            expected = report.expected_run_count
            expected_text = "unknown" if expected is None else str(expected)
            print(f"Experiment: {report.experiment_name}")
            print(
                "Runs: "
                f"{report.materialized_run_count} materialized / "
                f"{expected_text} expected"
            )
            print(f"Filesystem: {report.filesystem_counts}")
            if report.submitted_jobs:
                print(f"Scheduler: {report.scheduler_counts}")
            if report.warnings:
                print(f"Warnings: {len(report.warnings)}")
        return 0

    if args.command == "collect":
        report = collect_existing_results(args.results_root, kind=args.kind)
        data = report.as_dict()
        if args.json_output:
            print(json.dumps(data, indent=2, sort_keys=True))
        else:
            print(f"Collection kind: {report.kind}")
            for output_path in report.output_paths:
                print(output_path)
        return 0

    if args.command == "submit":
        submitted = submit_manifest(args.manifest)
        if args.json_output:
            print(json.dumps({"submitted": submitted}, indent=2, sort_keys=True))
        else:
            for name, job_id in submitted.items():
                print(f"{name}: {job_id}")
        return 0

    manifest, commands = _render(args)
    if args.command == "launch" and args.submit:
        manifest_path = Path(args.artifacts_dir) / "submission_manifest.json"
        submitted = submit_manifest(manifest_path)
        if args.json_output:
            print(json.dumps({"submitted": submitted}, indent=2, sort_keys=True))
        else:
            for name, job_id in submitted.items():
                print(f"{name}: {job_id}")
        return 0

    _print_render_summary(
        manifest,
        commands,
        json_output=args.json_output,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return execute(args)
    except (SlurmConfigError, RenderError, CollectionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except SubmissionError as exc:
        print(f"submission error: {exc}", file=sys.stderr)
        return 4


__all__ = ["build_parser", "execute", "main"]
