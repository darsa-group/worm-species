#!/usr/bin/env python3
"""Upload curated GBIF image ZIPs to CVAT Online, one archive per task."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cvat_sdk import models
from cvat_sdk.core.auth import ClientAuthParameters, make_client_from_cli
from cvat_sdk.core.helpers import DeferredTqdmProgressReporter
from cvat_sdk.core.proxies.tasks import ResourceType


DEFAULT_ARCHIVE_DIR = Path("gbif_oligochaeta/curation")
DEFAULT_MANIFEST = DEFAULT_ARCHIVE_DIR / "cvat_upload_manifest.csv"
DEFAULT_LABELS = (
    "keep",
    "reject_low_quality",
    "reject_non_worm",
    "reject_text_or_label",
    "unsure",
)
ARCHIVE_PATTERN = re.compile(r"curated_images_(\d+)\.zip")
IMAGE_SUFFIXES = {".gif", ".jpg", ".jpeg", ".png"}
MANIFEST_FIELDS = (
    "timestamp_utc",
    "archive_path",
    "archive_bytes",
    "image_count",
    "project_id",
    "project_name",
    "task_id",
    "task_name",
    "task_url",
    "status",
    "attempt",
    "error",
)


@dataclass(frozen=True)
class ArchiveInfo:
    index: int
    path: Path
    image_count: int
    uncompressed_bytes: int
    members: frozenset[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="https://app.cvat.ai")
    parser.add_argument("--organization", default="DARSA")
    parser.add_argument(
        "--profile",
        help="Saved cvat-cli profile. Omit to use CVAT_ACCESS_TOKEN.",
    )
    parser.add_argument("--project-name", default="GBIF Earthworm Image Quality")
    parser.add_argument("--project-id", type=int, help="Reuse this project instead of name lookup.")
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-archives", type=int, default=11)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=int, default=30)
    parser.add_argument("--image-quality", type=int, default=100)
    parser.add_argument("--staging-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--staging-reserve-gb", type=int, default=2)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate local archives without contacting CVAT.",
    )
    return parser.parse_args()


def discover_archives(directory: Path, expected_count: int) -> list[ArchiveInfo]:
    directory = directory.resolve()
    candidates: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = ARCHIVE_PATTERN.fullmatch(path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    candidates.sort()
    expected_indices = list(range(expected_count))
    actual_indices = [index for index, _path in candidates]
    if actual_indices != expected_indices:
        raise RuntimeError(
            f"Expected archive indices {expected_indices}, found {actual_indices} in {directory}"
        )

    archives: list[ArchiveInfo] = []
    all_members: set[str] = set()
    for index, path in candidates:
        print(f"Validating {path.name} ({path.stat().st_size:,} bytes)...", flush=True)
        try:
            with zipfile.ZipFile(path) as archive:
                infos = [info for info in archive.infolist() if not info.is_dir()]
                members = [info.filename for info in infos]
                bad_member = archive.testzip()
        except (OSError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"Invalid ZIP archive {path}: {exc}") from exc
        if bad_member is not None:
            raise RuntimeError(f"CRC validation failed for {bad_member!r} in {path}")
        if not members:
            raise RuntimeError(f"Archive has no images: {path}")
        unsafe = [
            name
            for name in members
            if Path(name).name != name
            or name.startswith(("/", "\\"))
            or Path(name).suffix.lower() not in IMAGE_SUFFIXES
        ]
        if unsafe:
            raise RuntimeError(f"Unsafe or non-image members in {path}: {unsafe[:5]}")
        member_set = frozenset(members)
        if len(member_set) != len(members):
            raise RuntimeError(f"Duplicate member names inside {path}")
        overlap = all_members.intersection(member_set)
        if overlap:
            raise RuntimeError(
                f"Images occur in more than one archive; first duplicates: {sorted(overlap)[:5]}"
            )
        all_members.update(member_set)
        archives.append(
            ArchiveInfo(
                index=index,
                path=path,
                image_count=len(member_set),
                uncompressed_bytes=sum(info.file_size for info in infos),
                members=member_set,
            )
        )
    print(
        f"Validated {len(archives)} archives containing {len(all_members):,} unique images "
        f"({sum(item.path.stat().st_size for item in archives):,} bytes).",
        flush=True,
    )
    return archives


def next_numbered_name(base: str, existing_names: set[str]) -> str:
    if base not in existing_names:
        return base
    number = 2
    while f"{base}_{number}" in existing_names:
        number += 1
    return f"{base}_{number}"


def read_successful_archives(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    successful: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "complete":
                successful[str(Path(row["archive_path"]).resolve())] = row
    return successful


def append_manifest(path: Path, row: dict[str, object]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def manifest_row(
    *,
    archive: ArchiveInfo,
    project_id: int,
    project_name: str,
    task_id: int | None,
    task_name: str,
    server: str,
    status: str,
    attempt: int,
    error: str = "",
) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(archive.path.resolve()),
        "archive_bytes": archive.path.stat().st_size,
        "image_count": archive.image_count,
        "project_id": project_id,
        "project_name": project_name,
        "task_id": task_id or "",
        "task_name": task_name,
        "task_url": f"{server.rstrip('/')}/tasks/{task_id}" if task_id else "",
        "status": status,
        "attempt": attempt,
        "error": error,
    }


def connect(args: argparse.Namespace):
    if args.profile:
        parameters = ClientAuthParameters(
            profile=args.profile,
            organization=args.organization,
        )
    else:
        if not os.environ.get("CVAT_ACCESS_TOKEN"):
            raise RuntimeError(
                "CVAT_ACCESS_TOKEN is not set. Export it in this terminal or pass --profile."
            )
        parameters = ClientAuthParameters(
            server_host=args.server,
            organization=args.organization,
        )
    return make_client_from_cli(parameters, check_server_version=True)


def required_label_names() -> set[str]:
    return set(DEFAULT_LABELS)


def select_or_create_project(client, args: argparse.Namespace):
    if args.project_id is not None:
        project = client.projects.retrieve(args.project_id)
    else:
        matches = [project for project in client.projects.list() if project.name == args.project_name]
        if len(matches) > 1:
            raise RuntimeError(
                f"More than one project is named {args.project_name!r}; rerun with --project-id."
            )
        if matches:
            project = matches[0]
            print(f"Reusing project #{project.id}: {project.name}", flush=True)
        else:
            project = client.projects.create(
                spec=models.ProjectWriteRequest(
                    name=args.project_name,
                    labels=[{"name": name} for name in DEFAULT_LABELS],
                )
            )
            print(f"Created project #{project.id}: {project.name}", flush=True)

    actual_labels = {label.name for label in project.get_labels()}
    if actual_labels != required_label_names():
        raise RuntimeError(
            f"Project #{project.id} labels are {sorted(actual_labels)}, expected "
            f"{sorted(required_label_names())}. Refusing to modify an existing project."
        )
    return project


def completed_task_still_valid(client, row: dict[str, str], archive: ArchiveInfo) -> bool:
    try:
        task = client.tasks.retrieve(int(row["task_id"]))
    except Exception as exc:  # CVAT SDK exception classes differ between releases.
        print(f"Could not verify prior task #{row.get('task_id')}: {exc}", flush=True)
        return False
    return int(task.size or 0) == archive.image_count


def stage_archive(archive: ArchiveInfo, staging_root: Path, reserve_gb: int):
    staging_root = staging_root.resolve()
    staging_root.mkdir(parents=True, exist_ok=True)
    required_bytes = archive.uncompressed_bytes + reserve_gb * 1024**3
    available_bytes = shutil.disk_usage(staging_root).free
    if available_bytes < required_bytes:
        raise RuntimeError(
            f"Insufficient staging space in {staging_root}: need "
            f"{required_bytes:,} bytes including reserve, have {available_bytes:,}."
        )
    temporary = tempfile.TemporaryDirectory(
        prefix=f"cvat-{archive.path.stem}-",
        dir=staging_root,
    )
    destination = Path(temporary.name)
    print(
        f"Extracting {archive.path.name} to temporary staging "
        f"({archive.uncompressed_bytes:,} bytes)...",
        flush=True,
    )
    try:
        with zipfile.ZipFile(archive.path) as source:
            source.extractall(destination)
        resources = sorted(path for path in destination.iterdir() if path.is_file())
        if len(resources) != archive.image_count:
            raise RuntimeError(
                f"Staged {len(resources)} files from {archive.path}; "
                f"expected {archive.image_count}."
            )
        return temporary, resources
    except Exception:
        temporary.cleanup()
        raise


def upload_archive(
    *,
    client,
    archive: ArchiveInfo,
    project,
    existing_names: set[str],
    manifest_path: Path,
    server: str,
    attempts: int,
    retry_delay_seconds: int,
    image_quality: int,
    staging_root: Path,
    staging_reserve_gb: int,
) -> None:
    temporary, resources = stage_archive(archive, staging_root, staging_reserve_gb)
    try:
        last_error = ""
        for attempt in range(1, attempts + 1):
            task_name = next_numbered_name(archive.path.stem, existing_names)
            existing_names.add(task_name)
            task = None
            try:
                task = client.tasks.create(
                    spec=models.TaskWriteRequest(name=task_name, project_id=project.id)
                )
                append_manifest(
                    manifest_path,
                    manifest_row(
                        archive=archive,
                        project_id=project.id,
                        project_name=project.name,
                        task_id=task.id,
                        task_name=task_name,
                        server=server,
                        status="created",
                        attempt=attempt,
                    ),
                )
                print(
                    f"Uploading {archive.image_count:,} extracted images from "
                    f"{archive.path.name} to task #{task.id} ({task_name}), "
                    f"attempt {attempt}/{attempts}...",
                    flush=True,
                )
                task.upload_data(
                    resources=resources,
                    resource_type=ResourceType.LOCAL,
                    pbar=DeferredTqdmProgressReporter(),
                    params={
                        "image_quality": image_quality,
                        "sorting_method": "lexicographical",
                    },
                    wait_for_completion=True,
                )
                task.fetch()
                if int(task.size or 0) != archive.image_count:
                    raise RuntimeError(
                        f"Task #{task.id} contains {task.size} frames; "
                        f"expected {archive.image_count}."
                    )
                append_manifest(
                    manifest_path,
                    manifest_row(
                        archive=archive,
                        project_id=project.id,
                        project_name=project.name,
                        task_id=task.id,
                        task_name=task_name,
                        server=server,
                        status="complete",
                        attempt=attempt,
                    ),
                )
                print(
                    f"Completed task #{task.id}: {archive.image_count:,} images from "
                    f"{archive.path.name}",
                    flush=True,
                )
                return
            except Exception as exc:  # Preserve the task and record it for inspection.
                last_error = f"{type(exc).__name__}: {exc}"
                task_id = getattr(task, "id", None)
                append_manifest(
                    manifest_path,
                    manifest_row(
                        archive=archive,
                        project_id=project.id,
                        project_name=project.name,
                        task_id=task_id,
                        task_name=task_name,
                        server=server,
                        status="failed",
                        attempt=attempt,
                        error=last_error,
                    ),
                )
                print(f"Upload attempt failed: {last_error}", file=sys.stderr, flush=True)
                if attempt < attempts:
                    print(
                        f"Retrying with a numbered task name in "
                        f"{retry_delay_seconds} seconds...",
                        flush=True,
                    )
                    time.sleep(retry_delay_seconds)
        raise RuntimeError(f"All upload attempts failed for {archive.path}: {last_error}")
    finally:
        print(f"Removing temporary staging directory {temporary.name}", flush=True)
        temporary.cleanup()


def main() -> None:
    args = parse_args()
    if args.expected_archives <= 0:
        raise ValueError("--expected-archives must be positive")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if not 0 <= args.image_quality <= 100:
        raise ValueError("--image-quality must be between 0 and 100")
    if args.staging_reserve_gb < 0:
        raise ValueError("--staging-reserve-gb cannot be negative")

    archives = discover_archives(args.archive_dir, args.expected_archives)
    if args.validate_only:
        return

    successful = read_successful_archives(args.manifest)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with connect(args) as client:
        project = select_or_create_project(client, args)
        existing_names = {task.name for task in project.get_tasks()}
        print(
            f"Using organization {args.organization!r}, project #{project.id}, "
            f"and {len(existing_names)} existing project tasks.",
            flush=True,
        )
        for archive in archives:
            archive_key = str(archive.path.resolve())
            previous = successful.get(archive_key)
            if previous and completed_task_still_valid(client, previous, archive):
                print(
                    f"Skipping {archive.path.name}; verified prior task "
                    f"#{previous['task_id']} is complete.",
                    flush=True,
                )
                continue
            upload_archive(
                client=client,
                archive=archive,
                project=project,
                existing_names=existing_names,
                manifest_path=args.manifest,
                server=args.server,
                attempts=args.attempts,
                retry_delay_seconds=args.retry_delay_seconds,
                image_quality=args.image_quality,
                staging_root=args.staging_dir,
                staging_reserve_gb=args.staging_reserve_gb,
            )
    print(f"Upload manifest: {args.manifest.resolve()}", flush=True)


if __name__ == "__main__":
    main()
