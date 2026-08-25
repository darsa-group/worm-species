from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

from .pipeline import file_sha256
from .pipeline import resolve_manifest_image_path


REQUIRED_MEDIA_COLUMNS = {
    "image_id", "gbif_id", "source_url", "genus", "genus_key", "dataset_key"
}
REQUIRED_DOWNLOAD_COLUMNS = REQUIRED_MEDIA_COLUMNS | {
    "download_status",
    "local_path",
    "sha256",
}


def _read_manifest(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def validate_transfer_bundle(
    bundle_root: str | Path,
    *,
    verify_image_hashes: bool = False,
    required_dataset_key: str | None = None,
) -> dict:
    """Require a complete one-row-per-media manifest before data transfer."""
    root = Path(bundle_root).resolve()
    media_path = root / "manifests" / "media_manifest.csv"
    downloaded_path = root / "manifests" / "downloaded_manifest.csv"
    media = _read_manifest(media_path, REQUIRED_MEDIA_COLUMNS)
    downloaded = _read_manifest(downloaded_path, REQUIRED_DOWNLOAD_COLUMNS)
    excluded_paths = sorted((root / "manifests").glob("excluded_*.csv"))
    excluded_frames = [
        _read_manifest(path, REQUIRED_MEDIA_COLUMNS | {"exclusion_reason"})
        for path in excluded_paths
    ]
    excluded = (
        pd.concat(excluded_frames, ignore_index=True)
        if excluded_frames
        else pd.DataFrame(columns=list(REQUIRED_MEDIA_COLUMNS | {"exclusion_reason"}))
    )

    if media.empty:
        raise ValueError("The source media manifest is empty")
    if media["image_id"].duplicated().any():
        raise ValueError("The source media manifest contains duplicate image_id values")
    if downloaded["image_id"].duplicated().any():
        raise ValueError("The downloaded manifest contains duplicate image_id values")
    if excluded["image_id"].duplicated().any():
        raise ValueError("The excluded manifest contains duplicate image_id values")

    expected_ids = set(media["image_id"])
    actual_ids = set(downloaded["image_id"])
    excluded_ids = set(excluded["image_id"])
    overlap_ids = actual_ids & excluded_ids
    missing_rows = expected_ids - actual_ids - excluded_ids
    extra_rows = actual_ids - expected_ids
    failed_rows = downloaded.loc[downloaded["download_status"] != "downloaded"]
    blank_paths = downloaded.loc[downloaded["local_path"] == ""]
    blank_hashes = downloaded.loc[downloaded["sha256"] == ""]
    outside_dataset = (
        downloaded.loc[downloaded["dataset_key"] != required_dataset_key]
        if required_dataset_key
        else downloaded.iloc[0:0]
    )
    problems = []
    if len(downloaded) + len(excluded) != len(media):
        problems.append(
            f"active plus excluded row count is {len(downloaded) + len(excluded):,}, "
            f"expected {len(media):,}"
        )
    if overlap_ids:
        problems.append(f"{len(overlap_ids):,} image_id rows are both active and excluded")
    if missing_rows:
        problems.append(f"{len(missing_rows):,} image_id rows are missing")
    if extra_rows:
        problems.append(f"{len(extra_rows):,} unexpected image_id rows are present")
    if not failed_rows.empty:
        problems.append(f"{len(failed_rows):,} rows are not downloaded")
    if not blank_paths.empty:
        problems.append(f"{len(blank_paths):,} rows have no local_path")
    if not blank_hashes.empty:
        problems.append(f"{len(blank_hashes):,} rows have no SHA-256")
    if not outside_dataset.empty:
        problems.append(
            f"{len(outside_dataset):,} active rows are outside required dataset "
            f"{required_dataset_key}; run the dataset-filter command"
        )
    if problems:
        raise RuntimeError("Transfer bundle is incomplete: " + "; ".join(problems))

    resolved = downloaded["local_path"].map(
        lambda value: resolve_manifest_image_path(downloaded_path, value).resolve()
    )
    missing_files = [path for path in resolved.drop_duplicates() if not path.is_file()]
    if missing_files:
        preview = ", ".join(str(path) for path in missing_files[:3])
        raise RuntimeError(
            f"Transfer bundle has {len(missing_files):,} missing image files; "
            f"first: {preview}"
        )

    asset_rows = pd.DataFrame({"path": resolved, "sha256": downloaded["sha256"]})
    conflicting = asset_rows.groupby("path")["sha256"].nunique()
    conflicting = conflicting.loc[conflicting > 1]
    if not conflicting.empty:
        raise RuntimeError(
            f"{len(conflicting):,} local image paths have conflicting manifest hashes"
        )

    unique_assets = asset_rows.drop_duplicates("path")
    if verify_image_hashes:
        mismatches = []
        for row in unique_assets.itertuples(index=False):
            actual_hash = file_sha256(row.path)
            if actual_hash != row.sha256:
                mismatches.append(str(row.path))
        if mismatches:
            raise RuntimeError(
                f"{len(mismatches):,} image files failed SHA-256 verification; "
                f"first: {mismatches[0]}"
            )

    image_root = root / "images"
    files_on_disk = {
        path.resolve()
        for path in image_root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    }
    referenced_files = set(unique_assets["path"])
    return {
        "ready": True,
        "bundle_root": str(root),
        "media_rows": int(len(media)),
        "active_media_rows": int(len(downloaded)),
        "excluded_media_rows": int(len(excluded)),
        "exclusion_manifests": [str(path) for path in excluded_paths],
        "occurrences": int(media["gbif_id"].nunique()),
        "unique_source_urls": int(media["source_url"].nunique()),
        "unique_image_files": int(len(unique_assets)),
        "unreferenced_image_files": int(len(files_on_disk - referenced_files)),
        "image_hashes_verified": bool(verify_image_hashes),
        "required_dataset_key": required_dataset_key,
    }


def prepare_transfer_bundle(
    bundle_root: str | Path,
    *,
    verify_image_hashes: bool = True,
    show_progress: bool = False,
) -> dict:
    """Validate and write checksums for every payload file in the bundle."""
    root = Path(bundle_root).resolve()
    summary = validate_transfer_bundle(root, verify_image_hashes=False)
    partial_files = [path for path in root.rglob("*.part") if path.is_file()]
    if partial_files:
        raise RuntimeError(
            f"Refusing to prepare a bundle with {len(partial_files):,} partial files"
        )

    transfer_dir = root / "transfer"
    payload_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and transfer_dir not in path.parents
    )
    if not payload_files:
        raise RuntimeError("No files were found in the transfer bundle")

    expected_image_hashes: dict[Path, str] = {}
    if verify_image_hashes:
        downloaded_path = root / "manifests" / "downloaded_manifest.csv"
        downloaded = _read_manifest(downloaded_path, REQUIRED_DOWNLOAD_COLUMNS)
        for row in downloaded.itertuples(index=False):
            image_path = resolve_manifest_image_path(
                downloaded_path, row.local_path
            ).resolve()
            expected_image_hashes[image_path] = row.sha256

    checksum_lines = []
    payload_bytes = 0
    image_hash_mismatches = []
    total_files = len(payload_files)
    if show_progress:
        print(
            f"Hashing {total_files:,} local payload files before transfer...",
            file=sys.stderr,
            flush=True,
        )
    for file_number, path in enumerate(payload_files, start=1):
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise ValueError(f"Unsupported newline in payload filename: {relative!r}")
        digest = file_sha256(path)
        expected = expected_image_hashes.get(path.resolve())
        if expected is not None and digest != expected:
            image_hash_mismatches.append(relative)
        checksum_lines.append(f"{digest}  {relative}")
        payload_bytes += path.stat().st_size
        if show_progress and (file_number % 500 == 0 or file_number == total_files):
            print(
                f"  hashed {file_number:,}/{total_files:,} files "
                f"({payload_bytes / (1024 ** 3):.1f} GiB read)",
                file=sys.stderr,
                flush=True,
            )
    if image_hash_mismatches:
        raise RuntimeError(
            f"{len(image_hash_mismatches):,} image files failed SHA-256 "
            f"verification; first: {image_hash_mismatches[0]}"
        )

    transfer_dir.mkdir(parents=True, exist_ok=True)
    checksum_path = transfer_dir / "SHA256SUMS"
    checksum_temporary = checksum_path.with_suffix(".tmp")
    checksum_temporary.write_text("\n".join(checksum_lines) + "\n")
    os.replace(checksum_temporary, checksum_path)

    summary.update(
        payload_files=len(payload_files),
        payload_bytes=payload_bytes,
        checksum_manifest="transfer/SHA256SUMS",
        checksum_manifest_sha256=hashlib.sha256(checksum_path.read_bytes()).hexdigest(),
        image_hashes_verified=bool(verify_image_hashes),
    )
    ready_path = transfer_dir / "ready.json"
    ready_temporary = ready_path.with_suffix(".tmp")
    ready_temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(ready_temporary, ready_path)
    return summary


def write_transfer_file_list(
    bundle_root: str | Path,
    output_path: str | Path | None = None,
    *,
    required_dataset_key: str | None = None,
) -> dict:
    """Write an rsync file list containing only active images and provenance."""
    root = Path(bundle_root).resolve()
    summary = validate_transfer_bundle(
        root,
        verify_image_hashes=False,
        required_dataset_key=required_dataset_key,
    )
    downloaded_path = root / "manifests" / "downloaded_manifest.csv"
    downloaded = _read_manifest(downloaded_path, REQUIRED_DOWNLOAD_COLUMNS)

    relative_files: set[str] = set()
    for value in downloaded["local_path"].drop_duplicates():
        path = resolve_manifest_image_path(downloaded_path, value).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"Active image is outside the bundle: {path}") from exc
        relative_files.add(relative)
    for directory_name in ("download", "manifests"):
        directory = root / directory_name
        if directory.is_dir():
            relative_files.update(
                path.relative_to(root).as_posix()
                for path in directory.rglob("*")
                if path.is_file()
            )

    transfer_dir = root / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    output = Path(output_path) if output_path else transfer_dir / "FILES.txt"
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    summary_path = transfer_dir / "files_summary.json"
    relative_files.add(output.relative_to(root).as_posix())
    relative_files.add(summary_path.relative_to(root).as_posix())
    ordered = sorted(relative_files)
    if any("\n" in value or "\r" in value for value in ordered):
        raise ValueError("Transfer file paths may not contain newlines")

    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(ordered) + "\n")
    os.replace(temporary, output)
    payload_bytes = sum((root / relative).stat().st_size for relative in ordered if (root / relative).is_file())
    result = {
        **summary,
        "transfer_file_list": str(output),
        "transfer_files": len(ordered),
        "transfer_bytes_before_summary": payload_bytes,
        "content_hashing": False,
    }
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
