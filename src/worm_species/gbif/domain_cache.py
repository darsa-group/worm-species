"""Persistent preprocessing and node-local loading for GBIF/Petri training."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image

from .domain_data import DOMAINS, SPLITS, file_sha256, prepared_paths


CACHE_SCHEMA_VERSION = 1
READY_MARKER = "CACHE_READY.json"
NODE_CACHE_ENV = "WORM_GBIF_NODE_CACHE"


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prepared_summary(config: dict) -> tuple[Path, dict, str]:
    path = Path(config["paths"]["output_root"]) / "prepared" / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8")), file_sha256(path)


def domain_cache_identity(config: dict) -> dict:
    summary_path, summary, summary_sha256 = _prepared_summary(config)
    source_inventory = summary.get("source_inventory")
    if not isinstance(source_inventory, dict) or not source_inventory.get("sha256"):
        raise ValueError(
            f"Prepared summary predates source-inventory caching; rerun prepare: {summary_path}"
        )
    settings = config["preprocessed_cache"]
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "prepared_summary": str(summary_path.resolve()),
        "prepared_summary_sha256": summary_sha256,
        "source_inventory_sha256": source_inventory["sha256"],
        "image_size": int(config["data"]["image_size"]),
        "format": str(settings["format"]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["cache_identity"] = hashlib.sha256(encoded).hexdigest()
    return payload


def domain_cache_directory(config: dict) -> Path:
    identity = domain_cache_identity(config)
    return Path(config["preprocessed_cache"]["root"]) / (
        f"v{CACHE_SCHEMA_VERSION}-{identity['cache_identity'][:20]}"
    )


def _cache_destination(sample_id: str, domain: str, suffix: str) -> Path:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return Path("images") / domain / digest[:2] / f"{digest}{suffix}"


def _preprocess_one(args: tuple[str, str, int]) -> tuple[str, str]:
    source_value, destination_value, image_size = args
    source = Path(source_value)
    destination = Path(destination_value)
    if destination.is_file():
        try:
            with Image.open(destination) as cached:
                if cached.mode == "RGB" and cached.size == (image_size, image_size):
                    cached.verify()
                    return destination_value, "reused"
        except (OSError, ValueError):
            pass
        destination.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with Image.open(source) as image:
            prepared = image.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            prepared.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination_value, "created"


def domain_cache_status(
    config: dict,
    *,
    cache_root: str | Path | None = None,
    verify_files: bool = True,
) -> dict:
    identity = domain_cache_identity(config)
    root = Path(cache_root) if cache_root is not None else domain_cache_directory(config)
    marker = root / READY_MARKER
    result = {
        "ready": False,
        "cache_root": str(root),
        "ready_marker": str(marker),
        "cache_identity": identity["cache_identity"],
    }
    if not marker.is_file():
        result["reason"] = "missing_ready_marker"
        return result
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result["reason"] = "invalid_ready_marker"
        return result
    if recorded.get("cache_identity") != identity["cache_identity"]:
        result["reason"] = "cache_identity_mismatch"
        return result
    manifest_rows = 0
    referenced_paths: list[Path] = []
    for domain in DOMAINS:
        for split in SPLITS:
            manifest = root / "manifests" / f"{domain}_{split}.csv"
            if not manifest.is_file():
                result["reason"] = f"missing_manifest:{manifest.name}"
                return result
            frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
            manifest_rows += len(frame)
            if verify_files:
                referenced_paths.extend(root / value for value in frame["image_path"])
    if manifest_rows != int(recorded.get("rows", -1)):
        result["reason"] = "manifest_row_count_mismatch"
        return result
    if verify_files:
        missing = [path for path in referenced_paths if not path.is_file()]
        if missing:
            result["reason"] = f"missing_cached_images:{len(missing)}"
            return result
    result.update(recorded)
    result["ready"] = True
    result["verified_files"] = bool(verify_files)
    return result


def build_domain_cache(config: dict) -> dict:
    identity = domain_cache_identity(config)
    root = domain_cache_directory(config)
    current = domain_cache_status(config, cache_root=root, verify_files=True)
    if current["ready"]:
        current["status"] = "reused"
        current["copied_this_call"] = 0
        return current

    marker = root / READY_MARKER
    marker.unlink(missing_ok=True)
    image_size = int(config["data"]["image_size"])
    suffix = ".png"
    tasks: list[tuple[str, str, int]] = []
    cached_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for domain in DOMAINS:
        for split in SPLITS:
            frame = pd.read_csv(
                prepared_paths(config, domain, split), dtype=str, keep_default_na=False
            )
            cached = frame.copy()
            relative_paths = []
            for row in frame.itertuples(index=False):
                source = Path(row.image_path)
                if not source.is_file():
                    raise FileNotFoundError(f"Prepared source image is missing: {source}")
                relative = _cache_destination(str(row.sample_id), domain, suffix)
                relative_paths.append(relative.as_posix())
                tasks.append((str(source), str(root / relative), image_size))
            cached["image_path"] = relative_paths
            cached_frames[(domain, split)] = cached

    workers = int(config["preprocessed_cache"]["workers"])
    interval = int(config["preprocessed_cache"].get("progress_interval_images", 1000))
    counts = {"created": 0, "reused": 0}
    if workers == 1:
        completed = map(_preprocess_one, tasks)
        for index, (_path, status) in enumerate(completed, start=1):
            counts[status] += 1
            if index % interval == 0 or index == len(tasks):
                print(f"Preprocessed cache: {index}/{len(tasks)} images", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_preprocess_one, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                _path, status = future.result()
                counts[status] += 1
                if index % interval == 0 or index == len(tasks):
                    print(f"Preprocessed cache: {index}/{len(tasks)} images", flush=True)

    manifests_root = root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {}
    for (domain, split), frame in cached_frames.items():
        destination = manifests_root / f"{domain}_{split}.csv"
        frame.to_csv(destination, index=False)
        manifest_hashes[destination.name] = file_sha256(destination)
    missing_images = sum(
        not (root / value).is_file()
        for frame in cached_frames.values()
        for value in frame["image_path"]
    )
    if missing_images:
        raise RuntimeError(f"Preprocessed cache is incomplete: {missing_images} images missing")
    ready = {
        **identity,
        "rows": int(len(tasks)),
        "created_this_call": counts["created"],
        "reused_this_call": counts["reused"],
        "manifest_sha256": manifest_hashes,
    }
    _atomic_json(marker, ready)
    result = domain_cache_status(config, cache_root=root, verify_files=True)
    if not result["ready"]:
        raise RuntimeError(f"Preprocessed cache failed final validation: {result}")
    result["status"] = "built"
    result["copied_this_call"] = counts["created"]
    return result


def load_cached_domain_frames(config: dict, cache_root: str | Path) -> dict:
    root = Path(cache_root)
    status = domain_cache_status(config, cache_root=root, verify_files=False)
    if not status["ready"]:
        raise RuntimeError(f"Node-local GBIF/Petri cache is not ready: {status}")
    frames = {}
    for domain in DOMAINS:
        frames[domain] = {}
        for split in SPLITS:
            frame = pd.read_csv(
                root / "manifests" / f"{domain}_{split}.csv",
                dtype=str,
                keep_default_na=False,
            )
            frame["image_path"] = frame["image_path"].map(lambda value: str(root / value))
            frames[domain][split] = frame
    return frames
