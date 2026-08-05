"""Versioned, deterministic caches for paper-ablation input conditions."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import shutil
import socket
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from src.cache import attach_existing_image_cache

from ..config.loading import load_config
from ..config.normalization import normalize_config
from ..data.conditions import build_condition_operations
from ..data.metadata import prepare_metadata
from .maintenance import CacheMaintenanceError


SCHEMA_VERSION = 2
READY_MARKER = "CACHE_READY"
MANIFEST_FILE = "condition_cache_manifest.json"
SUBSET_MANIFEST_FILE = "condition_cache_subset.json"
DEFAULT_TRANSFORMS = frozenset(
    {
        "gaussian_blur_percent",
        "patch_shuffle",
        "resolution_loss",
        "binary_mask",
        "composed",
    }
)
TENSOR_COLUMN = "_condition_cached_tensor_path"


@dataclass(frozen=True)
class ConditionCacheResult:
    status: str
    cache_dir: str
    ready_marker: str
    manifest_path: str
    condition: str
    rows: int | None = None
    cached_rows: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_condition(condition: dict[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(condition)
    name = raw.get("condition") or raw.get("name")
    transform_name = str(raw.get("transform", "original")).lower()
    if not isinstance(name, str) or not name:
        raise ValueError("condition cache requires a non-empty condition name")
    parameters = raw.get("parameters", {}) or {}
    if not isinstance(parameters, dict):
        raise TypeError("condition parameters must be a mapping")
    parameters = copy.deepcopy(parameters)
    for key in (
        "retention",
        "order",
        "diameter",
        "sigma_colour",
        "sigma_space",
        "sigma",
        "grid_size",
        "seed",
        "percent",
        "max_sigma",
        "threshold",
        "operations",
    ):
        if key in raw:
            parameters[key] = copy.deepcopy(raw[key])

    float_parameters = {
        "retention",
        "sigma_colour",
        "sigma_space",
        "sigma",
        "percent",
        "max_sigma",
        "threshold",
    }
    integer_parameters = {"diameter", "grid_size", "seed"}
    for key in float_parameters:
        if key in parameters:
            parameters[key] = float(parameters[key])
    for key in integer_parameters:
        if key in parameters:
            parameters[key] = int(parameters[key])
    if "order" in parameters:
        order = parameters["order"]
        if isinstance(order, str):
            order = order.split(",")
        parameters["order"] = [int(value) for value in order]

    return {
        "name": name,
        "feature": str(raw.get("feature", "baseline")),
        "transform": transform_name,
        "strength": float(raw.get("strength", 0.0)),
        "parameters": parameters,
    }


def _condition_digest(
    condition: dict[str, Any],
    protocol_version: int,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": int(protocol_version),
        "condition": _canonical_condition(condition),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slug(value: str) -> str:
    cleaned = "".join(
        character.lower()
        if character.isalnum() or character in {"-", "_", "."}
        else "_"
        for character in value
    ).strip("._")
    return cleaned or "condition"


def condition_cache_directory(
    cache_root: str | Path,
    condition: dict[str, Any],
    *,
    protocol_version: int = 1,
) -> Path:
    canonical = _canonical_condition(condition)
    digest = _condition_digest(canonical, protocol_version)
    return (
        Path(cache_root).expanduser()
        / f"v{int(protocol_version)}"
        / f"{_slug(canonical['name'])}--{digest[:16]}"
    )


def _tensor_path(cache_dir: Path, base_image_path: str | Path) -> Path:
    source = Path(base_image_path)
    key = source.stem
    return cache_dir / "tensors" / key[:2] / f"{key}.pt"


def condition_cache_settings(config: dict[str, Any]) -> dict[str, Any]:
    raw = (config.get("cache", {}) or {}).get("condition_variants", {}) or {}
    if not isinstance(raw, dict):
        raise TypeError("cache.condition_variants must be a mapping")
    return {
        "enabled": bool(raw.get("enabled", False)),
        "protocol_version": int(raw.get("protocol_version", 1)),
        "storage": str(raw.get("storage", "torch_float32")),
    }


def cacheable_conditions(
    config: dict[str, Any],
    transforms_to_cache: Iterable[str] = DEFAULT_TRANSFORMS,
) -> list[dict[str, Any]]:
    canonical = normalize_config(config)
    sweep = canonical.get("sweep", {}) or {}
    conditions = sweep.get("conditions", []) or []
    selected = {str(name).lower() for name in transforms_to_cache}
    return [
        _canonical_condition(condition)
        for condition in conditions
        if str(condition.get("transform", "")).lower() in selected
    ]


def attach_condition_cache(
    frame: pd.DataFrame,
    *,
    cache_root: str | Path,
    condition: dict[str, Any],
    protocol_version: int,
) -> pd.DataFrame:
    """Point rows at complete cached tensors for one resolved condition."""
    cache_dir = condition_cache_directory(
        cache_root, condition, protocol_version=protocol_version
    )
    verify_condition_cache(cache_dir)
    if "_cached_image_path" not in frame.columns:
        raise CacheMaintenanceError(
            "condition caching requires '_cached_image_path'"
        )
    result = frame.copy()
    paths = [
        _tensor_path(cache_dir, path) if isinstance(path, str) else None
        for path in result["_cached_image_path"].tolist()
    ]
    missing = [
        str(path)
        for path in paths
        if path is None or not path.is_file()
    ]
    if missing:
        raise CacheMaintenanceError(
            f"condition cache {cache_dir} is incomplete: "
            f"{len(missing)} tensors are missing"
        )
    result[TENSOR_COLUMN] = [str(path) for path in paths]
    return result


def stage_condition_cache_subset(
    frame: pd.DataFrame,
    *,
    source_cache_root: str | Path,
    staging_cache_root: str | Path,
    condition: dict[str, Any],
    protocol_version: int,
) -> pd.DataFrame:
    """Stage only the tensors referenced by ``frame`` into shared scratch.

    The persistent condition cache is complete for every metadata row, while a
    post-training evaluator needs only its test frame. A per-condition lock lets
    concurrent GPU tasks on one node publish and reuse that smaller subset.
    """
    source_dir = condition_cache_directory(
        source_cache_root,
        condition,
        protocol_version=protocol_version,
    ).resolve()
    staging_dir = condition_cache_directory(
        staging_cache_root,
        condition,
        protocol_version=protocol_version,
    ).resolve()
    if source_dir == staging_dir:
        return attach_condition_cache(
            frame,
            cache_root=source_cache_root,
            condition=condition,
            protocol_version=protocol_version,
        )
    if "_cached_image_path" not in frame.columns:
        raise CacheMaintenanceError(
            "condition caching requires '_cached_image_path'"
        )
    if (staging_dir / READY_MARKER).is_file():
        return attach_condition_cache(
            frame,
            cache_root=staging_cache_root,
            condition=condition,
            protocol_version=protocol_version,
        )

    ready_path = source_dir / READY_MARKER
    manifest_path = source_dir / MANIFEST_FILE
    if not ready_path.is_file() or not manifest_path.is_file():
        raise CacheMaintenanceError(
            f"condition cache is not ready: {source_dir}"
        )
    try:
        source_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        source_rows = int(source_manifest["rows"])
        source_cached_rows = int(source_manifest["cached_rows"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CacheMaintenanceError(
            f"condition cache manifest is invalid: {manifest_path}"
        ) from exc
    if (
        source_manifest.get("schema_version") != SCHEMA_VERSION
        or source_manifest.get("status") != "complete"
        or source_rows < 1
        or source_cached_rows != source_rows
    ):
        raise CacheMaintenanceError(
            f"condition cache manifest is incomplete: {manifest_path}"
        )

    stat = ready_path.stat()
    source_signature = (
        f"{source_dir}|{stat.st_dev}:{stat.st_ino}:{stat.st_size}:"
        f"{stat.st_mtime_ns}"
    )
    destination_paths = [
        _tensor_path(staging_dir, path)
        for path in frame["_cached_image_path"].tolist()
    ]
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = staging_dir.parent / f".{staging_dir.name}.subset.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        subset_manifest_path = staging_dir / SUBSET_MANIFEST_FILE
        existing_signature = None
        if subset_manifest_path.is_file():
            try:
                existing_signature = json.loads(
                    subset_manifest_path.read_text(encoding="utf-8")
                ).get("source_signature")
            except (OSError, json.JSONDecodeError):
                existing_signature = None
        if staging_dir.exists() and existing_signature != source_signature:
            shutil.rmtree(staging_dir)

        source_paths = [
            _tensor_path(source_dir, path) if isinstance(path, str) else None
            for path in frame["_cached_image_path"].tolist()
        ]
        missing_sources = [
            str(source_path)
            for source_path, destination_path in zip(
                source_paths, destination_paths
            )
            if (
                not destination_path.is_file()
                and (source_path is None or not source_path.is_file())
            )
        ]
        if missing_sources:
            raise CacheMaintenanceError(
                f"condition cache {source_dir} is incomplete for this "
                f"evaluation: {len(missing_sources)} tensors are missing"
            )
        copied = 0
        for source_path, destination_path in zip(
            source_paths, destination_paths
        ):
            if destination_path.is_file():
                continue
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination_path.parent,
                prefix=f".{destination_path.name}.",
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                shutil.copyfile(source_path, temporary)
                os.replace(temporary, destination_path)
            finally:
                temporary.unlink(missing_ok=True)
            copied += 1

        staging_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "condition": _canonical_condition(condition),
            "source_cache_dir": str(source_dir),
            "source_signature": source_signature,
            "staged_tensor_count": sum(
                1 for _ in (staging_dir / "tensors").rglob("*.pt")
            ),
            "required_tensor_count": len(destination_paths),
            "copied_this_call": copied,
        }
        temporary_manifest = subset_manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, subset_manifest_path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    missing_staged = [
        str(path) for path in destination_paths if not path.is_file()
    ]
    if missing_staged:
        raise CacheMaintenanceError(
            f"staged condition cache {staging_dir} is incomplete: "
            f"{len(missing_staged)} tensors are missing"
        )
    result = frame.copy()
    result[TENSOR_COLUMN] = [str(path) for path in destination_paths]
    return result


def _save_tensor(
    args: tuple[str, str, dict[str, Any], int],
) -> tuple[str, str | None]:
    source_raw, destination_raw, condition, image_size = args
    source = Path(source_raw)
    destination = Path(destination_raw)
    try:
        with Image.open(source) as handle:
            image = handle.convert("RGB")
        preparation = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                *build_condition_operations(condition),
            ]
        )
        tensor = preparation(image).to(dtype=torch.float32).contiguous()
        if tuple(tensor.shape) != (3, image_size, image_size):
            raise ValueError(
                f"unexpected cached tensor shape {tuple(tensor.shape)}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(tensor, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return str(destination), None
    except Exception as exc:
        return str(destination), repr(exc)


def _base_manifest_digest(base_cache_dir: Path) -> str:
    manifest = base_cache_dir / "cache_manifest.txt"
    ready = base_cache_dir / READY_MARKER
    if not manifest.is_file() or not ready.is_file():
        raise CacheMaintenanceError(
            f"persistent base cache is not ready: {base_cache_dir}"
        )
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _manifest_matches(
    manifest_path: Path,
    *,
    condition: dict[str, Any],
    protocol_version: int,
    base_manifest_sha256: str,
) -> bool:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("protocol_version") == protocol_version
        and payload.get("condition") == _canonical_condition(condition)
        and payload.get("base_manifest_sha256") == base_manifest_sha256
        and payload.get("status") == "complete"
    )


def build_condition_cache(
    config_path: str | Path,
    *,
    data_root: str | Path,
    metadata_csv: str | Path,
    base_cache_dir: str | Path,
    condition_cache_root: str | Path,
    condition_index: int,
    transforms_to_cache: Iterable[str] = DEFAULT_TRANSFORMS,
    image_col: str = "rel_path_seg",
    num_workers: int = 8,
    force: bool = False,
) -> ConditionCacheResult:
    """Build one condition selected from the canonical visual sweep."""
    if num_workers < 1:
        raise ValueError("condition cache num_workers must be at least 1")
    source_config = load_config(Path(config_path).expanduser().resolve())
    conditions = cacheable_conditions(source_config, transforms_to_cache)
    if condition_index < 0 or condition_index >= len(conditions):
        raise ValueError(
            f"condition index must be in [0, {len(conditions) - 1}]"
        )
    condition = conditions[condition_index]
    settings = condition_cache_settings(source_config)
    protocol_version = int(settings["protocol_version"])
    if settings["storage"] != "torch_float32":
        raise ValueError(
            "cache.condition_variants.storage must be torch_float32"
        )

    data_root = Path(data_root).expanduser().resolve()
    metadata_csv = Path(metadata_csv).expanduser().resolve()
    base_cache_dir = Path(base_cache_dir).expanduser().resolve()
    condition_cache_root = Path(condition_cache_root).expanduser().resolve()
    if not data_root.is_dir() or not metadata_csv.is_file():
        raise ValueError("condition cache data root or metadata CSV is missing")
    if condition_cache_root == data_root or condition_cache_root == Path("/"):
        raise ValueError("condition cache root must be a dedicated directory")
    condition_cache_root.mkdir(parents=True, exist_ok=True)
    base_manifest_sha256 = _base_manifest_digest(base_cache_dir)
    cache_dir = condition_cache_directory(
        condition_cache_root,
        condition,
        protocol_version=protocol_version,
    )
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / f".{cache_dir.name}.lock"
    manifest_path = cache_dir / MANIFEST_FILE
    ready_path = cache_dir / READY_MARKER

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if (
            not force
            and ready_path.is_file()
            and _manifest_matches(
                manifest_path,
                condition=condition,
                protocol_version=protocol_version,
                base_manifest_sha256=base_manifest_sha256,
            )
        ):
            verified = verify_condition_cache(cache_dir)
            return ConditionCacheResult(
                status="already_ready",
                cache_dir=verified.cache_dir,
                ready_marker=verified.ready_marker,
                manifest_path=verified.manifest_path,
                condition=verified.condition,
                rows=verified.rows,
                cached_rows=verified.cached_rows,
            )

        runtime = copy.deepcopy(source_config)
        runtime.setdefault("data", {}).update(
            root_dir=str(data_root),
            metadata_csv=str(metadata_csv),
            image_col=image_col,
        )
        runtime.setdefault("cache", {}).update(
            enabled=True,
            dir=str(base_cache_dir),
            root_dir_cache=str(base_cache_dir.parent),
        )
        metadata = prepare_metadata(runtime)
        cached = attach_existing_image_cache(runtime, metadata)
        missing_base = int(cached["_cached_image_path"].isna().sum())
        if missing_base:
            raise CacheMaintenanceError(
                f"base cache is incomplete: {missing_base} images are missing"
            )

        preprocessing = runtime.get("preprocessing", {}) or {}
        image_size = int(
            preprocessing.get(
                "image_size", runtime.get("data", {}).get("image_size", 224)
            )
        )
        temporary = Path(
            tempfile.mkdtemp(
                dir=cache_dir.parent,
                prefix=f".{cache_dir.name}.building-",
            )
        )
        errors: list[str] = []
        try:
            tasks = [
                (
                    str(base_path),
                    str(_tensor_path(temporary, base_path)),
                    condition,
                    image_size,
                )
                for base_path in cached["_cached_image_path"].tolist()
            ]
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(_save_tensor, task) for task in tasks
                ]
                for future in as_completed(futures):
                    _, error = future.result()
                    if error is not None:
                        errors.append(error)
            if errors:
                raise CacheMaintenanceError(
                    f"condition cache build failed for {len(errors)} images; "
                    f"first error: {errors[0]}"
                )
            tensor_count = sum(
                1 for _ in (temporary / "tensors").rglob("*.pt")
            )
            if tensor_count != len(cached):
                raise CacheMaintenanceError(
                    f"condition cache wrote {tensor_count}/{len(cached)} tensors"
                )
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "protocol_version": protocol_version,
                "status": "complete",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "host": socket.gethostname(),
                "condition": condition,
                "condition_sha256": _condition_digest(
                    condition, protocol_version
                ),
                "base_cache_dir": str(base_cache_dir),
                "base_manifest_sha256": base_manifest_sha256,
                "data_root": str(data_root),
                "metadata_csv": str(metadata_csv),
                "image_col": image_col,
                "image_size": image_size,
                "storage": "torch_float32",
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "rows": len(cached),
                "cached_rows": tensor_count,
            }
            (temporary / MANIFEST_FILE).write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (temporary / READY_MARKER).write_text("", encoding="utf-8")
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            os.replace(temporary, cache_dir)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    return ConditionCacheResult(
        status="built",
        cache_dir=str(cache_dir),
        ready_marker=str(ready_path),
        manifest_path=str(manifest_path),
        condition=condition["name"],
        rows=len(cached),
        cached_rows=len(cached),
    )


def verify_condition_cache(cache_dir: str | Path) -> ConditionCacheResult:
    cache_dir = Path(cache_dir).expanduser().resolve()
    ready_path = cache_dir / READY_MARKER
    manifest_path = cache_dir / MANIFEST_FILE
    if not ready_path.is_file() or not manifest_path.is_file():
        raise CacheMaintenanceError(
            f"condition cache is not ready: {cache_dir}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = int(manifest["rows"])
        cached_rows = int(manifest["cached_rows"])
        condition = str(manifest["condition"]["name"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CacheMaintenanceError(
            f"condition cache manifest is invalid: {manifest_path}"
        ) from exc
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or rows < 1
        or cached_rows != rows
    ):
        raise CacheMaintenanceError(
            f"condition cache manifest is incomplete: {manifest_path}"
        )
    actual = sum(1 for _ in (cache_dir / "tensors").rglob("*.pt"))
    if actual != rows:
        raise CacheMaintenanceError(
            f"condition cache contains {actual}/{rows} tensor files"
        )
    return ConditionCacheResult(
        status="ready",
        cache_dir=str(cache_dir),
        ready_marker=str(ready_path),
        manifest_path=str(manifest_path),
        condition=condition,
        rows=rows,
        cached_rows=cached_rows,
    )


def resolved_condition_cache_directory(
    config_path: str | Path,
    cache_root: str | Path,
    *,
    require_cacheable: bool = True,
) -> Path | None:
    """Resolve the condition-cache directory for one external run config."""
    from ..training.loaders import get_input_condition

    config = normalize_config(
        load_config(Path(config_path).expanduser().resolve())
    )
    settings = condition_cache_settings(config)
    if not settings["enabled"]:
        raise ValueError("condition variants are not enabled")
    condition = get_input_condition(config)
    if condition["transform"] not in DEFAULT_TRANSFORMS:
        if require_cacheable:
            raise ValueError(
                "resolved condition is not cacheable: "
                f"{condition['transform']}"
            )
        return None
    return condition_cache_directory(
        cache_root,
        condition,
        protocol_version=int(settings["protocol_version"]),
    )
