"""Configuration loading and validation for canonical SLURM workflows.

This module deliberately has no scheduler dependency.  Loading and validating
a configuration is safe on laptops and login nodes and never submits a job or
creates an output directory.
"""

from __future__ import annotations

import copy
import os
import re
import shlex
from pathlib import Path
from typing import Any, Mapping

import yaml

from ..config.overrides import parse_scalar, set_nested


class SlurmConfigError(ValueError):
    """A submission configuration is incomplete or internally inconsistent."""


_SLURM_KEYS: dict[str, object] = {
    "enabled": None,
    "cluster_profile": None,
    "account": None,
    "partition": None,
    "nodes": None,
    "ntasks": None,
    "cpus_per_task": None,
    "memory": None,
    "time_limit": None,
    "gpus_per_task": None,
    "array": {"max_active": None},
    "planning": {
        "experiment_type": None,
        "external_expansion": None,
        "training_profile": None,
    },
    "setup": {
        "enabled": None,
        "per_node": None,
        "cpus_per_task": None,
        "memory": None,
        "time_limit": None,
    },
    "collection": {
        "enabled": None,
        "partition": None,
        "cpus_per_task": None,
        "memory": None,
        "time_limit": None,
    },
    "cleanup": {
        "enabled": None,
        "partition": None,
        "per_node": None,
        "cpus_per_task": None,
        "memory": None,
        "time_limit": None,
    },
    "scratch": {
        "mode": None,
        "root": None,
        "nodes": None,
        "copy_project": None,
        "copy_data": None,
        "data_include": None,
        "reuse_ready_cache": None,
        "cleanup_after_run": None,
        "copy_cache_to_tmp": None,
        "tmp_reserve_gb": None,
        "ready_marker": None,
    },
    "environment": {"conda_sh": None, "conda_env": None},
    "paths": {
        "project_root": None,
        "data_root": None,
        "metadata_csv": None,
        "results_root": None,
        "cache_root": None,
    },
    "logging": {"directory": None, "separate_stdout_stderr": None},
    "monitoring": {"enabled": None, "interval_seconds": None},
    "submission": {"extra_sbatch_args": None, "exclude_nodes": None},
}

_PATH_FIELDS = (
    ("scratch", "root"),
    ("environment", "conda_sh"),
    ("paths", "project_root"),
    ("paths", "data_root"),
    ("paths", "metadata_csv"),
    ("paths", "results_root"),
    ("paths", "cache_root"),
    ("logging", "directory"),
)
_MANAGED_SBATCH_OPTIONS = frozenset({
    "--array",
    "--dependency",
    "--error",
    "--export",
    "--job-name",
    "--output",
})
_SHELL_META = re.compile(r"[\n\r;&|`<>]")
_MEMORY_RE = re.compile(r"^(?P<number>[1-9][0-9]*)(?P<unit>[KMGTkmgt]?)$")
_TIME_RE = re.compile(
    r"^(?:(?P<days>[0-9]+)-)?(?P<hours>[0-9]+):(?P<minutes>[0-5][0-9]):(?P<seconds>[0-5][0-9])$"
)


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings without mutating either input."""
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_extended(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise SlurmConfigError(f"Configuration extends cycle: {cycle}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise SlurmConfigError(f"Configuration not found: {path}") from exc
    if not isinstance(loaded, dict):
        raise SlurmConfigError(f"Configuration must be a YAML mapping: {path}")
    parent = loaded.pop("extends", None)
    if parent is None:
        return loaded
    if not isinstance(parent, str) or not parent.strip():
        raise SlurmConfigError(f"extends must be a non-empty path string: {path}")
    parent_config = _load_extended(path.parent / parent, (*stack, path))
    return deep_merge(parent_config, loaded)


def _apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise SlurmConfigError(f"Override must look like key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SlurmConfigError("Override key must not be empty")
        value = parse_scalar(raw_value)
        if key in {"slurm.scratch.nodes", "slurm.submission.exclude_nodes"}:
            if isinstance(value, str):
                parsed = yaml.safe_load(raw_value)
                if isinstance(parsed, list):
                    value = parsed
                else:
                    value = [item for item in re.split(r"[\s,]+", raw_value) if item]
        set_nested(resolved, key, value)
    return resolved


def _expand_selected_paths(config: dict[str, Any]) -> None:
    slurm = config.get("slurm", {})
    for section, key in _PATH_FIELDS:
        mapping = slurm.get(section, {})
        if isinstance(mapping, dict) and isinstance(mapping.get(key), str):
            mapping[key] = os.path.expandvars(os.path.expanduser(mapping[key]))


def load_submission_config(
    experiment_config: str | Path,
    cluster_config: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve base, cluster defaults, experiment overlay, and CLI overrides.

    Cluster files contain machine capabilities rather than scientific settings.
    They may replace generic experiment resources, while explicit command-line
    overrides take precedence over both.
    """
    experiment = _load_extended(Path(experiment_config))
    if cluster_config is not None:
        cluster = _load_extended(Path(cluster_config))
        unexpected = set(cluster) - {"slurm"}
        if unexpected:
            raise SlurmConfigError(
                "Cluster profiles may only define 'slurm'; unexpected keys: "
                + ", ".join(sorted(unexpected))
            )
        # Cluster profiles own machine capabilities and may override generic
        # experiment resources (for example Genome's shorter GPU queue).
        # GHPC intentionally leaves experiment-dependent resources unset.
        resolved = deep_merge(experiment, cluster)
        experiment_collection = (
            experiment.get("slurm", {}).get("collection", {})
            if isinstance(experiment.get("slurm"), dict)
            else {}
        )
        if (
            resolved.get("slurm", {}).get("enabled", True)
            and "enabled" in experiment_collection
        ):
            resolved["slurm"].setdefault("collection", {})["enabled"] = (
                experiment_collection["enabled"]
            )
    else:
        resolved = copy.deepcopy(experiment)
    resolved = _apply_overrides(resolved, overrides or [])
    _expand_selected_paths(resolved)
    validate_slurm_config(resolved)
    return resolved


def parse_memory(value: Any, path: str = "memory") -> int:
    """Return configured memory in MiB; bare historical integers mean MiB."""
    if isinstance(value, bool):
        raise SlurmConfigError(f"{path} must be a positive memory value")
    if isinstance(value, int):
        if value <= 0:
            raise SlurmConfigError(f"{path} must be positive")
        return value
    if not isinstance(value, str):
        raise SlurmConfigError(f"{path} must be an integer MiB value or K/M/G/T string")
    match = _MEMORY_RE.fullmatch(value.strip())
    if match is None:
        raise SlurmConfigError(f"{path} has invalid memory syntax: {value!r}")
    number = int(match.group("number"))
    multiplier = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[
        match.group("unit").upper()
    ]
    return max(1, int(number * multiplier))


def parse_time_limit(value: Any, path: str = "time_limit") -> int:
    """Return a SLURM time limit in seconds."""
    if not isinstance(value, str):
        raise SlurmConfigError(f"{path} must use HH:MM:SS or D-HH:MM:SS syntax")
    match = _TIME_RE.fullmatch(value.strip())
    if match is None:
        raise SlurmConfigError(f"{path} has invalid time-limit syntax: {value!r}")
    days = int(match.group("days") or 0)
    hours = int(match.group("hours"))
    seconds = ((days * 24 + hours) * 60 + int(match.group("minutes"))) * 60
    seconds += int(match.group("seconds"))
    if seconds <= 0:
        raise SlurmConfigError(f"{path} must be greater than zero")
    return seconds


def _reject_unknown(mapping: Any, schema: dict[str, object], path: str) -> None:
    if not isinstance(mapping, dict):
        raise SlurmConfigError(f"{path} must be a mapping")
    unknown = set(mapping) - set(schema)
    if unknown:
        raise SlurmConfigError(
            f"Unknown {path} option(s): " + ", ".join(sorted(unknown))
        )
    for key, child_schema in schema.items():
        if key in mapping and isinstance(child_schema, dict):
            _reject_unknown(mapping[key], child_schema, f"{path}.{key}")


def _require_bool(mapping: dict[str, Any], key: str, path: str) -> None:
    if key in mapping and not isinstance(mapping[key], bool):
        raise SlurmConfigError(f"{path}.{key} must be boolean")


def _require_positive_int(mapping: dict[str, Any], key: str, path: str) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SlurmConfigError(f"{path}.{key} must be a positive integer")


def _validate_resource(mapping: dict[str, Any], path: str) -> None:
    _require_positive_int(mapping, "cpus_per_task", path)
    if "memory" in mapping:
        parse_memory(mapping["memory"], f"{path}.memory")
    if "time_limit" in mapping:
        parse_time_limit(mapping["time_limit"], f"{path}.time_limit")


def _validate_extra_args(args: Any) -> None:
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise SlurmConfigError("slurm.submission.extra_sbatch_args must be a list of strings")
    for item in args:
        if not item.startswith("--") or _SHELL_META.search(item):
            raise SlurmConfigError(f"Unsafe or unsupported extra sbatch argument: {item!r}")
        try:
            pieces = shlex.split(item)
        except ValueError as exc:
            raise SlurmConfigError(f"Invalid extra sbatch argument: {item!r}") from exc
        if len(pieces) != 1:
            raise SlurmConfigError(
                "Each slurm.submission.extra_sbatch_args item must be one argv element"
            )
        option = pieces[0].split("=", 1)[0]
        if option in _MANAGED_SBATCH_OPTIONS:
            raise SlurmConfigError(f"{option} is managed by the canonical launcher")


def validate_slurm_config(config: dict[str, Any]) -> None:
    """Validate scheduler-specific structure and resource constraints."""
    slurm = config.get("slurm")
    if not isinstance(slurm, dict):
        raise SlurmConfigError("slurm must be a mapping")
    _reject_unknown(slurm, _SLURM_KEYS, "slurm")
    _require_bool(slurm, "enabled", "slurm")
    for key in ("nodes", "ntasks", "cpus_per_task"):
        _require_positive_int(slurm, key, "slurm")
    if "gpus_per_task" in slurm:
        value = slurm["gpus_per_task"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SlurmConfigError("slurm.gpus_per_task must be a non-negative integer")
    _validate_resource(slurm, "slurm")

    array = slurm.get("array", {})
    _require_positive_int(array, "max_active", "slurm.array")
    expansion = slurm.get("planning", {}).get("external_expansion", "sweep")
    if expansion not in {"sweep", "colour_ablation", "dual_cue"}:
        raise SlurmConfigError(
            f"Unsupported slurm.planning.external_expansion: {expansion!r}"
        )
    profile = slurm.get("planning", {}).get("training_profile")
    if not isinstance(profile, str) or not profile:
        raise SlurmConfigError("slurm.planning.training_profile is required")
    experiment_type = slurm.get("planning", {}).get("experiment_type")
    if not isinstance(experiment_type, str) or not experiment_type:
        raise SlurmConfigError("slurm.planning.experiment_type is required")
    for name in ("setup", "collection", "cleanup"):
        stage = slurm.get(name, {})
        _require_bool(stage, "enabled", f"slurm.{name}")
        if name in {"setup", "cleanup"}:
            _require_bool(stage, "per_node", f"slurm.{name}")
        _validate_resource(stage, f"slurm.{name}")

    scratch = slurm.get("scratch", {})
    mode = scratch.get("mode", "none")
    if mode not in {"none", "node_local", "persistent_cache", "job_local_cache"}:
        raise SlurmConfigError(f"Unsupported slurm.scratch.mode: {mode!r}")
    for key in ("copy_project", "copy_data", "reuse_ready_cache", "cleanup_after_run"):
        _require_bool(scratch, key, "slurm.scratch")
    nodes = scratch.get("nodes", [])
    if not isinstance(nodes, list) or any(not isinstance(node, str) or not node for node in nodes):
        raise SlurmConfigError("slurm.scratch.nodes must be a list of non-empty node names")
    if len(nodes) != len(set(nodes)):
        raise SlurmConfigError("slurm.scratch.nodes contains duplicate node names")
    copy_mode = scratch.get("copy_cache_to_tmp", "auto")
    if copy_mode not in {"auto", 0, 1, "0", "1", False, True}:
        raise SlurmConfigError("slurm.scratch.copy_cache_to_tmp must be auto, 0, or 1")
    reserve = scratch.get("tmp_reserve_gb", 0)
    if isinstance(reserve, bool) or not isinstance(reserve, int) or reserve < 0:
        raise SlurmConfigError("slurm.scratch.tmp_reserve_gb must be a non-negative integer")
    data_include = scratch.get("data_include", [])
    if not isinstance(data_include, list) or any(not isinstance(item, str) for item in data_include):
        raise SlurmConfigError("slurm.scratch.data_include must be a list of strings")

    for section, key in _PATH_FIELDS:
        value = slurm.get(section, {}).get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise SlurmConfigError(f"slurm.{section}.{key} must be a non-empty path string")
    environment = slurm.get("environment", {})
    for key in ("conda_sh", "conda_env"):
        value = environment.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise SlurmConfigError(f"slurm.environment.{key} must be a non-empty string")
    logging = slurm.get("logging", {})
    _require_bool(logging, "separate_stdout_stderr", "slurm.logging")
    monitoring = slurm.get("monitoring", {})
    _require_bool(monitoring, "enabled", "slurm.monitoring")
    _require_positive_int(monitoring, "interval_seconds", "slurm.monitoring")
    submission = slurm.get("submission", {})
    _validate_extra_args(submission.get("extra_sbatch_args", []))
    excluded = submission.get("exclude_nodes", [])
    if not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded):
        raise SlurmConfigError("slurm.submission.exclude_nodes must be a list of strings")

    enabled = slurm.get("enabled", True)
    if enabled:
        for key in ("partition", "account"):
            value = slurm.get(key)
            if not isinstance(value, str) or not value.strip():
                raise SlurmConfigError(f"slurm.{key} is required when SLURM is enabled")
    if mode == "node_local":
        if not nodes:
            raise SlurmConfigError(
                "slurm.scratch.nodes is required for node-local scratch; explicitly set GPU_NODES/profile nodes"
            )
        if not bool(slurm.get("setup", {}).get("enabled", False)):
            raise SlurmConfigError("node-local scratch requires slurm.setup.enabled=true")
        if not bool(slurm.get("cleanup", {}).get("enabled", False)):
            raise SlurmConfigError("node-local scratch requires slurm.cleanup.enabled=true")
    if mode in {"persistent_cache", "job_local_cache"}:
        cache_root = slurm.get("paths", {}).get("cache_root")
        if not isinstance(cache_root, str) or not cache_root:
            raise SlurmConfigError(f"slurm.paths.cache_root is required for {mode}")
