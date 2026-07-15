"""Compatibility adapters for canonical read-only result discovery."""

from src.worm_species.results.discovery import (
    DEFAULT_MAX_DEPTH,
    EXPERIMENT_ARTIFACT_NAMES,
    POSSIBLY_ACTIVE_SECONDS,
    artifact_as_dict,
    artifact_record,
    discover_results_root,
)
from src.worm_species.results.readers import (
    CHECKPOINT_SUFFIXES,
    MAX_JSON_BYTES,
    MAX_TABULAR_BYTES,
    MAX_TEXT_BYTES,
    load_csv_rows,
    load_json,
    load_text,
)


def discover_results(results_root, *, max_depth=DEFAULT_MAX_DEPTH):
    """Retain the dashboard draft API for a parent results directory."""

    return discover_results_root(results_root, max_depth=max_depth)


__all__ = [
    "CHECKPOINT_SUFFIXES",
    "DEFAULT_MAX_DEPTH",
    "EXPERIMENT_ARTIFACT_NAMES",
    "MAX_JSON_BYTES",
    "MAX_TABULAR_BYTES",
    "MAX_TEXT_BYTES",
    "POSSIBLY_ACTIVE_SECONDS",
    "artifact_as_dict",
    "artifact_record",
    "discover_results",
    "load_csv_rows",
    "load_json",
    "load_text",
]
