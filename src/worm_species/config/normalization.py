"""Pure normalization of the canonical experiment sweep representation."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .ranges import (
    DecimalRange,
    RangeExpansionError,
    decimal_to_number,
    format_range_name,
)


class ConfigNormalizationError(ValueError):
    """Canonical configuration syntax cannot be normalized safely."""


def _require_text(raw: Mapping[str, Any], key: str, path: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigNormalizationError(f"{path}.{key} must be a non-empty string")
    return value


def _explicit_condition(raw: Mapping[str, Any], path: str) -> dict[str, Any]:
    name = _require_text(raw, "name", path)
    transform = _require_text(raw, "transform", path)
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ConfigNormalizationError(f"{path}.parameters must be a mapping")
    condition = copy.deepcopy(dict(raw))
    condition["name"] = name
    condition["transform"] = transform
    condition["parameters"] = copy.deepcopy(dict(parameters))
    return condition


def _range_conditions(raw: Mapping[str, Any], path: str) -> list[dict[str, Any]]:
    template = _require_text(raw, "name_template", path)
    transform = _require_text(raw, "transform", path)
    parameter = _require_text(raw, "parameter", path)
    range_raw = raw.get("range")
    if not isinstance(range_raw, Mapping):
        raise ConfigNormalizationError(f"{path}.range must be a mapping")
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ConfigNormalizationError(f"{path}.parameters must be a mapping")
    if parameter in parameters:
        raise ConfigNormalizationError(
            f"{path}.parameters.{parameter} duplicates the ranged parameter"
        )
    try:
        number_range = DecimalRange.from_mapping(range_raw)
    except RangeExpansionError as exc:
        raise ConfigNormalizationError(f"{path}.{exc}") from exc

    metadata = {
        key: copy.deepcopy(value)
        for key, value in raw.items()
        if key not in {"name_template", "parameter", "range", "parameters"}
    }
    conditions: list[dict[str, Any]] = []
    for index, value in enumerate(number_range.decimals()):
        try:
            name = format_range_name(
                template,
                parameter=parameter,
                value=value,
                index=index,
            )
        except RangeExpansionError as exc:
            raise ConfigNormalizationError(f"{path}.{exc}") from exc
        condition = copy.deepcopy(metadata)
        condition["name"] = name
        condition["transform"] = transform
        condition["parameters"] = copy.deepcopy(dict(parameters))
        condition["parameters"][parameter] = decimal_to_number(
            value,
            prefer_float=number_range.prefer_float,
        )
        conditions.append(condition)
    return conditions


def normalize_conditions(raw_conditions: Any) -> list[dict[str, Any]]:
    """Return complete, atomic condition objects in configured order."""
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ConfigNormalizationError("sweep.conditions must be a non-empty list")
    conditions: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_conditions):
        path = f"sweep.conditions[{index}]"
        if not isinstance(raw, Mapping):
            raise ConfigNormalizationError(f"{path} must be a mapping")
        has_name = "name" in raw
        has_range = "name_template" in raw or "range" in raw or "parameter" in raw
        if has_name and has_range:
            raise ConfigNormalizationError(
                f"{path} must be either an explicit condition or a range, not both"
            )
        if has_name:
            conditions.append(_explicit_condition(raw, path))
        elif has_range:
            conditions.extend(_range_conditions(raw, path))
        else:
            raise ConfigNormalizationError(
                f"{path} requires either name or name_template/range/parameter"
            )

    first_index: dict[str, int] = {}
    for index, condition in enumerate(conditions):
        name = condition["name"]
        if name in first_index:
            raise ConfigNormalizationError(
                "sweep.conditions contains duplicate condition identifier "
                f"{name!r} at indices {first_index[name]} and {index}"
            )
        first_index[name] = index
    return conditions


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy and normalize canonical sweep ranges into condition objects.

    The operation is non-mutating and idempotent. It deliberately leaves
    evaluation sections untouched because evaluation schedules do not create
    model fits.
    """
    if not isinstance(config, dict):
        raise ConfigNormalizationError("config must be a mapping")
    normalized = copy.deepcopy(config)
    sweep = normalized.get("sweep")
    if sweep is None:
        return normalized
    if not isinstance(sweep, dict):
        raise ConfigNormalizationError("sweep must be a mapping")
    if "conditions" in sweep:
        sweep["conditions"] = normalize_conditions(sweep["conditions"])
    return normalized


# British spelling is convenient beside the public ``normalisation`` config
# key while the module and primary API retain the project's Python spelling.
normalise_config = normalize_config


__all__ = [
    "ConfigNormalizationError",
    "normalise_config",
    "normalize_conditions",
    "normalize_config",
]
