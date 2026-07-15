from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any

from .normalization import normalize_conditions
from .overrides import parse_scalar, set_nested


@dataclass
class SweepItem:
    """One externally resolvable training fit from the canonical sweep."""

    index: int
    assignments: tuple[tuple[str, Any], ...]
    condition: dict[str, Any] | None = None

    @property
    def parameter_values(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.assignments))


def parse_sweep_item(item: str) -> tuple[str, list[Any]]:
    """Parse ``key=v1,v2`` using the legacy scalar rules."""
    if "=" not in item:
        raise ValueError(f"Sweep item must look like key=v1,v2. Got: {item}")

    key, values = item.split("=", 1)
    parsed = [parse_scalar(value) for value in values.split(",") if value.strip()]
    if len(parsed) == 0:
        raise ValueError(f"No values supplied for sweep key: {key}")
    return key, parsed


def get_sweep_parameters_from_config(config: dict[str, Any]) -> dict[str, list[Any]]:
    """Return the ordinary configured sweep without expanding it."""
    sweep_config = config.get("sweep", {})
    if not sweep_config.get("enabled", False):
        return {}
    parameters = sweep_config.get("parameters", {})
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise ValueError("sweep.parameters must be a dictionary.")
    return parameters


def get_sweep_conditions_from_config(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return complete canonical conditions without creating a product within them."""
    sweep_config = config.get("sweep", {})
    if not sweep_config.get("enabled", False) or "conditions" not in sweep_config:
        return []
    return normalize_conditions(sweep_config["conditions"])


def generate_colour_retention_values(config: dict[str, Any]) -> list[float]:
    """Generate the legacy inclusive colour-retention percentage sequence."""
    ablation_config = config.get("colour_ablation", {}) or {}
    if not ablation_config.get("enabled", False):
        return []

    start = int(ablation_config.get("start_percent", 100))
    stop = int(ablation_config.get("stop_percent", 0))
    step = int(ablation_config.get("step_percent", 1))
    if not 0 <= start <= 100 or not 0 <= stop <= 100:
        raise ValueError("Colour-ablation percentages must be between 0 and 100.")
    if step <= 0:
        raise ValueError("colour_ablation.step_percent must be greater than zero.")

    if start >= stop:
        percentages = list(range(start, stop - 1, -step))
    else:
        percentages = list(range(start, stop + 1, step))
    if not percentages or percentages[-1] != stop:
        percentages.append(stop)
    return [percentage / 100.0 for percentage in percentages]


def get_colour_sweep_parameters_from_config(
    config: dict[str, Any],
) -> dict[str, list[Any]]:
    """Combine ordinary and colour-ablation dimensions using legacy safeguards."""
    sweep_config = config.get("sweep", {}) or {}
    parameters: dict[str, list[Any]] = {}
    if sweep_config.get("enabled", False):
        configured = sweep_config.get("parameters", {}) or {}
        if not isinstance(configured, dict):
            raise ValueError("sweep.parameters must be a dictionary.")
        parameters = copy.deepcopy(configured)

    colour_values = generate_colour_retention_values(config)
    if colour_values:
        ablation_config = config.get("colour_ablation", {}) or {}
        if parameters and not ablation_config.get("combine_with_sweep", False):
            raise ValueError(
                "Colour ablation is enabled while an ordinary parameter sweep is also enabled. "
                "Disable sweep.enabled for a controlled colour-only experiment, or set "
                "colour_ablation.combine_with_sweep=true intentionally."
            )
        if "data.colour_retention" in parameters:
            raise ValueError(
                "data.colour_retention is defined both by sweep.parameters and colour_ablation."
            )
        parameters["data.colour_retention"] = colour_values
    return parameters


def get_sweep_parameters_from_cli(items: list[str]) -> dict[str, list[Any]]:
    parameters: dict[str, list[Any]] = {}
    for item in items:
        key, values = parse_sweep_item(item)
        parameters[key] = values
    return parameters


def expand_sweep_items(
    base_config: dict[str, Any],
    cli_sweep_items: list[str] | None = None,
    *,
    include_colour_ablation: bool = False,
) -> list[SweepItem]:
    """Expand one canonical training layer in deterministic configured order.

    Ordinary dotted parameters form a Cartesian product. Complete condition
    objects are an additional single dimension and remain atomic. Sections
    outside ``sweep`` -- notably ``evaluation`` -- never add training fits.
    """
    cli_sweep_items = cli_sweep_items or []
    if cli_sweep_items:
        parameters = get_sweep_parameters_from_cli(cli_sweep_items)
    elif include_colour_ablation:
        parameters = get_colour_sweep_parameters_from_config(base_config)
    else:
        parameters = get_sweep_parameters_from_config(base_config)

    conditions = get_sweep_conditions_from_config(base_config)
    keys = list(parameters)
    parameter_products = (
        itertools.product(*(parameters[key] for key in keys))
        if keys
        else [()]
    )
    condition_values: list[dict[str, Any] | None] = conditions or [None]
    items: list[SweepItem] = []
    for combination in parameter_products:
        assignments = tuple(
            (key, copy.deepcopy(value))
            for key, value in zip(keys, combination)
        )
        for condition in condition_values:
            items.append(
                SweepItem(
                    index=len(items),
                    assignments=assignments,
                    condition=copy.deepcopy(condition),
                )
            )
    return items


def apply_sweep_item(
    base_config: dict[str, Any],
    item: SweepItem,
    *,
    disable_sweep: bool = False,
) -> dict[str, Any]:
    """Apply one sweep item to a deep copy of ``base_config``."""
    config = copy.deepcopy(base_config)
    for key, value in item.assignments:
        set_nested(config, key, copy.deepcopy(value))
    if item.condition is not None:
        input_condition = copy.deepcopy(item.condition)
        input_condition["enabled"] = True
        config["input_condition"] = input_condition
    if disable_sweep:
        sweep = config.setdefault("sweep", {})
        if not isinstance(sweep, dict):
            raise TypeError("sweep must be a dictionary")
        sweep["enabled"] = False
    return config


def generate_sweep_configs(
    base_config: dict[str, Any],
    cli_sweep_items: list[str] | None = None,
    *,
    include_colour_ablation: bool = False,
) -> list[dict[str, Any]]:
    """Expand exactly one sweep layer into independent deep-copied configs."""
    items = expand_sweep_items(
        base_config,
        cli_sweep_items,
        include_colour_ablation=include_colour_ablation,
    )
    if (
        len(items) == 1
        and not items[0].assignments
        and items[0].condition is None
    ):
        return [base_config]
    return [apply_sweep_item(base_config, item) for item in items]
