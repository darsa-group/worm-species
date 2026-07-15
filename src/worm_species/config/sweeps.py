from __future__ import annotations

import copy
import itertools
from typing import Any

from .overrides import parse_scalar, set_nested


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


def generate_sweep_configs(
    base_config: dict[str, Any],
    cli_sweep_items: list[str] | None = None,
    *,
    include_colour_ablation: bool = False,
) -> list[dict[str, Any]]:
    """Expand exactly one sweep layer into independent deep-copied configs."""
    cli_sweep_items = cli_sweep_items or []
    if len(cli_sweep_items) > 0:
        parameters = get_sweep_parameters_from_cli(cli_sweep_items)
    elif include_colour_ablation:
        parameters = get_colour_sweep_parameters_from_config(base_config)
    else:
        parameters = get_sweep_parameters_from_config(base_config)

    if len(parameters) == 0:
        return [base_config]

    keys = list(parameters)
    values = [parameters[key] for key in keys]
    configs: list[dict[str, Any]] = []
    for combination in itertools.product(*values):
        config = copy.deepcopy(base_config)
        for key, value in zip(keys, combination):
            set_nested(config, key, value)
        configs.append(config)
    return configs
