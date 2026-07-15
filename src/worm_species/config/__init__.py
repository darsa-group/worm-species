"""Configuration loading, dotted overrides, and sweep expansion."""

from .loading import load_config
from .overrides import apply_overrides, parse_scalar, set_nested
from .sweeps import (
    generate_colour_retention_values,
    generate_sweep_configs,
    get_colour_sweep_parameters_from_config,
    get_sweep_parameters_from_cli,
    get_sweep_parameters_from_config,
    parse_sweep_item,
)

__all__ = [
    "apply_overrides",
    "generate_colour_retention_values",
    "generate_sweep_configs",
    "get_colour_sweep_parameters_from_config",
    "get_sweep_parameters_from_cli",
    "get_sweep_parameters_from_config",
    "load_config",
    "parse_scalar",
    "parse_sweep_item",
    "set_nested",
]
