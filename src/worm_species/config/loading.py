from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load the existing YAML mapping contract without applying defaults."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config is None:
        return {}
    if not isinstance(config, dict):
        raise TypeError(f"Config must contain a YAML mapping: {config_path}")
    return config
