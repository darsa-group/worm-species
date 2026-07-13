from __future__ import annotations

import json
import random
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_scalar(value: str) -> Any:
    value = value.strip()

    if value.lower() in {"true", "false"}:
        return value.lower() == "true"

    if value.lower() in {"none", "null"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def set_nested(cfg: dict[str, Any], key: str, value: Any) -> None:
    """
    Example:
    set_nested(cfg, "model.name", "vit_b_16")
    """
    parts = key.split(".")
    d = cfg

    for p in parts[:-1]:
        if p not in d:
            d[p] = {}
        d = d[p]

    d[parts[-1]] = value


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    import copy

    cfg = copy.deepcopy(cfg)

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item}")

        key, value = item.split("=", 1)
        set_nested(cfg, key, parse_scalar(value))

    return cfg


def short_hash(obj: Any, length: int = 8) -> str:
    text = json.dumps(obj, sort_keys=True)
    return hashlib.md5(text.encode()).hexdigest()[:length]


def make_run_name(cfg: dict[str, Any]) -> str:
    parts = [
        cfg["model"]["name"],
        cfg["data"]["image_col"],
        cfg["data"]["target_col"],
        short_hash(cfg),
    ]
    return "__".join(str(p) for p in parts)