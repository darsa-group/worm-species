#!/usr/bin/env python3
"""
Download all pretrained torchvision model weights required by config.yaml.

Default cache target:
    /usr/home/qgg/mehrot/.cache/torch/hub/checkpoints

Usage:
    cd ~/worm-species
    conda activate wormspecies
    python download_pretrained_from_config.py --config config.yaml
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
from pathlib import Path
from typing import Any

import yaml


def set_nested(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = cfg

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def generate_sweep_configs(base_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sweep_cfg = base_cfg.get("sweep", {}) or {}

    if not sweep_cfg.get("enabled", False):
        return [base_cfg]

    params = sweep_cfg.get("parameters", {}) or {}

    if len(params) == 0:
        return [base_cfg]

    if not isinstance(params, dict):
        raise TypeError("sweep.parameters must be a dictionary.")

    keys = list(params.keys())
    values = []

    for key in keys:
        vals = params[key]
        if not isinstance(vals, list):
            raise TypeError(f"sweep.parameters.{key} must be a list.")
        if len(vals) == 0:
            raise ValueError(f"sweep.parameters.{key} is empty.")
        values.append(vals)

    configs: list[dict[str, Any]] = []

    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base_cfg)

        for key, value in zip(keys, combo):
            set_nested(cfg, key, value)

        configs.append(cfg)

    return configs


def collect_pretrained_model_names(configs: list[dict[str, Any]]) -> list[str]:
    """
    Collect model names that require pretrained weights.

    Missing model.pretrained defaults to True because most training scripts
    use pretrained=True as the default unless explicitly disabled.
    """
    names = set()

    for cfg in configs:
        model_cfg = cfg.get("model", {}) or {}

        name = model_cfg.get("name")
        pretrained = model_cfg.get("pretrained", True)

        if name is None:
            continue

        if bool(pretrained):
            names.add(str(name))

    return sorted(names)


def download_torchvision_model(name: str) -> None:
    import torchvision.models as models

    registry = {
        # ResNet
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),

        # EfficientNet
        "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
        "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
        "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
        "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),

        # Vision Transformer
        "vit_b_16": (models.vit_b_16, models.ViT_B_16_Weights.DEFAULT),
        "vit_b_32": (models.vit_b_32, models.ViT_B_32_Weights.DEFAULT),
        "vit_l_16": (models.vit_l_16, models.ViT_L_16_Weights.DEFAULT),
        "vit_l_32": (models.vit_l_32, models.ViT_L_32_Weights.DEFAULT),

        # ConvNeXt
        "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
        "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT),

        # Swin
        "swin_t": (models.swin_t, models.Swin_T_Weights.DEFAULT),
        "swin_s": (models.swin_s, models.Swin_S_Weights.DEFAULT),
    }

    if name not in registry:
        print(f"[SKIP] No torchvision download rule for model.name={name!r}")
        print("       Add it to the registry in this script if needed.")
        return

    fn, weights = registry[name]

    print(f"\n[DOWNLOAD/CHECK] {name}")
    print(f"  weights: {weights}")

    _ = fn(weights=weights)

    print(f"[OK] {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=Path("/usr/home/qgg/mehrot/.cache/torch"),
        help="Torch cache root. Checkpoints go under <torch-home>/hub/checkpoints.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    args.torch_home.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.torch_home / "hub" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_HOME"] = str(args.torch_home)

    print("Config:", args.config.resolve())
    print("TORCH_HOME:", os.environ["TORCH_HOME"])
    print("Checkpoint directory:", checkpoints_dir)

    with args.config.open("r") as f:
        base_cfg = yaml.safe_load(f)

    configs = generate_sweep_configs(base_cfg)
    model_names = collect_pretrained_model_names(configs)

    print(f"\nNumber of expanded configs: {len(configs)}")

    if len(model_names) == 0:
        print("No pretrained models required by this config.")
        return

    print("Pretrained models required:")
    for name in model_names:
        print(f"  - {name}")

    for name in model_names:
        download_torchvision_model(name)

    print("\nDone.")
    print("Cached checkpoint files:")
    for path in sorted(checkpoints_dir.glob("*")):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name:45s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
