#!/usr/bin/env python3
"""
Generate one run-spec file per sweep entry from config.yaml.

Expected config format:

sweep:
  enabled: true
  parameters:
    model.name:
      - efficientnet_b0
      - resnet18
    data.image_col:
      - rel_path_seg
      - rel_path_raw
    training.lr:
      - 0.0001
      - 0.00005

Each output file contains one override per line:

model.name=efficientnet_b0
data.image_col=rel_path_seg
training.lr=0.0001

The launcher passes these lines to:

python train_multitask_masked.py --config config.yaml --override ...
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import yaml


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.config.open("r") as f:
        cfg = yaml.safe_load(f)

    sweep_cfg = cfg.get("sweep", {})
    enabled = bool(sweep_cfg.get("enabled", False))
    params = sweep_cfg.get("parameters", {}) or {}

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not enabled or len(params) == 0:
        run_file = args.out_dir / "run_000.args"
        run_file.write_text("")
        plan_file = args.out_dir.parent / "sweep_plan.tsv"
        plan_file.write_text("run_index\trun_name\toverrides\n0\trun_000\t<no sweep enabled>\n")
        print("1")
        return

    if not isinstance(params, dict):
        raise TypeError("sweep.parameters must be a dictionary.")

    keys = list(params.keys())
    values = []

    for key in keys:
        v = params[key]
        if not isinstance(v, list):
            raise TypeError(f"sweep.parameters.{key} must be a list.")
        if len(v) == 0:
            raise ValueError(f"sweep.parameters.{key} is empty.")
        values.append(v)

    plan_lines = ["run_index\trun_name\toverrides"]

    n = 0
    for n, combo in enumerate(itertools.product(*values)):
        run_name = f"run_{n:03d}"
        override_lines = [
            f"{key}={format_value(value)}"
            for key, value in zip(keys, combo)
        ]

        (args.out_dir / f"{run_name}.args").write_text(
            "\n".join(override_lines) + "\n"
        )

        plan_lines.append(
            f"{n}\t{run_name}\t" + " ".join(override_lines)
        )

    (args.out_dir.parent / "sweep_plan.tsv").write_text(
        "\n".join(plan_lines) + "\n"
    )

    print(n + 1)


if __name__ == "__main__":
    main()
