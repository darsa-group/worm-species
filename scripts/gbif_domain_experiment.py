#!/usr/bin/env python3
"""Prepare, render, submit, resume, and inspect the GBIF/Petri experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worm_species.gbif.domain_data import load_domain_config
from worm_species.gbif.domain_data import prepare_domain_manifests
from worm_species.gbif.domain_orchestration import experiment_status
from worm_species.gbif.domain_orchestration import render_inference
from worm_species.gbif.domain_orchestration import render_training
from worm_species.gbif.domain_orchestration import submit_inference
from worm_species.gbif.domain_orchestration import submit_training
from worm_species.gbif.domain_training import train_stage


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gbif_training.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check-config")
    commands.add_parser("prepare")
    for name in ("render-training", "submit-training"):
        command = commands.add_parser(name)
        command.add_argument("--phase", choices=("primary", "dino"), required=True)
    for name in ("render-inference", "submit-inference"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
    stage = commands.add_parser("train-stage")
    stage.add_argument("--spec", required=True)
    commands.add_parser("status")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_domain_config(args.config)
    if args.command == "check-config":
        _print({
            "valid": True,
            "partition": config["slurm"]["partition"],
            "array_max_active": config["slurm"]["array_max_active"],
            "training_models": config["models"]["primary"],
            "dino_models": config["models"]["dino"],
            "inference_shards": config["inference"]["shards"],
        })
    elif args.command == "prepare":
        _print(prepare_domain_manifests(config))
    elif args.command == "render-training":
        _print(render_training(config, args.config, args.phase, prepare=True))
    elif args.command == "submit-training":
        _print(submit_training(config, args.config, args.phase))
    elif args.command == "render-inference":
        _print(render_inference(config, args.config, args.checkpoint))
    elif args.command == "submit-inference":
        _print(submit_inference(config, args.config, args.checkpoint))
    elif args.command == "train-stage":
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        try:
            status = train_stage(config, spec)
        except Exception as exc:
            output = Path(spec["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            (output / "run_status.json").write_text(json.dumps({
                "status": "failed",
                "run_id": spec["run_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise
        _print(status)
        if status.get("status") != "complete":
            raise SystemExit(3)
    elif args.command == "status":
        _print(experiment_status(config))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
