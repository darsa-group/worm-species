#!/usr/bin/env python3
"""Prepare, render, submit, resume, and inspect the GBIF/Petri experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worm_species.gbif.domain_cache import build_domain_cache
from worm_species.gbif.domain_cache import domain_cache_directory
from worm_species.gbif.domain_cache import domain_cache_status
from worm_species.gbif.domain_data import load_domain_config
from worm_species.gbif.domain_data import prepare_domain_manifests
from worm_species.gbif.domain_orchestration import experiment_status
from worm_species.gbif.domain_orchestration import _training_specs
from worm_species.gbif.domain_orchestration import render_inference
from worm_species.gbif.domain_orchestration import render_primary_pipeline
from worm_species.gbif.domain_orchestration import render_training
from worm_species.gbif.domain_orchestration import submit_primary_pipeline
from worm_species.gbif.domain_orchestration import submit_inference
from worm_species.gbif.domain_orchestration import submit_training
from worm_species.gbif.domain_training import train_stage
from worm_species.gbif.domain_training import stage_is_complete


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gbif_training.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("check-config")
    commands.add_parser("prepare")
    commands.add_parser("build-cache")
    commands.add_parser("cache-path")
    cache_status = commands.add_parser("cache-status")
    cache_status.add_argument("--cache-root")
    cache_status.add_argument("--verify-files", action="store_true")
    commands.add_parser("render-primary-pipeline")
    commands.add_parser("submit-primary-pipeline")
    for name in ("render-training", "submit-training"):
        command = commands.add_parser(name)
        command.add_argument("--phase", choices=("primary", "dino"), required=True)
    for name in ("render-inference", "submit-inference"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
    stage = commands.add_parser("train-stage")
    stage.add_argument("--spec", required=True)
    stage_complete = commands.add_parser("stage-complete")
    stage_complete.add_argument("--spec", required=True)
    commands.add_parser("status")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_domain_config(args.config)
    if args.command == "check-config":
        wave1, wave2 = _training_specs(config, "primary")
        _print({
            "valid": True,
            "partition": config["slurm"]["partition"],
            "array_max_active": config["slurm"]["array_max_active"],
            "slurm_resources": {
                "training": config["slurm"]["training"],
                "inference": config["slurm"]["inference"],
                "preprocessing": config["slurm"]["preprocessing"],
                "merge": config["slurm"]["merge"],
                "analysis": config["slurm"]["analysis"],
            },
            "training_models": config["models"]["primary"],
            "dino_models": config["models"]["dino"],
            "inference_shards": config["inference"]["shards"],
            "preprocessed_cache_root": config["preprocessed_cache"]["root"],
            "node_cache_root": config["preprocessed_cache"]["node_root"],
            "experiment": {
                "strategies": ["gbif_only", "peti_to_gbif", "gbif_to_peti", "mixed"],
                "seeds": config["models"]["primary_seeds"],
                "hierarchy_loss_weights": config["training"]["hierarchy_loss"]["weights"],
                "wave1_jobs": len(wave1), "wave2_jobs": len(wave2),
                "total_stage_jobs": len(wave1) + len(wave2),
                "final_trajectories": sum(spec["final_model"] for spec in wave1 + wave2),
                "fixed_budget": bool(config["training"]["fixed_budget"]),
                "training_batch_size": int(config["training"]["batch_size"]),
                "mixed_batch_per_domain": int(config["training"]["batch_size"]) // 2,
                "checkpoint_selection": {
                    "gbif_only": ["gbif"],
                    "peti_to_gbif": {"stage1": ["petri"], "stage2": ["gbif"]},
                    "gbif_to_peti": {"stage1": ["gbif"], "stage2": ["petri"]},
                    "mixed": ["gbif", "petri"],
                },
            },
        })
    elif args.command == "prepare":
        _print(prepare_domain_manifests(config))
    elif args.command == "build-cache":
        _print(build_domain_cache(config))
    elif args.command == "cache-path":
        print(domain_cache_directory(config))
    elif args.command == "cache-status":
        status = domain_cache_status(
            config,
            cache_root=args.cache_root,
            verify_files=bool(args.verify_files),
        )
        _print(status)
        if not status["ready"]:
            raise SystemExit(2)
    elif args.command == "render-primary-pipeline":
        _print(render_primary_pipeline(config, args.config))
    elif args.command == "submit-primary-pipeline":
        _print(submit_primary_pipeline(config, args.config))
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
    elif args.command == "stage-complete":
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
        if stage_is_complete(spec):
            print(f"{spec['run_id']} is already complete; skipping before cache staging.")
        else:
            raise SystemExit(1)
    elif args.command == "status":
        _print(experiment_status(config))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
