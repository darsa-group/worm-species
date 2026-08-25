"""Render and optionally submit direct-on-Genome GBIF experiment jobs."""

from __future__ import annotations

import json
import importlib.util
import os
import shlex
import subprocess
from pathlib import Path

from .domain_data import prepare_domain_manifests


REGIMES = {
    "curated_then_petri": ("gbif", "petri"),
    "petri_then_curated": ("petri", "gbif"),
}


def _require_training_runtime(phase: str) -> None:
    required = ["torch", "torchvision", "wandb"]
    if phase == "dino":
        required.append("timm")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"The active Genome conda environment lacks required {phase} packages: {missing}"
        )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _training_specs(config: dict, phase: str) -> tuple[list[dict], list[dict]]:
    if phase not in {"primary", "dino"}:
        raise ValueError("phase must be primary or dino")
    models = config["models"][phase]
    seeds = config["models"][f"{phase}_seeds"]
    root = Path(config["paths"]["output_root"]) / "runs" / phase
    steps = int(config["training"]["steps_per_domain"])
    mixed_steps = int(config["training"]["mixed_steps"])
    wave1: list[dict] = []
    wave2: list[dict] = []
    for model in models:
        for seed in seeds:
            for regime, domains in REGIMES.items():
                stage1_id = f"{phase}-{model}-seed{seed}-{regime}-stage1-{domains[0]}"
                stage1_output = root / model / f"seed-{seed}" / regime / "stage1"
                stage1 = {
                    "phase": phase, "model": model, "seed": int(seed),
                    "regime": regime, "stage": "stage1", "domain": domains[0],
                    "max_steps": steps, "run_id": stage1_id,
                    "output_dir": str(stage1_output), "initial_checkpoint": None,
                    "final_model": False,
                }
                stage2_id = f"{phase}-{model}-seed{seed}-{regime}-stage2-{domains[1]}"
                stage2_output = root / model / f"seed-{seed}" / regime / "stage2"
                stage2 = {
                    "phase": phase, "model": model, "seed": int(seed),
                    "regime": regime, "stage": "stage2", "domain": domains[1],
                    "max_steps": steps, "run_id": stage2_id,
                    "output_dir": str(stage2_output),
                    "initial_checkpoint": str(stage1_output / "last_model.pt"),
                    "final_model": True,
                }
                wave1.append(stage1)
                wave2.append(stage2)
            mixed_id = f"{phase}-{model}-seed{seed}-mixed"
            wave1.append({
                "phase": phase, "model": model, "seed": int(seed),
                "regime": "mixed", "stage": "mixed", "domain": "mixed",
                "max_steps": mixed_steps, "run_id": mixed_id,
                "output_dir": str(root / model / f"seed-{seed}" / "mixed"),
                "initial_checkpoint": None, "final_model": True,
            })
    return wave1, wave2


def _write_index(path: Path, specs: list[dict], spec_root: Path) -> None:
    lines = ["array_index\tspec_path"]
    for index, spec in enumerate(specs):
        spec_path = spec_root / f"{spec['run_id']}.json"
        _atomic_json(spec_path, spec)
        lines.append(f"{index}\t{spec_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _training_array_script(config: dict, config_path: Path, index_path: Path, count: int, phase: str, wave: int) -> str:
    slurm = config["slurm"]
    paths = config["paths"]
    log_dir = Path(paths["output_root"]) / "logs" / phase
    max_active = int(slurm["array_max_active"])
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=gbif-{phase}-w{wave}
#SBATCH --account={slurm['account']}
#SBATCH --partition={slurm['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --mem={slurm['memory']}
#SBATCH --time={slurm['time_limit']}
#SBATCH --gres=gpu:{slurm['gpus_per_task']}
#SBATCH --array=0-{count - 1}%{max_active}
#SBATCH --signal=B:USR1@300
#SBATCH --output={log_dir}/%x-%A_%a.out
#SBATCH --error={log_dir}/%x-%A_%a.err

set -euo pipefail
mkdir -p {shlex.quote(str(log_dir))}
source {shlex.quote(paths['conda_sh'])}
conda activate {shlex.quote(paths['conda_env'])}
spec=$(awk -F '\t' -v index="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == index {{print $2}}' {shlex.quote(str(index_path))})
[[ -n "$spec" && -f "$spec" ]] || {{ echo "Missing spec for array index $SLURM_ARRAY_TASK_ID" >&2; exit 2; }}
cd {shlex.quote(paths['project_root'])}
export PYTHONPATH={shlex.quote(paths['project_root'])}/src
srun python scripts/gbif_domain_experiment.py --config {shlex.quote(str(config_path))} train-stage --spec "$spec"
"""


def render_training(config: dict, config_path: str | Path, phase: str, *, prepare: bool) -> dict:
    if prepare:
        prepare_domain_manifests(config)
    wave1, wave2 = _training_specs(config, phase)
    root = Path(config["paths"]["output_root"]) / "generated" / phase
    specs_root = root / "specs"
    specs_root.mkdir(parents=True, exist_ok=True)
    (Path(config["paths"]["output_root"]) / "logs" / phase).mkdir(parents=True, exist_ok=True)
    wave1_index = root / "wave1.tsv"
    wave2_index = root / "wave2.tsv"
    _write_index(wave1_index, wave1, specs_root)
    _write_index(wave2_index, wave2, specs_root)
    config_source = Path(config_path)
    if not config_source.is_absolute():
        config_source = Path(config["paths"]["project_root"]) / config_source
    wave1_script = root / "wave1.sbatch"
    wave2_script = root / "wave2.sbatch"
    wave1_script.write_text(
        _training_array_script(config, config_source, wave1_index, len(wave1), phase, 1),
        encoding="utf-8",
    )
    wave2_script.write_text(
        _training_array_script(config, config_source, wave2_index, len(wave2), phase, 2),
        encoding="utf-8",
    )
    wave1_script.chmod(0o755)
    wave2_script.chmod(0o755)
    manifest = {
        "schema_version": 1, "phase": phase,
        "array_max_active": int(config["slurm"]["array_max_active"]),
        "gpus_per_task": 1,
        "wave1": {"count": len(wave1), "script": str(wave1_script)},
        "wave2": {"count": len(wave2), "script": str(wave2_script), "dependency": "afterok:wave1"},
        "final_model_count": sum(spec["final_model"] for spec in wave1 + wave2),
        "stage_job_count": len(wave1) + len(wave2),
    }
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def _sbatch(script: str | Path, dependency: str | None = None) -> str:
    argv = ["sbatch", "--parsable"]
    if dependency:
        argv.append(f"--dependency={dependency}")
    argv.append(str(script))
    completed = subprocess.run(argv, check=False, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"sbatch failed: {completed.stderr.strip()}")
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Invalid sbatch response: {completed.stdout!r}")
    return job_id


def submit_training(config: dict, config_path: str | Path, phase: str) -> dict:
    _require_training_runtime(phase)
    if phase == "primary":
        inference = (
            Path(config["paths"]["output_root"])
            / "inference" / "baseline" / "predictions.csv"
        )
        if not inference.is_file():
            raise RuntimeError(
                "Baseline curated-GBIF inference must finish before primary training; "
                f"missing {inference}"
            )
    else:
        primary = experiment_status(config)["primary"]["counts"]
        if primary["complete"] != 75:
            raise RuntimeError(
                "DINO is the final phase and requires all 75 primary stage jobs "
                f"to be complete; current counts: {primary}"
            )
    manifest = render_training(config, config_path, phase, prepare=True)
    wave1_id = _sbatch(manifest["wave1"]["script"])
    root = Path(config["paths"]["output_root"]) / "generated" / phase
    receipt = {"phase": phase, "wave1_job_id": wave1_id, "wave2_job_id": None}
    _atomic_json(root / "submission_receipt.json", receipt)
    wave2_id = _sbatch(manifest["wave2"]["script"], f"afterok:{wave1_id}")
    receipt["wave2_job_id"] = wave2_id
    _atomic_json(root / "submission_receipt.json", receipt)
    return receipt


def _inference_array_script(config: dict, config_path: Path, checkpoint: Path, root: Path) -> str:
    slurm = config["slurm"]
    paths = config["paths"]
    inference = config["inference"]
    shards = int(inference["shards"])
    shard_dir = root / "shards"
    log_dir = Path(paths["output_root"]) / "logs" / "inference"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=gbif-infer
#SBATCH --account={slurm['account']}
#SBATCH --partition={slurm['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --mem={slurm['memory']}
#SBATCH --time={slurm['time_limit']}
#SBATCH --gres=gpu:1
#SBATCH --array=0-{shards - 1}%{slurm['array_max_active']}
#SBATCH --output={log_dir}/%x-%A_%a.out
#SBATCH --error={log_dir}/%x-%A_%a.err

set -euo pipefail
mkdir -p {shlex.quote(str(shard_dir))} {shlex.quote(str(log_dir))}
source {shlex.quote(paths['conda_sh'])}
conda activate {shlex.quote(paths['conda_env'])}
cd {shlex.quote(paths['project_root'])}
export PYTHONPATH={shlex.quote(paths['project_root'])}/src
output=$(printf {shlex.quote(str(shard_dir / 'shard-%03d.csv'))} "$SLURM_ARRAY_TASK_ID")
srun python scripts/gbif_oligochaeta_pipeline.py infer-existing \
  --manifest {shlex.quote(paths['gbif_manifest'])} \
  --checkpoint {shlex.quote(str(checkpoint))} --output "$output" \
  --batch-size {inference['batch_size']} --workers {inference['num_workers']} \
  --prefetch-factor {inference['prefetch_factor']} --device cuda \
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count {shards}
"""


def _inference_merge_script(config: dict, root: Path) -> str:
    merge = config["slurm"]["merge"]
    paths = config["paths"]
    shards = int(config["inference"]["shards"])
    log_dir = Path(paths["output_root"]) / "logs" / "inference"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=gbif-infer-merge
#SBATCH --account={config['slurm']['account']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={merge['cpus_per_task']}
#SBATCH --mem={merge['memory']}
#SBATCH --time={merge['time_limit']}
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err

set -euo pipefail
source {shlex.quote(paths['conda_sh'])}
conda activate {shlex.quote(paths['conda_env'])}
cd {shlex.quote(paths['project_root'])}
export PYTHONPATH={shlex.quote(paths['project_root'])}/src
srun python scripts/gbif_oligochaeta_pipeline.py merge-inference \
  --manifest {shlex.quote(paths['gbif_manifest'])} \
  --shard-dir {shlex.quote(str(root / 'shards'))} \
  --output {shlex.quote(str(root / 'predictions.csv'))} --shard-count {shards}
"""


def render_inference(config: dict, config_path: str | Path, checkpoint: str | Path) -> dict:
    checkpoint_path = Path(checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    root = Path(config["paths"]["output_root"]) / "inference" / "baseline"
    generated = root / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    (Path(config["paths"]["output_root"]) / "logs" / "inference").mkdir(
        parents=True, exist_ok=True
    )
    array_script = generated / "inference_array.sbatch"
    merge_script = generated / "merge.sbatch"
    array_script.write_text(
        _inference_array_script(config, Path(config_path).resolve(), checkpoint_path, root),
        encoding="utf-8",
    )
    merge_script.write_text(_inference_merge_script(config, root), encoding="utf-8")
    array_script.chmod(0o755)
    merge_script.chmod(0o755)
    manifest = {
        "schema_version": 1, "checkpoint": str(checkpoint_path),
        "shards": int(config["inference"]["shards"]),
        "array_max_active": int(config["slurm"]["array_max_active"]),
        "gpus_per_array_task": 1,
        "array_script": str(array_script), "merge_script": str(merge_script),
    }
    _atomic_json(generated / "manifest.json", manifest)
    return manifest


def submit_inference(config: dict, config_path: str | Path, checkpoint: str | Path) -> dict:
    manifest = render_inference(config, config_path, checkpoint)
    array_id = _sbatch(manifest["array_script"])
    root = Path(config["paths"]["output_root"]) / "inference" / "baseline" / "generated"
    receipt = {"array_job_id": array_id, "merge_job_id": None}
    _atomic_json(root / "submission_receipt.json", receipt)
    merge_id = _sbatch(manifest["merge_script"], f"afterok:{array_id}")
    receipt["merge_job_id"] = merge_id
    _atomic_json(root / "submission_receipt.json", receipt)
    return receipt


def experiment_status(config: dict) -> dict:
    result = {}
    for phase in ("primary", "dino"):
        wave1, wave2 = _training_specs(config, phase)
        counts = {"complete": 0, "interrupted": 0, "missing": 0, "failed": 0}
        rows = []
        for spec in wave1 + wave2:
            status_path = Path(spec["output_dir"]) / "run_status.json"
            if not status_path.is_file():
                state = "missing"
            else:
                try:
                    state = str(json.loads(status_path.read_text(encoding="utf-8")).get("status", "failed"))
                except (OSError, json.JSONDecodeError):
                    state = "failed"
            counts[state if state in counts else "failed"] += 1
            rows.append({"run_id": spec["run_id"], "status": state})
        result[phase] = {"counts": counts, "runs": rows}
    inference = Path(config["paths"]["output_root"]) / "inference" / "baseline" / "predictions.csv"
    result["inference"] = {"complete": inference.is_file(), "output": str(inference)}
    return result
