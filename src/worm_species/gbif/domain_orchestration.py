"""Render and optionally submit direct-on-Genome GBIF experiment jobs."""

from __future__ import annotations

import json
import importlib.util
import os
import shlex
import subprocess
from pathlib import Path

from .domain_cache import READY_MARKER, domain_cache_status
from .domain_data import prepare_domain_manifests


STRATEGIES = {
    "gbif_only": (("gbif", None, True),),
    "peti_to_gbif": (("gbif", "peti_checkpoint", True),),
    "gbif_to_peti": (("gbif", None, False), ("petri", "stage1", True)),
    "mixed": (("mixed", None, True),),
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
    hierarchy = config["training"]["hierarchy_loss"]
    for model in models:
        for seed in seeds:
            for hierarchy_weight in hierarchy["weights"]:
                hierarchy_weight = float(hierarchy_weight)
                hierarchy_enabled = hierarchy_weight > 0.0
                hierarchy_slug = str(hierarchy_weight).replace(".", "p")
                run_suffix = f"-hloss{hierarchy_slug}"
                seed_root = root / model / f"seed-{seed}"
                seed_root = seed_root / f"hloss-{hierarchy_slug}"
                hierarchy_spec = {
                    "enabled": hierarchy_enabled,
                    "parent_task": hierarchy["parent_task"],
                    "child_task": hierarchy["child_task"],
                    "weight": hierarchy_weight,
                }
                for strategy, stages in STRATEGIES.items():
                    first_domain, first_initial, first_final = stages[0]
                    strategy_root = seed_root / strategy
                    first_id = f"{phase}-{model}-seed{seed}{run_suffix}-{strategy}-stage1"
                    first_output = strategy_root / "stage1"
                    first_checkpoint = None
                    freeze_age = False
                    if first_initial == "peti_checkpoint":
                        first_checkpoint = str(config["paths"]["peti_checkpoint"])
                        freeze_age = True
                    first = {
                        "phase": phase, "model": model, "seed": int(seed),
                        "strategy": strategy, "regime": strategy,
                        "stage": "stage1", "domain": first_domain,
                        "max_steps": mixed_steps if strategy == "mixed" else steps,
                        "run_id": first_id, "output_dir": str(first_output),
                        "initial_checkpoint": first_checkpoint,
                        "final_model": bool(first_final and len(stages) == 1),
                        "freeze_age_head": freeze_age,
                        "hierarchy_loss": hierarchy_spec,
                        "hierarchy_loss_weight": hierarchy_weight,
                    }
                    wave1.append(first)
                    if len(stages) > 1:
                        second_domain, second_initial, second_final = stages[1]
                        second_id = f"{phase}-{model}-seed{seed}{run_suffix}-{strategy}-stage2"
                        second_output = strategy_root / "stage2"
                        wave2.append({
                            "phase": phase, "model": model, "seed": int(seed),
                            "strategy": strategy, "regime": strategy,
                            "stage": "stage2", "domain": second_domain,
                            "max_steps": steps, "run_id": second_id,
                            "output_dir": str(second_output),
                            "initial_checkpoint": str(first_output / "last_model.pt"),
                            "final_model": bool(second_final),
                            "freeze_age_head": False,
                            "hierarchy_loss": hierarchy_spec,
                            "hierarchy_loss_weight": hierarchy_weight,
                        })
    return wave1, wave2


def _write_index(path: Path, specs: list[dict], spec_root: Path) -> None:
    lines = ["array_index\tspec_path"]
    for index, spec in enumerate(specs):
        spec_path = spec_root / f"{spec['run_id']}.json"
        _atomic_json(spec_path, spec)
        lines.append(f"{index}\t{spec_path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _preprocessing_script(config: dict, config_path: Path, phase: str) -> str:
    slurm = config["slurm"]
    resources = slurm["preprocessing"]
    paths = config["paths"]
    log_dir = Path(paths["output_root"]) / "logs" / "preprocessing"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=gbif-preprocess-{phase}
#SBATCH --account={slurm['account']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={resources['cpus_per_task']}
#SBATCH --mem={resources['memory']}
#SBATCH --time={resources['time_limit']}
#SBATCH --output={log_dir}/%x-%j.out
#SBATCH --error={log_dir}/%x-%j.err

set -euo pipefail
mkdir -p {shlex.quote(str(log_dir))}
source {shlex.quote(paths['conda_sh'])}
conda activate {shlex.quote(paths['conda_env'])}
cd {shlex.quote(paths['project_root'])}
export PYTHONPATH={shlex.quote(paths['project_root'])}/src
srun python scripts/gbif_domain_experiment.py --config {shlex.quote(str(config_path))} build-cache
"""


def _training_array_script(config: dict, config_path: Path, index_path: Path, count: int, phase: str, wave: int) -> str:
    slurm = config["slurm"]
    paths = config["paths"]
    log_dir = Path(paths["output_root"]) / "logs" / phase
    max_active = int(slurm["array_max_active"])
    cache = config["preprocessed_cache"]
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
spec=$(awk -F '\t' -v array_id="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == array_id {{print $2}}' {shlex.quote(str(index_path))})
[[ -n "$spec" && -f "$spec" ]] || {{ echo "Missing spec for array index $SLURM_ARRAY_TASK_ID" >&2; exit 2; }}
cd {shlex.quote(paths['project_root'])}
export PYTHONPATH={shlex.quote(paths['project_root'])}/src
PERSISTENT_CACHE=$(python scripts/gbif_domain_experiment.py --config {shlex.quote(str(config_path))} cache-path)
CACHE_READY="$PERSISTENT_CACHE/{READY_MARKER}"
[[ -d "$PERSISTENT_CACHE" && -f "$CACHE_READY" ]] || {{
    echo "Persistent preprocessed cache is not ready: $PERSISTENT_CACHE" >&2
    exit 3
}}
NODE_ROOT={shlex.quote(cache['node_root'])}
NODE_CACHE="$NODE_ROOT/cache"
NODE_READY="$NODE_CACHE/{READY_MARKER}"
NODE_SIGNATURE="$NODE_CACHE/SOURCE_READY.signature"
CACHE_LOCK="$NODE_ROOT/CACHE_COPY.lock"
mkdir -p "$NODE_ROOT"
source_signature="$PERSISTENT_CACHE|$(sha256sum "$CACHE_READY" | awk '{{print $1}}')"
(
    flock -x 200
    cached_signature=""
    [[ -f "$NODE_SIGNATURE" ]] && cached_signature=$(<"$NODE_SIGNATURE")
    if [[ ! -f "$NODE_READY" || "$cached_signature" != "$source_signature" ]]; then
        cache_bytes=$(du -sb "$PERSISTENT_CACHE" | awk '{{print $1}}')
        available_bytes=$(df -PB1 "$NODE_ROOT" | awk 'NR == 2 {{print $4}}')
        reserve_bytes=$(({int(cache['tmp_reserve_gb'])} * 1024 * 1024 * 1024))
        [[ "$cache_bytes" =~ ^[0-9]+$ && "$available_bytes" =~ ^[0-9]+$ ]] || {{
            echo "Could not determine cache size or node-local free space" >&2
            exit 4
        }}
        (( available_bytes >= cache_bytes + reserve_bytes )) || {{
            echo "Insufficient node-local space for preprocessed cache" >&2
            exit 5
        }}
        partial="$NODE_ROOT/.cache.partial.$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID"
        rm -rf -- "$partial"
        mkdir -p "$partial"
        rsync -a "$PERSISTENT_CACHE/" "$partial/"
        python scripts/gbif_domain_experiment.py --config {shlex.quote(str(config_path))} \
            cache-status --cache-root "$partial" --verify-files >/dev/null
        source_image_count=$(find "$PERSISTENT_CACHE/images" -type f -name '*.png' | wc -l)
        copied_image_count=$(find "$partial/images" -type f -name '*.png' | wc -l)
        [[ "$copied_image_count" == "$source_image_count" ]] || {{
            echo "Node-cache image count mismatch: copied=$copied_image_count source=$source_image_count" >&2
            exit 6
        }}
        printf '%s\n' "$source_signature" > "$partial/SOURCE_READY.signature"
        rm -rf -- "$NODE_CACHE"
        mv "$partial" "$NODE_CACHE"
        echo "Published $copied_image_count preprocessed GBIF/Petri images on $(hostname)"
    else
        echo "Reusing preprocessed GBIF/Petri cache on $(hostname)"
    fi
) 200>"$CACHE_LOCK"
export WORM_GBIF_NODE_CACHE="$NODE_CACHE"
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
    (Path(config["paths"]["output_root"]) / "logs" / "preprocessing").mkdir(
        parents=True, exist_ok=True
    )
    wave1_index = root / "wave1.tsv"
    wave2_index = root / "wave2.tsv"
    _write_index(wave1_index, wave1, specs_root)
    _write_index(wave2_index, wave2, specs_root)
    config_source = Path(config_path)
    if not config_source.is_absolute():
        config_source = Path(config["paths"]["project_root"]) / config_source
    wave1_script = root / "wave1.sbatch"
    wave2_script = root / "wave2.sbatch"
    preprocessing_script = root / "preprocessing.sbatch"
    preprocessing_script.write_text(
        _preprocessing_script(config, config_source, phase), encoding="utf-8"
    )
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
    preprocessing_script.chmod(0o755)
    manifest = {
        "schema_version": 1, "phase": phase,
        "array_max_active": int(config["slurm"]["array_max_active"]),
        "gpus_per_task": 1,
        "preprocessed_cache": {
            "script": str(preprocessing_script),
            "base_root": str(config["preprocessed_cache"]["root"]),
            "dependency": "before:wave1",
        },
        "wave1": {"count": len(wave1), "script": str(wave1_script)},
        "wave2": {"count": len(wave2), "script": str(wave2_script), "dependency": "afterok:wave1"},
        "final_model_count": sum(spec["final_model"] for spec in wave1 + wave2),
        "stage_job_count": len(wave1) + len(wave2),
    }
    _atomic_json(root / "manifest.json", manifest)
    _atomic_json(root / "plan.json", {
        "phase": phase,
        "total_stage_jobs": len(wave1) + len(wave2),
        "wave1_jobs": len(wave1),
        "wave2_jobs": len(wave2),
        "final_trajectory_count": sum(spec["final_model"] for spec in wave1 + wave2),
        "strategies": sorted(STRATEGIES),
        "seeds": sorted({int(spec["seed"]) for spec in wave1 + wave2}),
        "hierarchy_loss_weights": sorted({float(spec["hierarchy_loss_weight"]) for spec in wave1 + wave2}),
        "runs": [
            {
                "run_id": spec["run_id"], "strategy": spec["strategy"],
                "stage": spec["stage"], "seed": spec["seed"],
                "hierarchy_loss_weight": spec["hierarchy_loss_weight"],
                "initial_checkpoint": spec.get("initial_checkpoint"),
                "output_dir": spec["output_dir"], "final_model": spec["final_model"],
            }
            for spec in wave1 + wave2
        ],
    })
    manifest["plan"] = str(root / "plan.json")
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
        primary_wave1, primary_wave2 = _training_specs(config, "primary")
        required_primary_jobs = len(primary_wave1) + len(primary_wave2)
        if primary["complete"] != required_primary_jobs:
            raise RuntimeError(
                f"DINO is the final phase and requires all {required_primary_jobs} "
                "primary stage jobs "
                f"to be complete; current counts: {primary}"
            )
    manifest = render_training(config, config_path, phase, prepare=True)
    specs = _training_specs(config, phase)
    reference_paths = {
        Path(spec["initial_checkpoint"])
        for spec in specs[0] + specs[1]
        if spec.get("initial_checkpoint") == config["paths"]["peti_checkpoint"]
    }
    missing_reference = [path for path in reference_paths if not path.is_file()]
    if missing_reference:
        raise FileNotFoundError(
            "PETI_ONLY reference checkpoint is missing: "
            + ", ".join(str(path) for path in missing_reference)
        )
    cache_status = domain_cache_status(config, verify_files=True)
    preprocessing_id = None
    wave1_dependency = None
    if not cache_status["ready"]:
        preprocessing_id = _sbatch(manifest["preprocessed_cache"]["script"])
        wave1_dependency = f"afterok:{preprocessing_id}"
    wave1_id = _sbatch(manifest["wave1"]["script"], wave1_dependency)
    root = Path(config["paths"]["output_root"]) / "generated" / phase
    receipt = {
        "phase": phase,
        "preprocessing_job_id": preprocessing_id,
        "preprocessed_cache_reused": bool(cache_status["ready"]),
        "wave1_job_id": wave1_id,
        "wave2_job_id": None,
    }
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
    try:
        result["preprocessed_cache"] = domain_cache_status(
            config, verify_files=False
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        result["preprocessed_cache"] = {
            "ready": False,
            "reason": "prepared_summary_missing_or_stale",
            "detail": str(exc),
        }
    return result
