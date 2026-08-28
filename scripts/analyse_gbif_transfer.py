#!/usr/bin/env python3
"""Post-hoc GBIF transfer, hierarchy, rarity, quality, and checkpoint analysis.

The workflow is inference/reporting only.  It never starts or resumes training.
`run --mode dry-run` validates completed checkpoints and renders Slurm scripts;
`run --mode submit` submits one GPU task per checkpoint and one dependent CPU
report job.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from worm_species.data.datasets import MultiTaskWormImageDataset
from worm_species.data.transforms import build_split_transform
from worm_species.gbif.domain_data import TASK_COLUMNS, file_sha256, load_domain_config
from worm_species.gbif.domain_orchestration import _training_specs
from worm_species.models.multitask import build_multitask_model


ANALYSIS_STRATEGIES = ("gbif_only", "peti_to_gbif")
AUDIT_STRATEGY = "gbif_to_peti"
TASKS = ("genus", "species")
PALETTE = {
    "gbif_only": "#0072B2",
    "peti_to_gbif": "#D55E00",
    "h0": "#009E73",
    "h05": "#CC79A7",
}


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _analysis_root(config: dict) -> Path:
    return Path(config["paths"]["output_root"]) / config["analysis"]["output_dir"]


def _prediction_path(config: dict, spec: dict) -> Path:
    return _analysis_root(config) / "predictions" / f"{spec['run_id']}.csv.gz"


def _checkpoint_specs(config: dict) -> tuple[list[dict], list[tuple[dict, dict]]]:
    wave1, wave2 = _training_specs(config, "primary")
    all_specs = wave1 + wave2
    selected = [
        spec for spec in all_specs
        if spec["strategy"] in ANALYSIS_STRATEGIES and bool(spec["final_model"])
    ]
    expected = (
        len(config["models"]["primary"])
        * len(config["models"]["primary_seeds"])
        * len(config["training"]["hierarchy_loss"]["weights"])
        * len(ANALYSIS_STRATEGIES)
    )
    if len(selected) != expected:
        raise RuntimeError(
            f"Expected {expected} final GBIF analysis checkpoints, found {len(selected)}"
        )
    audit_pairs = []
    for base in [spec for spec in wave1 if spec["strategy"] == "gbif_only"]:
        matches = [
            spec for spec in wave1
            if spec["strategy"] == AUDIT_STRATEGY
            and spec["model"] == base["model"]
            and int(spec["seed"]) == int(base["seed"])
            and float(spec["hierarchy_loss_weight"])
            == float(base["hierarchy_loss_weight"])
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one gbif_to_peti stage-1 audit match for {base['run_id']}"
            )
        audit_pairs.append((base, matches[0]))
    return sorted(selected, key=lambda row: row["run_id"]), audit_pairs


def _validate_completed_checkpoint(spec: dict) -> Path:
    output = Path(spec["output_dir"])
    checkpoint = output / "best_model.pt"
    status_path = output / "run_status.json"
    test_metrics = output / "test_metrics.json"
    missing = [str(path) for path in (checkpoint, status_path, test_metrics) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Completed-run artifacts are missing for {spec['run_id']}: {missing}"
        )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: {spec['run_id']} ({status.get('status')})")
    return checkpoint


def _prediction_complete(config: dict, spec: dict, checkpoint: Path) -> bool:
    output = _prediction_path(config, spec)
    summary_path = output.with_suffix("").with_suffix(".summary.json")
    if not output.is_file() or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stat = checkpoint.stat()
    return (
        summary.get("status") == "complete"
        and summary.get("run_id") == spec["run_id"]
        and Path(summary.get("checkpoint", "")).resolve() == checkpoint.resolve()
        and int(summary.get("checkpoint_bytes", -1)) == stat.st_size
        and int(summary.get("checkpoint_mtime_ns", -1)) == stat.st_mtime_ns
    )


def render_workflow(config: dict, config_path: Path) -> dict:
    selected, audit_pairs = _checkpoint_specs(config)
    for spec in selected:
        _validate_completed_checkpoint(spec)
    for left, right in audit_pairs:
        _validate_completed_checkpoint(left)
        _validate_completed_checkpoint(right)

    root = _analysis_root(config)
    generated = root / "generated"
    predictions = root / "predictions"
    logs = root / "logs"
    for directory in (generated, predictions, logs):
        directory.mkdir(parents=True, exist_ok=True)

    pending = []
    rows = []
    for spec in selected:
        checkpoint = Path(spec["output_dir"]) / "best_model.pt"
        complete = _prediction_complete(config, spec, checkpoint)
        if not complete:
            pending.append(spec)
        rows.append({
            "run_id": spec["run_id"],
            "model": spec["model"],
            "seed": int(spec["seed"]),
            "hierarchy_loss_weight": float(spec["hierarchy_loss_weight"]),
            "strategy": spec["strategy"],
            "stage": spec["stage"],
            "checkpoint": str(checkpoint),
            "prediction": str(_prediction_path(config, spec)),
            "already_complete": complete,
        })
    inventory_path = generated / "checkpoint_inventory.tsv"
    pd.DataFrame(rows).to_csv(inventory_path, sep="\t", index=False)

    pending_path = generated / "pending_inference.tsv"
    pending_rows = []
    for index, spec in enumerate(pending):
        pending_rows.append({
            "array_index": index,
            "run_id": spec["run_id"],
            "checkpoint": str(Path(spec["output_dir"]) / "best_model.pt"),
            "prediction": str(_prediction_path(config, spec)),
        })
    pd.DataFrame(
        pending_rows,
        columns=("array_index", "run_id", "checkpoint", "prediction"),
    ).to_csv(pending_path, sep="\t", index=False)

    array_script = generated / "inference_array.sbatch"
    report_script = generated / "report.sbatch"
    array_script.write_text(
        _render_inference_script(config, config_path, pending_path, len(pending)),
        encoding="utf-8",
    )
    report_script.write_text(
        _render_report_script(config, config_path), encoding="utf-8"
    )
    array_script.chmod(0o755)
    report_script.chmod(0o755)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_is_inference_only": True,
        "checkpoint_count": len(selected),
        "pending_inference_count": len(pending),
        "checkpoint_audit_pair_count": len(audit_pairs),
        "array_max_active": int(config["slurm"]["array_max_active"]),
        "inference_array_script": str(array_script),
        "report_script": str(report_script),
        "checkpoint_inventory": str(inventory_path),
        "pending_inference_index": str(pending_path),
        "report_resources": config["slurm"]["analysis"],
        "interpretation": (
            "Raw GBIF labels are retained. Exact species metrics are restricted to "
            "rows whose species maps into the trained classifier vocabulary."
        ),
    }
    _json_dump(generated / "workflow_manifest.json", manifest)
    return manifest


def _render_inference_script(
    config: dict, config_path: Path, index_path: Path, count: int
) -> str:
    paths = config["paths"]
    slurm = config["slurm"]
    resources = slurm["inference"]
    logs = _analysis_root(config) / "logs"
    array_line = (
        f"#SBATCH --array=0-{count - 1}%{slurm['array_max_active']}"
        if count else "# No pending inference tasks; this script is not submitted."
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=gbif-transfer-infer
#SBATCH --account={slurm['account']}
#SBATCH --partition={slurm['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={resources['cpus_per_task']}
#SBATCH --mem={resources['memory']}
#SBATCH --time={resources['time_limit']}
#SBATCH --gres=gpu:1
{array_line}
#SBATCH --output={logs}/%x-%A_%a.out
#SBATCH --error={logs}/%x-%A_%a.err

set -euo pipefail
source {shlex.quote(paths['conda_sh'])}
conda activate {shlex.quote(paths['conda_env'])}
cd {shlex.quote(paths['project_root'])}
export PYTHONPATH={shlex.quote(paths['project_root'])}/src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun python scripts/analyse_gbif_transfer.py \
  --config {shlex.quote(str(config_path))} infer-task \
  --index {shlex.quote(str(index_path))} \
  --array-index "$SLURM_ARRAY_TASK_ID"
"""


def _render_report_script(config: dict, config_path: Path) -> str:
    paths = config["paths"]
    slurm = config["slurm"]
    resources = slurm["analysis"]
    logs = _analysis_root(config) / "logs"
    partition = (
        f"#SBATCH --partition={resources['partition']}\n"
        if resources.get("partition") else ""
    )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=gbif-transfer-report
#SBATCH --account={slurm['account']}
{partition}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={resources['cpus_per_task']}
#SBATCH --mem={resources['memory']}
#SBATCH --time={resources['time_limit']}
#SBATCH --output={logs}/%x-%j.out
#SBATCH --error={logs}/%x-%j.err

set -euo pipefail
source {shlex.quote(paths['conda_sh'])}
conda activate {shlex.quote(paths['conda_env'])}
cd {shlex.quote(paths['project_root'])}
export PYTHONPATH={shlex.quote(paths['project_root'])}/src
export MPLCONFIGDIR="${{TMPDIR:-/tmp}}/worm-gbif-mpl-${{SLURM_JOB_ID:-local}}"
mkdir -p "$MPLCONFIGDIR"
srun python scripts/analyse_gbif_transfer.py \
  --config {shlex.quote(str(config_path))} report
"""


def _sbatch(script: Path, dependency: str | None = None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(str(script))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {result.stdout!r}")
    return job_id


def run_workflow(config: dict, config_path: Path, mode: str) -> dict:
    manifest = render_workflow(config, config_path)
    if mode == "dry-run":
        manifest["mode"] = "dry-run"
        return manifest
    generated = _analysis_root(config) / "generated"
    pending = int(manifest["pending_inference_count"])
    array_id = _sbatch(Path(manifest["inference_array_script"])) if pending else None
    dependency = f"afterok:{array_id}" if array_id else None
    report_id = _sbatch(Path(manifest["report_script"]), dependency)
    receipt = {
        "mode": "submit",
        "inference_array_job_id": array_id,
        "inference_task_count": pending,
        "report_job_id": report_id,
        "report_dependency": dependency,
    }
    _json_dump(generated / "submission_receipt.json", receipt)
    return {**manifest, **receipt}


def _load_checkpoint_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    label_maps = payload["label_to_index_by_task"]
    cfg = payload.get("cfg", {})
    model_name = (
        cfg.get("model", {}).get("name")
        or payload.get("experiment_spec", {}).get("model")
    )
    model = build_multitask_model(
        {"model": {"name": model_name, "pretrained": False}},
        {task: len(mapping) for task, mapping in label_maps.items()},
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, payload, label_maps


def infer_checkpoint(config: dict, spec: dict, checkpoint: Path, output: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Transfer-analysis inference requires a CUDA GPU")
    prepared = Path(config["paths"]["output_root"]) / "prepared"
    frame = pd.read_csv(prepared / "gbif_test.csv", dtype=str, keep_default_na=False)
    device = torch.device("cuda")
    model, payload, label_maps = _load_checkpoint_model(checkpoint, device)
    recorded = payload.get("experiment_spec", {})
    for field in ("run_id", "model", "strategy", "stage"):
        if str(recorded.get(field)) != str(spec[field]):
            raise ValueError(
                f"Checkpoint metadata mismatch for {field}: "
                f"{recorded.get(field)!r} != {spec[field]!r}"
            )
    if int(recorded.get("seed")) != int(spec["seed"]):
        raise ValueError("Checkpoint seed does not match the inference specification")
    if float(recorded.get("hierarchy_loss_weight", 0.0)) != float(
        spec["hierarchy_loss_weight"]
    ):
        raise ValueError("Checkpoint hierarchy weight does not match the specification")

    preprocessing = payload.get("cfg", {}).get("preprocessing", {})
    transform = build_split_transform(
        split="test", preprocessing=preprocessing,
        augmentation=payload.get("cfg", {}).get("augmentation", {}),
        condition={"transform": "original"}, apply_augmentation=False,
    )
    dataset = MultiTaskWormImageDataset(
        frame, root_dir="/", image_col="image_path", target_cols=TASK_COLUMNS,
        label_to_index_by_task=label_maps, transform=transform,
        crop_to_foreground=False,
    )
    inference = config["inference"]
    loader = DataLoader(
        dataset, batch_size=int(inference["batch_size"]), shuffle=False,
        num_workers=int(inference["num_workers"]), pin_memory=True,
        prefetch_factor=int(inference["prefetch_factor"]), persistent_workers=True,
    )
    inverse = {
        task: {int(index): str(label) for label, index in mapping.items()}
        for task, mapping in label_maps.items()
    }
    records: list[dict] = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            outputs = model(images)
            batch_count = len(images)
            batch_probabilities = {
                task: torch.softmax(outputs[task].float(), dim=1).cpu().numpy()
                for task in TASKS
            }
            for row_index in range(batch_count):
                source = frame.iloc[offset + row_index]
                record = {
                    "sample_id": source["sample_id"],
                    "group_id": source["group_id"],
                    "gbif_id": source.get("gbif_id", ""),
                    "image_path": source["image_path"],
                    "raw_true_genus": source.get("true_genus", ""),
                    "raw_true_species": source.get("true_species", ""),
                    "mapped_true_genus": source.get("genus", ""),
                    "mapped_true_species": source.get("species", ""),
                    "model": spec["model"],
                    "seed": int(spec["seed"]),
                    "hierarchy_loss_weight": float(spec["hierarchy_loss_weight"]),
                    "strategy": spec["strategy"],
                    "stage": spec["stage"],
                    "run_id": spec["run_id"],
                }
                for task in TASKS:
                    probabilities = batch_probabilities[task][row_index]
                    order = np.argsort(probabilities)[::-1]
                    mapped_true = str(source.get(task, ""))
                    record[f"{task}_evaluable"] = bool(
                        mapped_true and mapped_true in label_maps[task]
                    )
                    record[f"predicted_{task}"] = inverse[task][int(order[0])]
                    record[f"predicted_{task}_probability"] = float(probabilities[order[0]])
                    record[f"true_{task}_probability"] = (
                        float(probabilities[int(label_maps[task][mapped_true])])
                        if record[f"{task}_evaluable"] else np.nan
                    )
                    record[f"{task}_probabilities_json"] = json.dumps(
                        [float(value) for value in probabilities], separators=(",", ":")
                    )
                    for k in (1, 3, 5):
                        indices = order[: min(k, len(order))]
                        top = [
                            {
                                "label": inverse[task][int(index)],
                                "probability": float(probabilities[index]),
                            }
                            for index in indices
                        ]
                        record[f"{task}_top{k}_json"] = json.dumps(
                            top, separators=(",", ":")
                        )
                        record[f"{task}_top{k}_correct"] = (
                            bool(mapped_true in {item["label"] for item in top})
                            if record[f"{task}_evaluable"] else pd.NA
                        )
                records.append(record)
            offset += batch_count
    if offset != len(frame):
        raise AssertionError(f"Inference coverage mismatch: {offset} != {len(frame)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False, compression="gzip")
    stat = checkpoint.stat()
    summary = {
        "status": "complete",
        "run_id": spec["run_id"],
        "rows": len(records),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "gpu": torch.cuda.get_device_name(0),
        "species_vocabulary_size": len(label_maps["species"]),
        "genus_vocabulary_size": len(label_maps["genus"]),
    }
    summary_path = output.with_suffix("").with_suffix(".summary.json")
    _json_dump(summary_path, summary)
    return summary


def infer_task(config: dict, index_path: Path, array_index: int) -> dict:
    index = pd.read_csv(index_path, sep="\t", dtype=str, keep_default_na=False)
    selected = index.loc[index["array_index"].astype(int).eq(array_index)]
    if len(selected) != 1:
        raise ValueError(f"Expected one inference row for array index {array_index}")
    row = selected.iloc[0]
    specs, _ = _checkpoint_specs(config)
    matches = [spec for spec in specs if spec["run_id"] == row["run_id"]]
    if len(matches) != 1:
        raise ValueError(f"Unknown run_id in inference index: {row['run_id']}")
    return infer_checkpoint(
        config, matches[0], Path(row["checkpoint"]), Path(row["prediction"])
    )


def _quality_record(item: tuple[str, str]) -> dict:
    sample_id, image_path = item
    result = {"sample_id": sample_id, "image_path": image_path, "quality_status": "ok"}
    try:
        with Image.open(image_path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            working = rgb.copy()
            working.thumbnail((512, 512), Image.Resampling.BILINEAR)
        array = np.asarray(working, dtype=np.float32) / 255.0
        luminance = (
            0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
        )
        gray = np.clip(np.rint(luminance * 255), 0, 255).astype(np.uint8)
        histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
        probability = histogram[histogram > 0] / histogram.sum()
        entropy = float(-(probability * np.log2(probability)).sum())
        if min(gray.shape) >= 3:
            center = gray[1:-1, 1:-1].astype(np.float32)
            laplacian = (
                -4.0 * center
                + gray[:-2, 1:-1]
                + gray[2:, 1:-1]
                + gray[1:-1, :-2]
                + gray[1:-1, 2:]
            )
            sharpness = float(np.var(laplacian))
        else:
            sharpness = np.nan
        maximum = array.max(axis=2)
        minimum = array.min(axis=2)
        saturation = np.divide(
            maximum - minimum, maximum,
            out=np.zeros_like(maximum), where=maximum > 0,
        )
        result.update({
            "width_px": int(width),
            "height_px": int(height),
            "megapixels": float(width * height / 1_000_000),
            "aspect_ratio": float(width / max(height, 1)),
            "luminance_mean": float(luminance.mean()),
            "luminance_contrast_sd": float(luminance.std()),
            "luminance_dynamic_range_p99_p01": float(
                np.quantile(luminance, 0.99) - np.quantile(luminance, 0.01)
            ),
            "dark_clip_fraction": float((luminance <= 0.02).mean()),
            "bright_clip_fraction": float((luminance >= 0.98).mean()),
            "exposure_clip_fraction": float(
                ((luminance <= 0.02) | (luminance >= 0.98)).mean()
            ),
            "saturation_mean": float(saturation.mean()),
            "grayscale_entropy_bits": entropy,
            "laplacian_variance": sharpness,
        })
    except Exception as exc:  # keep a row-level audit rather than losing the job
        result.update(quality_status="error", quality_error=f"{type(exc).__name__}: {exc}")
    return result


def extract_image_quality(config: dict, test: pd.DataFrame) -> pd.DataFrame:
    root = _analysis_root(config)
    output = root / "tables" / "gbif_test_image_quality.csv"
    expected_ids = set(test["sample_id"].astype(str))
    if output.is_file():
        existing = pd.read_csv(output, dtype={"sample_id": str})
        if set(existing["sample_id"].astype(str)) == expected_ids:
            return existing
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", config["analysis"]["quality_workers"]))
    workers = min(int(config["analysis"]["quality_workers"]), allocated)
    items = list(test[["sample_id", "image_path"]].itertuples(index=False, name=None))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_quality_record, items, chunksize=max(1, len(items) // (workers * 8))))
    quality = pd.DataFrame(rows)
    good = quality["quality_status"].eq("ok")
    if not good.all():
        failures = quality.loc[~good, ["sample_id", "quality_error"]].head().to_dict("records")
        raise RuntimeError(f"Image-quality extraction failed for {(~good).sum()} rows: {failures}")
    badness = pd.DataFrame(index=quality.index)
    badness["blur"] = (-quality["laplacian_variance"]).rank(pct=True)
    badness["low_contrast"] = (-quality["luminance_contrast_sd"]).rank(pct=True)
    badness["low_detail"] = (-quality["grayscale_entropy_bits"]).rank(pct=True)
    badness["clipping"] = quality["exposure_clip_fraction"].rank(pct=True)
    quality["quality_challenge_index"] = badness.mean(axis=1)
    quality["quality_quartile"] = pd.qcut(
        quality["quality_challenge_index"].rank(method="first"),
        4, labels=["Q1_cleaner", "Q2", "Q3", "Q4_messier"],
    ).astype(str)
    output.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(output, index=False)
    return quality


def _rarity_label(count: int, bands: list[dict]) -> str:
    for band in bands:
        lower = int(band["minimum"])
        upper = band.get("maximum")
        if count >= lower and (upper is None or count <= int(upper)):
            return str(band["label"])
    raise ValueError(f"GBIF training count {count} is outside configured rarity bands")


def build_species_metadata(config: dict, prepared: Path) -> pd.DataFrame:
    gbif_train = pd.read_csv(prepared / "gbif_train.csv", dtype=str, keep_default_na=False)
    gbif_test = pd.read_csv(prepared / "gbif_test.csv", dtype=str, keep_default_na=False)
    petri_train = pd.read_csv(prepared / "petri_train.csv", dtype=str, keep_default_na=False)
    petri_seen_exact = set(
        petri_train.loc[petri_train["true_species"].ne(""), "true_species"]
    )
    petri_seen_mapped = set(
        petri_train.loc[petri_train["species"].ne(""), "species"]
    )
    keys = ["true_genus", "true_species"]

    def counts(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        valid = frame.loc[frame["true_species"].ne("")]
        return valid.groupby(keys, dropna=False).agg(
            **{
                f"{prefix}_images": ("sample_id", "size"),
                f"{prefix}_groups": ("group_id", "nunique"),
                "mapped_genus": ("genus", lambda values: next((v for v in values if v), "")),
                "mapped_species": ("species", lambda values: next((v for v in values if v), "")),
            }
        ).reset_index()

    train_counts = counts(gbif_train, "gbif_train")
    test_counts = counts(gbif_test, "gbif_test")
    metadata = test_counts.merge(
        train_counts, on=keys, how="outer", suffixes=("_test", "_train")
    )
    for column in ("gbif_train_images", "gbif_train_groups", "gbif_test_images", "gbif_test_groups"):
        metadata[column] = metadata[column].fillna(0).astype(int)
    mapped_genus_test = metadata["mapped_genus_test"].fillna("")
    mapped_genus_train = metadata["mapped_genus_train"].fillna("")
    mapped_species_test = metadata["mapped_species_test"].fillna("")
    mapped_species_train = metadata["mapped_species_train"].fillna("")
    metadata["mapped_genus"] = mapped_genus_test.where(
        mapped_genus_test.ne(""), mapped_genus_train
    )
    metadata["mapped_species"] = mapped_species_test.where(
        mapped_species_test.ne(""), mapped_species_train
    )
    metadata["petri_seen_exact_raw_species"] = metadata["true_species"].isin(
        petri_seen_exact
    )
    metadata["petri_seen_species"] = (
        metadata["mapped_species"].ne("")
        & metadata["mapped_species"].isin(petri_seen_mapped)
    )
    metadata["genus_evaluable"] = metadata["mapped_genus"].ne("")
    metadata["species_evaluable"] = metadata["mapped_species"].ne("")
    metadata["rarity_band"] = metadata["gbif_train_images"].map(
        lambda value: _rarity_label(int(value), config["analysis"]["rarity_bands"])
    )
    keep = [
        "true_genus", "true_species", "mapped_genus", "mapped_species",
        "gbif_train_images", "gbif_train_groups", "gbif_test_images",
        "gbif_test_groups", "rarity_band", "petri_seen_species",
        "petri_seen_exact_raw_species",
        "genus_evaluable", "species_evaluable",
    ]
    return metadata[keep].sort_values(
        ["gbif_train_images", "true_genus", "true_species"]
    ).reset_index(drop=True)


def _species_to_genus(prepared: Path) -> dict[str, str]:
    frames = [
        pd.read_csv(prepared / f"{domain}_{split}.csv", dtype=str, keep_default_na=False)
        for domain in ("gbif", "petri") for split in ("train", "validation", "test")
    ]
    pairs = pd.concat(frames, ignore_index=True).loc[
        lambda frame: frame["species"].ne("") & frame["genus"].ne(""),
        ["species", "genus"],
    ].drop_duplicates()
    conflicts = pairs.groupby("species")["genus"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("A model species maps to more than one genus")
    return pairs.drop_duplicates("species").set_index("species")["genus"].to_dict()


def _run_metrics(frame: pd.DataFrame, species_to_genus: dict[str, str]) -> pd.DataFrame:
    identity = ["run_id", "model", "seed", "hierarchy_loss_weight", "strategy"]
    rows = []
    for values, group in frame.groupby(identity, sort=True):
        base = dict(zip(identity, values))
        for task in TASKS:
            valid = group[f"{task}_evaluable"].astype(bool)
            subset = group.loc[valid]
            if subset.empty:
                continue
            true = subset[f"mapped_true_{task}"].astype(str)
            pred = subset[f"predicted_{task}"].astype(str)
            prevalence = true.value_counts()
            majority_accuracy = float(prevalence.iloc[0] / len(true))
            balanced_chance = float(1.0 / len(prevalence))
            for metric, value in (
                ("top1_accuracy", accuracy_score(true, pred)),
                ("balanced_accuracy", balanced_accuracy_score(true, pred)),
                ("macro_f1", f1_score(true, pred, average="macro", zero_division=0)),
                ("top3_accuracy", subset[f"{task}_top3_correct"].astype(bool).mean()),
                ("top5_accuracy", subset[f"{task}_top5_correct"].astype(bool).mean()),
                ("majority_class_accuracy", majority_accuracy),
                ("balanced_chance_1_over_k", balanced_chance),
            ):
                rows.append({**base, "task": task, "metric": metric, "value": float(value), "n": len(subset)})
        consistent = frame.loc[group.index, "predicted_species"].map(species_to_genus).eq(
            frame.loc[group.index, "predicted_genus"]
        )
        rows.append({**base, "task": "taxonomy", "metric": "genus_species_consistency", "value": float(consistent.mean()), "n": len(group)})
        species = group.loc[group["species_evaluable"].astype(bool)].copy()
        if not species.empty:
            species["species_correct"] = species["species_top1_correct"].astype(bool)
            species["predicted_species_genus"] = species["predicted_species"].map(species_to_genus)
            errors = species.loc[~species["species_correct"]]
            within = errors["predicted_species_genus"].eq(errors["mapped_true_genus"])
            severity = np.where(
                species["species_correct"], 0,
                np.where(species["predicted_species_genus"].eq(species["mapped_true_genus"]), 1, 2),
            )
            rows.extend([
                {**base, "task": "taxonomy", "metric": "within_genus_error_fraction", "value": float(within.mean()) if len(errors) else np.nan, "n": len(errors)},
                {**base, "task": "taxonomy", "metric": "between_genus_error_fraction", "value": float((~within).mean()) if len(errors) else np.nan, "n": len(errors)},
                {**base, "task": "taxonomy", "metric": "taxonomic_error_severity_0_1_2", "value": float(np.mean(severity)), "n": len(species)},
            ])
    return pd.DataFrame(rows)


def _paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.loc[~metrics["metric"].isin(
        ["majority_class_accuracy", "balanced_chance_1_over_k"]
    )]
    keys = ["model", "seed", "hierarchy_loss_weight", "task", "metric"]
    wide = metrics.pivot_table(index=keys, columns="strategy", values="value").reset_index()
    wide = wide.dropna(subset=list(ANALYSIS_STRATEGIES))
    wide["petri_minus_gbif"] = wide["peti_to_gbif"] - wide["gbif_only"]
    return wide


def _paired_test(values: Iterable[float]) -> dict:
    array = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if not len(array):
        return {"n": 0, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "wilcoxon_p": np.nan}
    mean = float(array.mean())
    if len(array) > 1:
        sem = stats.sem(array)
        low, high = stats.t.interval(0.95, len(array) - 1, loc=mean, scale=sem)
    else:
        low = high = np.nan
    if np.allclose(array, 0):
        pvalue = 1.0
    else:
        pvalue = float(stats.wilcoxon(array, alternative="two-sided").pvalue)
    return {
        "n": int(len(array)), "mean": mean,
        "ci_low": float(low), "ci_high": float(high), "wilcoxon_p": pvalue,
    }


def _effect_tests(effects: pd.DataFrame, group_columns: list[str], value: str) -> pd.DataFrame:
    rows = []
    for keys, group in effects.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({
            **dict(zip(group_columns, keys)),
            **_paired_test(group[value]),
        })
    return pd.DataFrame(rows)


def _per_image_effects(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "sample_id", "model", "seed", "hierarchy_loss_weight",
        "raw_true_genus", "raw_true_species", "mapped_true_genus",
        "mapped_true_species", "genus_evaluable", "species_evaluable",
    ]
    value_columns = []
    for task in TASKS:
        value_columns.extend([
            f"predicted_{task}", f"true_{task}_probability",
            f"{task}_top1_correct", f"{task}_top3_correct", f"{task}_top5_correct",
        ])
    left = predictions.loc[predictions["strategy"].eq("gbif_only"), keys + value_columns]
    right = predictions.loc[predictions["strategy"].eq("peti_to_gbif"), keys + value_columns]
    merged = left.merge(right, on=keys, validate="one_to_one", suffixes=("_gbif_only", "_peti_to_gbif"))
    for task in TASKS:
        for k in (1, 3, 5):
            left_col = f"{task}_top{k}_correct_gbif_only"
            right_col = f"{task}_top{k}_correct_peti_to_gbif"
            merged[f"{task}_top{k}_effect"] = (
                merged[right_col].astype("boolean").astype("Int64")
                - merged[left_col].astype("boolean").astype("Int64")
            )
        merged[f"{task}_true_probability_effect"] = (
            merged[f"true_{task}_probability_peti_to_gbif"]
            - merged[f"true_{task}_probability_gbif_only"]
        )
    return merged


def _stratified_effects(
    effects: pd.DataFrame, stratum: str, task: str
) -> pd.DataFrame:
    valid = effects[f"{task}_evaluable"].astype(bool)
    subset = effects.loc[valid]
    group_columns = ["model", "seed", "hierarchy_loss_weight", stratum]
    rows = []
    for keys, group in subset.groupby(group_columns, dropna=False, sort=True):
        rows.append({
            **dict(zip(group_columns, keys)), "task": task,
            "gbif_only_accuracy": float(group[f"{task}_top1_correct_gbif_only"].astype(bool).mean()),
            "peti_to_gbif_accuracy": float(group[f"{task}_top1_correct_peti_to_gbif"].astype(bool).mean()),
            "petri_minus_gbif": float(group[f"{task}_top1_effect"].astype(float).mean()),
            "n_images": len(group), "n_species": group["raw_true_species"].nunique(),
        })
    return pd.DataFrame(rows)


def _species_effects(effects: pd.DataFrame, task: str) -> pd.DataFrame:
    valid = effects[f"{task}_evaluable"].astype(bool) & effects["raw_true_species"].ne("")
    group_columns = [
        "model", "seed", "hierarchy_loss_weight", "raw_true_genus", "raw_true_species",
        "gbif_train_images", "gbif_train_groups", "rarity_band", "petri_seen_species",
    ]
    rows = []
    for keys, group in effects.loc[valid].groupby(group_columns, dropna=False, sort=True):
        rows.append({
            **dict(zip(group_columns, keys)), "task": task,
            "gbif_only_accuracy": float(group[f"{task}_top1_correct_gbif_only"].astype(bool).mean()),
            "peti_to_gbif_accuracy": float(group[f"{task}_top1_correct_peti_to_gbif"].astype(bool).mean()),
            "petri_minus_gbif": float(group[f"{task}_top1_effect"].astype(float).mean()),
            "n_test_images": len(group),
        })
    return pd.DataFrame(rows)


def _rarity_correlations(species_effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in species_effects.groupby(
        ["model", "seed", "hierarchy_loss_weight", "task"], sort=True
    ):
        if group["gbif_train_images"].nunique() < 2 or len(group) < 3:
            rho = pvalue = np.nan
        else:
            rho, pvalue = stats.spearmanr(
                np.log1p(group["gbif_train_images"].astype(float)),
                group["petri_minus_gbif"].astype(float),
            )
        rows.append({
            **dict(zip(["model", "seed", "hierarchy_loss_weight", "task"], keys)),
            "spearman_rho_log1p_train_images": rho,
            "spearman_p": pvalue,
            "n_species": len(group),
        })
    return pd.DataFrame(rows)


def _hierarchy_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.loc[~metrics["metric"].isin(
        ["majority_class_accuracy", "balanced_chance_1_over_k"]
    )]
    keys = ["model", "seed", "strategy", "task", "metric"]
    wide = metrics.pivot_table(index=keys, columns="hierarchy_loss_weight", values="value").reset_index()
    if 0.0 not in wide or 0.5 not in wide:
        raise ValueError("Hierarchy comparison requires both h=0 and h=0.5")
    wide = wide.dropna(subset=[0.0, 0.5]).rename(columns={0.0: "h0", 0.5: "h0_5"})
    wide["h0_5_minus_h0"] = wide["h0_5"] - wide["h0"]
    return wide


def _interaction_effects(paired: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "seed", "task", "metric"]
    wide = paired.pivot_table(index=keys, columns="hierarchy_loss_weight", values="petri_minus_gbif").reset_index()
    wide = wide.dropna(subset=[0.0, 0.5]).rename(columns={0.0: "petri_effect_h0", 0.5: "petri_effect_h0_5"})
    wide["petri_by_hierarchy_interaction"] = wide["petri_effect_h0_5"] - wide["petri_effect_h0"]
    return wide


def audit_checkpoints(config: dict) -> pd.DataFrame:
    _, pairs = _checkpoint_specs(config)
    rows = []
    training_fields = (
        "phase", "model", "seed", "stage", "domain", "max_steps",
        "initial_checkpoint", "freeze_age_head", "selection_domains",
        "hierarchy_loss", "hierarchy_loss_weight",
    )
    for base_spec, transfer_spec in pairs:
        left_path = _validate_completed_checkpoint(base_spec)
        right_path = _validate_completed_checkpoint(transfer_spec)
        left_hash = file_sha256(left_path)
        right_hash = file_sha256(right_path)
        left = torch.load(left_path, map_location="cpu", weights_only=False)
        right = torch.load(right_path, map_location="cpu", weights_only=False)
        keys_equal = set(left["model_state"]) == set(right["model_state"])
        exact = keys_equal
        max_delta = 0.0
        differing = 0
        if keys_equal:
            for key in left["model_state"]:
                a = left["model_state"][key]
                b = right["model_state"][key]
                same = torch.equal(a, b)
                exact = exact and same
                if not same:
                    differing += 1
                    max_delta = max(max_delta, float((a.float() - b.float()).abs().max()))
        comparable_spec = all(base_spec.get(field) == transfer_spec.get(field) for field in training_fields)
        if left_path.resolve() == right_path.resolve():
            reuse_status = "shared_path"
        elif exact:
            reuse_status = "duplicate_paths_exact_model_state"
        else:
            reuse_status = "duplicate_paths_diverged_model_state"
        left_metrics = json.loads((Path(base_spec["output_dir"]) / "test_metrics.json").read_text())
        right_metrics = json.loads((Path(transfer_spec["output_dir"]) / "test_metrics.json").read_text())
        row = {
            "model": base_spec["model"], "seed": int(base_spec["seed"]),
            "hierarchy_loss_weight": float(base_spec["hierarchy_loss_weight"]),
            "gbif_only_checkpoint": str(left_path),
            "gbif_to_peti_stage1_checkpoint": str(right_path),
            "paths_are_shared": left_path.resolve() == right_path.resolve(),
            "training_specs_equivalent": comparable_spec,
            "checkpoint_sha256_equal": left_hash == right_hash,
            "model_state_keys_equal": keys_equal,
            "model_state_exact_equal": exact,
            "differing_tensors": differing,
            "maximum_absolute_tensor_delta": max_delta,
            "gbif_only_best_step": left.get("best_step"),
            "gbif_to_peti_best_step": right.get("best_step"),
            "gbif_only_best_validation_score": left.get("best_val_score"),
            "gbif_to_peti_best_validation_score": right.get("best_val_score"),
            "reuse_status": reuse_status,
        }
        for task in TASKS:
            key = f"{task}_balanced_accuracy"
            row[f"gbif_test_{task}_balanced_accuracy_delta"] = (
                float(right_metrics["gbif"][key]) - float(left_metrics["gbif"][key])
            )
        rows.append(row)
        del left, right
    return pd.DataFrame(rows)


def _save_figure(fig, figures: Path, stem: str, source: pd.DataFrame) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    source_root = figures / "figure_sources"
    source_root.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_root / f"{stem}.csv", index=False)
    for extension in ("svg", "pdf"):
        fig.savefig(figures / f"{stem}.{extension}", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plots(
    paired: pd.DataFrame,
    rarity: pd.DataFrame,
    seen: pd.DataFrame,
    hierarchy: pd.DataFrame,
    interaction: pd.DataFrame,
    quality: pd.DataFrame,
    figures: Path,
) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.dpi": 160, "savefig.transparent": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    selected = paired.loc[
        paired["metric"].isin(["top1_accuracy", "top3_accuracy", "top5_accuracy"])
        & paired["task"].isin(TASKS)
    ].copy()
    summary = selected.groupby(["task", "metric", "hierarchy_loss_weight"])["petri_minus_gbif"].agg(["mean", "sem", "count"]).reset_index()
    summary["ci"] = summary["sem"] * summary["count"].map(lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    labels = summary["task"] + " · " + summary["metric"].str.replace("_", " ") + " · h=" + summary["hierarchy_loss_weight"].astype(str)
    y = np.arange(len(summary))
    colors = [PALETTE["h0"] if value == 0 else PALETTE["h05"] for value in summary["hierarchy_loss_weight"]]
    ax.errorbar(
        summary["mean"], y, xerr=summary["ci"], fmt="none",
        ecolor="#555555", elinewidth=1, capsize=3,
    )
    ax.scatter(summary["mean"], y, c=colors, s=30, zorder=3)
    ax.axvline(0, color="#555555", lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Petri→GBIF minus GBIF-only")
    ax.set_title("Paired GBIF transfer effects across model–seed runs")
    _save_figure(fig, figures, "01_overall_petri_effects", summary.assign(label=labels))

    if not rarity.empty:
        order = ["0", "1-10", "11-25", "26-100", ">100"]
        summary = rarity.groupby(["task", "hierarchy_loss_weight", "rarity_band"])["petri_minus_gbif"].agg(["mean", "sem", "count"]).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
        for ax, task in zip(axes, TASKS):
            for h, color in ((0.0, PALETTE["h0"]), (0.5, PALETTE["h05"])):
                subset = summary.loc[(summary["task"] == task) & (summary["hierarchy_loss_weight"] == h)].set_index("rarity_band").reindex(order)
                ax.plot(order, subset["mean"], marker="o", color=color, label=f"h={h:g}")
            ax.axhline(0, color="#666666", lw=1)
            ax.set_title(task.capitalize())
            ax.set_xlabel("GBIF training images per raw species")
            ax.tick_params(axis="x", rotation=35)
        axes[0].set_ylabel("Petri transfer accuracy effect")
        axes[1].legend(frameon=False)
        _save_figure(fig, figures, "02_species_rarity_effects", summary)

    if not seen.empty:
        summary = seen.groupby(["hierarchy_loss_weight", "petri_seen_species"])["petri_minus_gbif"].agg(["mean", "sem", "count"]).reset_index()
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        positions = np.arange(len(summary))
        labels = [f"{'seen' if seen_value else 'unseen'} · h={h:g}" for h, seen_value in zip(summary["hierarchy_loss_weight"], summary["petri_seen_species"])]
        ax.bar(positions, summary["mean"], color=[PALETTE["h0"] if h == 0 else PALETTE["h05"] for h in summary["hierarchy_loss_weight"]])
        ax.axhline(0, color="#555555", lw=1)
        ax.set_xticks(positions, labels, rotation=25, ha="right")
        ax.set_ylabel("Petri transfer genus-accuracy effect")
        ax.set_title("Petri-seen versus Petri-unseen GBIF species")
        _save_figure(fig, figures, "03_seen_unseen_genus_effects", summary.assign(label=labels))

    taxonomy_metrics = ["genus_species_consistency", "within_genus_error_fraction", "taxonomic_error_severity_0_1_2"]
    subset = hierarchy.loc[hierarchy["metric"].isin(taxonomy_metrics)].copy()
    if not subset.empty:
        summary = subset.groupby(["strategy", "metric"])["h0_5_minus_h0"].agg(["mean", "sem", "count"]).reset_index()
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        y = np.arange(len(summary))
        colors = summary["strategy"].map(PALETTE)
        ax.scatter(summary["mean"], y, c=colors, s=34)
        ax.axvline(0, color="#555555", lw=1)
        ax.set_yticks(y, summary["strategy"] + " · " + summary["metric"].str.replace("_", " "))
        ax.set_xlabel("Hierarchy h=0.5 minus h=0")
        ax.set_title("Hierarchy consistency and taxonomic error structure")
        _save_figure(fig, figures, "04_hierarchy_taxonomic_effects", summary)

    if not interaction.empty:
        subset = interaction.loc[interaction["metric"].isin(["balanced_accuracy", "genus_species_consistency", "taxonomic_error_severity_0_1_2"])]
        summary = subset.groupby(["task", "metric"])["petri_by_hierarchy_interaction"].agg(["mean", "sem", "count"]).reset_index()
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        y = np.arange(len(summary))
        ax.scatter(summary["mean"], y, color="#E69F00", s=34)
        ax.axvline(0, color="#555555", lw=1)
        ax.set_yticks(y, summary["task"] + " · " + summary["metric"].str.replace("_", " "))
        ax.set_xlabel("Petri × hierarchy difference-in-differences")
        ax.set_title("Interaction between Petri pretraining and hierarchy loss")
        _save_figure(fig, figures, "05_petri_hierarchy_interaction", summary)

    if not quality.empty:
        order = ["Q1_cleaner", "Q2", "Q3", "Q4_messier"]
        summary = quality.groupby(["task", "hierarchy_loss_weight", "quality_quartile"])["petri_minus_gbif"].agg(["mean", "sem", "count"]).reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
        for ax, task in zip(axes, TASKS):
            for h, color in ((0.0, PALETTE["h0"]), (0.5, PALETTE["h05"])):
                item = summary.loc[(summary["task"] == task) & (summary["hierarchy_loss_weight"] == h)].set_index("quality_quartile").reindex(order)
                ax.plot(order, item["mean"], marker="o", color=color, label=f"h={h:g}")
            ax.axhline(0, color="#666666", lw=1)
            ax.set_title(task.capitalize())
            ax.tick_params(axis="x", rotation=25)
        axes[0].set_ylabel("Petri transfer accuracy effect")
        axes[1].legend(frameon=False)
        _save_figure(fig, figures, "06_image_quality_effects", summary)


def build_report(config: dict) -> dict:
    root = _analysis_root(config)
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    specs, _ = _checkpoint_specs(config)
    frames = []
    for spec in specs:
        checkpoint = _validate_completed_checkpoint(spec)
        if not _prediction_complete(config, spec, checkpoint):
            raise RuntimeError(f"Inference output is incomplete for {spec['run_id']}")
        frames.append(pd.read_csv(_prediction_path(config, spec), low_memory=False))
    predictions = pd.concat(frames, ignore_index=True)
    expected_rows = len(frames[0]) * len(specs)
    if len(predictions) != expected_rows:
        raise AssertionError("Combined per-image prediction coverage is incomplete")

    prepared = Path(config["paths"]["output_root"]) / "prepared"
    test = pd.read_csv(prepared / "gbif_test.csv", dtype=str, keep_default_na=False)
    species = build_species_metadata(config, prepared)
    species.to_csv(tables / "gbif_species_metadata.csv", index=False)
    quality = extract_image_quality(config, test)
    species_annotations = species[[
        "true_genus", "true_species", "gbif_train_images", "gbif_train_groups",
        "gbif_test_images", "gbif_test_groups", "rarity_band", "petri_seen_species",
        "petri_seen_exact_raw_species",
    ]]
    predictions = predictions.merge(
        species_annotations,
        left_on=["raw_true_genus", "raw_true_species"],
        right_on=["true_genus", "true_species"], how="left", validate="many_to_one",
    ).drop(columns=["true_genus", "true_species"])
    predictions = predictions.merge(
        quality.drop(columns=["image_path"]), on="sample_id", how="left", validate="many_to_one"
    )
    predictions.to_csv(tables / "per_image_predictions.csv.gz", index=False, compression="gzip")

    species_to_genus = _species_to_genus(prepared)
    metrics = _run_metrics(predictions, species_to_genus)
    metrics.to_csv(tables / "per_run_metrics.csv", index=False)
    paired = _paired_effects(metrics)
    paired.to_csv(tables / "per_seed_effects.csv", index=False)
    overall_tests = _effect_tests(
        paired, ["hierarchy_loss_weight", "task", "metric"], "petri_minus_gbif"
    )
    overall_tests["analysis"] = "overall_petri_effect"

    image_effects = _per_image_effects(predictions)
    image_effects = image_effects.merge(
        species_annotations,
        left_on=["raw_true_genus", "raw_true_species"],
        right_on=["true_genus", "true_species"], how="left", validate="many_to_one",
    ).drop(columns=["true_genus", "true_species"])
    image_effects = image_effects.merge(
        quality.drop(columns=["image_path"]), on="sample_id", how="left", validate="many_to_one"
    )
    image_effects.to_csv(tables / "per_image_petri_effects.csv.gz", index=False, compression="gzip")

    species_effects = pd.concat(
        [_species_effects(image_effects, task) for task in TASKS], ignore_index=True
    )
    species_effects.to_csv(tables / "per_species_petri_effects.csv", index=False)
    rarity = pd.concat(
        [_stratified_effects(image_effects, "rarity_band", task) for task in TASKS],
        ignore_index=True,
    )
    rarity.to_csv(tables / "rarity_band_effects_per_seed.csv", index=False)
    rarity_tests = _effect_tests(
        rarity, ["hierarchy_loss_weight", "task", "rarity_band"], "petri_minus_gbif"
    )
    rarity_tests["analysis"] = "rarity_band_petri_effect"
    rarity_correlations = _rarity_correlations(species_effects)
    rarity_correlations.to_csv(tables / "rarity_continuous_correlations.csv", index=False)
    rarity_correlation_tests = _effect_tests(
        rarity_correlations,
        ["hierarchy_loss_weight", "task"],
        "spearman_rho_log1p_train_images",
    )
    rarity_correlation_tests["analysis"] = "rarity_continuous_spearman_across_runs"

    seen = _stratified_effects(image_effects, "petri_seen_species", "genus")
    seen.to_csv(tables / "petri_seen_unseen_genus_effects_per_seed.csv", index=False)
    seen_tests = _effect_tests(
        seen, ["hierarchy_loss_weight", "task", "petri_seen_species"], "petri_minus_gbif"
    )
    seen_tests["analysis"] = "petri_seen_unseen_effect"
    seen_contrast = seen.pivot_table(
        index=["model", "seed", "hierarchy_loss_weight", "task"],
        columns="petri_seen_species", values="petri_minus_gbif",
    ).reset_index()
    if False in seen_contrast and True in seen_contrast:
        seen_contrast = seen_contrast.dropna(subset=[False, True]).rename(
            columns={False: "petri_unseen_effect", True: "petri_seen_effect"}
        )
        seen_contrast["unseen_minus_seen_petri_effect"] = (
            seen_contrast["petri_unseen_effect"] - seen_contrast["petri_seen_effect"]
        )
    else:
        seen_contrast = pd.DataFrame(columns=[
            "model", "seed", "hierarchy_loss_weight", "task",
            "petri_unseen_effect", "petri_seen_effect", "unseen_minus_seen_petri_effect",
        ])
    seen_contrast.to_csv(tables / "petri_seen_unseen_contrast_per_seed.csv", index=False)
    seen_contrast_tests = _effect_tests(
        seen_contrast,
        ["hierarchy_loss_weight", "task"],
        "unseen_minus_seen_petri_effect",
    ) if not seen_contrast.empty else pd.DataFrame()
    if not seen_contrast_tests.empty:
        seen_contrast_tests["analysis"] = "petri_unseen_minus_seen_interaction"

    hierarchy = _hierarchy_effects(metrics)
    hierarchy.to_csv(tables / "hierarchy_effects_per_seed.csv", index=False)
    hierarchy_tests = _effect_tests(
        hierarchy, ["strategy", "task", "metric"], "h0_5_minus_h0"
    )
    hierarchy_tests["analysis"] = "hierarchy_h0_5_minus_h0"
    interaction = _interaction_effects(paired)
    interaction.to_csv(tables / "petri_hierarchy_interaction_per_seed.csv", index=False)
    interaction_tests = _effect_tests(
        interaction, ["task", "metric"], "petri_by_hierarchy_interaction"
    )
    interaction_tests["analysis"] = "petri_by_hierarchy_interaction"

    quality_effects = pd.concat(
        [_stratified_effects(image_effects, "quality_quartile", task) for task in TASKS],
        ignore_index=True,
    )
    quality_effects.to_csv(tables / "quality_quartile_effects_per_seed.csv", index=False)
    quality_tests = _effect_tests(
        quality_effects,
        ["hierarchy_loss_weight", "task", "quality_quartile"],
        "petri_minus_gbif",
    )
    quality_tests["analysis"] = "quality_quartile_petri_effect"
    quality_contrast = quality_effects.pivot_table(
        index=["model", "seed", "hierarchy_loss_weight", "task"],
        columns="quality_quartile", values="petri_minus_gbif",
    ).reset_index()
    required_quality = ["Q1_cleaner", "Q4_messier"]
    if all(column in quality_contrast for column in required_quality):
        quality_contrast = quality_contrast.dropna(subset=required_quality)
        quality_contrast["messier_minus_cleaner_petri_effect"] = (
            quality_contrast["Q4_messier"] - quality_contrast["Q1_cleaner"]
        )
    else:
        quality_contrast = pd.DataFrame(columns=[
            "model", "seed", "hierarchy_loss_weight", "task",
            "Q1_cleaner", "Q4_messier", "messier_minus_cleaner_petri_effect",
        ])
    quality_contrast.to_csv(tables / "quality_messier_vs_cleaner_contrast_per_seed.csv", index=False)
    quality_contrast_tests = _effect_tests(
        quality_contrast,
        ["hierarchy_loss_weight", "task"],
        "messier_minus_cleaner_petri_effect",
    ) if not quality_contrast.empty else pd.DataFrame()
    if not quality_contrast_tests.empty:
        quality_contrast_tests["analysis"] = "quality_messier_minus_cleaner_interaction"
    quality_correlations = []
    for task in TASKS:
        valid = image_effects[f"{task}_evaluable"].astype(bool)
        for keys, group in image_effects.loc[valid].groupby(
            ["model", "seed", "hierarchy_loss_weight"], sort=True
        ):
            rho, pvalue = stats.spearmanr(
                group["quality_challenge_index"], group[f"{task}_top1_effect"].astype(float)
            )
            quality_correlations.append({
                **dict(zip(["model", "seed", "hierarchy_loss_weight"], keys)),
                "task": task, "spearman_rho": rho, "spearman_p": pvalue,
                "n_images": len(group),
            })
    quality_correlations = pd.DataFrame(quality_correlations)
    quality_correlations.to_csv(
        tables / "quality_continuous_correlations.csv", index=False
    )
    quality_correlation_tests = _effect_tests(
        quality_correlations,
        ["hierarchy_loss_weight", "task"],
        "spearman_rho",
    )
    quality_correlation_tests["analysis"] = "quality_continuous_spearman_across_runs"

    rescue_rows = []
    for keys, group in image_effects.groupby(
        ["sample_id", "raw_true_genus", "raw_true_species", "rarity_band", "petri_seen_species", "quality_quartile"],
        dropna=False, sort=True,
    ):
        for task in TASKS:
            valid = group[f"{task}_evaluable"].astype(bool)
            subset = group.loc[valid]
            if subset.empty:
                continue
            effect = subset[f"{task}_top1_effect"].astype(float)
            rescue_rows.append({
                **dict(zip(["sample_id", "raw_true_genus", "raw_true_species", "rarity_band", "petri_seen_species", "quality_quartile"], keys)),
                "task": task, "paired_comparisons": len(subset),
                "rescued_count": int(effect.eq(1).sum()),
                "harmed_count": int(effect.eq(-1).sum()),
                "net_rescue_count": int(effect.sum()),
                "mean_true_probability_effect": float(subset[f"{task}_true_probability_effect"].mean()),
            })
    rescued = pd.DataFrame(rescue_rows).sort_values(
        ["net_rescue_count", "rescued_count", "harmed_count"],
        ascending=[False, False, True],
    )
    rescued.to_csv(tables / "rescued_harmed_images.csv", index=False)

    audit = audit_checkpoints(config)
    audit.to_csv(tables / "gbif_only_vs_gbif_to_peti_stage1_checkpoint_audit.csv", index=False)
    test_frames = [
        overall_tests, rarity_tests, rarity_correlation_tests, seen_tests,
        hierarchy_tests, interaction_tests, quality_tests, quality_correlation_tests,
    ]
    if not seen_contrast_tests.empty:
        test_frames.append(seen_contrast_tests)
    if not quality_contrast_tests.empty:
        test_frames.append(quality_contrast_tests)
    statistical_tests = pd.concat(
        test_frames,
        ignore_index=True, sort=False,
    )
    statistical_tests.to_csv(tables / "statistical_tests.csv", index=False)
    label_maps = json.loads((prepared / "label_maps.json").read_text(encoding="utf-8"))
    _json_dump(tables / "probability_label_order.json", {
        task: [
            label for label, _index in sorted(mapping.items(), key=lambda item: int(item[1]))
        ]
        for task, mapping in label_maps.items() if task in TASKS
    })
    _plots(paired, rarity, seen, hierarchy, interaction, quality_effects, figures)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_count": len(specs),
        "prediction_rows": len(predictions),
        "unique_gbif_test_images": int(predictions["sample_id"].nunique()),
        "raw_gbif_test_species": int(predictions["raw_true_species"].replace("", np.nan).nunique()),
        "species_evaluable_prediction_rows": int(predictions["species_evaluable"].astype(bool).sum()),
        "quality_workers": int(config["analysis"]["quality_workers"]),
        "report_resources": config["slurm"]["analysis"],
        "tables": sorted(str(path.relative_to(root)) for path in tables.glob("*")),
        "figures": sorted(str(path.relative_to(root)) for path in figures.glob("*.svg")),
        "interpretation": [
            "GBIF agreement uses occurrence labels and is not independently verified taxonomic accuracy.",
            "Exact species metrics include only raw GBIF species mapped into the trained classifier vocabulary.",
            "Petri-unseen species are compared primarily at genus level.",
            "Quality challenge is a preregistered composite of blur, low contrast, low detail, and exposure clipping; component measures are retained.",
            "Inference and report generation do not retrain or alter any checkpoint.",
        ],
    }
    _json_dump(root / "analysis_manifest.json", manifest)
    (root / "README.md").write_text(
        "# GBIF transfer analysis\n\n"
        "This directory contains inference-only analyses of completed `gbif_only` "
        "and `peti_to_gbif` checkpoints. No checkpoint was trained or modified.\n\n"
        "- `tables/per_image_predictions.csv.gz`: raw/mapped truth, predictions, "
        "probabilities, top-1/3/5, species counts, Petri-seen status, and quality.\n"
        "- `tables/gbif_species_metadata.csv`: GBIF train/test counts and rarity for "
        "every raw species.\n"
        "- `tables/per_seed_effects.csv`: matched Petri-minus-GBIF effects.\n"
        "- `tables/per_species_petri_effects.csv`: model-seed species effects.\n"
        "- `tables/rescued_harmed_images.csv`: images consistently rescued or harmed.\n"
        "- `tables/gbif_only_vs_gbif_to_peti_stage1_checkpoint_audit.csv`: "
        "supposedly shared Stage-1 checkpoint audit.\n"
        "- `tables/statistical_tests.csv`: paired tests and 95% intervals.\n"
        "- `figures/`: editable SVG and PDF plots with source CSVs.\n\n"
        "Exact species metrics include only classifier-evaluable species. GBIF "
        "agreement is not independently verified taxonomic accuracy.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gbif_training.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("dry-run", "submit"), default="dry-run")
    infer = commands.add_parser("infer-task")
    infer.add_argument("--index", type=Path, required=True)
    infer.add_argument("--array-index", type=int, required=True)
    commands.add_parser("report")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_domain_config(config_path)
    if args.command == "run":
        result = run_workflow(config, config_path, args.mode)
    elif args.command == "infer-task":
        result = infer_task(config, args.index, args.array_index)
    elif args.command == "report":
        result = build_report(config)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
