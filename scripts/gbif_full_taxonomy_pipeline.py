#!/usr/bin/env python3
"""Render, submit, and execute the immutable full-GBIF three-phase pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from worm_species.data.datasets import MultiTaskWormImageDataset
from worm_species.data.transforms import build_split_transform
from worm_species.gbif.full_taxonomy import (
    GBIF_TASKS, atomic_json, file_sha256, load_full_taxonomy_config,
    run_full_taxonomy_audit,
)
from worm_species.gbif.full_taxonomy_training import (
    stage_complete, train_full_taxonomy_stage,
)
from worm_species.models.multitask import build_multitask_model


def _root(config: dict) -> Path:
    return Path(config["paths"]["experiment_root"])


def _run_id(model: str, seed: int, condition: str, hierarchy: str) -> str:
    return f"full-gbif-{model}-seed{seed}-{condition}-{hierarchy}"


def build_specs(config: dict) -> dict[str, list[dict]]:
    root = _root(config)
    petri, primary, hierarchy = [], [], []
    for model in config["models"]["backbones"]:
        for seed in config["models"]["seeds"]:
            petri_output = root / "runs" / "petri_pretrain" / model / f"seed-{seed}"
            petri.append({
                "run_id": f"petri-pretrain-{model}-seed{seed}",
                "model": model, "seed": int(seed), "domain": "petri",
                "condition": "petri_pretrain", "hierarchy_kind": "none",
                "initialisation": "imagenet",
                "max_steps": int(config["training"]["petri_steps"]),
                "output_dir": str(petri_output),
            })
            for condition in ("gbif_only", "peti_to_gbif"):
                initialisation = "imagenet" if condition == "gbif_only" else "petri_backbone"
                common = {
                    "model": model, "seed": int(seed), "domain": "gbif",
                    "condition": condition, "initialisation": initialisation,
                    "petri_checkpoint": str(petri_output / "best_model.pt") if condition == "peti_to_gbif" else None,
                    "max_steps": int(config["training"]["gbif_steps"]),
                }
                primary_id = _run_id(model, seed, condition, "h0")
                primary.append({
                    **common, "run_id": primary_id, "hierarchy_kind": "none",
                    "hierarchy_loss_weight": 0.0,
                    "output_dir": str(root / "runs" / "primary_h0" / model / f"seed-{seed}" / condition),
                })
                hierarchy_id = _run_id(model, seed, condition, "ground-truth-h0p5")
                hierarchy.append({
                    **common, "run_id": hierarchy_id, "hierarchy_kind": "ground_truth",
                    "hierarchy_loss_weight": float(config["training"]["revised_hierarchy"]["weight"]),
                    "output_dir": str(root / "runs" / "revised_hierarchy" / model / f"seed-{seed}" / condition),
                })
    return {"petri": petri, "primary": primary, "hierarchy": hierarchy}


def _write_specs(root: Path, phase: str, specs: list[dict]) -> Path:
    spec_root = root / "generated" / "specs" / phase
    spec_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, spec in enumerate(specs):
        path = spec_root / f"{spec['run_id']}.json"
        atomic_json(path, spec)
        rows.append({"array_index": index, "spec_path": str(path)})
    index_path = root / "generated" / f"{phase}_tasks.tsv"
    pd.DataFrame(rows).to_csv(index_path, sep="\t", index=False)
    return index_path


def _sbatch_header(config: dict, name: str, resources: dict, *, gpu: bool, array: str | None = None) -> str:
    root = _root(config)
    logs = root / "logs"
    partition = (
        f"#SBATCH --partition={config['slurm']['gpu_partition']}\n" if gpu
        else f"#SBATCH --partition={resources['partition']}\n" if resources.get("partition") else ""
    )
    gpu_line = "#SBATCH --gres=gpu:1\n" if gpu else ""
    array_line = f"#SBATCH --array={array}\n" if array else ""
    suffix = "%A_%a" if array else "%j"
    return f"""#!/usr/bin/env bash
#SBATCH --job-name={name}
#SBATCH --account={config['slurm']['account']}
{partition}#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={resources['cpus_per_task']}
#SBATCH --mem={resources['memory']}
#SBATCH --time={resources['time_limit']}
{gpu_line}{array_line}#SBATCH --output={logs}/%x-{suffix}.out
#SBATCH --error={logs}/%x-{suffix}.err
"""


def _environment(config: dict) -> str:
    paths = config["paths"]
    return f"""set -euo pipefail
source {shlex.quote(paths['conda_sh'])}
conda activate {shlex.quote(paths['conda_env'])}
cd {shlex.quote(paths['project_root'])}
export PYTHONPATH={shlex.quote(paths['project_root'])}/src
"""


def _array_script(config: dict, config_path: Path, phase: str, count: int, index: Path) -> str:
    resources = config["slurm"]["training"]
    array = f"0-{count - 1}%{config['slurm']['array_max_active']}"
    return _sbatch_header(config, f"gbif-full-{phase}", resources, gpu=True, array=array) + "\n" + _environment(config) + f"""
spec=$(awk -F '\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == idx {{print $2}}' {shlex.quote(str(index))})
[[ -n "$spec" ]] || {{ echo "Missing task specification" >&2; exit 2; }}
srun python scripts/gbif_full_taxonomy_pipeline.py --config {shlex.quote(str(config_path))} train-task --spec "$spec"
"""


def render_pipeline(config: dict, config_path: Path) -> dict:
    root = _root(config)
    generated = root / "generated"
    logs = root / "logs"
    generated.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    specs = build_specs(config)
    indices = {phase: _write_specs(root, phase, rows) for phase, rows in specs.items()}
    scripts = {}
    audit_script = generated / "phase_a_audit.sbatch"
    audit_script.write_text(
        _sbatch_header(config, "gbif-full-audit", config["slurm"]["audit"], gpu=False)
        + "\n" + _environment(config)
        + f"srun python scripts/gbif_full_taxonomy_pipeline.py --config {shlex.quote(str(config_path))} audit\n",
        encoding="utf-8",
    )
    scripts["audit"] = str(audit_script)
    for phase in ("petri", "primary", "hierarchy"):
        path = generated / f"phase_b_{phase}.sbatch"
        path.write_text(
            _array_script(config, config_path, phase, len(specs[phase]), indices[phase]),
            encoding="utf-8",
        )
        scripts[phase] = str(path)
    final_specs = specs["primary"] + specs["hierarchy"]
    inference_rows = []
    for index, spec in enumerate(final_specs):
        inference_rows.append({
            "array_index": index, "run_id": spec["run_id"],
            "checkpoint": str(Path(spec["output_dir"]) / "best_model.pt"),
            "output": str(root / "inference" / f"{spec['run_id']}.csv.gz"),
        })
    inference_index = generated / "inference_tasks.tsv"
    pd.DataFrame(inference_rows).to_csv(inference_index, sep="\t", index=False)
    inference_script = generated / "phase_c_inference.sbatch"
    inference_script.write_text(
        _sbatch_header(
            config, "gbif-full-infer", config["slurm"]["inference"], gpu=True,
            array=f"0-{len(inference_rows) - 1}%{config['slurm']['array_max_active']}",
        ) + "\n" + _environment(config) + f"""
srun python scripts/gbif_full_taxonomy_pipeline.py --config {shlex.quote(str(config_path))} infer-task --index {shlex.quote(str(inference_index))} --array-index "$SLURM_ARRAY_TASK_ID"
""", encoding="utf-8",
    )
    scripts["inference"] = str(inference_script)
    report_script = generated / "phase_c_report.sbatch"
    report_script.write_text(
        _sbatch_header(config, "gbif-full-report", config["slurm"]["report"], gpu=False)
        + "\n" + _environment(config)
        + "export MPLCONFIGDIR=\"${TMPDIR:-/tmp}/gbif-full-mpl-${SLURM_JOB_ID:-local}\"\n"
        + "mkdir -p \"$MPLCONFIGDIR\"\n"
        + f"srun python scripts/gbif_full_taxonomy_pipeline.py --config {shlex.quote(str(config_path))} report\n",
        encoding="utf-8",
    )
    scripts["report"] = str(report_script)
    for path in scripts.values():
        Path(path).chmod(0o755)
    manifest = {
        "schema_version": 1, "experiment_id": config["experiment_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path), "config_sha256": file_sha256(config_path),
        "new_immutable_experiment_root": str(root),
        "existing_restricted_experiment_reused": False,
        "primary_final_model_count": len(specs["primary"]),
        "secondary_hierarchy_model_count": len(specs["hierarchy"]),
        "petri_pretraining_stage_count": len(specs["petri"]),
        "inference_task_count": len(inference_rows), "scripts": scripts,
        "dag": ["audit", "petri", "primary_h0", "revised_hierarchy", "inference", "report"],
    }
    atomic_json(generated / "pipeline_manifest.json", manifest)
    return manifest


def _submit(script: str, dependency: str | None = None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(script)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch output: {result.stdout!r}")
    return job_id


def submit_pipeline(config: dict, config_path: Path) -> dict:
    manifest = render_pipeline(config, config_path)
    receipt_path = _root(config) / "generated" / "submission_receipt.json"
    if receipt_path.is_file():
        previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        ids = [str(value) for key, value in previous.items() if key.endswith("_job_id") and value]
        if ids:
            active = subprocess.run(
                ["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%i"],
                text=True, capture_output=True, check=False,
            )
            if active.returncode:
                raise RuntimeError(f"Could not check prior receipt: {active.stderr.strip()}")
            if active.stdout.strip():
                raise RuntimeError(f"Full-taxonomy pipeline still has active jobs: {active.stdout.strip()}")
    scripts = manifest["scripts"]
    audit = _submit(scripts["audit"])
    petri = _submit(scripts["petri"], audit)
    primary = _submit(scripts["primary"], petri)
    hierarchy = _submit(scripts["hierarchy"], primary)
    inference = _submit(scripts["inference"], hierarchy)
    report = _submit(scripts["report"], inference)
    receipt = {
        "audit_job_id": audit, "petri_job_id": petri,
        "primary_job_id": primary, "hierarchy_job_id": hierarchy,
        "inference_job_id": inference, "report_job_id": report,
        "all_jobs_submitted": True,
    }
    atomic_json(receipt_path, receipt)
    return {**manifest, "submission": receipt}


def _model_for_inference(checkpoint: Path, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    label_maps = payload["label_maps"]
    spec = payload["spec"]
    model = build_multitask_model(
        {"model": {"name": spec["model"], "pretrained": False}},
        {task: len(mapping) for task, mapping in label_maps.items()},
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval().to(device)
    return model, payload, label_maps


def infer_checkpoint(config: dict, checkpoint: Path, output: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Full-taxonomy inference requires CUDA")
    model, payload, label_maps = _model_for_inference(checkpoint, torch.device("cuda"))
    spec = payload["spec"]
    test = pd.read_csv(_root(config) / "prepared" / "gbif_test.csv", dtype=str, keep_default_na=False)
    transform = build_split_transform(
        split="test", preprocessing=payload["preprocessing"],
        augmentation=config["training"]["augmentation"],
        condition={"transform": "original"}, apply_augmentation=False,
    )
    dataset = MultiTaskWormImageDataset(
        test, root_dir="/", image_col="image_path", target_cols=GBIF_TASKS,
        label_to_index_by_task=label_maps, transform=transform,
        crop_to_foreground=False,
    )
    loader = DataLoader(
        dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False,
        num_workers=int(config["training"]["num_workers"]), pin_memory=True,
        prefetch_factor=int(config["training"]["prefetch_factor"]), persistent_workers=True,
    )
    inverse = {
        task: {int(index): label for label, index in mapping.items()}
        for task, mapping in label_maps.items()
    }
    records = []
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch["image"].to("cuda", non_blocking=True))
            probabilities = {task: torch.softmax(outputs[task].float(), 1).cpu().numpy() for task in GBIF_TASKS}
            count = len(batch["image"])
            for index in range(count):
                source = test.iloc[offset + index]
                record = {
                    "image_id": source["image_id"], "gbif_id": source["gbif_id"],
                    "occurrence_id": source["occurrence_id"], "group_id": source["group_id"],
                    "image_path": source["image_path"], "true_genus": source["genus"],
                    "true_species": source["species"], "raw_true_genus": source["raw_genus"],
                    "raw_true_species": source["raw_species"], "run_id": spec["run_id"],
                    "model": spec["model"], "seed": spec["seed"],
                    "condition": spec["condition"], "hierarchy_kind": spec["hierarchy_kind"],
                    "hierarchy_loss_weight": spec.get("hierarchy_loss_weight", 0.0),
                }
                for task in GBIF_TASKS:
                    probs = probabilities[task][index]
                    order = np.argsort(probs)[::-1]
                    true_label = str(source[task])
                    true_index = label_maps[task].get(true_label)
                    true_rank = int(np.flatnonzero(order == true_index)[0] + 1) if true_index is not None else pd.NA
                    record[f"predicted_{task}"] = inverse[task][int(order[0])]
                    record[f"predicted_{task}_probability"] = float(probs[order[0]])
                    record[f"true_{task}_probability"] = float(probs[true_index]) if true_index is not None else np.nan
                    record[f"true_{task}_rank"] = true_rank
                    record[f"{task}_probabilities_json"] = json.dumps([float(value) for value in probs], separators=(",", ":"))
                    for k in (1, 3, 5):
                        selected = order[: min(k, len(order))]
                        top = [{"label": inverse[task][int(item)], "probability": float(probs[item])} for item in selected]
                        record[f"{task}_top{k}_json"] = json.dumps(top, separators=(",", ":"))
                        record[f"{task}_top{k}_correct"] = bool(true_index is not None and true_index in selected)
                records.append(record)
            offset += count
    if offset != len(test):
        raise AssertionError("Full-taxonomy inference coverage mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False, compression="gzip")
    summary = {
        "status": "complete", "run_id": spec["run_id"], "rows": len(records),
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "genus_classes": len(label_maps["genus"]), "species_classes": len(label_maps["species"]),
    }
    atomic_json(output.with_suffix("").with_suffix(".summary.json"), summary)
    return summary


def infer_task(config: dict, index_path: Path, array_index: int) -> dict:
    table = pd.read_csv(index_path, sep="\t", dtype=str, keep_default_na=False)
    row = table.loc[table["array_index"].astype(int).eq(array_index)]
    if len(row) != 1:
        raise ValueError(f"Invalid inference array index {array_index}")
    item = row.iloc[0]
    checkpoint, output = Path(item["checkpoint"]), Path(item["output"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    summary_path = output.with_suffix("").with_suffix(".summary.json")
    if output.is_file() and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("checkpoint_sha256") == file_sha256(checkpoint):
            return {**summary, "status": "reused_complete"}
    return infer_checkpoint(config, checkpoint, output)


def _quality_one(item: tuple[str, str]) -> dict:
    from PIL import Image
    image_id, path = item
    result = {"image_id": image_id, "image_path": path, "quality_status": "ok"}
    try:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            working = rgb.copy()
            working.thumbnail((512, 512), Image.Resampling.BILINEAR)
        array = np.asarray(working, dtype=np.float32) / 255.0
        luminance = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
        gray = np.clip(np.rint(luminance * 255), 0, 255).astype(np.uint8)
        histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
        probability = histogram[histogram > 0] / histogram.sum()
        entropy = float(-(probability * np.log2(probability)).sum())
        center = gray[1:-1, 1:-1].astype(np.float32)
        laplacian = (
            -4 * center + gray[:-2, 1:-1] + gray[2:, 1:-1]
            + gray[1:-1, :-2] + gray[1:-1, 2:]
        ) if min(gray.shape) >= 3 else np.asarray([np.nan])
        result.update({
            "width_px": int(width), "height_px": int(height),
            "megapixels": float(width * height / 1_000_000),
            "luminance_mean": float(luminance.mean()),
            "luminance_contrast_sd": float(luminance.std()),
            "exposure_clip_fraction": float(((luminance <= 0.02) | (luminance >= 0.98)).mean()),
            "grayscale_entropy_bits": entropy,
            "laplacian_variance": float(np.nanvar(laplacian)),
        })
    except Exception as exc:
        result.update(quality_status="error", quality_error=f"{type(exc).__name__}: {exc}")
    return result


def _technical_quality(config: dict, test: pd.DataFrame) -> pd.DataFrame:
    output = _root(config) / "analysis" / "technical_image_quality.csv"
    if output.is_file():
        existing = pd.read_csv(output, dtype={"image_id": str})
        if set(existing["image_id"].astype(str)) == set(test["image_id"].astype(str)):
            return existing
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", config["analysis"]["quality_workers"]))
    workers = min(int(config["analysis"]["quality_workers"]), allocated)
    items = list(test[["image_id", "image_path"]].itertuples(index=False, name=None))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_quality_one, items, chunksize=max(1, len(items) // (workers * 8))))
    quality = pd.DataFrame(rows)
    errors = quality.loc[~quality["quality_status"].eq("ok")]
    if not errors.empty:
        raise RuntimeError(f"Technical quality failed for {len(errors)} required test images")
    components = pd.DataFrame(index=quality.index)
    components["blur"] = (-quality["laplacian_variance"]).rank(pct=True)
    components["low_contrast"] = (-quality["luminance_contrast_sd"]).rank(pct=True)
    components["low_detail"] = (-quality["grayscale_entropy_bits"]).rank(pct=True)
    components["clipping"] = quality["exposure_clip_fraction"].rank(pct=True)
    quality["technical_quality_challenge"] = components.mean(axis=1)
    quality["technical_quality_quartile"] = pd.qcut(
        quality["technical_quality_challenge"].rank(method="first"), 4,
        labels=["Q1_cleaner", "Q2", "Q3", "Q4_lower_quality"],
    ).astype(str)
    output.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(output, index=False)
    return quality


def _parent_mapping(root: Path) -> dict[str, str]:
    table = pd.read_csv(root / "prepared" / "species_to_genus.csv", dtype=str, keep_default_na=False)
    if table["species"].duplicated().any():
        raise ValueError("Species-to-genus mapping is not one-to-one")
    return table.set_index("species")["genus"].to_dict()


def _run_metrics(predictions: pd.DataFrame, parent: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    identities = ["run_id", "model", "seed", "condition", "hierarchy_kind", "hierarchy_loss_weight"]
    run_rows, species_rows = [], []
    for keys, group in predictions.groupby(identities, sort=True):
        base = dict(zip(identities, keys))
        for task in ("genus", "species"):
            valid = group[f"true_{task}"].ne("")
            subset = group.loc[valid]
            true, pred = subset[f"true_{task}"], subset[f"predicted_{task}"]
            if subset.empty:
                continue
            values = {
                "top1_accuracy": accuracy_score(true, pred),
                "balanced_accuracy": balanced_accuracy_score(true, pred),
                "macro_f1": f1_score(true, pred, average="macro", zero_division=0),
                "top3_accuracy": subset[f"{task}_top3_correct"].astype(bool).mean(),
                "top5_accuracy": subset[f"{task}_top5_correct"].astype(bool).mean(),
            }
            for metric, value in values.items():
                run_rows.append({**base, "level": "image", "task": task, "metric": metric, "value": float(value), "n_images": len(subset), "n_classes": true.nunique()})
            if task == "species":
                labels = sorted(true.unique())
                precision, recall, f1, support = precision_recall_fscore_support(
                    true, pred, labels=labels, zero_division=0
                )
                for label, p, r, score, n in zip(labels, precision, recall, f1, support):
                    species_rows.append({
                        **base, "species": label, "precision": float(p),
                        "recall": float(r), "species_image_accuracy": float(r),
                        "f1": float(score), "test_images": int(n),
                    })
        consistency = group["predicted_species"].map(parent).eq(group["predicted_genus"])
        species_valid = group["true_species"].ne("")
        species = group.loc[species_valid].copy()
        correct = species["predicted_species"].eq(species["true_species"])
        predicted_parent = species["predicted_species"].map(parent)
        within = (~correct) & predicted_parent.eq(species["true_genus"])
        between = (~correct) & ~predicted_parent.eq(species["true_genus"])
        severity = np.where(correct, 0, np.where(within, 1, 2))
        taxonomy_values = {
            "head_consistency": consistency.mean(),
            "within_genus_error_fraction": within.sum() / max((~correct).sum(), 1),
            "between_genus_error_fraction": between.sum() / max((~correct).sum(), 1),
            "taxonomic_severity_0_1_2": np.mean(severity),
        }
        for metric, value in taxonomy_values.items():
            run_rows.append({**base, "level": "image", "task": "taxonomy", "metric": metric, "value": float(value), "n_images": len(species), "n_classes": species["true_species"].nunique()})
        # Occurrence/group-level probabilities are averaged before classification.
        for task in ("genus", "species"):
            label_order = json.loads((_root_from_predictions(group) / "analysis" / "probability_label_order.json").read_text())[task]
            grouped_true, grouped_pred = [], []
            for _group_id, images in group.loc[group[f"true_{task}"].ne("")].groupby("group_id"):
                matrix = np.vstack(images[f"{task}_probabilities_json"].map(json.loads))
                grouped_true.append(images[f"true_{task}"].iloc[0])
                grouped_pred.append(label_order[int(matrix.mean(axis=0).argmax())])
            if grouped_true:
                for metric, value in (
                    ("top1_accuracy", accuracy_score(grouped_true, grouped_pred)),
                    ("balanced_accuracy", balanced_accuracy_score(grouped_true, grouped_pred)),
                    ("macro_f1", f1_score(grouped_true, grouped_pred, average="macro", zero_division=0)),
                ):
                    run_rows.append({**base, "level": "group", "task": task, "metric": metric, "value": float(value), "n_images": len(grouped_true), "n_classes": len(set(grouped_true))})
    return pd.DataFrame(run_rows), pd.DataFrame(species_rows)


def _root_from_predictions(_group: pd.DataFrame) -> Path:
    # The report sets this process-local environment variable before metric collection.
    return Path(os.environ["WORM_FULL_TAXONOMY_ROOT"])


def _pair_primary(predictions: pd.DataFrame) -> pd.DataFrame:
    primary = predictions.loc[predictions["hierarchy_kind"].eq("none")]
    keys = [
        "image_id", "group_id", "true_genus", "true_species", "model", "seed",
    ]
    value = []
    for task in ("genus", "species"):
        value += [
            f"predicted_{task}", f"true_{task}_probability", f"true_{task}_rank",
            f"{task}_top1_correct", f"{task}_top3_correct", f"{task}_top5_correct",
        ]
    left = primary.loc[primary["condition"].eq("gbif_only"), keys + value]
    right = primary.loc[primary["condition"].eq("peti_to_gbif"), keys + value]
    paired = left.merge(right, on=keys, validate="one_to_one", suffixes=("_gbif_only", "_peti_to_gbif"))
    for task in ("genus", "species"):
        for k in (1, 3, 5):
            paired[f"{task}_top{k}_effect"] = (
                paired[f"{task}_top{k}_correct_peti_to_gbif"].astype(int)
                - paired[f"{task}_top{k}_correct_gbif_only"].astype(int)
            )
        paired[f"{task}_rank_improvement"] = (
            pd.to_numeric(paired[f"true_{task}_rank_gbif_only"], errors="coerce")
            - pd.to_numeric(paired[f"true_{task}_rank_peti_to_gbif"], errors="coerce")
        )
    return paired


def _rarity_band(value: int, bands: list[dict]) -> str:
    for band in bands:
        upper = band.get("maximum")
        if value >= int(band["minimum"]) and (upper is None or value <= int(upper)):
            return str(band["label"])
    raise ValueError(f"Training group count {value} has no rarity band")


def _effect_tables(
    config: dict, run_metrics: pd.DataFrame, species_results: pd.DataFrame,
    paired_images: pd.DataFrame, quality: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    root = _root(config)
    taxon = pd.read_csv(root / "audit" / "species_inclusion_evaluability_audit.csv")
    taxon = taxon.rename(columns={
        "label": "species", "train_groups": "training_groups",
        "train_images": "training_images", "test_groups": "testing_groups",
    })
    crosswalk = pd.read_csv(root / "audit" / "gbif_petri_species_crosswalk.csv")
    primary_metrics = run_metrics.loc[run_metrics["hierarchy_kind"].eq("none")]
    index = ["model", "seed", "level", "task", "metric"]
    wide = primary_metrics.pivot_table(index=index, columns="condition", values="value").reset_index()
    wide["petri_minus_gbif"] = wide["peti_to_gbif"] - wide["gbif_only"]
    primary_species = species_results.loc[species_results["hierarchy_kind"].eq("none")]
    keys = ["model", "seed", "species"]
    columns = ["f1", "recall", "species_image_accuracy", "precision"]
    left = primary_species.loc[primary_species["condition"].eq("gbif_only"), keys + columns]
    right = primary_species.loc[primary_species["condition"].eq("peti_to_gbif"), keys + columns]
    species_effect = left.merge(right, on=keys, validate="one_to_one", suffixes=("_gbif_only", "_peti_to_gbif"))
    for metric in columns:
        species_effect[f"{metric}_petri_effect"] = species_effect[f"{metric}_peti_to_gbif"] - species_effect[f"{metric}_gbif_only"]
    species_effect = species_effect.merge(
        taxon[["species", "genus", "training_groups", "training_images", "testing_groups", "test_images"]],
        on="species", how="left", validate="many_to_one",
    ).merge(
        crosswalk[["gbif_species", "petri_seen", "mapping_rule"]],
        left_on="species", right_on="gbif_species", how="left", validate="many_to_one",
    ).drop(columns="gbif_species")
    species_effect["rarity_band"] = species_effect["training_groups"].map(
        lambda value: _rarity_band(int(value), config["analysis"]["rarity_group_bands"])
    )
    regressions = []
    for count_column in ("training_groups", "training_images"):
        for keys_value, group in species_effect.groupby(["model", "seed"], sort=True):
            x = np.log10(group[count_column].astype(float) + 1)
            y = group["f1_petri_effect"].astype(float)
            regression = stats.linregress(x, y) if x.nunique() > 1 else None
            regressions.append({
                "model": keys_value[0], "seed": keys_value[1], "count_measure": count_column,
                "slope": regression.slope if regression else np.nan,
                "intercept": regression.intercept if regression else np.nan,
                "r": regression.rvalue if regression else np.nan,
                "p": regression.pvalue if regression else np.nan,
                "n_species": len(group),
            })
    rarity_regression = pd.DataFrame(regressions)
    rarity_summary = species_effect.groupby(
        ["model", "seed", "rarity_band"], sort=True
    ).agg(
        mean_f1_effect=("f1_petri_effect", "mean"),
        species=("species", "nunique"),
        training_groups=("training_groups", "sum"),
        training_images=("training_images", "sum"),
        test_images=("test_images", "sum"),
    ).reset_index()
    seen = species_effect.groupby(["model", "seed", "petri_seen"], dropna=False).agg(
        mean_f1_effect=("f1_petri_effect", "mean"),
        mean_recall_effect=("recall_petri_effect", "mean"),
        species=("species", "nunique"), test_images=("test_images", "sum"),
    ).reset_index()
    paired_images = paired_images.merge(
        taxon[["species", "training_groups", "training_images"]],
        left_on="true_species", right_on="species", how="left", validate="many_to_one",
    ).drop(columns="species").merge(
        quality.drop(columns="image_path"), on="image_id", how="left", validate="many_to_one",
    )
    both_wrong = paired_images.loc[
        ~paired_images["species_top1_correct_gbif_only"].astype(bool)
        & ~paired_images["species_top1_correct_peti_to_gbif"].astype(bool)
        & paired_images["true_species"].ne("")
    ].copy()
    ranking = both_wrong.groupby(["model", "seed"]).agg(
        images=("image_id", "size"),
        mean_true_rank_improvement=("species_rank_improvement", "mean"),
        mean_reciprocal_rank_gbif=("true_species_rank_gbif_only", lambda values: np.mean(1 / pd.to_numeric(values))),
        mean_reciprocal_rank_petri=("true_species_rank_peti_to_gbif", lambda values: np.mean(1 / pd.to_numeric(values))),
        top3_effect=("species_top3_effect", "mean"), top5_effect=("species_top5_effect", "mean"),
    ).reset_index()
    valid = paired_images.loc[paired_images["true_species"].ne("")].copy()
    valid["change"] = np.select([
        valid["species_top1_effect"].eq(1), valid["species_top1_effect"].eq(-1),
    ], ["rescued_wrong_to_correct", "harmed_correct_to_wrong"], default="unchanged")
    changed = valid.loc[~valid["change"].eq("unchanged")].copy()
    quality_effect = valid.groupby(
        ["model", "seed", "technical_quality_quartile"], sort=True
    ).agg(
        petri_top1_effect=("species_top1_effect", "mean"), images=("image_id", "size"),
        mean_training_groups=("training_groups", "mean"),
    ).reset_index()
    return {
        "paired_run_effects": wide, "per_species_results": species_results,
        "per_species_petri_effects": species_effect,
        "rarity_regressions": rarity_regression, "rarity_band_summary": rarity_summary,
        "petri_seen_unseen": seen, "both_wrong_ranking": ranking,
        "rescued_harmed_images": changed, "technical_quality_effects": quality_effect,
        "paired_images": paired_images,
    }


def _hierarchy_effects(run_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "seed", "condition", "level", "task", "metric"]
    table = run_metrics.pivot_table(index=keys, columns="hierarchy_kind", values="value").reset_index()
    table = table.dropna(subset=["none", "ground_truth"])
    table["revised_minus_h0"] = table["ground_truth"] - table["none"]
    return table


def _save_plot(fig, root: Path, name: str, source: pd.DataFrame) -> None:
    figures = root / "figures"
    sources = figures / "figure_sources"
    sources.mkdir(parents=True, exist_ok=True)
    source.to_csv(sources / f"{name}.csv", index=False)
    for extension in ("svg", "pdf"):
        fig.savefig(figures / f"{name}.{extension}", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _figures(root: Path, tables: dict, hierarchy: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    paired = tables["paired_run_effects"]
    primary = paired.loc[(paired["level"] == "image") & paired["task"].isin(["species", "genus"])]
    primary = primary.loc[primary["metric"].isin(["macro_f1", "balanced_accuracy", "top1_accuracy", "top3_accuracy", "top5_accuracy"])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, model in zip(axes, ["convnext_base", "vit_b_16", "resnet50"]):
        subset = primary.loc[primary["model"].eq(model)]
        labels = subset["task"] + " · " + subset["metric"].str.replace("_", " ")
        positions = {label: index for index, label in enumerate(sorted(labels.unique()))}
        ax.scatter(subset["petri_minus_gbif"], labels.map(positions), alpha=.75)
        ax.axvline(0, color="#555555", lw=1)
        ax.set_title(model)
        ax.set_xlabel("Petri − GBIF-only")
        ax.set_yticks(list(positions.values()), list(positions.keys()))
    _save_plot(fig, root, "01_primary_paired_effects_by_backbone", primary)
    rarity = tables["rarity_band_summary"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, model in zip(axes, ["convnext_base", "vit_b_16", "resnet50"]):
        subset = rarity.loc[rarity["model"].eq(model)]
        for seed, group in subset.groupby("seed"):
            ax.plot(group["rarity_band"], group["mean_f1_effect"], marker="o", alpha=.7, label=str(seed))
        ax.axhline(0, color="#555555", lw=1)
        ax.set_title(model); ax.tick_params(axis="x", rotation=35)
        ax.set_xlabel("Training groups per species")
    axes[0].set_ylabel("Species F1 Petri effect")
    axes[-1].legend(title="seed", frameon=False)
    _save_plot(fig, root, "02_rarity_effects", rarity)
    seen = tables["petri_seen_unseen"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)
    for ax, model in zip(axes, ["convnext_base", "vit_b_16", "resnet50"]):
        subset = seen.loc[seen["model"].eq(model)]
        for flag, group in subset.groupby("petri_seen"):
            ax.scatter(["Petri-seen" if flag else "Petri-unseen"] * len(group), group["mean_f1_effect"], label=str(flag))
        ax.axhline(0, color="#555555", lw=1); ax.set_title(model)
    axes[0].set_ylabel("Mean per-species F1 effect")
    _save_plot(fig, root, "03_petri_seen_unseen", seen)
    quality = tables["technical_quality_effects"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, model in zip(axes, ["convnext_base", "vit_b_16", "resnet50"]):
        subset = quality.loc[quality["model"].eq(model)]
        for seed, group in subset.groupby("seed"):
            ax.plot(group["technical_quality_quartile"], group["petri_top1_effect"], marker="o")
        ax.axhline(0, color="#555555", lw=1); ax.set_title(model); ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Petri top-1 effect")
    _save_plot(fig, root, "04_technical_quality_interaction", quality)
    selected = hierarchy.loc[hierarchy["metric"].isin([
        "macro_f1", "head_consistency", "within_genus_error_fraction",
        "between_genus_error_fraction", "taxonomic_severity_0_1_2",
    ])]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, model in zip(axes, ["convnext_base", "vit_b_16", "resnet50"]):
        subset = selected.loc[selected["model"].eq(model)]
        labels = subset["condition"] + " · " + subset["task"] + " · " + subset["metric"].str.replace("_", " ")
        positions = {label: index for index, label in enumerate(sorted(labels.unique()))}
        ax.scatter(subset["revised_minus_h0"], labels.map(positions), alpha=.7)
        ax.axvline(0, color="#555555", lw=1); ax.set_title(model)
        ax.set_yticks(list(positions.values()), list(positions.keys()))
    _save_plot(fig, root, "05_revised_hierarchy_effects", selected)


def _mean_by_backbone(table: pd.DataFrame, value: str) -> list[str]:
    lines = []
    for model, group in table.groupby("model", sort=True):
        values = group[value].dropna().astype(float)
        lines.append(f"- {model}: mean {value} = {values.mean():+.4f} across {len(values)} paired seeds.")
    return lines


def build_report(config: dict) -> dict:
    root = _root(config)
    audit = json.loads((root / "audit" / "audit_manifest.json").read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or audit.get("fatal_leakage_rows") != 0:
        raise RuntimeError("Cannot report without a clean immutable audit")
    specs = build_specs(config)
    final_specs = specs["primary"] + specs["hierarchy"]
    frames = []
    for spec in final_specs:
        output = root / "inference" / f"{spec['run_id']}.csv.gz"
        summary = output.with_suffix("").with_suffix(".summary.json")
        if not output.is_file() or not summary.is_file():
            raise FileNotFoundError(f"Missing inference output for {spec['run_id']}")
        frames.append(pd.read_csv(output, low_memory=False))
    predictions = pd.concat(frames, ignore_index=True)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(analysis / "per_image_predictions_probabilities.csv.gz", index=False, compression="gzip")
    label_maps = json.loads((root / "prepared" / "gbif_label_maps.json").read_text())
    label_order = {
        task: [label for label, _index in sorted(mapping.items(), key=lambda item: item[1])]
        for task, mapping in label_maps.items()
    }
    atomic_json(analysis / "probability_label_order.json", label_order)
    os.environ["WORM_FULL_TAXONOMY_ROOT"] = str(root)
    parent = _parent_mapping(root)
    run_metrics, species_results = _run_metrics(predictions, parent)
    run_metrics.to_csv(analysis / "run_level_metrics.csv", index=False)
    species_results.to_csv(analysis / "per_species_results_all_runs.csv", index=False)
    test = pd.read_csv(root / "prepared" / "gbif_test.csv", dtype=str, keep_default_na=False)
    quality = _technical_quality(config, test)
    paired_images = _pair_primary(predictions)
    tables = _effect_tables(config, run_metrics, species_results, paired_images, quality)
    for name, table in tables.items():
        suffix = ".csv.gz" if name in {"paired_images"} else ".csv"
        table.to_csv(analysis / f"{name}{suffix}", index=False, compression="gzip" if suffix.endswith("gz") else None)
    hierarchy = _hierarchy_effects(run_metrics)
    hierarchy.to_csv(analysis / "revised_hierarchy_effects.csv", index=False)
    # Verify matched step-zero head tensors across both conditions and hierarchy arms.
    head_rows = []
    for spec in final_specs:
        step0 = json.loads((Path(spec["output_dir"]) / "step0_head_audit.json").read_text())
        head_rows.append({**{key: spec[key] for key in ("run_id", "model", "seed", "condition", "hierarchy_kind")}, **step0})
    head_audit = pd.DataFrame(head_rows)
    head_audit["matched_head_hash_count"] = head_audit.groupby(["model", "seed"])["head_tensor_sha256"].transform("nunique")
    if head_audit["matched_head_hash_count"].ne(1).any():
        raise RuntimeError("Matched GBIF heads were not identical before optimisation")
    head_audit.to_csv(analysis / "matched_head_initialisation_audit.csv", index=False)
    _figures(root, tables, hierarchy)
    primary_species = tables["paired_run_effects"].loc[
        (tables["paired_run_effects"]["level"] == "image")
        & (tables["paired_run_effects"]["task"] == "species")
        & (tables["paired_run_effects"]["metric"] == "macro_f1")
    ]
    rarity_group = tables["rarity_regressions"].loc[tables["rarity_regressions"]["count_measure"].eq("training_groups")]
    unseen = tables["petri_seen_unseen"].loc[~tables["petri_seen_unseen"]["petri_seen"].astype(bool)]
    quality_summary = tables["technical_quality_effects"].loc[
        tables["technical_quality_effects"]["technical_quality_quartile"].eq("Q4_lower_quality")
    ]
    hierarchy_species = hierarchy.loc[
        (hierarchy["level"] == "image") & (hierarchy["task"] == "species")
        & (hierarchy["metric"] == "macro_f1")
    ]
    report_lines = [
        "# Full-GBIF Petri-pretraining and hierarchy report", "",
        "## Immutable dataset", "",
        f"Raw species: {audit['raw_species_count']}; canonical valid: {audit['canonical_valid_species_count']}; "
        f"trainable: {audit['trainable_species_count']}; test-evaluable: {audit['test_evaluable_species_count']}.",
        "", "## Does Petri pretraining improve full-GBIF classification?", "",
        *_mean_by_backbone(primary_species, "petri_minus_gbif"),
        "", "## Does it help data-poor species more?", "",
        *_mean_by_backbone(rarity_group, "slope"),
        "", "Negative slopes mean larger benefit at lower training-group counts. Image-count sensitivity is in `rarity_regressions.csv`.",
        "", "## Does it transfer to species never seen in Petri?", "",
        *_mean_by_backbone(unseen, "mean_f1_effect"),
        "", "## Does it help difficult or messy photographs?", "",
        *_mean_by_backbone(quality_summary, "petri_top1_effect"),
        "", "Technical quality is measured automatically. Semantic messiness is not inferred: blinded fields remain in `audit/semantic_messiness_annotation_manifest.csv` until human annotation is completed.",
        "", "## Does revised hierarchy loss improve taxonomic correctness?", "",
        *_mean_by_backbone(hierarchy_species, "revised_minus_h0"),
        "", "Consistency, within-/between-genus errors and 0/1/2 taxonomic severity are in `revised_hierarchy_effects.csv`.",
        "", "All backbone results remain separate; the three backbones are not pooled as nine replicates.",
    ]
    (root / "final_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    result = {
        "schema_version": 1, "status": "complete",
        "experiment_id": config["experiment_id"], "audit_manifest_sha256": file_sha256(root / "audit" / "audit_manifest.json"),
        "split_manifest_sha256": audit["final_split_sha256"],
        "label_maps_sha256": audit["gbif_label_maps_sha256"],
        "run_level_rows": len(run_metrics), "per_image_rows": len(predictions),
        "per_species_rows": len(species_results), "final_report": str(root / "final_report.md"),
    }
    atomic_json(root / "final_manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gbif_full_taxonomy.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("dry-run", "submit"), default="dry-run")
    commands.add_parser("audit")
    train = commands.add_parser("train-task")
    train.add_argument("--spec", type=Path, required=True)
    infer = commands.add_parser("infer-task")
    infer.add_argument("--index", type=Path, required=True)
    infer.add_argument("--array-index", type=int, required=True)
    commands.add_parser("report")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_full_taxonomy_config(config_path)
    if args.command == "run":
        result = render_pipeline(config, config_path) if args.mode == "dry-run" else submit_pipeline(config, config_path)
    elif args.command == "audit":
        result = run_full_taxonomy_audit(config, config_path)
    elif args.command == "train-task":
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        result = train_full_taxonomy_stage(config, spec)
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
