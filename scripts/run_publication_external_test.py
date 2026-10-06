"""Evaluate every publication baseline seed on the two new camera cohorts."""

from __future__ import annotations

import argparse

from contextlib import contextmanager

import copy

import csv

import hashlib

import json

import os

from pathlib import Path

import shlex

import subprocess

import sys

import tempfile

PROJECT = Path(__file__).resolve().parents[1]

for path in (PROJECT, PROJECT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MODELS = ("convnext_base", "vit_b_16", "resnet50")

SEEDS = tuple(range(40, 2941, 100))

CAMERAS = ("gphoto2", "webcam")

TASKS = ("age", "genus", "species")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)

def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def stamp(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield

def protocol(cfg):
    """Only settings affecting evaluation; cluster/training paths are irrelevant."""
    return {"preprocessing": cfg["preprocessing"], "data": {
        key: cfg["data"].get(key) for key in (
            "crop_to_foreground", "crop_pad", "target_cols", "barcode_col",
            "taxonomic_uncertainty", "missing_label_values", "colour_retention")
    }, "use_amp": cfg["training"].get("use_amp", True)}

def discover_checkpoints(result_root):
    runs = {}
    for path in sorted((result_root / "runs/baseline").glob("**/config.json")):
        cfg = json.loads(path.read_text())
        model, seed = cfg["model"]["name"], cfg["seed"]
        if model not in MODELS or seed not in SEEDS:
            raise ValueError(f"Unexpected baseline model/seed in {path}: {model}, {seed}")
        key = (model, seed)
        if key in runs:
            raise ValueError(f"Multiple checkpoints for {key}; resolve duplicates before testing")
        condition = cfg.get("input_condition", {})
        if condition.get("transform", "original") != "original" or cfg.get("data_holdout", {}).get("enabled"):
            raise ValueError(f"Not a full-data original-image baseline: {path}")
        if cfg["data"].get("crop_to_foreground", True):
            raise ValueError(f"Expected the paper's no-recrop preprocessing: {path}")
        if cfg.get("multi_task", {}).get("hierarchy_loss", {}).get("enabled"):
            raise ValueError(f"Unexpected hierarchy-loss baseline: {path}")
        marker = path.parent.parent / "run_status.txt"
        if not marker.is_file() or marker.read_text().strip() != "0":
            raise ValueError(f"Baseline has no successful completion marker: {path}")
        checkpoint = path.parent / "best_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint (check uploaded symlink targets): {checkpoint}")
        mapping_path = path.parent / "label_to_index_by_task.json"
        mapping = json.loads(mapping_path.read_text())
        if set(mapping) != set(TASKS):
            raise ValueError(f"Unexpected model heads in {mapping_path}")
        for task, labels in mapping.items():
            if not labels or sorted(labels.values()) != list(range(len(labels))):
                raise ValueError(f"Non-contiguous class map: {mapping_path}/{task}")
        old_predictions = path.parent / "test_predictions_best.csv"
        if not old_predictions.is_file():
            raise FileNotFoundError(f"Original-test predictions required for comparison: {old_predictions}")
        runs[key] = {"model": model, "seed": seed, "checkpoint": str(checkpoint.absolute()),
                     "checkpoint_stamp": stamp(checkpoint), "config": cfg,
                     "label_maps": mapping, "original_predictions": str(old_predictions.absolute()),
                     "original_predictions_sha256": sha(old_predictions)}
    expected = {(m, s) for m in MODELS for s in SEEDS}
    if set(runs) != expected:
        raise ValueError(f"Need all 90 baselines. Missing: {sorted(expected - set(runs))}")
    ordered = [runs[(m, s)] for m in MODELS for s in SEEDS]
    first = ordered[0]
    if any(r["label_maps"] != first["label_maps"] or protocol(r["config"]) != protocol(first["config"]) for r in ordered):
        raise ValueError("Baseline class maps or preprocessing/taxonomy differ; cannot aggregate these seeds")
    return ordered

def camera_cohorts(dataset_root, year):
    manifest = dataset_root / "metadata/images.csv"
    rows = read_csv(manifest)
    selected = [r for r in rows if r["year"] == str(year) and r["camera"] in CAMERAS and r["kind"] == "worm"]
    if any(r["segmentation_status"] not in {"segmented", "reused", "failed"} for r in selected):
        raise ValueError("Segmentation is incomplete; finish it before freezing the test cohort")
    if len({r["image_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate image IDs in external test metadata")
    historical = {r["barcode"] for r in rows if r.get("original_paper_split")}
    overlap = historical & {r["barcode"] for r in selected}
    if overlap:
        raise ValueError(f"External and paper specimen barcodes overlap: {sorted(overlap)}")
    cohorts, excluded = {}, []
    for camera in CAMERAS:
        usable = []
        for row in sorted((r for r in selected if r["camera"] == camera), key=lambda r: r["image_id"]):
            if row["segmentation_status"] == "failed":
                excluded.append({**row, "exclusion_reason": "failed_segmentation"})
                continue
            relative = Path(row["rel_path_seg"])
            path = (dataset_root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(dataset_root.resolve()):
                raise ValueError(f"Invalid segmented image path: {relative}")
            if not path.is_file():
                raise FileNotFoundError(path)
            usable.append({**row, "file_bytes": str(path.stat().st_size), "file_mtime_ns": str(path.stat().st_mtime_ns)})
        if not usable:
            raise ValueError(f"No segmented worm images for {year}/{camera}")
        cohorts[camera] = usable
    return cohorts, excluded, sha(manifest)

def load_plan(root):
    plan = json.loads((root / "plan.json").read_text())
    expected = plan["identity"]
    if identity({k: v for k, v in plan.items() if k != "identity"}) != expected:
        raise ValueError("Plan identity mismatch")
    for path, expected_sha in plan["source_sha256"].items():
        if sha(path) != expected_sha:
            raise ValueError(f"Evaluation code changed after planning: {path}")
    return plan

def labeled_frame(rows, cfg):
    import pandas as pd
    from worm_species.data.taxonomy import parse_taxonomy_from_barcode, apply_taxonomic_uncertainty_rules
    frame = pd.DataFrame(rows).copy()
    frame["taxon_label"] = frame["taxon"]
    # Do not apply training rare-class filters to an external test cohort.
    frame = parse_taxonomy_from_barcode(frame, cfg)
    return apply_taxonomic_uncertainty_rules(frame, cfg)
