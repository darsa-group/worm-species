"""RGB external-camera inference for the paper; no classifier training."""

from __future__ import annotations

import argparse

import copy

import json

import math

import os

from pathlib import Path

import sys

PROJECT = Path(__file__).resolve().parents[1]

for path in (PROJECT, PROJECT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_publication_external_test as reference

CAMERAS = reference.CAMERAS

TASKS = reference.TASKS

INPUTS = ("rgb",)

def settings(path):
    import yaml
    cfg = yaml.safe_load(path.read_text())
    for name in ("dataset", "publication_results", "output", "external_reference"):
        if cfg.get(name) is not None:
            cfg[name] = str((path.parent / cfg[name]).resolve())
    if cfg["inputs"] != list(INPUTS):
        raise ValueError("External evaluation requires inputs: [rgb]")
    from worm_species.domain_analysis import METHODS
    if not set(cfg["aggregation"]) <= set(METHODS) or "mean_probabilities" not in cfg["aggregation"]:
        raise ValueError("Aggregation must include mean_probabilities and use known methods")
    if len(set(cfg["aggregation"])) != len(cfg["aggregation"]):
        raise ValueError("Duplicate aggregation methods")
    if cfg["batch_size"] < 1 or cfg["workers"] < 0 or cfg["cpu_threads"] < 1:
        raise ValueError("Invalid local execution resources")
    rep = cfg["representation"]
    if rep["model"] not in reference.MODELS or rep["selection_mode"] not in {"min", "max"}:
        raise ValueError("Invalid representation model or selection direction")
    if rep["bootstrap_samples"] < 100 or rep["projection"] != "pca":
        raise ValueError("Use at least 100 bootstrap samples and projection: pca")
    return cfg

def choose_validation_seed(runs, spec):
    """Read only original validation selection fields; never rank new-domain scores."""
    ranking = []
    for run in runs:
        if run["model"] != spec["model"]:
            continue
        cfg = run["config"]
        metric = cfg["multi_task"]["selection_metric"]
        mode = cfg.get("early_stopping", {}).get("mode") or ("min" if metric == "loss" else "max")
        path = Path(run["checkpoint"]).parent / "run_summary.json"
        summary = json.loads(path.read_text())
        if metric != spec["selection_metric"] or mode != spec["selection_mode"] or summary["selection_metric"] != metric:
            raise ValueError("Requested validation criterion differs from saved checkpoint selection")
        score = float(summary["best_val_score"])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite original validation score: {path}")
        ranking.append({"model": run["model"], "seed": run["seed"], "best_val_score": score,
                        "best_epoch": int(summary["best_epoch"]), "selection_metric": metric,
                        "selection_mode": mode, "source": str(path), "source_sha256": reference.sha(path)})
    if not ranking:
        raise ValueError("No candidate validation seeds")
    ranking.sort(key=lambda r: ((1 if spec["selection_mode"] == "min" else -1)*r["best_val_score"], r["seed"]))
    return ranking[0], ranking

def checked_image(dataset, row):
    relative = Path(row["rel_path_seg"])
    path = (dataset / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(dataset) or not path.is_file():
        raise ValueError(f"Missing or invalid segmented image: {relative}")
    return {**row, "file_bytes": str(path.stat().st_size), "file_mtime_ns": str(path.stat().st_mtime_ns)}

def original_truth(rows, prediction_path):
    """Map the original scored population to the relocated dataset without changing it."""
    lookup = {(r["barcode"], r["filename"]): r["image_id"] for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("Historical image keys are not unique")
    result = {task: {} for task in TASKS}
    for record in reference.read_csv(prediction_path):
        key = (record["individual_id"], record["filename"])
        if key not in lookup or record["task"] not in result:
            raise ValueError(f"Original-test reference cannot be mapped to metadata: {key}")
        task, image_id = record["task"], lookup[key]
        if image_id in result[task]:
            raise ValueError("Duplicate original-test prediction")
        result[task][image_id] = int(record["true_index"])
    if set().union(*(set(v) for v in result.values())) != {r["image_id"] for r in rows}:
        raise ValueError("Original test manifest differs from the original scored image population")
    return result

def make_plan(cfg):
    dataset, root = Path(cfg["dataset"]), Path(cfg["output"])
    for source in (dataset, Path(cfg["publication_results"]), *([Path(cfg["external_reference"])] if cfg.get("external_reference") else [])):
        if root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError("Analysis outputs must be separate from datasets and reference results")
    runs = reference.discover_checkpoints(Path(cfg["publication_results"]))
    selected, ranking = choose_validation_seed(runs, cfg["representation"])
    cohorts, excluded, manifest_sha = reference.camera_cohorts(dataset, cfg["year"])
    truth = None
    if cfg["include_original_test"]:
        rows = reference.read_csv(dataset / "metadata/images.csv")
        original = [r for r in rows if r["kind"] == "worm" and r["camera"] == "original_camera"
                    and r["original_paper_split"] == "test"]
        if not original or any(r["segmentation_status"] not in {"segmented", "reused"} for r in original):
            raise ValueError("The original test cohort must be complete")
        cohorts["original"] = [checked_image(dataset, r) for r in sorted(original, key=lambda r: r["image_id"])]
        for run in runs:
            actual = original_truth(original, run["original_predictions"])
            if truth is None:
                truth = actual
            elif actual != truth:
                raise ValueError("Original test identities or true labels differ between baseline seeds")
    all_rows = [row for rows in cohorts.values() for row in rows]
    if len({r["image_id"] for r in all_rows}) != len(all_rows):
        raise ValueError("Image IDs must be unique across domains")
    if any(Path(r["image_id"]).name != r["image_id"] or r["image_id"] in {"", ".", ".."} for r in all_rows):
        raise ValueError("Image IDs must be valid cache filename components")
    paired = sorted(set(r["individual_id"] for r in cohorts["gphoto2"]) & set(r["individual_id"] for r in cohorts["webcam"]))
    if len(paired) < 2:
        raise ValueError("At least two paired individuals are required")
    # Check individual labels before expensive inference.
    frame = reference.labeled_frame(all_rows, runs[0]["config"])
    for individual, group in frame.groupby("individual_id"):
        for column in ("barcode", "taxon", "life_stage", "genus", "species_label"):
            if group[column].fillna("").nunique() != 1:
                raise ValueError(f"Conflicting {column} metadata for {individual}")
    external_files = []
    if cfg.get("external_reference"):
        external_root = Path(cfg["external_reference"])
        external_plan = json.loads((external_root / "plan.json").read_text())
        external_runs = {(r["model"], r["seed"]): r for r in external_plan["runs"]}
        if set(external_runs) != {(r["model"], r["seed"]) for r in runs}:
            raise ValueError("Completed external reference uses a different checkpoint set")
        for camera in CAMERAS:
            if {r["image_id"] for r in external_plan["cohorts"][camera]} != {r["image_id"] for r in cohorts[camera]}:
                raise ValueError("Completed external reference uses a different camera cohort")
        for run in runs:
            external_run = external_runs[(run["model"], run["seed"])]
            if (external_run["label_maps"] != run["label_maps"]
                    or reference.protocol(external_run["config"]) != reference.protocol(run["config"])):
                raise ValueError("Completed external reference uses a different vocabulary or preprocessing")
            for camera in CAMERAS:
                directory = reference.run_dir(external_root, run, camera)
                if not reference.complete(directory, external_plan):
                    raise ValueError(f"Incomplete external reference: {directory}")
                path = directory / "predictions.csv"
                completion = json.loads((directory / "complete.json").read_text())
                external_files.append({"model": run["model"], "seed": run["seed"], "domain": camera,
                                       "path": str(path), "sha256": reference.sha(path),
                                       "checkpoint_sha256": completion["checkpoint_sha256"]})
    source_paths = [Path(__file__).resolve(), PROJECT / "scripts/run_publication_external_test.py",
                    PROJECT / "src/worm_species/domain_analysis.py",
                    *sorted((PROJECT / "src/worm_species/models").glob("*.py")),
                    *sorted((PROJECT / "src/worm_species/data").glob("*.py"))]
    # Device, batch size and worker count are runtime choices; scientific settings are frozen.
    scientific = {k: v for k, v in cfg.items() if k not in {"device", "batch_size", "workers", "cpu_threads"}}
    historical_tables = [{"path": str(p), "sha256": reference.sha(p), "stage": p.parent.name}
                         for p in sorted((Path(cfg["publication_results"]) / "runs").glob("*/multi_run_results.csv"))]
    plan = {"schema": 1, "settings": scientific, "dataset_root": str(dataset), "runs": runs,
            "cohorts": cohorts, "original_truth": truth, "paired_individuals": paired,
            "excluded": excluded, "manifest_sha256": manifest_sha, "selected": selected,
            "validation_ranking": ranking, "external_references": external_files,
            "historical_reference_tables": historical_tables,
            "source_sha256": {str(p): reference.sha(p) for p in source_paths}}
    plan["identity"] = reference.identity(plan)
    with reference.lock(root / ".plan.lock"):
        path = root / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError("This output belongs to a different analysis snapshot; choose a new output directory")
        reference.save_json(path, plan)
        reference.save_json(root / "selected_validation_seed.json", selected)
        reference.save_json(root / "reference_inventory.json", {"publication_results": cfg["publication_results"],
                                                                 "historical_tables": historical_tables,
                                                                 "external_reference": cfg.get("external_reference")})
        reference.save_csv(root / "validation_ranking.csv", ranking)
        for domain, rows in cohorts.items():
            reference.save_csv(root / "cohorts" / f"{domain}.csv", rows)
        reference.save_csv(root / "cohorts/excluded.csv", excluded,
                           list(cohorts["gphoto2"][0]) + ["exclusion_reason"])
    print(f"{len(runs)} checkpoints × {len(cohorts)} domains × {len(INPUTS)} inputs; no training")
    print(f"Paired individuals: {len(paired)}; best validation seed: {selected['model']}/{selected['seed']}")
    print(f"Frozen plan: {root / 'plan.json'}")
    return plan

def load_plan(root):
    return reference.load_plan(root)

def directory(root, run, domain, input_name):
    return root / "predictions" / run["model"] / f"seed_{run['seed']}" / domain / input_name

def has_embeddings(plan, run, domain):
    return domain in CAMERAS and (run["model"], run["seed"]) == (plan["selected"]["model"], plan["selected"]["seed"])

def complete(path, plan, embeddings=False):
    try:
        record = json.loads((path / "complete.json").read_text())
        expected = {"predictions.csv", "embeddings.npz"} if embeddings else {"predictions.csv"}
        return (record["plan_identity"] == plan["identity"] and set(record["files"]) == expected
                and all((path / name).is_file() and reference.sha(path / name) == checksum
                        for name, checksum in record["files"].items()))
    except (OSError, ValueError, KeyError, TypeError):
        return False

class FeatureHooks:
    """Early/middle global pools and the actual input to a classification head."""
    MODULES = {"convnext_base": ("backbone.features.1", "backbone.features.5"),
               "resnet50": ("backbone.layer1", "backbone.layer3"),
               "vit_b_16": ("backbone.encoder.layers.encoder_layer_2", "backbone.encoder.layers.encoder_layer_5")}

    def __init__(self, model, architecture):
        self.values, self.handles = {}, []
        for name, path in zip(("early", "intermediate"), self.MODULES[architecture]):
            self.handles.append(model.get_submodule(path).register_forward_hook(self.capture(name)))
        head = next(iter(model.heads.values()))
        self.handles.append(head.register_forward_pre_hook(lambda module, inputs: self.store("final", inputs[0])))

    def store(self, name, tensor):
        if tensor.ndim == 4:
            tensor = tensor.mean(dim=(2, 3))
        elif tensor.ndim == 3:
            tensor = tensor[:, 1:, :].mean(dim=1)  # ViT patch tokens, excluding CLS.
        if tensor.ndim != 2:
            raise ValueError(f"Unsupported feature shape for {name}: {tensor.shape}")
        self.values[name] = tensor.detach().float().cpu().numpy()

    def capture(self, name):
        def hook(module, inputs, output):
            self.store(name, output)
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()

def cache_images(root, plan):
    """Cache lossless, once-resized RGB images; greyscale remains a tensor transform."""
    from PIL import Image
    cache = root / ".cache/resized_rgb"
    cache.mkdir(parents=True, exist_ok=True)
    size = plan["runs"][0]["config"]["preprocessing"]["image_size"]
    rows = [r for records in plan["cohorts"].values() for r in records]
    with reference.lock(root / ".locks/cache.lock"):
        marker = cache / "manifest.json"
        record = json.loads(marker.read_text()) if marker.exists() else {}
        if record and record.get("plan_identity") != plan["identity"]:
            raise ValueError("Image cache belongs to a different plan")
        stamps = record.get("images", {})
        dirty = False
        for index, row in enumerate(rows):
            source = Path(plan["dataset_root"]) / row["rel_path_seg"]
            if reference.stamp(source) != {"bytes": int(row["file_bytes"]), "mtime_ns": int(row["file_mtime_ns"])}:
                raise ValueError("Dataset image changed after planning")
            dest = cache / (row["image_id"] + ".png")
            if not dest.is_file() or stamps.get(row["image_id"]) != reference.stamp(dest):
                temp = dest.with_suffix(".partial.png")
                with Image.open(source) as image:
                    image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR).save(temp)
                temp.replace(dest)
                stamps[row["image_id"]] = reference.stamp(dest)
                dirty = True
            if (index+1) % 100 == 0 and dirty:
                reference.save_json(marker, {"plan_identity": plan["identity"], "images": stamps})
                dirty = False
                print(f"Caching resized images: {index+1}/{len(rows)}", flush=True)
        if dirty or not marker.exists():
            reference.save_json(marker, {"plan_identity": plan["identity"], "images": stamps})
    return cache

def infer(model, run, plan, root, domain, input_name, runtime, device, checkpoint_hash, cache):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from worm_species.data.datasets import MultiTaskWormImageDataset
    from worm_species.data.transforms import build_split_transform
    path = directory(root, run, domain, input_name)
    path.mkdir(parents=True, exist_ok=True)
    (path / "complete.json").unlink(missing_ok=True)
    cfg, maps = run["config"], run["label_maps"]
    frame = reference.labeled_frame(plan["cohorts"][domain], cfg)
    for row in frame.to_dict("records"):
        actual = reference.stamp(Path(plan["dataset_root"]) / row["rel_path_seg"])
        if actual != {"bytes": int(row["file_bytes"]), "mtime_ns": int(row["file_mtime_ns"])}:
            raise ValueError("Dataset image changed after planning")
    transform = build_split_transform(split="test", preprocessing=cfg["preprocessing"],
                                      condition={"transform": "original" if input_name == "rgb" else "grayscale"},
                                      original_colour_retention=cfg["data"].get("colour_retention", 1.0))
    frame["cached_image"] = frame.image_id + ".png"
    data = MultiTaskWormImageDataset(frame, root_dir=cache, image_col="cached_image",
                                    transform=transform, crop_to_foreground=False,
                                    target_cols=cfg["data"]["target_cols"], label_to_index_by_task=maps)
    loader = DataLoader(data, batch_size=runtime["batch_size"], shuffle=False, num_workers=runtime["workers"],
                        pin_memory=device.type == "cuda")
    hooks = FeatureHooks(model, run["model"]) if has_embeddings(plan, run, domain) else None
    feature_batches, records, offset = {}, [], 0
    names = {task: [label for label, index in sorted(mapping.items(), key=lambda kv: kv[1])] for task, mapping in maps.items()}
    try:
        with torch.inference_mode():
            for batch in loader:
                if hooks:
                    hooks.values.clear()
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda" and cfg["training"].get("use_amp", True)):
                    outputs = model(batch["image"].to(device, non_blocking=True))
                for task in TASKS:
                    logits = outputs[task].float().cpu().numpy()
                    probs = outputs[task].float().softmax(dim=1).cpu().numpy()
                    if not np.isfinite(logits).all() or not np.isfinite(probs).all():
                        raise ValueError("Non-finite model output")
                    if hooks:
                        feature_batches.setdefault("probabilities_" + task, []).append(probs)
                    for j, true in enumerate(batch["labels"][task].numpy()):
                        row = frame.iloc[offset+j]
                        if domain == "original":
                            recorded = plan["original_truth"][task].get(row.image_id, -1)
                            if recorded >= 0 and int(true) != recorded:
                                raise ValueError("Historical truth differs from the saved reference")
                            true = recorded
                        value = row[cfg["data"]["target_cols"][task]]
                        label = "" if pd.isna(value) else str(value)
                        pred = int(probs[j].argmax())
                        records.append({"model": run["model"], "seed": run["seed"], "domain": domain,
                                        "input": input_name, "task": task, "image_id": row.image_id,
                                        "individual_id": row.individual_id, "barcode": row.barcode,
                                        "filename": row.filename, "location_code": row.location_code,
                                        "taxon": row.taxon, "genus": "" if pd.isna(row.genus) else str(row.genus),
                                        "life_stage": "" if pd.isna(row.life_stage) else str(row.life_stage),
                                        "true_label": label, "true_index": int(true), "predicted_index": pred,
                                        "predicted_label": names[task][pred],
                                        "unscored_reason": "" if true >= 0 else "original_reference_unscored" if domain == "original" else
                                        "missing_or_uncertain_label" if not label else "outside_checkpoint_vocabulary",
                                        "probabilities_json": json.dumps(probs[j].tolist()),
                                        "logits_json": json.dumps(logits[j].tolist())})
                if hooks:
                    if set(hooks.values) != {"early", "intermediate", "final"}:
                        raise ValueError("Required representation hooks did not all execute")
                    for name, values in hooks.values.items():
                        feature_batches.setdefault(name, []).append(values)
                offset += len(batch["image"])
    finally:
        if hooks:
            hooks.close()
    if offset != len(frame):
        raise ValueError("Incomplete image inference")
    reference.save_csv(path / "predictions.csv", records)
    files = ["predictions.csv"]
    if hooks:
        np.savez_compressed(path / "embeddings.npz", image_id=frame.image_id.to_numpy(dtype=str),
                            **{name: np.concatenate(values) for name, values in feature_batches.items()})
        files.append("embeddings.npz")
    runtime_record = {**runtime, "python": sys.version.split()[0], "torch": str(torch.__version__),
                      "cuda_build": torch.version.cuda, "numpy": np.__version__,
                      "actual_device": str(device), "amp": device.type == "cuda" and cfg["training"].get("use_amp", True)}
    reference.save_json(path / "complete.json", {"plan_identity": plan["identity"], "checkpoint_sha256": checkpoint_hash,
                                                "runtime": runtime_record, "images": len(frame),
                                                "files": {name: reference.sha(path / name) for name in files}})
    print(f"Completed {run['model']}/{run['seed']} {domain}/{input_name}: {len(frame)} images", flush=True)

def worker(root, plan, index, runtime):
    import torch
    from worm_species.models.multitask import build_multitask_model
    if not 0 <= index < len(plan["runs"]):
        raise ValueError("Checkpoint index outside the plan")
    run = plan["runs"][index]
    with reference.lock(root / ".locks" / f"run_{index}.lock"):
        pending = [(domain, condition) for domain in plan["cohorts"] for condition in INPUTS
                   if not complete(directory(root, run, domain, condition), plan, has_embeddings(plan, run, domain))]
        if not pending:
            print(f"Checkpoint {index} complete; skipping", flush=True)
            return
        device = torch.device(runtime["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Set device: cpu in the local YAML or fix the local GPU runtime")
        torch.set_num_threads(runtime["cpu_threads"])
        torch.manual_seed(run["seed"])
        if reference.stamp(run["checkpoint"]) != run["checkpoint_stamp"]:
            raise ValueError("Checkpoint changed after planning")
        cache = cache_images(root, plan)
        checkpoint_hash = reference.sha(run["checkpoint"])
        for saved in plan["external_references"]:
            if (saved["model"], saved["seed"]) == (run["model"], run["seed"]) and saved["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Checkpoint weights differ from the completed external reference")
        ckpt = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        if (cfg["model"]["name"] != run["model"] or cfg["seed"] != run["seed"]
                or reference.protocol(cfg) != reference.protocol(run["config"])
                or ckpt["label_to_index_by_task"] != run["label_maps"]):
            raise ValueError("Saved checkpoint differs from planned config or vocabulary")
        if has_embeddings(plan, run, "gphoto2") and (float(ckpt["best_val_score"]) != plan["selected"]["best_val_score"]
                                                       or int(ckpt["best_epoch"]) != plan["selected"]["best_epoch"]):
            raise ValueError("Validation selection provenance differs from the saved checkpoint")
        cfg = copy.deepcopy(cfg)
        cfg["model"]["pretrained"] = False
        model = build_multitask_model(cfg, {task: len(labels) for task, labels in run["label_maps"].items()})
        model.load_state_dict(ckpt["model_state"], strict=True)
        del ckpt
        model.to(device).eval()
        for domain, condition in pending:
            infer(model, run, plan, root, domain, condition, runtime, device, checkpoint_hash, cache)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

def status(root, plan):
    result = {"completed": 0, "expected": len(plan["runs"])*len(plan["cohorts"])*len(INPUTS), "pending_checkpoint_indices": []}
    for index, run in enumerate(plan["runs"]):
        flags = [complete(directory(root, run, domain, condition), plan, has_embeddings(plan, run, domain))
                 for domain in plan["cohorts"] for condition in INPUTS]
        result["completed"] += sum(flags)
        if not all(flags):
            result["pending_checkpoint_indices"].append(index)
    return result
