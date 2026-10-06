"""Native RGB feature extraction from saved ConvNeXt checkpoints."""

from __future__ import annotations

import argparse

import copy

import json

import os

from pathlib import Path

import sys

PROJECT = Path(__file__).resolve().parents[1]

for p in (PROJECT, PROJECT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

import pandas as pd

from scripts import run_publication_external_test as io

from scripts import run_publication_domain_analysis as previous

from worm_species import domain_transfer as transfer

TASKS = io.TASKS

PREPROCESSING = ("native",)

def metadata_frame(base):
    records = [{**r, "domain": domain} for domain, rows in base["cohorts"].items() for r in rows]
    frame = previous.reference.labeled_frame(records, base["runs"][0]["config"]).fillna("")
    frame["biological_group"] = [r.species_label if r.species_label else "unresolved:"+r.taxon for r in frame.itertuples()]
    frame["group_kind"] = np.where(frame.species_label.eq(""), "unresolved_taxon", "resolved_species")
    return frame

def run_root(root, run):
    return root / run["training_condition"] / f"seed_{run['seed']}"

def complete(path, identity):
    try:
        record = json.loads((path / "complete.json").read_text())
        return record["plan_identity"] == identity and bool(record["files"]) and all(
            (path / name).is_file() and io.sha(path / name) == sha for name, sha in record["files"].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False

def mark_complete(path, plan, files, **extra):
    io.save_json(path / "complete.json", {"plan_identity": plan["identity"],
                                         "files": {name: io.sha(path / name) for name in files}, **extra})

def extract_features(root, plan, run, cfg):
    import torch
    from torch.utils.data import DataLoader
    from worm_species.data.datasets import MultiTaskWormImageDataset
    from worm_species.data.transforms import build_split_transform
    from worm_species.models.multitask import build_multitask_model
    base = run_root(root, run) / "features"
    pending = [(domain, preprocessing) for domain in plan["cohorts"] for preprocessing in PREPROCESSING
               if not complete(base / domain / preprocessing, plan["identity"])]
    if not pending:
        return
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available to this process; use the local GPU terminal or set device: cpu")
    torch.set_num_threads(cfg["cpu_threads"])
    if io.stamp(run["checkpoint"]) != run["checkpoint_stamp"]:
        raise ValueError("Checkpoint changed after planning")
    cache = previous.cache_images(root, plan)
    ckpt = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    if (ckpt["cfg"] != run["config"] or ckpt["label_to_index_by_task"] != run["label_maps"]):
        raise ValueError("Checkpoint configuration or labels differ from discovery")
    model_cfg = copy.deepcopy(ckpt["cfg"]); model_cfg["model"]["pretrained"] = False
    model = build_multitask_model(model_cfg, {t: len(v) for t, v in run["label_maps"].items()})
    model.load_state_dict(ckpt["model_state"], strict=True)
    del ckpt
    model.to(device).eval()
    weights = {f"{task}_{name}": getattr(model.heads[task], name).detach().float().cpu().numpy()
               for task in TASKS for name in ("weight", "bias")}
    meta = pd.DataFrame(plan["metadata"])
    for domain, preprocessing in pending:
        path = base / domain / preprocessing
        path.mkdir(parents=True, exist_ok=True)
        (path / "complete.json").unlink(missing_ok=True)
        frame = meta.loc[meta.domain.eq(domain)].reset_index(drop=True)
        frame["cached_image"] = frame.image_id + ".png"
        transform = build_split_transform(split="test", preprocessing=run["config"]["preprocessing"],
                                          condition={"transform": "original"})
        dataset = MultiTaskWormImageDataset(frame, root_dir=cache, image_col="cached_image", transform=transform,
                                            crop_to_foreground=False, target_cols=run["config"]["data"]["target_cols"],
                                            label_to_index_by_task=run["label_maps"])
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["workers"], pin_memory=device.type=="cuda")
        hooks = previous.FeatureHooks(model, "convnext_base")
        batches = {}
        try:
            with torch.inference_mode():
                for batch in loader:
                    hooks.values.clear()
                    with torch.amp.autocast(device_type=device.type, enabled=device.type=="cuda" and run["config"]["training"].get("use_amp", True)):
                        outputs = model(batch["image"].to(device))
                    for name, values in hooks.values.items():
                        batches.setdefault(name, []).append(values)
                    for task in TASKS:
                        batches.setdefault("probabilities_"+task, []).append(outputs[task].float().softmax(dim=1).cpu().numpy())
        finally:
            hooks.close()
        arrays = {name: np.concatenate(values) for name, values in batches.items()}
        if len(arrays["final"]) != len(frame) or any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Incomplete or non-finite feature extraction")
        np.savez_compressed(path / "features.npz", image_id=frame.image_id.to_numpy(dtype=str), **arrays, **weights)
        io.save_csv(path / "metadata.csv", frame.to_dict("records"))
        mark_complete(path, plan, ["features.npz", "metadata.csv"], torch_version=str(torch.__version__), device=str(device),
                      checkpoint_sha256=io.sha(run["checkpoint"]), runtime={k: cfg[k] for k in ("batch_size", "workers", "cpu_threads")})
        print(f"Extracted {run['training_condition']}/{run['seed']} {domain}/{preprocessing}", flush=True)
        if cfg.get("_tracker"):
            cfg["_tracker"].event(**{"current/stage":"feature_extraction", "current/seed":run["seed"],
                                      "current/training_condition":run["training_condition"],
                                      "current/domain":domain, "current/preprocessing":preprocessing,
                                      "current/images":len(frame)})
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
