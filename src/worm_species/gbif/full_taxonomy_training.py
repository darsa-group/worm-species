"""Controlled full-taxonomy Petri/GBIF training and inference primitives."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from ..data.datasets import MultiTaskWormImageDataset
from ..data.labels import MISSING_LABEL
from ..data.transforms import build_split_transform
from ..models.multitask import build_multitask_model
from ..training.losses import build_child_to_parent_matrix
from ..training.losses import ground_truth_taxonomic_mass_loss
from .full_taxonomy import GBIF_TASKS, PETRI_TASKS, atomic_json, file_sha256


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _head_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.heads.state_dict().items()}


def ensure_gbif_head_initialisation(
    config: dict, model: nn.Module, model_name: str, seed: int, label_maps: dict
) -> tuple[Path, str]:
    root = Path(config["paths"]["experiment_root"])
    path = root / "head_initialisations" / model_name / f"seed-{seed}.pt"
    lock_path = path.with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not path.is_file():
            offset = int(config["training"]["rng_offsets"]["head"])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed) + offset)
                for task in sorted(model.heads):
                    model.heads[task].reset_parameters()
            state = _head_state(model)
            payload = {
                "schema_version": 1, "model": model_name, "seed": int(seed),
                "label_maps": label_maps, "head_state": state,
                "head_tensor_sha256": _tensor_state_hash(state),
            }
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            torch.save(payload, temporary)
            os.replace(temporary, path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["model"] != model_name or int(payload["seed"]) != int(seed):
        raise ValueError("GBIF head initialisation identity mismatch")
    if payload["label_maps"] != label_maps:
        raise ValueError("GBIF head initialisation label maps changed")
    observed = _tensor_state_hash(payload["head_state"])
    if observed != payload["head_tensor_sha256"]:
        raise ValueError("GBIF head initialisation tensor hash is invalid")
    model.heads.load_state_dict(payload["head_state"], strict=True)
    return path, observed


def _load_backbone_only(model: nn.Module, checkpoint_path: Path) -> str:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {
        key[len("backbone."):]: value
        for key, value in payload["model_state"].items()
        if key.startswith("backbone.")
    }
    if not state:
        raise ValueError("Petri checkpoint contains no backbone tensors")
    model.backbone.load_state_dict(state, strict=True)
    return _tensor_state_hash(state)


def _loader(
    frame: pd.DataFrame, tasks: dict[str, str], label_maps: dict, transform,
    config: dict, *, shuffle: bool, seed: int,
) -> DataLoader:
    workers = int(config["training"]["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    dataset = MultiTaskWormImageDataset(
        frame, root_dir="/", image_col="image_path", target_cols=tasks,
        label_to_index_by_task=label_maps, transform=transform,
        crop_to_foreground=False,
    )
    kwargs = {
        "batch_size": int(config["training"]["batch_size"]),
        "shuffle": shuffle, "num_workers": workers,
        "pin_memory": bool(config["training"]["pin_memory"]),
        "persistent_workers": bool(config["training"]["persistent_workers"]),
        "generator": generator, "drop_last": shuffle,
    }
    if workers:
        kwargs["prefetch_factor"] = int(config["training"]["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def _criteria(frame: pd.DataFrame, tasks: dict, maps: dict, config: dict, device) -> dict:
    result = {}
    for task, column in tasks.items():
        counts = frame.loc[frame[column].isin(maps[task]), column].value_counts()
        if bool(config["training"]["class_weight"]):
            weights = torch.tensor(
                [1.0 / max(float(counts.get(label, 0)), 1.0) for label in maps[task]],
                dtype=torch.float32, device=device,
            )
            weights /= weights.mean()
        else:
            weights = None
        result[task] = nn.CrossEntropyLoss(weight=weights)
    return result


def _scheduler(optimizer, warmup: int, total: int):
    def schedule(step: int) -> float:
        if step < warmup:
            return max(step, 1) / max(warmup, 1)
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _evaluate(model, loader, tasks: dict, criteria: dict, weights: dict, device) -> dict:
    model.eval()
    losses = []
    truths = {task: [] for task in tasks}
    predictions = {task: [] for task in tasks}
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = {task: batch["labels"][task].to(device, non_blocking=True) for task in tasks}
            outputs = model(images)
            total = torch.zeros((), device=device)
            active = 0.0
            for task in tasks:
                valid = labels[task].ne(MISSING_LABEL)
                if not valid.any():
                    continue
                loss = criteria[task](outputs[task][valid], labels[task][valid])
                total += float(weights[task]) * loss
                active += float(weights[task])
                truths[task].extend(labels[task][valid].cpu().tolist())
                predictions[task].extend(outputs[task][valid].argmax(1).cpu().tolist())
            if active:
                losses.append(float((total / active).item()))
    result = {"loss": float(np.mean(losses)) if losses else None}
    for task in tasks:
        true, pred = truths[task], predictions[task]
        if not true:
            result[task] = {"n": 0, "accuracy": None, "balanced_accuracy": None, "macro_f1": None}
            continue
        result[task] = {
            "n": len(true), "observed_classes": len(set(true)),
            "accuracy": float(accuracy_score(true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        }
    return result


def _cycle(loader):
    while True:
        yield from loader


def stage_complete(spec: dict) -> bool:
    output = Path(spec["output_dir"])
    status = output / "run_status.json"
    if not status.is_file() or not (output / "best_model.pt").is_file():
        return False
    try:
        return json.loads(status.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def train_full_taxonomy_stage(config: dict, spec: dict) -> dict:
    output = Path(spec["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if stage_complete(spec):
        return json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("Full-taxonomy training requires CUDA")
    audit_path = Path(config["paths"]["experiment_root"]) / "audit" / "audit_manifest.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Immutable Phase-A audit is missing")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete" or int(audit.get("fatal_leakage_rows", -1)) != 0:
        raise RuntimeError("Phase-A integrity gates did not pass")
    prepared = Path(config["paths"]["experiment_root"]) / "prepared"
    domain = spec["domain"]
    tasks = PETRI_TASKS if domain == "petri" else GBIF_TASKS
    maps_path = prepared / f"{domain}_label_maps.json"
    label_maps = json.loads(maps_path.read_text(encoding="utf-8"))
    frames = {
        split: pd.read_csv(prepared / f"{domain}_{split}.csv", dtype=str, keep_default_na=False)
        for split in ("train", "validation", "test")
    }
    preprocessing = {
        "image_size": int(config["data"]["image_size"]),
        "normalisation": {
            "enabled": True, "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    train_transform = build_split_transform(
        split="train", preprocessing=preprocessing,
        augmentation=config["training"]["augmentation"],
        condition={"transform": "original"},
    )
    eval_transform = build_split_transform(
        split="validation", preprocessing=preprocessing,
        augmentation=config["training"]["augmentation"],
        condition={"transform": "original"}, apply_augmentation=False,
    )
    offsets = config["training"]["rng_offsets"]
    seed = int(spec["seed"])
    data_seed = seed + int(offsets["data"])
    train_loader = _loader(frames["train"], tasks, label_maps, train_transform, config, shuffle=True, seed=data_seed)
    validation_loader = _loader(frames["validation"], tasks, label_maps, eval_transform, config, shuffle=False, seed=data_seed)
    test_loader = _loader(frames["test"], tasks, label_maps, eval_transform, config, shuffle=False, seed=data_seed)
    device = torch.device("cuda")
    _seed_all(seed + int(offsets["training"]))
    pretrained = bool(config["models"]["imagenet_pretrained"]) and spec["initialisation"] == "imagenet"
    model = build_multitask_model(
        {"model": {"name": spec["model"], "pretrained": pretrained}},
        {task: len(mapping) for task, mapping in label_maps.items()},
    )
    backbone_hash = None
    if spec["initialisation"] == "petri_backbone":
        checkpoint = Path(spec["petri_checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        backbone_hash = _load_backbone_only(model, checkpoint)
    head_path = head_hash = None
    if domain == "gbif":
        head_path, head_hash = ensure_gbif_head_initialisation(
            config, model, spec["model"], seed, label_maps
        )
    model.to(device)
    criteria = _criteria(frames["train"], tasks, label_maps, config, device)
    weights = config["training"]["loss_weights"][domain]
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": float(config["training"]["backbone_lr"])},
        {"params": model.heads.parameters(), "lr": float(config["training"]["head_lr"])},
    ], weight_decay=float(config["training"]["weight_decay"]))
    total_steps = int(spec["max_steps"])
    scheduler = _scheduler(optimizer, int(config["training"]["warmup_steps"]), total_steps)
    scaler = torch.amp.GradScaler(enabled=bool(config["training"]["use_amp"]))
    matrix = None
    if spec.get("hierarchy_kind") == "ground_truth":
        matrix = build_child_to_parent_matrix(label_maps, "genus", "species", device)
    resume = output / "last_model.pt"
    step = 0
    best_key = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    best_step = 0
    if resume.is_file():
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload["label_maps"] != label_maps or payload["spec"]["run_id"] != spec["run_id"]:
            raise ValueError("Resume checkpoint identity mismatch")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        scaler.load_state_dict(payload["scaler_state"])
        step = int(payload["step"])
        best_key = tuple(payload["best_key"])
        best_step = int(payload["best_step"])
    atomic_json(output / "spec.json", spec)
    if domain == "gbif":
        atomic_json(output / "step0_head_audit.json", {
            "head_initialisation": str(head_path), "head_tensor_sha256": head_hash,
            "petri_backbone_tensor_sha256": backbone_hash,
            "condition": spec["condition"], "model": spec["model"], "seed": seed,
        })
    iterator = _cycle(train_loader)
    training_seed = seed + int(offsets["augmentation"])
    _seed_all(training_seed)
    validation_interval = int(config["training"]["validation_interval_steps"])
    checkpoint_interval = int(config["training"]["checkpoint_interval_steps"])
    history = output / "history.jsonl"
    while step < total_steps:
        model.train()
        batch = next(iterator)
        images = batch["image"].to(device, non_blocking=True)
        labels = {task: batch["labels"][task].to(device, non_blocking=True) for task in tasks}
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=bool(config["training"]["use_amp"])):
            outputs = model(images)
            total = torch.zeros((), device=device)
            active_weight = 0.0
            task_losses = {}
            for task in tasks:
                valid = labels[task].ne(MISSING_LABEL)
                if not valid.any():
                    continue
                loss = criteria[task](outputs[task][valid], labels[task][valid])
                total += float(weights[task]) * loss
                active_weight += float(weights[task])
                task_losses[task] = float(loss.detach().item())
            if matrix is not None:
                valid = labels["genus"].ne(MISSING_LABEL)
                hierarchy = ground_truth_taxonomic_mass_loss(
                    outputs["species"], labels["genus"], matrix, valid
                )
                if hierarchy is not None:
                    hierarchy_weight = float(config["training"]["revised_hierarchy"]["weight"])
                    total += hierarchy_weight * hierarchy
                    active_weight += hierarchy_weight
                    task_losses["hierarchy"] = float(hierarchy.detach().item())
            if active_weight == 0:
                raise ValueError("Training batch contains no active labels")
            total /= active_weight
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        record = {"step": step, "train_loss": float(total.detach().item()), **{f"train_{key}_loss": value for key, value in task_losses.items()}}
        validate = step % validation_interval == 0 or step == total_steps
        improved = False
        if validate:
            validation = _evaluate(model, validation_loader, tasks, criteria, weights, device)
            species_f1 = float(validation["species"]["macro_f1"])
            genus_f1 = float(validation["genus"]["macro_f1"])
            selection_key = (species_f1, genus_f1, -float(validation["loss"]), -float(step))
            improved = selection_key > best_key
            if improved:
                best_key, best_step = selection_key, step
            record["validation"] = validation
            record["selection_key"] = list(selection_key)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if validate or step % checkpoint_interval == 0:
            payload = {
                "schema_version": 1, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(), "step": step, "best_key": list(best_key),
                "best_step": best_step, "label_maps": label_maps, "tasks": tasks,
                "spec": spec, "preprocessing": preprocessing, "head_tensor_sha256": head_hash,
            }
            temporary = output / f".last_model.pt.tmp-{os.getpid()}"
            torch.save(payload, temporary)
            os.replace(temporary, resume)
            if improved:
                temporary = output / f".best_model.pt.tmp-{os.getpid()}"
                torch.save(payload, temporary)
                os.replace(temporary, output / "best_model.pt")
    best = torch.load(output / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state"], strict=True)
    test_metrics = _evaluate(model, test_loader, tasks, criteria, weights, device)
    atomic_json(output / "test_metrics.json", test_metrics)
    status = {
        "status": "complete", "run_id": spec["run_id"], "best_step": best_step,
        "best_validation_species_macro_f1": best_key[0],
        "best_validation_genus_macro_f1": best_key[1],
        "head_tensor_sha256": head_hash, "fixed_budget_steps": total_steps,
    }
    atomic_json(output / "run_status.json", status)
    return status
