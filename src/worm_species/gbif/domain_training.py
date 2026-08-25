"""Step-balanced supervised training across curated GBIF and Petri domains."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Sampler

from ..data.datasets import MultiTaskWormImageDataset
from ..data.labels import MISSING_LABEL
from ..data.transforms import build_split_transform
from ..models.multitask import build_multitask_model
from .domain_data import DOMAINS, TASK_COLUMNS, file_sha256, prepared_paths


_STOP_REQUESTED = False


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"Received signal {signum}; checkpointing at the next completed step.", flush=True)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    frame: pd.DataFrame,
    label_maps: dict,
    transform,
    config: dict,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers_override: int | None = None,
    persistent_workers_override: bool | None = None,
) -> DataLoader:
    training = config["training"]
    workers = (
        int(num_workers_override)
        if num_workers_override is not None
        else int(training["num_workers"])
    )
    generator = torch.Generator().manual_seed(seed)
    dataset = MultiTaskWormImageDataset(
        frame,
        root_dir="/",
        image_col="image_path",
        target_cols=TASK_COLUMNS,
        label_to_index_by_task=label_maps,
        transform=transform,
        crop_to_foreground=False,
    )
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", True)),
        "generator": generator,
        "drop_last": shuffle,
    }
    if workers:
        kwargs.update(
            prefetch_factor=int(training["prefetch_factor"]),
            persistent_workers=(
                bool(persistent_workers_override)
                if persistent_workers_override is not None
                else bool(training.get("persistent_workers", True))
            ),
        )
    return DataLoader(dataset, **kwargs)


class _BalancedDomainBatchSampler(Sampler[list[int]]):
    def __init__(self, gbif_size: int, petri_size: int, per_domain: int, steps: int, seed: int):
        self.gbif_size = gbif_size
        self.petri_size = petri_size
        self.per_domain = per_domain
        self.steps = steps
        self.seed = seed

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.steps):
            gbif = torch.randint(self.gbif_size, (self.per_domain,), generator=generator)
            petri = torch.randint(self.petri_size, (self.per_domain,), generator=generator) + self.gbif_size
            indices = torch.cat([gbif, petri])
            order = torch.randperm(len(indices), generator=generator)
            yield indices[order].tolist()


def _mixed_loader(frames: dict[str, pd.DataFrame], label_maps: dict, transform, config: dict, seed: int) -> DataLoader:
    training = config["training"]
    gbif = frames["gbif"].reset_index(drop=True)
    petri = frames["petri"].reset_index(drop=True)
    combined = pd.concat([gbif, petri], ignore_index=True)
    dataset = MultiTaskWormImageDataset(
        combined, root_dir="/", image_col="image_path", target_cols=TASK_COLUMNS,
        label_to_index_by_task=label_maps, transform=transform,
        crop_to_foreground=False,
    )
    sampler = _BalancedDomainBatchSampler(
        len(gbif), len(petri), int(training["mixed_batch_per_domain"]),
        int(training["mixed_steps"]), seed,
    )
    return DataLoader(
        dataset, batch_sampler=sampler, num_workers=int(training["num_workers"]),
        pin_memory=bool(training.get("pin_memory", True)),
        prefetch_factor=int(training["prefetch_factor"]),
        persistent_workers=bool(training.get("persistent_workers", True)),
    )


def _cycle(loader: DataLoader):
    while True:
        yield from loader


def _criteria(config: dict, train_frames: dict[str, pd.DataFrame], label_maps: dict, device) -> dict:
    combined = pd.concat(list(train_frames.values()), ignore_index=True)
    criteria = {}
    for task, column in TASK_COLUMNS.items():
        mapping = label_maps[task]
        valid = combined[column].isin(mapping)
        counts = combined.loc[valid, column].value_counts()
        if bool(config["training"].get("class_weight", True)):
            weights = torch.tensor(
                [1.0 / max(float(counts.get(label, 0)), 1.0) for label in mapping],
                dtype=torch.float32,
                device=device,
            )
            weights = weights / weights.mean()
        else:
            weights = None
        criteria[task] = nn.CrossEntropyLoss(weight=weights)
    return criteria


def _loss_for_batch(model, batch, criteria, config, device, scaler, optimizer=None):
    train = optimizer is not None
    images = batch["image"].to(device, non_blocking=True)
    labels = {
        task: batch["labels"][task].to(device, non_blocking=True)
        for task in TASK_COLUMNS
    }
    if train:
        optimizer.zero_grad(set_to_none=True)
    weights = config["training"]["loss_weights"]
    with torch.set_grad_enabled(train):
        with torch.amp.autocast(
            device_type=device.type,
            enabled=bool(config["training"]["use_amp"]) and device.type == "cuda",
        ):
            outputs = model(images)
            total = torch.zeros((), device=device)
            active_weight = 0.0
            task_losses = {}
            for task in TASK_COLUMNS:
                valid = labels[task].ne(MISSING_LABEL)
                if not valid.any():
                    continue
                task_loss = criteria[task](outputs[task][valid], labels[task][valid])
                weight = float(weights[task])
                total = total + weight * task_loss
                active_weight += weight
                task_losses[task] = float(task_loss.detach().item())
            if active_weight == 0:
                raise ValueError("A training batch contains no active task labels")
            total = total / active_weight
        if train:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
    return float(total.detach().item()), task_losses, outputs, labels


def _evaluate(model, loader, criteria, config, device) -> dict:
    model.eval()
    losses = []
    truths = {task: [] for task in TASK_COLUMNS}
    predictions = {task: [] for task in TASK_COLUMNS}
    dummy_scaler = torch.amp.GradScaler(enabled=False)
    with torch.inference_mode():
        for batch in loader:
            loss, _, outputs, labels = _loss_for_batch(
                model, batch, criteria, config, device, dummy_scaler, optimizer=None
            )
            losses.append(loss)
            for task in TASK_COLUMNS:
                valid = labels[task].ne(MISSING_LABEL)
                if valid.any():
                    truths[task].extend(labels[task][valid].cpu().tolist())
                    predictions[task].extend(outputs[task][valid].argmax(1).cpu().tolist())
    metrics = {"loss": float(np.mean(losses)) if losses else None}
    for task in TASK_COLUMNS:
        y_true = truths[task]
        y_pred = predictions[task]
        metrics[f"{task}_n"] = len(y_true)
        if not y_true:
            metrics.update({
                f"{task}_accuracy": None,
                f"{task}_balanced_accuracy": None,
                f"{task}_macro_f1": None,
            })
            continue
        observed = sorted(set(y_true))
        metrics.update({
            f"{task}_accuracy": float(accuracy_score(y_true, y_pred)),
            f"{task}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            f"{task}_macro_f1": float(f1_score(
                y_true, y_pred, labels=observed, average="macro", zero_division=0
            )),
        })
    return metrics


def _domain_metrics(model, loaders, criteria, config, device) -> dict:
    metrics = {
        domain: _evaluate(model, loader, criteria, config, device)
        for domain, loader in loaders.items()
    }
    values = []
    for domain in DOMAINS:
        for task in TASK_COLUMNS:
            value = metrics[domain].get(f"{task}_macro_f1")
            if value is not None:
                values.append(float(value))
    metrics["domain_balanced_macro_f1"] = float(np.mean(values)) if values else None
    return metrics


def _scheduler(optimizer, warmup_steps: int, total_steps: int):
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _checkpoint_payload(
    *, model, optimizer, scheduler, scaler, config, label_maps, spec,
    global_step, stage_step, best_score, best_step, stale,
) -> dict:
    inference_cfg = {
        "model": {"name": spec["model"], "pretrained": False},
        "preprocessing": {
            "image_size": int(config["data"]["image_size"]),
            "normalisation": {
                "enabled": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "augmentation": copy.deepcopy(config["training"]["augmentation"]),
    }
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "cfg": inference_cfg,
        "label_to_index_by_task": label_maps,
        "index_to_label_by_task": {
            task: {index: label for label, index in mapping.items()}
            for task, mapping in label_maps.items()
        },
        "best_val_score": best_score,
        "selection_metric": "domain_balanced_macro_f1",
        "best_epoch": None,
        "best_step": best_step,
        "global_step": global_step,
        "stage_step": stage_step,
        "stale_validations": stale,
        "experiment_spec": spec,
    }


def _save_checkpoint(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _wandb_run(config: dict, spec: dict, output: Path):
    settings = config.get("wandb", {})
    if not bool(settings.get("enabled", True)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B is enabled but wandb is not installed") from exc
    run_id = hashlib.sha256(spec["run_id"].encode("utf-8")).hexdigest()[:16]
    return wandb.init(
        project=settings["project"],
        entity=settings.get("entity") or None,
        group=settings["group"],
        name=spec["run_id"],
        id=run_id,
        resume="allow",
        mode=os.environ.get("WANDB_MODE", settings.get("mode", "online")),
        job_type=spec["stage"],
        config={"experiment": config, "spec": spec},
        dir=str(output),
        save_code=bool(settings.get("save_code", False)),
        tags=[spec["phase"], spec["regime"], spec["model"]],
    )


def train_stage(config: dict, spec: dict) -> dict:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    spec = copy.deepcopy(spec)
    output = Path(spec["output_dir"])
    status_path = output / "run_status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete" and (output / "last_model.pt").is_file():
            print(f"{spec['run_id']} is already complete; skipping.")
            return status
    output.mkdir(parents=True, exist_ok=True)
    _seed_everything(int(spec["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("GBIF training requires a CUDA GPU")
    device = torch.device("cuda")
    spec["runtime"] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    prepared = Path(config["paths"]["output_root"]) / "prepared"
    prepared_summary_path = prepared / "summary.json"
    if not prepared_summary_path.is_file():
        raise FileNotFoundError(prepared_summary_path)
    spec["prepared_summary_sha256"] = file_sha256(prepared_summary_path)
    label_maps = json.loads((prepared / "label_maps.json").read_text(encoding="utf-8"))
    frames = {
        domain: {
            split: pd.read_csv(prepared_paths(config, domain, split), dtype=str, keep_default_na=False)
            for split in ("train", "validation", "test")
        }
        for domain in DOMAINS
    }
    preprocessing = {
        "image_size": int(config["data"]["image_size"]),
        "normalisation": {
            "enabled": True,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }
    train_transform = build_split_transform(
        split="train",
        preprocessing=preprocessing,
        augmentation=config["training"]["augmentation"],
        condition={"transform": "original"},
    )
    eval_transform = build_split_transform(
        split="validation",
        preprocessing=preprocessing,
        augmentation=config["training"]["augmentation"],
        condition={"transform": "original"},
        apply_augmentation=False,
    )
    batch_size = int(config["training"]["batch_size"])
    if spec["domain"] == "mixed":
        train_loader = _mixed_loader(
            {domain: frames[domain]["train"] for domain in DOMAINS},
            label_maps, train_transform, config, int(spec["seed"]),
        )
        train_iterator = _cycle(train_loader)
    else:
        train_loader = _loader(
            frames[spec["domain"]]["train"], label_maps, train_transform, config,
            batch_size=batch_size, shuffle=True, seed=int(spec["seed"]),
        )
        train_iterator = _cycle(train_loader)
    validation_loaders = {
        domain: _loader(
            frames[domain]["validation"], label_maps, eval_transform, config,
            batch_size=batch_size, shuffle=False, seed=int(spec["seed"]),
            num_workers_override=2, persistent_workers_override=False,
        )
        for domain in DOMAINS
    }
    test_loaders = {
        domain: _loader(
            frames[domain]["test"], label_maps, eval_transform, config,
            batch_size=batch_size, shuffle=False, seed=int(spec["seed"]),
            num_workers_override=2, persistent_workers_override=False,
        )
        for domain in DOMAINS
    }
    initial = spec.get("initial_checkpoint")
    resume_path = output / "last_model.pt"
    checkpoint_path = resume_path if resume_path.is_file() else Path(initial) if initial else None
    model_cfg = {"model": {
        "name": spec["model"],
        "pretrained": bool(config["models"]["pretrained"]) and checkpoint_path is None,
    }}
    model = build_multitask_model(
        model_cfg, {task: len(mapping) for task, mapping in label_maps.items()}
    ).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": float(config["training"]["backbone_lr"])},
        {"params": model.heads.parameters(), "lr": float(config["training"]["head_lr"])},
    ], weight_decay=float(config["training"]["weight_decay"]))
    total_budget = int(config["training"]["steps_per_domain"]) * 2
    scheduler = _scheduler(optimizer, int(config["training"]["warmup_steps"]), total_budget)
    scaler = torch.amp.GradScaler(enabled=bool(config["training"]["use_amp"]))
    criteria = _criteria(config, {domain: frames[domain]["train"] for domain in DOMAINS}, label_maps, device)

    global_step = 0
    stage_step = 0
    best_score = -float("inf")
    best_step = 0
    stale = 0
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        spec["initial_checkpoint_sha256"] = file_sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint["label_to_index_by_task"] != label_maps:
            raise ValueError("Checkpoint fixed label maps do not match prepared label maps")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        global_step = int(checkpoint.get("global_step", 0))
        if checkpoint_path == resume_path:
            stage_step = int(checkpoint.get("stage_step", 0))
            best_score = float(checkpoint.get("best_val_score", best_score))
            best_step = int(checkpoint.get("best_step", 0))
            stale = int(checkpoint.get("stale_validations", 0))

    _atomic_json(output / "spec.json", spec)

    run = _wandb_run(config, spec, output)
    history_path = output / "history.jsonl"
    max_steps = int(spec["max_steps"])
    validation_interval = int(config["training"]["validation_interval_steps"])
    checkpoint_interval = int(config["training"]["checkpoint_interval_steps"])
    min_steps = int(config["training"]["minimum_steps_before_early_stopping"])
    patience = int(config["training"]["early_stopping_patience"])
    min_delta = float(config["training"]["early_stopping_min_delta"])
    stopped_early = False
    while stage_step < max_steps:
        model.train()
        batch = next(train_iterator)
        loss, task_losses, _, _ = _loss_for_batch(
            model, batch, criteria, config, device, scaler, optimizer=optimizer
        )
        scheduler.step()
        stage_step += 1
        global_step += 1
        record = {
            "stage_step": stage_step, "global_step": global_step,
            "train_loss": loss, **{f"train_{task}_loss": value for task, value in task_losses.items()},
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        validate = stage_step % validation_interval == 0 or stage_step == max_steps
        if validate:
            validation = _domain_metrics(model, validation_loaders, criteria, config, device)
            score = float(validation["domain_balanced_macro_f1"])
            record["validation"] = validation
            improved = score > best_score + min_delta
            if improved:
                best_score = score
                best_step = stage_step
                stale = 0
            else:
                stale += 1
            record.update(best_score=best_score, best_step=best_step, stale_validations=stale)
            print(
                f"[{spec['run_id']}] step {stage_step}/{max_steps} "
                f"loss={loss:.4f} validation={score:.4f} stale={stale}/{patience}",
                flush=True,
            )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if run is not None:
            flat = {key: value for key, value in record.items() if not isinstance(value, dict)}
            if validate:
                for domain, values in validation.items():
                    if isinstance(values, dict):
                        for key, value in values.items():
                            if value is not None:
                                flat[f"validation/{domain}/{key}"] = value
                flat["validation/domain_balanced_macro_f1"] = score
            run.log(flat, step=global_step)

        should_checkpoint = validate or stage_step % checkpoint_interval == 0 or _STOP_REQUESTED
        if should_checkpoint:
            payload = _checkpoint_payload(
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                config=config, label_maps=label_maps, spec=spec,
                global_step=global_step, stage_step=stage_step,
                best_score=best_score, best_step=best_step, stale=stale,
            )
            _save_checkpoint(payload, output / "last_model.pt")
            if validate and improved:
                _save_checkpoint(payload, output / "best_model.pt")
        if _STOP_REQUESTED:
            status = {"status": "interrupted", "stage_step": stage_step, "global_step": global_step}
            _atomic_json(status_path, status)
            if run is not None:
                run.finish(exit_code=3)
            return status
        if validate and stage_step >= min_steps and stale >= patience:
            stopped_early = True
            break

    if bool(spec.get("final_model", False)) and (output / "best_model.pt").is_file():
        best_checkpoint = torch.load(output / "best_model.pt", map_location="cpu")
        model.load_state_dict(best_checkpoint["model_state"], strict=True)
    test_metrics = _domain_metrics(model, test_loaders, criteria, config, device)
    _atomic_json(output / "test_metrics.json", test_metrics)
    status = {
        "status": "complete", "run_id": spec["run_id"],
        "stage_step": stage_step, "global_step": global_step,
        "stopped_early": stopped_early, "best_step": best_step,
        "best_validation_score": best_score,
        "equivalent_epochs": {
            domain: (
                stage_step * (
                    int(config["training"]["mixed_batch_per_domain"])
                    if spec["domain"] == "mixed" else batch_size
                ) / max(len(frames[domain]["train"]), 1)
                if spec["domain"] in {domain, "mixed"} else 0.0
            )
            for domain in DOMAINS
        },
    }
    _atomic_json(status_path, status)
    if run is not None:
        for domain, values in test_metrics.items():
            if isinstance(values, dict):
                run.summary.update({f"test/{domain}/{key}": value for key, value in values.items()})
        run.summary.update(status)
        run.finish()
    return status
