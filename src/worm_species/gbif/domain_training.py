"""Step-balanced supervised training across curated GBIF and Petri domains."""

from __future__ import annotations

import copy
import hashlib
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
from ..training.losses import build_child_to_parent_matrix
from ..training.losses import hierarchy_consistency_loss
from .domain_cache import NODE_CACHE_ENV, load_cached_domain_frames
from .domain_data import DOMAINS, TASK_COLUMNS, file_sha256


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


def stage_is_complete(spec: dict) -> bool:
    """Return whether a stage has the artifacts required for a safe skip."""
    output = Path(spec["output_dir"])
    status_path = output / "run_status.json"
    if not status_path.is_file() or not (output / "last_model.pt").is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return status.get("status") == "complete"


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


def mixed_batch_per_domain(config: dict) -> int:
    """Return the enforced half-batch used for each mixed-training domain."""
    batch_size = int(config["training"]["batch_size"])
    if batch_size <= 0 or batch_size % 2:
        raise ValueError("Mixed training requires a positive even training.batch_size")
    return batch_size // 2


def _mixed_loader(frames: dict[str, pd.DataFrame], label_maps: dict, transform, config: dict, seed: int) -> DataLoader:
    training = config["training"]
    per_domain = mixed_batch_per_domain(config)
    gbif = frames["gbif"].reset_index(drop=True)
    petri = frames["petri"].reset_index(drop=True)
    combined = pd.concat([gbif, petri], ignore_index=True)
    dataset = MultiTaskWormImageDataset(
        combined, root_dir="/", image_col="image_path", target_cols=TASK_COLUMNS,
        label_to_index_by_task=label_maps, transform=transform,
        crop_to_foreground=False,
    )
    sampler = _BalancedDomainBatchSampler(
        len(gbif), len(petri), per_domain,
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


def _loss_for_batch(
    model, batch, criteria, config, device, scaler,
    hierarchy_config, child_to_parent_matrix, optimizer=None,
):
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
            if bool(hierarchy_config.get("enabled", False)):
                parent_task = hierarchy_config["parent_task"]
                child_task = hierarchy_config["child_task"]
                hierarchy_valid = (
                    labels[parent_task].ne(MISSING_LABEL)
                    & labels[child_task].ne(MISSING_LABEL)
                )
                hierarchy_loss = hierarchy_consistency_loss(
                    parent_logits=outputs[parent_task],
                    child_logits=outputs[child_task],
                    child_to_parent_matrix=child_to_parent_matrix,
                    valid_mask=hierarchy_valid,
                )
                if hierarchy_loss is not None:
                    hierarchy_weight = float(hierarchy_config["weight"])
                    total = total + hierarchy_weight * hierarchy_loss
                    active_weight += hierarchy_weight
                    task_losses["hierarchy"] = float(hierarchy_loss.detach().item())
            if active_weight == 0:
                raise ValueError("A training batch contains no active task labels")
            total = total / active_weight
        if train:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
    return float(total.detach().item()), task_losses, outputs, labels


def _evaluate(
    model, loader, criteria, config, device,
    hierarchy_config, child_to_parent_matrix,
) -> dict:
    model.eval()
    losses = []
    truths = {task: [] for task in TASK_COLUMNS}
    predictions = {task: [] for task in TASK_COLUMNS}
    dummy_scaler = torch.amp.GradScaler(enabled=False)
    with torch.inference_mode():
        for batch in loader:
            loss, _, outputs, labels = _loss_for_batch(
                model, batch, criteria, config, device, dummy_scaler,
                hierarchy_config, child_to_parent_matrix, optimizer=None,
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


def _domain_metrics(
    model, loaders, criteria, config, device,
    hierarchy_config, child_to_parent_matrix, selection_domains,
) -> dict:
    metrics = {
        domain: _evaluate(
            model, loader, criteria, config, device,
            hierarchy_config, child_to_parent_matrix,
        )
        for domain, loader in loaders.items()
    }
    domain_scores, selection_score = _domain_selection_score(
        metrics, selection_domains
    )
    metrics["domain_macro_f1"] = domain_scores
    metrics["selection_domains"] = list(selection_domains)
    metrics["domain_balanced_macro_f1"] = selection_score
    return metrics


def _domain_selection_score(
    metrics: dict,
    selection_domains,
) -> tuple[dict[str, float | None], float]:
    domain_scores = {}
    for domain in metrics:
        if domain not in DOMAINS:
            continue
        values = []
        for task in TASK_COLUMNS:
            value = metrics[domain].get(f"{task}_macro_f1")
            if value is not None:
                values.append(float(value))
        domain_scores[domain] = float(np.mean(values)) if values else None
    selected = [
        domain_scores[domain]
        for domain in selection_domains
        if domain_scores.get(domain) is not None
    ]
    if len(selected) != len(selection_domains):
        missing = [
            domain for domain in selection_domains
            if domain_scores.get(domain) is None
        ]
        raise ValueError(f"Selection domains have no validation metrics: {missing}")
    return domain_scores, float(np.mean(selected))


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
    if stage_is_complete(spec):
        status = json.loads(status_path.read_text(encoding="utf-8"))
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
    node_cache_root = os.environ.get(NODE_CACHE_ENV)
    if not node_cache_root:
        raise RuntimeError(
            f"{NODE_CACHE_ENV} is required; training must use the staged node-local cache"
        )
    frames = load_cached_domain_frames(config, node_cache_root)
    spec["runtime"]["node_cache_root"] = str(Path(node_cache_root).resolve())
    for domain in DOMAINS:
        train_frame = frames[domain]["train"]
        active = train_frame[list(TASK_COLUMNS.values())].ne("").any(axis=1)
        frames[domain]["train"] = train_frame.loc[active].reset_index(drop=True)
        if frames[domain]["train"].empty:
            raise ValueError(f"No PETI-vocabulary training labels remain for domain {domain}")
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
    per_domain_mixed_batch = mixed_batch_per_domain(config)
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
    selection_domains = tuple(spec["selection_domains"])
    if not selection_domains or any(domain not in DOMAINS for domain in selection_domains):
        raise ValueError(f"Invalid checkpoint-selection domains: {selection_domains}")
    validation_loaders = {
        domain: _loader(
            frames[domain]["validation"], label_maps, eval_transform, config,
            batch_size=batch_size, shuffle=False, seed=int(spec["seed"]),
            num_workers_override=2, persistent_workers_override=False,
        )
        for domain in selection_domains
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
    if bool(spec.get("freeze_age_head", False)):
        for parameter in model.heads["age"].parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW([
        {
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": float(config["training"]["backbone_lr"]),
        },
        {
            "params": [p for p in model.heads.parameters() if p.requires_grad],
            "lr": float(config["training"]["head_lr"]),
        },
    ], weight_decay=float(config["training"]["weight_decay"]))
    max_steps = int(spec["max_steps"])
    scheduler = _scheduler(optimizer, int(config["training"]["warmup_steps"]), max_steps)
    scaler = torch.amp.GradScaler(enabled=bool(config["training"]["use_amp"]))
    criteria = _criteria(config, {domain: frames[domain]["train"] for domain in DOMAINS}, label_maps, device)
    hierarchy_config = dict(spec.get("hierarchy_loss", {}))
    child_to_parent_matrix = None
    if bool(hierarchy_config.get("enabled", False)):
        child_to_parent_matrix = build_child_to_parent_matrix(
            label_maps,
            hierarchy_config["parent_task"],
            hierarchy_config["child_task"],
            device,
        )

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
        # External transfer checkpoints are model initialisers, not resumes.
        # Their optimizer parameter groups may differ (PETI -> GBIF freezes
        # the age head), so only restore optimizer state for this stage's own
        # interrupted checkpoint.  Sequential stages deliberately start with
        # the fixed Stage-2 optimizer policy.
        is_resume = checkpoint_path == resume_path
        if is_resume:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
        global_step = int(checkpoint.get("global_step", 0))
        if is_resume:
            stage_step = int(checkpoint.get("stage_step", 0))
            best_score = float(checkpoint.get("best_val_score", best_score))
            best_step = int(checkpoint.get("best_step", 0))
            stale = int(checkpoint.get("stale_validations", 0))

    _atomic_json(output / "spec.json", spec)

    run = _wandb_run(config, spec, output)
    history_path = output / "history.jsonl"
    validation_interval = int(config["training"]["validation_interval_steps"])
    checkpoint_interval = int(config["training"]["checkpoint_interval_steps"])
    min_delta = float(config["training"]["checkpoint_selection_min_delta"])
    samples_seen = {"gbif": 0, "petri": 0}
    while stage_step < max_steps:
        model.train()
        batch = next(train_iterator)
        loss, task_losses, _, _ = _loss_for_batch(
            model, batch, criteria, config, device, scaler,
            hierarchy_config, child_to_parent_matrix, optimizer=optimizer,
        )
        scheduler.step()
        stage_step += 1
        global_step += 1
        if spec["domain"] == "mixed":
            samples_seen["gbif"] += per_domain_mixed_batch
            samples_seen["petri"] += per_domain_mixed_batch
        else:
            samples_seen[spec["domain"]] += batch_size
        record = {
            "stage_step": stage_step, "global_step": global_step,
            "train_loss": loss, **{f"train_{task}_loss": value for task, value in task_losses.items()},
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "gbif_samples_seen": samples_seen["gbif"],
            "peti_samples_seen": samples_seen["petri"],
        }
        validate = stage_step % validation_interval == 0 or stage_step == max_steps
        if validate:
            validation = _domain_metrics(
                model, validation_loaders, criteria, config, device,
                hierarchy_config, child_to_parent_matrix, selection_domains,
            )
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
                f"loss={loss:.4f} validation={score:.4f} stale={stale}",
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
            status = {
                "status": "interrupted", "stage_step": stage_step, "global_step": global_step,
                "samples_seen": {"gbif": samples_seen["gbif"], "peti": samples_seen["petri"]},
            }
            _atomic_json(status_path, status)
            if run is not None:
                run.finish(exit_code=3)
            return status
    if (output / "best_model.pt").is_file():
        best_checkpoint = torch.load(output / "best_model.pt", map_location="cpu")
        model.load_state_dict(best_checkpoint["model_state"], strict=True)
    else:
        raise RuntimeError(f"Fixed-budget stage did not produce a best checkpoint: {output}")
    test_metrics = _domain_metrics(
        model, test_loaders, criteria, config, device,
        hierarchy_config, child_to_parent_matrix, DOMAINS,
    )
    _atomic_json(output / "test_metrics.json", test_metrics)
    status = {
        "status": "complete", "run_id": spec["run_id"],
        "stage_step": stage_step, "global_step": global_step,
        "fixed_budget_complete": stage_step == max_steps,
        "checkpoint_selection_domains": list(selection_domains),
        "best_step": best_step,
        "best_validation_score": best_score,
        "equivalent_epochs": {
            domain: (
                stage_step * (
                    per_domain_mixed_batch
                    if spec["domain"] == "mixed" else batch_size
                ) / max(len(frames[domain]["train"]), 1)
                if spec["domain"] in {domain, "mixed"} else 0.0
            )
            for domain in DOMAINS
        },
        "samples_seen": {
            "gbif": int(samples_seen["gbif"]),
            "peti": int(samples_seen["petri"]),
            "total": int(sum(samples_seen.values())),
            "gbif_fraction": float(samples_seen["gbif"] / max(sum(samples_seen.values()), 1)),
            "peti_fraction": float(samples_seen["petri"] / max(sum(samples_seen.values()), 1)),
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
