from __future__ import annotations

import argparse
import copy
import itertools
import math
from pathlib import Path

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.dataset_multitask import (
    MISSING_LABEL,
    MultiTaskWormImageDataset,
    build_transforms,
    get_target_cols,
    prepare_metadata,
)
from src.cache import build_image_cache
from src.models import build_model
from src.splits import make_individual_level_splits
from src.utils import (
    load_config,
    apply_overrides,
    set_nested,
    parse_scalar,
    set_seed,
    save_json,
    make_run_name,
)


# ---------------------------------------------------------------------
# Sweep utilities: same style as your current single-task script
# ---------------------------------------------------------------------


def parse_sweep_item(item: str) -> tuple[str, list]:
    """Parse command-line sweep item, e.g. model.name=resnet18,efficientnet_b0."""
    if "=" not in item:
        raise ValueError(f"Sweep item must look like key=v1,v2. Got: {item}")

    key, values = item.split("=", 1)
    vals = [parse_scalar(v) for v in values.split(",") if v.strip()]

    if len(vals) == 0:
        raise ValueError(f"No values supplied for sweep key: {key}")

    return key, vals


def get_sweep_parameters_from_config(cfg: dict) -> dict:
    sweep_cfg = cfg.get("sweep", {})
    if not sweep_cfg.get("enabled", False):
        return {}
    params = sweep_cfg.get("parameters", {})
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError("sweep.parameters must be a dictionary.")
    return params


def get_sweep_parameters_from_cli(sweep_items: list[str]) -> dict:
    params = {}
    for item in sweep_items:
        key, values = parse_sweep_item(item)
        params[key] = values
    return params


def generate_sweep_configs(
    base_cfg: dict,
    cli_sweep_items: list[str] | None = None,
) -> list[dict]:
    cli_sweep_items = cli_sweep_items or []

    if len(cli_sweep_items) > 0:
        sweep_params = get_sweep_parameters_from_cli(cli_sweep_items)
    else:
        sweep_params = get_sweep_parameters_from_config(base_cfg)

    if len(sweep_params) == 0:
        return [base_cfg]

    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        cfg = copy.deepcopy(base_cfg)
        for key, value in zip(keys, combo):
            set_nested(cfg, key, value)
        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class MultiTaskClassifier(nn.Module):
    """
    Shared image backbone with one classification head per task.

    Assumes build_model returns a torchvision-like classifier, for example:
    - ResNet with .fc
    - EfficientNet/MobileNet/DenseNet with .classifier
    - torchvision ViT with .heads.head
    """

    def __init__(self, base_model: nn.Module, num_classes_by_task: dict[str, int]):
        super().__init__()
        self.backbone = base_model
        feature_dim = self._remove_classifier_and_get_feature_dim()

        self.heads = nn.ModuleDict({
            task: nn.Linear(feature_dim, num_classes)
            for task, num_classes in num_classes_by_task.items()
        })

    def _remove_classifier_and_get_feature_dim(self) -> int:
        if hasattr(self.backbone, "fc") and isinstance(self.backbone.fc, nn.Linear):
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            return feature_dim

        if hasattr(self.backbone, "classifier"):
            classifier = self.backbone.classifier

            if isinstance(classifier, nn.Linear):
                feature_dim = classifier.in_features
                self.backbone.classifier = nn.Identity()
                return feature_dim

            if isinstance(classifier, nn.Sequential):
                for i in range(len(classifier) - 1, -1, -1):
                    if isinstance(classifier[i], nn.Linear):
                        feature_dim = classifier[i].in_features
                        classifier[i] = nn.Identity()
                        return feature_dim

        if hasattr(self.backbone, "heads") and hasattr(self.backbone.heads, "head"):
            head = self.backbone.heads.head
            if isinstance(head, nn.Linear):
                feature_dim = head.in_features
                self.backbone.heads.head = nn.Identity()
                return feature_dim

        raise ValueError(
            "Could not identify the final classifier layer. "
            "Add a case in MultiTaskClassifier._remove_classifier_and_get_feature_dim "
            "for your model."
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)

        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.ndim > 2:
            features = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(features, 1), 1)

        return {task: head(features) for task, head in self.heads.items()}


def build_multitask_model(cfg: dict, num_classes_by_task: dict[str, int]) -> nn.Module:
    # The single-task head is removed immediately after construction.
    temporary_num_classes = max(num_classes_by_task.values())

    base_model = build_model(
        name=cfg["model"]["name"],
        num_classes=temporary_num_classes,
        pretrained=cfg["model"].get("pretrained", True),
        freeze_backbone=cfg["model"].get("freeze_backbone", False),
    )

    return MultiTaskClassifier(
        base_model=base_model,
        num_classes_by_task=num_classes_by_task,
    )


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def build_label_maps(
    train_df: pd.DataFrame,
    target_cols: dict[str, str],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[int, str]]]:
    label_to_index_by_task = {}
    index_to_label_by_task = {}

    for task, col in target_cols.items():
        labels = sorted(train_df.loc[train_df[col].notna(), col].astype(str).unique())

        if len(labels) == 0:
            raise ValueError(
                f"Task '{task}' has no labelled examples in the training split. "
                f"Check column '{col}' or reduce min_individuals_per_class for this task."
            )

        if len(labels) == 1:
            print(
                f"Warning: task '{task}' has only one class in training: {labels}. "
                "The head can train, but evaluation is not biologically informative."
            )

        label_to_index = {label: i for i, label in enumerate(labels)}
        index_to_label = {i: label for label, i in label_to_index.items()}

        label_to_index_by_task[task] = label_to_index
        index_to_label_by_task[task] = index_to_label

    return label_to_index_by_task, index_to_label_by_task

#READ PREFINED CSV'S split_csv/train_split.csv
def read_csvs_from_dir(dir_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_csv_path = os.path.join(dir_path,"split_csv", "train_split.csv")
    val_csv_path = os.path.join(dir_path,"split_csv", "val_split.csv")
    test_csv_path = os.path.join(dir_path,"split_csv", "test_split.csv")

    if not train_csv_path.exists() or not val_csv_path.exists() or not test_csv_path.exists():
        raise FileNotFoundError(f"One or more split CSV files not found in {dir_path}")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)

    return train_df, val_df, test_df

def make_loaders(cfg: dict):
    df = prepare_metadata(cfg)

    target_cols = get_target_cols(cfg)
    group_col = cfg["data"]["group_col"]

    split_target_col = cfg["data"].get("split_target_col", "__taxon_for_split__")
    if split_target_col not in df.columns:
        raise ValueError(
            f"data.split_target_col={split_target_col!r} is not in the metadata dataframe. "
            "Use '__taxon_for_split__' or an existing column."
        )

    cache_enabled = cfg.get("cache", {}).get("enabled", False)

    if cache_enabled:
        df = build_image_cache(cfg, df)
        df = df[df["_cached_image_path"].notna()].reset_index(drop=True)
        image_col_for_dataset = "_cached_image_path"
        crop_to_foreground_for_dataset = False
    else:
        image_col_for_dataset = cfg["data"]["image_col"]
        crop_to_foreground_for_dataset = cfg["data"].get("crop_to_foreground", True)

    train_df, val_df, test_df = read_csvs_from_dir(cfg["data"]["root_dir"]) if cfg["split"].get("use_predefined_splits", False) else make_individual_level_splits(
        df=df,
        group_col=group_col,
        target_col=split_target_col,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        seed=cfg["seed"],
        root_dir=cfg["data"]["root_dir"] if cfg["data"].get("save_splits", False) else None,
    )

    label_to_index_by_task, index_to_label_by_task = build_label_maps(train_df, target_cols)

    image_size = cfg["data"]["image_size"]
    train_tf = build_transforms(image_size=image_size, train=True)
    eval_tf = build_transforms(image_size=image_size, train=False)

    common_kwargs = dict(
        root_dir=cfg["data"]["root_dir"],
        image_col=image_col_for_dataset,
        target_cols=target_cols,
        label_to_index_by_task=label_to_index_by_task,
        mask_col=cfg["data"].get("mask_col"),
        crop_to_foreground=crop_to_foreground_for_dataset,
        crop_pad=cfg["data"].get("crop_pad", 0.15),
    )

    train_ds = MultiTaskWormImageDataset(train_df, transform=train_tf, **common_kwargs)
    val_ds = MultiTaskWormImageDataset(val_df, transform=eval_tf, **common_kwargs)
    test_ds = MultiTaskWormImageDataset(test_df, transform=eval_tf, **common_kwargs)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 4)

    train_loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    eval_loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 4
        eval_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        **train_loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **eval_loader_kwargs,
    )

    split_summary = {
        "target_cols": target_cols,
        "split_target_col": split_target_col,
        "num_classes_by_task": {
            task: len(label_to_index)
            for task, label_to_index in label_to_index_by_task.items()
        },
        "classes_by_task": {
            task: list(label_to_index.keys())
            for task, label_to_index in label_to_index_by_task.items()
        },
        "labelled_rows_by_task": {
            task: {
                "train": int(train_df[col].notna().sum()),
                "val": int(val_df[col].notna().sum()),
                "test": int(test_df[col].notna().sum()),
            }
            for task, col in target_cols.items()
        },
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_individuals": train_df[group_col].nunique(),
        "val_individuals": val_df[group_col].nunique(),
        "test_individuals": test_df[group_col].nunique(),
    }

    return (
        train_loader,
        val_loader,
        test_loader,
        label_to_index_by_task,
        index_to_label_by_task,
        split_summary,
        train_df,
        target_cols,
    )


# ---------------------------------------------------------------------
# Loss and metrics
# ---------------------------------------------------------------------


def compute_individual_class_weights(
    train_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    label_to_index: dict[str, int],
) -> torch.Tensor:
    labelled = train_df[train_df[target_col].notna()]
    counts = labelled.groupby(target_col)[group_col].nunique().to_dict()

    weights = []
    for label in label_to_index:
        weights.append(1.0 / max(float(counts.get(label, 1.0)), 1.0))

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def build_criteria(
    train_df: pd.DataFrame,
    target_cols: dict[str, str],
    group_col: str,
    label_to_index_by_task: dict[str, dict[str, int]],
    device: torch.device,
) -> dict[str, nn.Module]:
    criteria = {}

    for task, col in target_cols.items():
        class_weights = compute_individual_class_weights(
            train_df=train_df,
            target_col=col,
            group_col=group_col,
            label_to_index=label_to_index_by_task[task],
        ).to(device)

        criteria[task] = nn.CrossEntropyLoss(weight=class_weights)

    return criteria


def _safe_metric(metric_fn, y_true: np.ndarray, y_pred: np.ndarray, default: float = float("nan")) -> float:
    if len(y_true) == 0:
        return default
    return float(metric_fn(y_true, y_pred))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criteria: dict[str, nn.Module],
    optimizer,
    device: torch.device,
    train: bool,
    scaler=None,
    use_amp: bool = True,
    task_loss_weights: dict[str, float] | None = None,
    normalize_loss_by_active_tasks: bool = True,
):
    if train:
        model.train()
    else:
        model.eval()

    tasks = list(criteria.keys())
    task_loss_weights = task_loss_weights or {task: 1.0 for task in tasks}

    losses = []
    task_losses = {task: [] for task in tasks}
    all_true = {task: [] for task in tasks}
    all_pred = {task: [] for task in tasks}

    complete_exact_correct = 0
    complete_exact_total = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = {
            task: batch["labels"][task].to(device, non_blocking=True)
            for task in tasks
        }

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(
                enabled=(use_amp and device.type == "cuda"),
                device_type=device.type,
            ):
                logits_by_task = model(x)

                total_loss = torch.zeros((), device=device)
                active_weight_sum = 0.0
                loss_by_task: dict[str, torch.Tensor | None] = {}

                for task in tasks:
                    valid = y[task] != MISSING_LABEL

                    if valid.any():
                        task_loss = criteria[task](logits_by_task[task][valid], y[task][valid])
                        weight = float(task_loss_weights.get(task, 1.0))
                        total_loss = total_loss + weight * task_loss
                        active_weight_sum += weight
                        loss_by_task[task] = task_loss
                    else:
                        loss_by_task[task] = None

                if active_weight_sum == 0:
                    # Extremely unlikely because prepare_metadata_multitask removes fully unlabelled rows.
                    continue

                if normalize_loss_by_active_tasks:
                    total_loss = total_loss / active_weight_sum

            if train:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

        losses.append(float(total_loss.item()))

        complete_mask = torch.ones(x.shape[0], dtype=torch.bool, device=device)
        complete_correct = torch.ones(x.shape[0], dtype=torch.bool, device=device)

        for task in tasks:
            pred = logits_by_task[task].argmax(dim=1)
            valid = y[task] != MISSING_LABEL

            complete_mask &= valid
            complete_correct &= pred.eq(y[task])

            if valid.any():
                if loss_by_task[task] is not None:
                    task_losses[task].append(float(loss_by_task[task].item()))
                all_true[task].extend(y[task][valid].detach().cpu().numpy().tolist())
                all_pred[task].extend(pred[valid].detach().cpu().numpy().tolist())

        if complete_mask.any():
            complete_exact_total += int(complete_mask.sum().item())
            complete_exact_correct += int((complete_correct & complete_mask).sum().item())

    metrics = {"loss": float(np.mean(losses)) if losses else float("nan")}
    if train:
        return metrics, all_true, all_pred
    else:
        macro_f1_values = []

        for task in tasks:
            y_true = np.array(all_true[task], dtype=int)
            y_pred = np.array(all_pred[task], dtype=int)

            if len(y_true) == 0:
                metrics[f"{task}_loss"] = float("nan")
                metrics[f"{task}_n"] = 0
                metrics[f"{task}_accuracy"] = float("nan")
                metrics[f"{task}_balanced_accuracy"] = float("nan")
                metrics[f"{task}_macro_f1"] = float("nan")
                continue

            task_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            macro_f1_values.append(task_macro_f1)

            metrics[f"{task}_loss"] = float(np.mean(task_losses[task])) if task_losses[task] else float("nan")
            metrics[f"{task}_n"] = int(len(y_true))
            metrics[f"{task}_accuracy"] = _safe_metric(accuracy_score, y_true, y_pred)
            metrics[f"{task}_balanced_accuracy"] = _safe_metric(balanced_accuracy_score, y_true, y_pred)
            metrics[f"{task}_macro_f1"] = float(task_macro_f1)

        metrics["mean_macro_f1"] = float(np.mean(macro_f1_values)) if macro_f1_values else float("nan")
        metrics["complete_exact_match_accuracy"] = (
            float(complete_exact_correct / complete_exact_total)
            if complete_exact_total > 0
            else float("nan")
        )
        metrics["complete_exact_match_n"] = int(complete_exact_total)

    return metrics, all_true, all_pred


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------


def _score_for_selection(metrics: dict, selection_metric: str) -> float:
    value = float(metrics.get(selection_metric, float("nan")))
    if math.isnan(value):
        return -float("inf")
    return value


def train_one_run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = make_run_name(cfg)
    out_dir = Path(cfg["output"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(cfg, out_dir / "config.json")

    (
        train_loader,
        val_loader,
        test_loader,
        label_to_index_by_task,
        index_to_label_by_task,
        split_summary,
        train_df,
        target_cols,
    ) = make_loaders(cfg)

    save_json(split_summary, out_dir / "split_summary.json")
    save_json(label_to_index_by_task, out_dir / "label_to_index_by_task.json")
    print(f"Split summary and label maps saved to {out_dir}")
    num_classes_by_task = {
        task: len(label_to_index)
        for task, label_to_index in label_to_index_by_task.items()
    }

    model = build_multitask_model(
        cfg=cfg,
        num_classes_by_task=num_classes_by_task,
    ).to(device)
    print("Model built and moved to device.")
    criteria = build_criteria(
        train_df=train_df,
        target_cols=target_cols,
        group_col=cfg["data"]["group_col"],
        label_to_index_by_task=label_to_index_by_task,
        device=device,
    )

    task_loss_weights = cfg.get("multi_task", {}).get(
        "loss_weights",
        {task: 1.0 for task in target_cols},
    )
    normalize_loss_by_active_tasks = cfg.get("multi_task", {}).get(
        "normalize_loss_by_active_tasks",
        True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
    )

    use_amp = cfg["training"].get("use_amp", True)
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    history = []

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", True)
    patience = early_cfg.get("patience", 3)
    min_delta = early_cfg.get("min_delta", 0.001)

    selection_metric = cfg.get("multi_task", {}).get("selection_metric", "mean_macro_f1")
    val_interval = cfg["training"].get("val_interval", 3)

    best_val_score = -float("inf")
    epochs_without_improvement = 0
    best_epoch = 0
    print(f"Training for {cfg['training']['epochs']} epochs with early stopping: {early_enabled}, patience: {patience}, min_delta: {min_delta}")
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criteria=criteria,
            optimizer=optimizer,
            device=device,
            train=True,
            scaler=scaler,
            use_amp=use_amp,
            task_loss_weights=task_loss_weights,
            normalize_loss_by_active_tasks=normalize_loss_by_active_tasks,
        )

        do_validation = (
            epoch == 1
            or epoch % val_interval == 0
            or epoch == cfg["training"]["epochs"]
        )

        if do_validation:
            val_metrics, _, _ = run_epoch(
                model=model,
                loader=val_loader,
                criteria=criteria,
                optimizer=None,
                device=device,
                train=False,
                scaler=None,
                use_amp=use_amp,
                task_loss_weights=task_loss_weights,
                normalize_loss_by_active_tasks=normalize_loss_by_active_tasks,
            )
        else:
            val_metrics = {}

        scheduler.step()

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        if do_validation:
            if selection_metric not in val_metrics:
                raise ValueError(
                    f"multi_task.selection_metric={selection_metric!r} is not available. "
                    f"Available validation metrics: {list(val_metrics.keys())}"
                )

            current_val_score = _score_for_selection(val_metrics, selection_metric)

            print(
                f"[{run_name}] Epoch {epoch:03d}/{cfg['training']['epochs']} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"val {selection_metric} {val_metrics[selection_metric]:.4f} | "
                f"complete exact-match {val_metrics['complete_exact_match_accuracy']:.4f} "
                f"n={val_metrics['complete_exact_match_n']}"
            )

            for task in target_cols:
                print(
                    f"    {task}: val macro-F1 {val_metrics[f'{task}_macro_f1']:.4f} | "
                    f"val bal-acc {val_metrics[f'{task}_balanced_accuracy']:.4f} | "
                    f"n={val_metrics[f'{task}_n']}"
                )

            improved = current_val_score > best_val_score + min_delta

            if improved or epoch == 1:
                best_val_score = current_val_score
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "cfg": cfg,
                        "label_to_index_by_task": label_to_index_by_task,
                        "index_to_label_by_task": index_to_label_by_task,
                        "best_val_score": best_val_score,
                        "selection_metric": selection_metric,
                        "best_epoch": best_epoch,
                    },
                    out_dir / "best_model.pt",
                )

                print(
                    f"[{run_name}] New best model saved | "
                    f"best val {selection_metric} {best_val_score:.4f} at epoch {best_epoch}"
                )
            else:
                epochs_without_improvement += 1
                print(f"[{run_name}] No improvement for {epochs_without_improvement}/{patience} validation checks")

            if early_enabled and epochs_without_improvement >= patience:
                print(
                    f"[{run_name}] Early stopping at epoch {epoch}. "
                    f"Best val {selection_metric} {best_val_score:.4f} at epoch {best_epoch}."
                )
                break

        else:
            print(
                f"[{run_name}] Epoch {epoch:03d}/{cfg['training']['epochs']} | "
                f"train loss {train_metrics['loss']:.4f} | "
                f"validation skipped"
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics, y_true_by_task, y_pred_by_task = run_epoch(
        model=model,
        loader=test_loader,
        criteria=criteria,
        optimizer=None,
        device=device,
        train=False,
        scaler=None,
        use_amp=use_amp,
        task_loss_weights=task_loss_weights,
        normalize_loss_by_active_tasks=normalize_loss_by_active_tasks,
    )

    for task in target_cols:
        index_to_label = index_to_label_by_task[task]
        labels = list(range(len(index_to_label)))
        label_names = [index_to_label[i] for i in labels]

        y_true = np.array(y_true_by_task[task], dtype=int)
        y_pred = np.array(y_pred_by_task[task], dtype=int)

        if len(y_true) == 0:
            pd.DataFrame([{"note": "No labelled test examples for this task."}]).to_csv(
                out_dir / f"classification_report_{task}.csv",
                index=False,
            )
            pd.DataFrame().to_csv(out_dir / f"confusion_matrix_{task}.csv")
            continue

        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        pd.DataFrame(report).transpose().to_csv(out_dir / f"classification_report_{task}.csv")
        pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
            out_dir / f"confusion_matrix_{task}.csv"
        )

    save_json(test_metrics, out_dir / "test_metrics.json")

    print("\nTest metrics:")
    print(test_metrics)

    return {
        "run_name": run_name,
        "out_dir": str(out_dir),
        "best_val_score": best_val_score,
        "selection_metric": selection_metric,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
    )

    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. model.name=vit_b_16 training.lr=1e-5",
    )

    parser.add_argument(
        "--sweep",
        nargs="*",
        default=[],
        help="Multi-run sweep, e.g. model.name=resnet18,vit_b_16 data.image_col=rel_path_seg,rel_path_raw",
    )

    args = parser.parse_args()

    base_cfg = load_config(args.config)
    base_cfg = apply_overrides(base_cfg, args.override)

    configs = generate_sweep_configs(
        base_cfg=base_cfg,
        cli_sweep_items=args.sweep,
    )

    all_results = []

    for i, cfg in enumerate(configs, start=1):
        print(f"\nRunning {i}/{len(configs)}")
        result = train_one_run(cfg)
        all_results.append(result)

    out_dir = Path(base_cfg["output"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "multi_run_results.csv", index=False)

    print("\nAll runs finished.")
    print(results_df)


if __name__ == "__main__":
    main()
