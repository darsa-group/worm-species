from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

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

from src.dataset import (
    prepare_metadata,
    WormImageDataset,
    build_transforms,
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

import copy
import itertools


def parse_sweep_item(item: str) -> tuple[str, list]:
    """
    Parse command-line sweep item.

    Example:
    model.name=resnet18,efficientnet_b0
    """
    if "=" not in item:
        raise ValueError(f"Sweep item must look like key=v1,v2. Got: {item}")

    key, values = item.split("=", 1)
    vals = [parse_scalar(v) for v in values.split(",") if v.strip()]

    if len(vals) == 0:
        raise ValueError(f"No values supplied for sweep key: {key}")

    return key, vals


def get_sweep_parameters_from_config(cfg: dict) -> dict:
    """
    Read sweep parameters from config.yaml.

    Expected format:

    sweep:
      enabled: true
      parameters:
        model.name:
          - resnet18
          - efficientnet_b0
        training.lr:
          - 0.0001
          - 0.00005
    """
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
    """
    Read sweep parameters from command-line --sweep.
    """
    params = {}

    for item in sweep_items:
        key, values = parse_sweep_item(item)
        params[key] = values

    return params


def generate_sweep_configs(
    base_cfg: dict,
    cli_sweep_items: list[str] | None = None,
) -> list[dict]:
    """
    Generate run configs from sweep settings.

    Priority:
    - If command-line --sweep is provided, use that.
    - Otherwise, use sweep.parameters from config.yaml.
    - If no sweep is enabled, return one config.
    """

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

        # Remove sweep section from saved run config if desired
        # This avoids confusion inside individual run folders.
        # You can comment this out if you prefer to keep it.
        # cfg.pop("sweep", None)

        for key, value in zip(keys, combo):
            set_nested(cfg, key, value)

        configs.append(cfg)

    return configs


def make_loaders(cfg: dict):
    df = prepare_metadata(cfg)
    
    cache_enabled = cfg.get("cache", {}).get("enabled", False)

    if cache_enabled:
        df = build_image_cache(cfg, df)
        df = df[df["_cached_image_path"].notna()].reset_index(drop=True)

        image_col_for_dataset = "_cached_image_path"
        crop_to_foreground_for_dataset = False
    else:
        image_col_for_dataset = cfg["data"]["image_col"]
        crop_to_foreground_for_dataset = cfg["data"].get("crop_to_foreground", True)

    target_col = cfg["data"]["target_col"]
    group_col = cfg["data"]["group_col"]

    train_df, val_df, test_df = make_individual_level_splits(
        df=df,
        target_col=target_col,
        group_col=group_col,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        seed=cfg["seed"],
    )

    labels = sorted(train_df[target_col].unique())
    label_to_index = {label: i for i, label in enumerate(labels)}
    index_to_label = {i: label for label, i in label_to_index.items()}

    # Remove val/test classes absent from training
    val_df = val_df[val_df[target_col].isin(label_to_index)].reset_index(drop=True)
    test_df = test_df[test_df[target_col].isin(label_to_index)].reset_index(drop=True)

    image_size = cfg["data"]["image_size"]

    train_tf = build_transforms(image_size=image_size, train=True)
    eval_tf = build_transforms(image_size=image_size, train=False)

    common_kwargs = dict(
        root_dir=cfg["data"]["root_dir"],
        image_col=image_col_for_dataset,
        target_col=target_col,
        label_to_index=label_to_index,
        mask_col=cfg["data"].get("mask_col"),
        crop_to_foreground=crop_to_foreground_for_dataset,
        crop_pad=cfg["data"].get("crop_pad", 0.15),
    )

    train_ds = WormImageDataset(train_df, transform=train_tf, **common_kwargs)
    val_ds = WormImageDataset(val_df, transform=eval_tf, **common_kwargs)
    test_ds = WormImageDataset(test_df, transform=eval_tf, **common_kwargs)

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 4)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2
    )

    split_summary = {
        "num_classes": len(labels),
        "classes": labels,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_individuals": train_df[group_col].nunique(),
        "val_individuals": val_df[group_col].nunique(),
        "test_individuals": test_df[group_col].nunique(),
    }

    return train_loader, val_loader, test_loader, label_to_index, index_to_label, split_summary, train_df


def compute_individual_class_weights(train_df, target_col, group_col, label_to_index):
    counts = (
        train_df.groupby(target_col)[group_col]
        .nunique()
        .to_dict()
    )

    weights = []

    for label in label_to_index:
        weights.append(1.0 / counts[label])

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    train: bool,
    scaler=None,
    use_amp: bool = True,
):
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    all_true = []
    all_pred = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(enabled=(use_amp and device.type == "cuda"),device_type=device.type):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                if scaler is not None and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        pred = logits.argmax(dim=1)

        losses.append(loss.item())
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy_score(all_true, all_pred),
        "balanced_accuracy": balanced_accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro"),
    }

    return metrics, np.array(all_true), np.array(all_pred)


def train_one_run(cfg: dict) -> dict:
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = make_run_name(cfg)
    out_dir = Path(cfg["output"]["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(cfg, out_dir / "config.json")

    (
        train_loader,
        val_loader,
        test_loader,
        label_to_index,
        index_to_label,
        split_summary,
        train_df,
    ) = make_loaders(cfg)

    save_json(split_summary, out_dir / "split_summary.json")
    save_json(label_to_index, out_dir / "label_to_index.json")

    model = build_model(
        name=cfg["model"]["name"],
        num_classes=len(label_to_index),
        pretrained=cfg["model"].get("pretrained", True),
        freeze_backbone=cfg["model"].get("freeze_backbone", False),
    ).to(device)

    class_weights = compute_individual_class_weights(
        train_df=train_df,
        target_col=cfg["data"]["target_col"],
        group_col=cfg["data"]["group_col"],
        label_to_index=label_to_index,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

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

    best_val_f1 = -1.0
    history = []

    early_cfg = cfg.get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", True)
    patience = early_cfg.get("patience", 12)
    min_delta = early_cfg.get("min_delta", 0.001)

    best_val_f1 = -float("inf")
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_metrics, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
            scaler=scaler,
            use_amp=use_amp,
        )

        val_metrics, _, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
            scaler=None,
            use_amp=use_amp,
        )

        scheduler.step()

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }

        history.append(row)

        current_val_f1 = val_metrics["macro_f1"]

        print(
            f"[{run_name}] "
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"val macro-F1 {current_val_f1:.4f} | "
            f"val bal-acc {val_metrics['balanced_accuracy']:.4f}"
        )

        # -----------------------------
        # Check improvement
        # -----------------------------
        improved = current_val_f1 > best_val_f1 + min_delta

        if improved:
            best_val_f1 = current_val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg,
                    "label_to_index": label_to_index,
                    "index_to_label": index_to_label,
                    "best_val_macro_f1": best_val_f1,
                    "best_epoch": best_epoch,
                },
                out_dir / "best_model.pt",
            )

            print(
                f"[{run_name}] New best model saved | "
                f"best val macro-F1 {best_val_f1:.4f} at epoch {best_epoch}"
            )

        else:
            epochs_without_improvement += 1

            print(
                f"[{run_name}] No improvement for "
                f"{epochs_without_improvement}/{patience} epochs"
            )

        # -----------------------------
        # Early stopping
        # -----------------------------
        if early_enabled and epochs_without_improvement >= patience:
            print(
                f"[{run_name}] Early stopping at epoch {epoch}. "
                f"Best val macro-F1 {best_val_f1:.4f} at epoch {best_epoch}."
            )
            break
    # Final test evaluation
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "history.csv", index=False)
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_metrics, y_true, y_pred = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
        scaler=None,
        use_amp=use_amp,
    )

    label_names = [index_to_label[i] for i in range(len(index_to_label))]

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")
    pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(out_dir / "confusion_matrix.csv")

    save_json(test_metrics, out_dir / "test_metrics.json")

    print("\nTest metrics:")
    print(test_metrics)

    return {
        "run_name": run_name,
        "out_dir": str(out_dir),
        "best_val_macro_f1": best_val_f1,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


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