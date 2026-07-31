"""Explicit optimiser ownership and staged-unfreezing support."""

from __future__ import annotations

from dataclasses import dataclass

import torch


GROUP_ORDER = (
    "early_backbone",
    "final_backbone_stage",
    "task_specific_branches",
    "classification_heads",
    "projection_heads",
)


def parameter_group_name(name: str) -> str:
    if name.startswith("heads.") or "species_adversary" in name:
        return "classification_heads"
    if "projection" in name:
        return "projection_heads"
    if any(token in name for token in (
        "taxonomy_final_stage", "age_final_stage",
        "taxonomy_adapter", "age_adapter", "taxonomy_pool", "age_pool",
    )):
        return "task_specific_branches"
    if any(token in name for token in (
        "backbone.features.7", "backbone.stages.3", "backbone.layer4",
        "backbone.encoder.layers.encoder_layer_11",
    )):
        return "final_backbone_stage"
    return "early_backbone"


def grouped_named_parameters(model: torch.nn.Module) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    groups = {name: [] for name in GROUP_ORDER}
    for name, parameter in model.named_parameters():
        groups[parameter_group_name(name)].append((name, parameter))
    return groups


def validate_optimizer_coverage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    expected = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    observed: list[int] = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    duplicates = {identifier for identifier in observed if observed.count(identifier) > 1}
    missing = set(expected) - set(observed)
    unexpected = set(observed) - set(expected)
    if duplicates or missing or unexpected:
        raise ValueError(
            "Optimizer parameter coverage violation: "
            f"missing={[expected[value] for value in missing]}, "
            f"duplicate_count={len(duplicates)}, unexpected_count={len(unexpected)}"
        )


def _learning_rates(cfg: dict) -> dict[str, float]:
    optimizer_cfg = cfg.get("optimizer", {}) or {}
    configured = optimizer_cfg.get("learning_rates")
    if not configured:
        return {name: float(cfg["training"]["lr"]) for name in GROUP_ORDER}
    missing = set(GROUP_ORDER) - set(configured)
    if missing:
        raise ValueError(
            "optimizer.learning_rates is missing groups: " + ", ".join(sorted(missing))
        )
    return {name: float(configured[name]) for name in GROUP_ORDER}


def optimizer_group_specification(model: torch.nn.Module, cfg: dict) -> list[dict]:
    rates = _learning_rates(cfg)
    decay = float(
        (cfg.get("optimizer", {}) or {}).get(
            "weight_decay", cfg["training"]["weight_decay"]
        )
    )
    grouped = grouped_named_parameters(model)
    return [
        {
            "name": name,
            "params": [parameter for _, parameter in grouped[name] if parameter.requires_grad],
            "lr": rates[name],
            "weight_decay": decay,
        }
        for name in GROUP_ORDER
        if any(parameter.requires_grad for _, parameter in grouped[name])
    ]


def build_optimizer(model: torch.nn.Module, cfg: dict) -> tuple[torch.optim.Optimizer, list[dict]]:
    optimizer_cfg = cfg.get("optimizer", {}) or {}
    optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if optimizer_type != "adamw":
        raise ValueError("optimizer.type must be adamw")
    staged = bool(((cfg.get("training", {}) or {}).get("staged_unfreezing", {}) or {}).get("enabled", False))
    if not optimizer_cfg.get("learning_rates") and not staged:
        specs = [{
            "name": "all_trainable",
            "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
            "lr": float(cfg["training"]["lr"]),
            "weight_decay": float(cfg["training"]["weight_decay"]),
        }]
    else:
        specs = optimizer_group_specification(model, cfg)
    optimizer = torch.optim.AdamW(specs)
    validate_optimizer_coverage(model, optimizer)
    summary = [
        {
            "group_name": group["name"],
            "number_of_parameters": int(sum(p.numel() for p in group["params"])),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }
        for group in optimizer.param_groups
    ]
    return optimizer, summary


@dataclass
class StagedUnfreezer:
    enabled: bool
    heads_only_epochs: int
    task_branches_epoch: int
    full_backbone_epoch: int

    @classmethod
    def from_config(cls, cfg: dict) -> "StagedUnfreezer":
        raw = (cfg.get("training", {}) or {}).get("staged_unfreezing", {}) or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            heads_only_epochs=int(raw.get("heads_only_epochs", 5)),
            task_branches_epoch=int(raw.get("task_branches_epoch", 5)),
            full_backbone_epoch=int(raw.get("full_backbone_epoch", 15)),
        )

    def allowed_groups(self, epoch: int) -> set[str]:
        if not self.enabled:
            return set(GROUP_ORDER)
        allowed = {"classification_heads", "projection_heads"}
        if epoch >= max(self.heads_only_epochs, self.task_branches_epoch):
            allowed.add("task_specific_branches")
        if epoch >= self.full_backbone_epoch:
            allowed.add("early_backbone")
            allowed.add("final_backbone_stage")
        return allowed

    def initialise(self, model: torch.nn.Module) -> int:
        if not self.enabled:
            return sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        allowed = self.allowed_groups(0)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = parameter_group_name(name) in allowed
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    def transition(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: dict,
        epoch: int,
        scheduler=None,
    ) -> tuple[bool, int]:
        allowed = self.allowed_groups(epoch)
        present = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        rates = _learning_rates(cfg)
        decay = float((cfg.get("optimizer", {}) or {}).get(
            "weight_decay", cfg["training"]["weight_decay"]
        ))
        grouped = grouped_named_parameters(model)
        changed = False
        for group_name in GROUP_ORDER:
            if group_name not in allowed:
                continue
            additions = []
            for _, parameter in grouped[group_name]:
                if not parameter.requires_grad:
                    parameter.requires_grad = True
                    changed = True
                if id(parameter) not in present:
                    additions.append(parameter)
                    present.add(id(parameter))
            if additions:
                optimizer.add_param_group({
                    "name": group_name,
                    "params": additions,
                    "lr": rates[group_name],
                    "weight_decay": decay,
                })
                if scheduler is not None and hasattr(scheduler, "base_lrs"):
                    scheduler.base_lrs.append(rates[group_name])
        validate_optimizer_coverage(model, optimizer)
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        return changed, int(count)

    def resolved_schedule(self) -> dict:
        return {
            "enabled": self.enabled,
            "heads_only_epochs": self.heads_only_epochs,
            "task_branches_epoch": self.task_branches_epoch,
            "full_backbone_epoch": self.full_backbone_epoch,
        }


__all__ = [
    "GROUP_ORDER",
    "StagedUnfreezer",
    "build_optimizer",
    "grouped_named_parameters",
    "parameter_group_name",
    "validate_optimizer_coverage",
]
