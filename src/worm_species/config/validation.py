from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from .schema import CONFIG_FIELDS, MISSING_DEFAULT, field_for_path, is_known_config_path


Workflow = Literal["auto", "training", "run_specs", "saved"]
KNOWN_TRANSFORMS = frozenset({
    "original",
    "saturation",
    "grayscale",
    "channel_shuffle",
    "bilateral_filter",
    "gaussian_blur",
    "patch_shuffle",
})
_ABSENT = object()


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class ConfigValidationError(ValueError):
    def __init__(self, issues: Iterable[ValidationIssue]):
        self.issues = tuple(issues)
        if not self.issues:
            raise ValueError("ConfigValidationError requires at least one issue")
        super().__init__("Invalid configuration:\n" + "\n".join(
            f"  - {issue}" for issue in self.issues
        ))


def _get(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _ABSENT
        value = value[part]
    return value


def _is_type(value: Any, expected: tuple[type, ...]) -> bool:
    if value is None:
        return type(None) in expected
    if isinstance(value, bool):
        return bool in expected
    numeric = tuple(item for item in expected if item in {int, float})
    if numeric and isinstance(value, (int, float)):
        return True
    return isinstance(value, expected)


def _type_names(expected: tuple[type, ...]) -> str:
    return " or ".join(item.__name__ for item in expected)


def _number(
    issues: list[ValidationIssue],
    path: str,
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(ValidationIssue(path, "must be a number"))
        return False
    numeric = float(value)
    if not math.isfinite(numeric):
        issues.append(ValidationIssue(path, "must be finite"))
        return False
    if minimum is not None:
        invalid = numeric <= minimum if exclusive_minimum else numeric < minimum
        if invalid:
            operator = ">" if exclusive_minimum else ">="
            issues.append(ValidationIssue(path, f"must be {operator} {minimum:g}"))
            return False
    if maximum is not None and numeric > maximum:
        issues.append(ValidationIssue(path, f"must be <= {maximum:g}"))
        return False
    return True


def resolve_workflow(config: dict[str, Any], workflow: Workflow = "auto") -> str:
    if workflow != "auto":
        return workflow
    if bool(_get(config, "input_condition.enabled") is True):
        return "training"
    if bool(_get(config, "matched_condition_training.enabled") is True):
        return "run_specs"
    return "training"


def validate_override_items(items: Iterable[str]) -> tuple[str, ...]:
    """Validate dotted override keys without changing legacy value parsing."""
    issues: list[ValidationIssue] = []
    keys: list[str] = []
    for item in items:
        if "=" not in item:
            issues.append(ValidationIssue("override", f"must look like key=value, got {item!r}"))
            continue
        raw_key, _ = item.split("=", 1)
        key = raw_key.strip()
        if not key:
            issues.append(ValidationIssue("override", "key must not be empty"))
            continue
        if not is_known_config_path(key):
            issues.append(ValidationIssue(key, "unknown configuration override path"))
            continue
        keys.append(key)
    if issues:
        raise ConfigValidationError(issues)
    return tuple(keys)


def _validate_channel_order(
    issues: list[ValidationIssue], path: str, order: Any
) -> None:
    if isinstance(order, str):
        try:
            order = [int(item.strip()) for item in order.split(",")]
        except ValueError:
            issues.append(ValidationIssue(path, "must be a permutation of 0,1,2"))
            return
    if not isinstance(order, (list, tuple)):
        issues.append(ValidationIssue(path, "must be a list, tuple, or comma-separated string"))
        return
    if any(isinstance(item, bool) or not isinstance(item, int) for item in order):
        issues.append(ValidationIssue(path, "must contain integer channel indices"))
        return
    if sorted(order) != [0, 1, 2]:
        issues.append(ValidationIssue(path, f"must be a permutation of [0, 1, 2], got {list(order)!r}"))


def _validate_transform_parameters(
    config: dict[str, Any], issues: list[ValidationIssue]
) -> None:
    cue = config.get("test_cue_suppression", {}) or {}
    if not isinstance(cue, dict):
        return

    saturation = cue.get("saturation", {}) or {}
    if isinstance(saturation, dict):
        values = saturation.get("values", _ABSENT)
        if values is not _ABSENT:
            if not isinstance(values, list) or not values:
                issues.append(ValidationIssue(
                    "test_cue_suppression.saturation.values", "must be a non-empty list"
                ))
            elif all(_number(issues, f"test_cue_suppression.saturation.values[{index}]", value, minimum=0, maximum=1)
                     for index, value in enumerate(values)):
                pass
        for key, default in (("start", 1.0), ("stop", 0.0)):
            _number(
                issues,
                f"test_cue_suppression.saturation.{key}",
                saturation.get(key, default),
                minimum=0,
                maximum=1,
            )
        _number(
            issues,
            "test_cue_suppression.saturation.step",
            saturation.get("step", 0.01),
            minimum=0,
            exclusive_minimum=True,
        )

    channel = cue.get("channel_shuffle", {}) or {}
    if isinstance(channel, dict):
        orders = channel.get("orders", [[2, 0, 1]])
        if not isinstance(orders, list) or not orders:
            issues.append(ValidationIssue(
                "test_cue_suppression.channel_shuffle.orders", "must be a non-empty list"
            ))
        else:
            for index, order in enumerate(orders):
                _validate_channel_order(
                    issues,
                    f"test_cue_suppression.channel_shuffle.orders[{index}]",
                    order,
                )

    gaussian = cue.get("gaussian_blur", {}) or {}
    if isinstance(gaussian, dict):
        sigmas = gaussian.get("sigmas", [0.5, 1.0, 2.0, 4.0])
        if not isinstance(sigmas, list) or not sigmas:
            issues.append(ValidationIssue(
                "test_cue_suppression.gaussian_blur.sigmas", "must be a non-empty list"
            ))
        else:
            for index, sigma in enumerate(sigmas):
                _number(
                    issues,
                    f"test_cue_suppression.gaussian_blur.sigmas[{index}]",
                    sigma,
                    minimum=0,
                    exclusive_minimum=True,
                )

    bilateral = cue.get("bilateral_filter", {}) or {}
    if isinstance(bilateral, dict):
        settings = bilateral.get("settings", [
            {"diameter": 5, "sigma_colour": 25, "sigma_space": 25},
            {"diameter": 7, "sigma_colour": 50, "sigma_space": 50},
            {"diameter": 9, "sigma_colour": 100, "sigma_space": 100},
        ])
        if not isinstance(settings, list) or not settings:
            issues.append(ValidationIssue(
                "test_cue_suppression.bilateral_filter.settings", "must be a non-empty list"
            ))
        else:
            for index, setting in enumerate(settings):
                path = f"test_cue_suppression.bilateral_filter.settings[{index}]"
                if not isinstance(setting, dict):
                    issues.append(ValidationIssue(path, "must be a mapping"))
                    continue
                diameter = setting.get("diameter", _ABSENT)
                if isinstance(diameter, bool) or not isinstance(diameter, int):
                    issues.append(ValidationIssue(f"{path}.diameter", "must be an integer"))
                elif diameter <= 0 or diameter % 2 == 0:
                    issues.append(ValidationIssue(f"{path}.diameter", "must be a positive odd integer"))
                for key in ("sigma_colour", "sigma_space"):
                    if key not in setting:
                        issues.append(ValidationIssue(f"{path}.{key}", "is required"))
                    else:
                        _number(
                            issues,
                            f"{path}.{key}",
                            setting[key],
                            minimum=0,
                            exclusive_minimum=True,
                        )

    patch = cue.get("patch_shuffle", {}) or {}
    if isinstance(patch, dict):
        grids = patch.get("grid_sizes", [2, 4, 8])
        image_size = _get(config, "data.image_size")
        if not isinstance(grids, list) or not grids:
            issues.append(ValidationIssue(
                "test_cue_suppression.patch_shuffle.grid_sizes", "must be a non-empty list"
            ))
        else:
            for index, grid in enumerate(grids):
                path = f"test_cue_suppression.patch_shuffle.grid_sizes[{index}]"
                if isinstance(grid, bool) or not isinstance(grid, int):
                    issues.append(ValidationIssue(path, "must be an integer"))
                elif grid < 2:
                    issues.append(ValidationIssue(path, "must be >= 2"))
                elif isinstance(image_size, int) and image_size % grid != 0:
                    issues.append(ValidationIssue(path, f"must divide data.image_size={image_size}"))

    raw = config.get("input_condition", {}) or {}
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        return
    transform = str(raw.get("transform", "original")).lower()
    if transform not in KNOWN_TRANSFORMS:
        issues.append(ValidationIssue(
            "input_condition.transform",
            f"unknown transformation {transform!r}; expected one of {sorted(KNOWN_TRANSFORMS)!r}",
        ))
        return
    if transform == "saturation":
        if "retention" not in raw:
            issues.append(ValidationIssue("input_condition.retention", "is required for saturation"))
        else:
            _number(issues, "input_condition.retention", raw["retention"], minimum=0, maximum=1)
    elif transform == "channel_shuffle":
        _validate_channel_order(issues, "input_condition.order", raw.get("order", [2, 0, 1]))
    elif transform == "gaussian_blur":
        if "sigma" not in raw:
            issues.append(ValidationIssue("input_condition.sigma", "is required for gaussian_blur"))
        else:
            _number(issues, "input_condition.sigma", raw["sigma"], minimum=0, exclusive_minimum=True)
    elif transform == "bilateral_filter":
        for key in ("diameter", "sigma_colour", "sigma_space"):
            if key not in raw:
                issues.append(ValidationIssue(f"input_condition.{key}", "is required for bilateral_filter"))
        diameter = raw.get("diameter")
        if diameter is not None and (
            isinstance(diameter, bool) or not isinstance(diameter, int)
            or diameter <= 0 or diameter % 2 == 0
        ):
            issues.append(ValidationIssue("input_condition.diameter", "must be a positive odd integer"))
        for key in ("sigma_colour", "sigma_space"):
            if key in raw:
                _number(issues, f"input_condition.{key}", raw[key], minimum=0, exclusive_minimum=True)
    elif transform == "patch_shuffle":
        grid = raw.get("grid_size", _ABSENT)
        if grid is _ABSENT:
            issues.append(ValidationIssue("input_condition.grid_size", "is required for patch_shuffle"))
        elif isinstance(grid, bool) or not isinstance(grid, int) or grid < 2:
            issues.append(ValidationIssue("input_condition.grid_size", "must be an integer >= 2"))
        else:
            image_size = _get(config, "data.image_size")
            if isinstance(image_size, int) and image_size % grid != 0:
                issues.append(ValidationIssue(
                    "input_condition.grid_size", f"must divide data.image_size={image_size}"
                ))


def _validate_sweeps(
    config: dict[str, Any], issues: list[ValidationIssue], workflow: str
) -> None:
    sweep = config.get("sweep", {}) or {}
    if isinstance(sweep, dict) and bool(sweep.get("enabled", False)):
        parameters = sweep.get("parameters", {})
        if not isinstance(parameters, dict):
            return
        for key, values in parameters.items():
            path = f"sweep.parameters.{key}"
            field = field_for_path(str(key))
            if field is None:
                issues.append(ValidationIssue(path, "references an unknown configuration path"))
            if not isinstance(values, list) or not values:
                issues.append(ValidationIssue(path, "must be a non-empty list"))
                continue
            if field is not None:
                for index, value in enumerate(values):
                    value_path = f"{path}[{index}]"
                    if not _is_type(value, field.expected_types):
                        issues.append(ValidationIssue(
                            value_path,
                            f"must be {_type_names(field.expected_types)}, "
                            f"got {type(value).__name__}",
                        ))
                    elif field.choices and value not in field.choices:
                        issues.append(ValidationIssue(
                            value_path, f"must be one of {list(field.choices)!r}"
                        ))

    colour = config.get("colour_ablation", {}) or {}
    if not isinstance(colour, dict) or not bool(colour.get("enabled", False)):
        return
    for key, default in (("start_percent", 100), ("stop_percent", 0)):
        value = colour.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if not 0 <= value <= 100:
            issues.append(ValidationIssue(f"colour_ablation.{key}", "must be in [0, 100]"))
    step = colour.get("step_percent", 1)
    if isinstance(step, int) and not isinstance(step, bool) and step <= 0:
        issues.append(ValidationIssue("colour_ablation.step_percent", "must be > 0"))
    parameters = sweep.get("parameters", {}) if isinstance(sweep, dict) else {}
    if bool(sweep.get("enabled", False)) and parameters:
        if not bool(colour.get("combine_with_sweep", False)):
            issues.append(ValidationIssue(
                "colour_ablation.combine_with_sweep",
                "must be true when colour ablation and an ordinary sweep are both enabled",
            ))
        if isinstance(parameters, dict) and "data.colour_retention" in parameters:
            issues.append(ValidationIssue(
                "sweep.parameters.data.colour_retention",
                "duplicates colour_ablation expansion",
            ))
    if workflow == "run_specs":
        issues.append(ValidationIssue(
            "colour_ablation.enabled",
            "must be false while external matched-condition run specifications are expanded",
        ))


def _validate_tasks(config: dict[str, Any], issues: list[ValidationIssue]) -> None:
    target_cols = _get(config, "data.target_cols")
    if target_cols is _ABSENT:
        return
    if not isinstance(target_cols, dict) or not target_cols:
        issues.append(ValidationIssue("data.target_cols", "must be a non-empty mapping"))
        return
    for task, column in target_cols.items():
        if not isinstance(task, str) or not task.strip():
            issues.append(ValidationIssue("data.target_cols", "task names must be non-empty strings"))
        if not isinstance(column, str) or not column.strip():
            issues.append(ValidationIssue(f"data.target_cols.{task}", "column must be a non-empty string"))

    multi = config.get("multi_task", {}) or {}
    if not isinstance(multi, dict):
        return
    weights = multi.get("loss_weights", {}) or {}
    if isinstance(weights, dict):
        for task in weights:
            if task != "hierarchy" and task not in target_cols:
                issues.append(ValidationIssue(
                    f"multi_task.loss_weights.{task}", "task is not present in data.target_cols"
                ))
    selection = multi.get("selection_metric", "mean_macro_f1")
    allowed_metrics = {"mean_macro_f1", *(f"{task}_macro_f1" for task in target_cols)}
    if isinstance(selection, str) and selection not in allowed_metrics:
        issues.append(ValidationIssue(
            "multi_task.selection_metric", f"must be one of {sorted(allowed_metrics)!r}"
        ))
    hierarchy = multi.get("hierarchy_loss", {}) or {}
    if isinstance(hierarchy, dict) and bool(hierarchy.get("enabled", False)):
        parent = hierarchy.get("parent_task", "genus")
        child = hierarchy.get("child_task", "species")
        if parent == child:
            issues.append(ValidationIssue("multi_task.hierarchy_loss", "parent_task and child_task must differ"))
        for key, task in (("parent_task", parent), ("child_task", child)):
            if task not in target_cols:
                issues.append(ValidationIssue(
                    f"multi_task.hierarchy_loss.{key}", "must name a task in data.target_cols"
                ))


def _validate_paths(config: dict[str, Any], issues: list[ValidationIssue]) -> None:
    data_root = _get(config, "data.root_dir")
    metadata = _get(config, "data.metadata_csv")
    if isinstance(data_root, str) and not Path(data_root).is_dir():
        issues.append(ValidationIssue("data.root_dir", f"directory does not exist: {data_root}"))
    if isinstance(metadata, str) and not Path(metadata).is_file():
        issues.append(ValidationIssue("data.metadata_csv", f"file does not exist: {metadata}"))

    use_predefined = _get(config, "split.use_predefined_splits")
    split_root = _get(config, "split.predefined_split_dir")
    if use_predefined is True and isinstance(split_root, str):
        split_dir = Path(split_root) / "split_csv"
        for filename in ("train_split.csv", "val_split.csv", "test_split.csv"):
            if not (split_dir / filename).is_file():
                issues.append(ValidationIssue(
                    "split.predefined_split_dir",
                    f"missing required predefined split: {split_dir / filename}",
                ))

    output = _get(config, "output.out_dir")
    if isinstance(output, str):
        output_path = Path(output).resolve(strict=False)
        protected = {Path.cwd().resolve(strict=False)}
        if isinstance(data_root, str):
            protected.add(Path(data_root).resolve(strict=False))
        if isinstance(split_root, str):
            protected.add(Path(split_root).resolve(strict=False))
        if output_path in protected:
            issues.append(ValidationIssue(
                "output.out_dir", "must not be the repository, data, or predefined-split root"
            ))


def _validate_model_name(
    config: dict[str, Any], issues: list[ValidationIssue]
) -> None:
    names: list[tuple[str, Any]] = [("model.name", _get(config, "model.name"))]
    sweep = config.get("sweep", {}) or {}
    parameters = sweep.get("parameters", {}) if isinstance(sweep, dict) else {}
    sweep_models = parameters.get("model.name") if isinstance(parameters, dict) else None
    if isinstance(sweep_models, list):
        names.extend(
            (f"sweep.parameters.model.name[{index}]", name)
            for index, name in enumerate(sweep_models)
        )
    names = [(path, name) for path, name in names if isinstance(name, str) and name]
    if not names:
        return
    try:
        from torchvision import models
    except Exception as exc:  # pragma: no cover - environment-specific dependency failure
        issues.append(ValidationIssue(
            "model.name", f"could not inspect torchvision model registry: {exc}"
        ))
        return
    for path, name in names:
        if not callable(getattr(models, name, None)):
            issues.append(ValidationIssue(path, f"unknown torchvision model {name!r}"))


def _validate_canonical_training_switches(
    config: dict[str, Any], issues: list[ValidationIssue]
) -> None:
    """Validate config-driven training without changing legacy profiles."""
    legacy_profile = _get(config, "training.profile")
    if legacy_profile is not _ABSENT:
        return
    try:
        from ..training.modes import infer_experiment_type
        from ..training.modes import resolve_configured_profile
        from ..training.modes import validate_training_semantics

        profile = resolve_configured_profile(config)
        validate_training_semantics(
            config,
            profile,
            infer_experiment_type(config),
        )
    except ValueError as exc:
        issues.append(ValidationIssue("training", str(exc)))


def validate_config(
    config: dict[str, Any],
    *,
    workflow: Workflow = "auto",
    check_paths: bool = False,
    check_model_registry: bool = True,
) -> dict[str, Any]:
    """Validate without coercing, default-injecting, or mutating ``config``."""
    if not isinstance(config, dict):
        raise ConfigValidationError((ValidationIssue("config", "must be a mapping"),))
    resolved_workflow = resolve_workflow(config, workflow)
    issues: list[ValidationIssue] = []

    for section in (
        "wandb", "data", "multi_task", "multitask", "early_stopping", "split",
        "model", "training", "output", "cache", "colour_ablation", "experiment",
        "test_cue_suppression", "matched_condition_training", "sweep", "input_condition",
    ):
        value = config.get(section, _ABSENT)
        if value is not _ABSENT and value is not None and not isinstance(value, dict):
            issues.append(ValidationIssue(section, "must be a mapping"))

    for field in CONFIG_FIELDS:
        if field.path.endswith(".*"):
            continue
        value = _get(config, field.path)
        if value is _ABSENT:
            if resolved_workflow in field.required_in and field.default is MISSING_DEFAULT:
                issues.append(ValidationIssue(field.path, f"is required for {resolved_workflow}"))
            continue
        if not _is_type(value, field.expected_types):
            issues.append(ValidationIssue(
                field.path,
                f"must be {_type_names(field.expected_types)}, got {type(value).__name__}",
            ))
            continue
        if isinstance(value, float) and not math.isfinite(value):
            issues.append(ValidationIssue(field.path, "must be finite"))
        if field.choices and value not in field.choices:
            issues.append(ValidationIssue(field.path, f"must be one of {list(field.choices)!r}"))

    for path, minimum, maximum, exclusive in (
        ("data.colour_retention", 0, 1, False),
        ("data.min_individuals_per_class", 0, None, False),
        ("data.crop_pad", 0, None, False),
        ("data.image_size", 0, None, True),
        ("training.epochs", 0, None, True),
        ("training.batch_size", 0, None, True),
        ("training.lr", 0, None, True),
        ("training.weight_decay", 0, None, False),
        ("training.num_workers", 0, None, False),
        ("training.val_interval", 0, None, True),
        ("early_stopping.patience", 0, None, False),
        ("early_stopping.min_delta", 0, None, False),
        ("multi_task.hierarchy_loss.weight", 0, None, False),
    ):
        value = _get(config, path)
        if value is not _ABSENT and _is_type(value, (int, float)):
            _number(
                issues, path, value, minimum=minimum, maximum=maximum,
                exclusive_minimum=exclusive,
            )

    test_size = _get(config, "split.test_size")
    val_size = _get(config, "split.val_size")
    if test_size is not _ABSENT and _is_type(test_size, (int, float)):
        _number(issues, "split.test_size", test_size, minimum=0, maximum=1, exclusive_minimum=True)
        if float(test_size) >= 1:
            issues.append(ValidationIssue("split.test_size", "must be < 1"))
    if val_size is not _ABSENT and _is_type(val_size, (int, float)):
        _number(issues, "split.val_size", val_size, minimum=0, maximum=1, exclusive_minimum=True)
        if float(val_size) >= 1:
            issues.append(ValidationIssue("split.val_size", "must be < 1"))
    if all(value is not _ABSENT and _is_type(value, (int, float)) for value in (test_size, val_size)):
        if float(test_size) + float(val_size) >= 1:
            issues.append(ValidationIssue("split", "test_size + val_size must be < 1"))

    for mapping_path in (
        "data.min_individuals_per_class_by_task", "multi_task.loss_weights"
    ):
        mapping = _get(config, mapping_path)
        if isinstance(mapping, dict):
            for key, value in mapping.items():
                if _is_type(value, (int, float)):
                    _number(issues, f"{mapping_path}.{key}", value, minimum=0)

    for path in ("training.mode", "experiment.type"):
        value = _get(config, path)
        if isinstance(value, str) and not value.strip():
            issues.append(ValidationIssue(path, "must be a non-empty string"))

    _validate_tasks(config, issues)
    _validate_transform_parameters(config, issues)
    _validate_sweeps(config, issues, resolved_workflow)
    _validate_canonical_training_switches(config, issues)

    if resolved_workflow == "run_specs" and _get(config, "input_condition.enabled") is True:
        issues.append(ValidationIssue(
            "input_condition.enabled",
            "must be false while external matched-condition run specifications are expanded",
        ))

    if resolved_workflow == "training":
        raw = config.get("input_condition", {}) or {}
        if isinstance(raw, dict) and bool(raw.get("enabled", False)):
            for path in (
                "matched_condition_training.enabled",
                "sweep.enabled",
                "colour_ablation.enabled",
            ):
                if _get(config, path) is True:
                    issues.append(ValidationIssue(
                        path, "must be false when executing one externally expanded input_condition"
                    ))
            transform = str(raw.get("transform", "original")).lower()
            if _get(config, "test_cue_suppression.enabled") is True and transform != "original":
                issues.append(ValidationIssue(
                    "test_cue_suppression.enabled",
                    "fixed-RGB stress evaluation is only valid for an original/RGB training condition",
                ))

    if bool(_get(config, "matched_condition_training.enabled") is True):
        try:
            from ..experiments.conditions import generate_conditions

            generate_conditions(config)
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(ValidationIssue("matched_condition_training", str(exc)))

    if check_model_registry:
        _validate_model_name(config, issues)
    if check_paths:
        _validate_paths(config, issues)
    if issues:
        raise ConfigValidationError(issues)
    return config


__all__ = [
    "ConfigValidationError",
    "KNOWN_TRANSFORMS",
    "ValidationIssue",
    "resolve_workflow",
    "validate_config",
    "validate_override_items",
]
