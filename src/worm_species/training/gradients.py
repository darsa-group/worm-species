"""Task-gradient diagnostics and PCGrad projection utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn


GradientList = list[torch.Tensor | None]


def trainable_shared_parameters(model: nn.Module) -> list[nn.Parameter]:
    provider = getattr(model, "shared_parameters", None)
    parameters: Iterable[nn.Parameter] = (
        provider() if callable(provider) else model.parameters()
    )
    return [parameter for parameter in parameters if parameter.requires_grad]


def task_gradients(
    objectives: dict[str, torch.Tensor],
    parameters: Sequence[nn.Parameter],
) -> dict[str, GradientList]:
    result = {}
    for task, objective in objectives.items():
        result[task] = list(torch.autograd.grad(
            objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        ))
    return result


def _flatten(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[nn.Parameter],
) -> torch.Tensor:
    pieces = [
        (
            gradient.detach().float().reshape(-1)
            if gradient is not None
            else torch.zeros(
                parameter.numel(),
                device=parameter.device,
                dtype=torch.float32,
            )
        )
        for gradient, parameter in zip(gradients, parameters)
    ]
    if not pieces:
        return torch.empty(0)
    return torch.cat(pieces)


def gradient_statistics(
    gradients: dict[str, GradientList],
    parameters: Sequence[nn.Parameter],
) -> dict[str, float]:
    flattened = {
        task: _flatten(values, parameters)
        for task, values in gradients.items()
    }
    result = {
        f"{task}_gradient_norm": float(vector.norm().item())
        for task, vector in flattened.items()
    }
    for left, right in (
        ("genus", "species"),
        ("genus", "age"),
        ("species", "age"),
    ):
        key = f"{left}_{right}_cosine"
        if left not in flattened or right not in flattened:
            result[key] = float("nan")
            continue
        left_vector = flattened[left]
        right_vector = flattened[right]
        denominator = left_vector.norm() * right_vector.norm()
        result[key] = (
            float(torch.dot(left_vector, right_vector).div(denominator).item())
            if float(denominator.item()) > 0
            else float("nan")
        )
    return result


def pcgrad_project(
    gradients: dict[str, GradientList],
    parameters: Sequence[nn.Parameter],
) -> tuple[GradientList, float]:
    """Project conflicting task gradients and return their summed update."""
    tasks = list(gradients)
    if not tasks:
        return [None for _ in parameters], 0.0
    vectors = {
        task: _flatten(gradients[task], parameters)
        for task in tasks
    }
    negative = 0
    pair_count = 0
    for left_index, left in enumerate(tasks):
        for right in tasks[left_index + 1:]:
            pair_count += 1
            if float(torch.dot(vectors[left], vectors[right]).item()) < 0:
                negative += 1

    projected = {}
    for task in tasks:
        current = vectors[task].clone()
        for other in tasks:
            if other == task:
                continue
            reference = vectors[other]
            dot = torch.dot(current, reference)
            norm_sq = torch.dot(reference, reference)
            if float(dot.item()) < 0 and float(norm_sq.item()) > 0:
                current = current - dot / norm_sq * reference
        projected[task] = current
    merged_vector = torch.stack(list(projected.values())).sum(dim=0)

    merged: GradientList = []
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        value = merged_vector[offset:offset + count].reshape_as(parameter)
        merged.append(value.to(dtype=parameter.dtype))
        offset += count
    negative_proportion = (
        float(negative / pair_count) if pair_count else 0.0
    )
    return merged, negative_proportion


def replace_shared_gradients(
    *,
    parameters: Sequence[nn.Parameter],
    raw_task_gradients: dict[str, GradientList],
    projected_gradients: Sequence[torch.Tensor | None],
) -> None:
    """Replace only shared gradients, preserving ordinary auxiliary gradients."""
    for index, parameter in enumerate(parameters):
        ordinary = parameter.grad
        raw_values = [
            values[index]
            for values in raw_task_gradients.values()
            if values[index] is not None
        ]
        raw_sum = (
            torch.stack(raw_values).sum(dim=0)
            if raw_values
            else torch.zeros_like(parameter)
        )
        auxiliary = (
            ordinary.detach() - raw_sum
            if ordinary is not None
            else -raw_sum
        )
        projected = projected_gradients[index]
        parameter.grad = (
            auxiliary
            if projected is None
            else auxiliary + projected.to(parameter)
        )


__all__ = [
    "gradient_statistics",
    "pcgrad_project",
    "replace_shared_gradients",
    "task_gradients",
    "trainable_shared_parameters",
]
