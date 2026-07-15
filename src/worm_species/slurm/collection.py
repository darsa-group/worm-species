"""Exact adapter for existing, schema-stable result aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..experiments.result_collection import collect_results as collect_dual_results


DUAL_OUTPUT_NAMES = (
    "matched_condition_results.csv",
    "matched_condition_macro_f1_long.csv",
    "failed_runs.csv",
    "rgb_model_cue_suppression_macro_f1_ratios.csv",
    "rgb_model_cue_suppression_test_metrics.csv",
    "rgb_model_cue_suppression_transform_summary.csv",
    "matched_vs_rgb_stress_test.csv",
    "condition_matrix_evaluations.csv",
    "condition_matrix_task_metrics.csv",
    "condition_matrix_collection_summary.json",
)


class CollectionError(ValueError):
    """A requested aggregation has no safe schema-preserving adapter."""


@dataclass(frozen=True)
class CollectionReport:
    results_root: str
    kind: str
    output_paths: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "results_root": self.results_root,
            "kind": self.kind,
            "output_paths": list(self.output_paths),
        }


def _normalise_kind(kind: str) -> str:
    value = kind.strip().lower().replace("_", "-")
    aliases = {
        "dual-cue": "dual-cue",
        "matched-condition": "dual-cue",
        "rgb-stress": "dual-cue",
        "matched-and-rgb-stress": "dual-cue",
    }
    if value in aliases:
        return aliases[value]
    if value in {"standard", "colour-ablation", "color-ablation"}:
        raise CollectionError(
            f"Collection kind {value!r} is intentionally unsupported until its "
            "historical collector is extracted without schema changes"
        )
    raise CollectionError(f"Unknown collection kind: {kind!r}")


def _detect_kind(root: Path) -> str:
    dual_markers = (
        "dual_cue_experiment_plan.json",
        "condition_manifest.json",
        "matched_condition_results.csv",
        "matched_vs_rgb_stress_test.csv",
    )
    if any((root / marker).is_file() for marker in dual_markers):
        return "dual-cue"
    if (root / "colour_ablation_results.csv").is_file():
        raise CollectionError(
            "Detected colour-ablation results; exact colour collection remains "
            "with the historical collector"
        )
    raise CollectionError(
        "Could not safely identify a dual-cue result root; pass kind='dual-cue' "
        "explicitly or use the historical standard/colour collector"
    )


def collect_existing_results(
    results_root: str | Path,
    *,
    kind: str = "auto",
) -> CollectionReport:
    """Delegate to the existing dual collector without adding another scan."""
    root = Path(results_root).expanduser().absolute()
    if not root.is_dir():
        raise CollectionError(f"Results root is not a directory: {root}")
    selected_kind = _detect_kind(root) if kind == "auto" else _normalise_kind(kind)
    if selected_kind != "dual-cue":
        raise CollectionError(f"Unsupported collection kind: {selected_kind}")
    collect_dual_results(root)
    outputs = tuple(
        str(root / name) for name in DUAL_OUTPUT_NAMES if (root / name).is_file()
    )
    return CollectionReport(
        results_root=str(root),
        kind=selected_kind,
        output_paths=outputs,
    )


__all__ = [
    "CollectionError",
    "CollectionReport",
    "DUAL_OUTPUT_NAMES",
    "collect_existing_results",
]
