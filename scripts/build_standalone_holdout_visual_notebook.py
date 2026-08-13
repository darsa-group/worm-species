#!/usr/bin/env python3
"""Embed all reporting code in the editable publication-figure notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "holdouts_and_visual_combinations.ipynb"


def _top_level_source(
    path: Path,
    *,
    functions: set[str] | None = None,
    classes: set[str] | None = None,
    include_assignments: bool = False,
    excluded_functions: set[str] | None = None,
) -> str:
    """Copy selected top-level definitions without importing a local script."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    blocks: list[str] = []
    excluded = excluded_functions or set()
    for node in tree.body:
        selected_function = (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name not in excluded
            and (functions is None or node.name in functions)
        )
        selected_assignment = include_assignments and isinstance(
            node, (ast.Assign, ast.AnnAssign)
        )
        selected_class = (
            isinstance(node, ast.ClassDef)
            and classes is not None
            and node.name in classes
        )
        if not (selected_function or selected_assignment or selected_class):
            continue
        segment = ast.get_source_segment(source, node)
        if segment:
            blocks.append(segment.rstrip())
    return "\n\n\n".join(blocks) + "\n"


def embedded_helpers() -> str:
    preamble = '''\
# Standalone notebook implementation. This cell intentionally contains the
# result readers, statistics, and plotting functions used by every figure.
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image as DisplayImage, display
from matplotlib.ticker import PercentFormatter
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as transform_functional
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

try:
    import cv2
except ImportError:
    cv2 = None
else:
    cv2.setNumThreads(0)

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

SplitName = Literal["train", "validation", "test"]
DEFAULT_IMAGE_SIZE = 224
DEFAULT_NORMALISATION_MEAN = (0.485, 0.456, 0.406)
DEFAULT_NORMALISATION_STD = (0.229, 0.224, 0.225)
'''
    paper_functions = {
        "_loss_name",
        "_condition_parameter",
        "collect_runs",
        "_split_paths",
        "load_split_frames",
        "dataset_composition_table",
    }
    adult_functions = {
        "_read_json",
        "_read_csv",
        "_hierarchy_weight",
        "_loss_recipe",
        "_control_definitions",
        "_collect_stage",
        "collect_adult_taxon_metrics",
    }
    metric_functions = {
        "_safe_divide",
        "_expected_calibration_error",
        "classification_metric_summary",
    }
    condition_classes = {
        "ColourRetention",
        "ChannelShuffle",
        "BinaryForegroundMask",
        "TensorGaussianBlur",
        "GaussianBlurPercent",
        "ResolutionLoss",
        "TensorBilateralFilter",
        "PatchShuffle",
    }
    condition_functions = {
        "_condition_parameters",
        "build_condition_operations",
    }
    transform_functions = {
        "_mapping",
        "_enabled_operation",
        "_augmentation_operations",
        "_normalisation_operation",
        "build_split_transform",
    }
    support = "\n\n".join((
        _top_level_source(
            PROJECT_ROOT / "scripts" / "build_paper_results.py",
            functions=paper_functions,
        ),
        _top_level_source(
            PROJECT_ROOT / "scripts" / "build_adult_taxon_ablation_results.py",
            functions=adult_functions,
        ),
        _top_level_source(
            PROJECT_ROOT / "src" / "worm_species" / "training" / "metrics.py",
            functions=metric_functions,
        ),
        _top_level_source(
            PROJECT_ROOT / "src" / "worm_species" / "data" / "conditions.py",
            functions=condition_functions,
            classes=condition_classes,
        ),
        _top_level_source(
            PROJECT_ROOT / "src" / "worm_species" / "data" / "transforms.py",
            functions=transform_functions,
        ),
    ))
    plotting = _top_level_source(
        PROJECT_ROOT / "scripts" / "build_holdout_visual_notebook.py",
        include_assignments=True,
        excluded_functions={"main"},
    )
    return preamble + "\n\n" + support + "\n\n" + plotting


def _source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source_lines(source),
    }


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _source_lines(source),
    }


def build_notebook() -> dict:
    existing = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    helper_source = embedded_helpers()
    settings = '''\
# Edit these paths and selections, then run the preparation cell once.
PAPER_ROOT = PROJECT_ROOT / "publication_30seed_result"
TAXON_STAGE_ROOT = PAPER_ROOT
OUTPUT_DIR = PAPER_ROOT / "publication_bundle" / "figures"
SOURCE_ROOT = OUTPUT_DIR / "figure_sources"
DATA_ROOT = PROJECT_ROOT.parent / "petridish-worm-images"
VISUAL_MODEL = "convnext_base"
SPECIES_ABLATION = "Aporrectodea_longa"

def show(stem: str) -> None:
    path = OUTPUT_DIR / f"{stem}.png"
    if path.exists():
        display(DisplayImage(filename=str(path)))
    else:
        print(f"Not created; completed input rows are missing: {path}")
'''
    preparation = '''\
runs = collect_runs(PAPER_ROOT)
baseline_frame = prepare_baseline_frame(runs)
age_metrics, age_confusions = prepare_developmental_stage_diagnostics(baseline_frame)
visual_frames = prepare_convnext_visual_frames(runs, VISUAL_MODEL)
chance_reference = visual_uniform_chance_reference(runs, PROJECT_ROOT)
chance_row = chance_reference[chance_reference["task"].eq("mean")]
visual_chance = (
    float(chance_row.iloc[0]["expected_uniform_macro_f1"])
    if not chance_row.empty else float("nan")
)
for frame in visual_frames.values():
    frame["chance"] = visual_chance

taxon_frame = _paper_design_only(
    _model_only(
        prepare_taxon_stage_holdout_frame(
            collect_adult_taxon_metrics(TAXON_STAGE_ROOT)
        ),
        TAXON_MODEL,
        context="data-ablation figures",
    ),
    require_loss_recipe=True,
)
paired = pair_taxon_metrics(taxon_frame)
paired, taxon_counts = attach_taxon_individual_counts(paired, PROJECT_ROOT)
species_paired = (
    paired[
        paired.get("species", pd.Series(index=paired.index, dtype=str))
        .astype(str).eq(SPECIES_ABLATION)
    ].copy()
    if not paired.empty else paired
)
biological_questions = prepare_biological_question_frame(
    taxon_frame, PROJECT_ROOT
)

print(
    f"Loaded {len(runs)} completed model runs, "
    f"{len(paired)} paired data-ablation rows, and "
    f"{len(biological_questions)} biological-question rows."
)
'''
    cells = [
        _markdown(
            "# Publication figures — standalone and editable\n\n"
            "This notebook embeds its result readers, statistics, uncertainty "
            "calculations, image transformations, and plotting functions. It "
            "does not import any local project module or reporting script. Run "
            "the helper, settings, and preparation cells "
            "once; then edit and rerun any individual figure cell. Completed "
            "validation-selected checkpoints and the independent test split "
            "remain the only scientific inputs.\n"
        ),
        _markdown(
            "## Embedded helper implementation\n\n"
            "Normally leave this cell unchanged. It makes this single notebook "
            "the complete reporting implementation.\n"
        ),
        _code(helper_source),
        _markdown("## Editable paths and figure selections\n"),
        _code(settings),
        _markdown(
            "## Prepare completed-run data\n\n"
            "This reads files only; it never trains or submits jobs.\n"
        ),
        _code(preparation),
    ]

    # Retain the approved one-markdown/one-code-cell figure layout while
    # replacing the old module-qualified calls with embedded function calls.
    figure_start = next(
        index
        for index, cell in enumerate(existing["cells"])
        if cell.get("cell_type") == "markdown"
        and "".join(cell.get("source", [])).startswith("## Figure 1 ")
    )
    for cell in existing["cells"][figure_start:]:
        copied = json.loads(json.dumps(cell))
        if copied.get("cell_type") == "code":
            source = "".join(copied.get("source", [])).replace(
                "figure_builder.", ""
            )
            copied["source"] = _source_lines(source)
            copied["execution_count"] = None
            copied["outputs"] = []
        cells.append(copied)

    return {
        "cells": cells,
        "metadata": existing.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    payload = build_notebook()
    NOTEBOOK_PATH.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote standalone notebook with {len(payload['cells'])} cells: "
        f"{NOTEBOOK_PATH}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
