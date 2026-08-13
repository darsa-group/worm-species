#!/usr/bin/env python3
"""Build the main/supplementary figures and an auditable publication record."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_holdout_visual_notebook import (
    PAPER_LOSS_WEIGHTS,
    PAPER_SEEDS,
    build_holdout_visual_notebook_figures,
)


CONFIG_NAMES = (
    "genome_publication_30seed_pipeline.yaml",
    "genome_publication_30seed_baseline.yaml",
    "genome_publication_30seed_visual.yaml",
    "genome_publication_30seed_interactions.yaml",
    "genome_publication_30seed_taxon_baseline.yaml",
    "genome_publication_30seed_taxon_holdouts.yaml",
    "paper_report_style.yaml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _file_inventory(paper_root: Path, pattern: str) -> pd.DataFrame:
    columns = ("path", "relative_path", "bytes", "sha256")
    rows = []
    for path in sorted(paper_root.glob(pattern)):
        if path.is_file():
            rows.append({
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(paper_root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    return pd.DataFrame(rows, columns=columns)


def build_publication_bundle(
    paper_root: Path,
    *,
    split_root: Path,
    data_root: Path,
) -> dict:
    paper_root = Path(paper_root).resolve()
    bundle_root = paper_root / "publication_bundle"
    figures_root = bundle_root / "figures"
    metadata_root = bundle_root / "metadata"
    configs_root = bundle_root / "configs"
    for directory in (figures_root, metadata_root, configs_root):
        directory.mkdir(parents=True, exist_ok=True)

    figure_manifest = build_holdout_visual_notebook_figures(
        paper_root,
        output_dir=figures_root,
        taxon_stage_root=paper_root,
        visual_model="convnext_base",
        split_root=split_root,
        data_root=data_root,
    )

    for name in CONFIG_NAMES:
        source = PROJECT_ROOT / "dev" / name
        if source.is_file():
            shutil.copy2(source, configs_root / name)

    inventories = {
        "best_checkpoints": _file_inventory(paper_root, "runs/**/best_model.pt"),
        "resolved_run_configs": _file_inventory(paper_root, "runs/**/config.json"),
        "label_maps": _file_inventory(paper_root, "runs/**/label_to_index_by_task.json"),
        "model_parameters": _file_inventory(paper_root, "runs/**/model_parameters.json"),
        "split_summaries": _file_inventory(paper_root, "runs/**/split_summary.json"),
        "test_metrics": _file_inventory(paper_root, "runs/**/test_metrics_best.json"),
        "test_predictions": _file_inventory(paper_root, "runs/**/test_predictions_best.csv"),
        "data_ablation_target_metrics": _file_inventory(
            paper_root, "runs/**/target_class_metrics_full_test.csv"
        ),
        "training_histories": _file_inventory(paper_root, "runs/**/history.csv"),
        "pipeline_manifests": _file_inventory(paper_root, "artifacts/**/pipeline_manifest.json"),
    }
    for name, frame in inventories.items():
        frame.to_csv(metadata_root / f"{name}.csv", index=False)

    split_rows = []
    for filename in ("train_split.csv", "val_split.csv", "test_split.csv"):
        candidates = (Path(split_root) / "split_csv" / filename, Path(split_root) / filename)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is not None:
            split_rows.append({"split": filename, "path": str(path.resolve()), "sha256": _sha256(path)})
    pd.DataFrame(split_rows).to_csv(metadata_root / "split_hashes.csv", index=False)

    packages = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    (metadata_root / "environment.txt").write_text(
        "\n".join([
            f"python={platform.python_version()}",
            f"platform={platform.platform()}",
            *packages,
        ]) + "\n",
        encoding="utf-8",
    )
    git_status = _git_value("status", "--porcelain") or ""
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_status),
        "git_status": git_status.splitlines(),
        "seeds": list(PAPER_SEEDS),
        "loss_weights": PAPER_LOSS_WEIGHTS,
        "hierarchy_loss_weight": 0.0,
        "selection_metric": "validation total weighted loss",
        "checkpoint_policy": "best only",
        "reported_split": "test only",
        "visual_protocol": "matched condition in training, validation, and test",
        "figure_manifest": figure_manifest,
        "inventory_counts": {name: int(len(frame)) for name, frame in inventories.items()},
    }
    (metadata_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle_root / "README.md").write_text(
        "# Publication bundle\n\n"
        "Main and supplementary figures are under `figures/`; every figure has PNG, PDF, SVG, source CSVs, and a manifest. "
        "`metadata/` records checksums and absolute locations of the retained best checkpoints, resolved configs, label maps, split summaries, test metrics, exact test predictions, and training histories. "
        "Only validation loss selected checkpoints; all reported performance is from the test split.\n",
        encoding="utf-8",
    )
    manifest = {
        "bundle_root": str(bundle_root),
        "figures": figure_manifest["figures"],
        "metadata": {name: str(metadata_root / f"{name}.csv") for name in inventories},
        "provenance": str(metadata_root / "provenance.json"),
    }
    (bundle_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-result", type=Path, default=Path("publication_30seed_result"))
    parser.add_argument("--split-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("../petridish-worm-images"))
    parser.add_argument("--style", type=Path, help="Accepted for pipeline compatibility; figure styling is fixed in code.")
    args = parser.parse_args()
    print(json.dumps(build_publication_bundle(
        args.paper_result, split_root=args.split_root, data_root=args.data_root,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
