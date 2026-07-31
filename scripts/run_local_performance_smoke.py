#!/usr/bin/env python3
"""Create a tiny local data/SLURM simulation and run five real CPU epochs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/train/performance/local_5epoch_smoke.yaml"


def build_synthetic_data(simulation_root: Path) -> tuple[Path, Path]:
    data_root = simulation_root / "data"
    split_root = simulation_root / "predefined_splits"
    image_root = data_root / "01_Segmented"
    split_dir = split_root / "split_csv"
    image_root.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    species = (
        ("Aporrectodea", "Aporrectodea_longa"),
        ("Aporrectodea", "Aporrectodea_rosea"),
        ("Lumbricus", "Lumbricus_terrestris"),
        ("Lumbricus", "Lumbricus_rubellus"),
    )
    rng = np.random.default_rng(20260731)
    split_rows = {"train": [], "val": [], "test": []}
    for genus_index, (genus, species_label) in enumerate(species):
        for age_index, age in enumerate(("Juvenile", "Adult")):
            for individual_index, split in enumerate(("train", "val", "test")):
                barcode = f"{species_label}_{age}_{individual_index}"
                for image_index in range(2):
                    relative = Path("01_Segmented") / barcode / f"image_{image_index}.jpg"
                    destination = data_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    array = rng.integers(0, 35, size=(64, 64, 3), dtype=np.uint8)
                    array[..., genus_index % 3] += 120
                    array[8 + age_index * 20:24 + age_index * 20, 8:56] += 70
                    Image.fromarray(array).save(destination, quality=90)
                    row = {
                        "timestamp": "2026-07-31T00:00:00",
                        "barcode": barcode,
                        "filename": destination.name,
                        "rel_path_raw": str(relative),
                        "rel_path_seg": str(relative),
                        "rel_path_segmask": "",
                        "species_dir": barcode,
                        "individual_dir": f"{barcode}_{image_index}",
                        "taxon_label": species_label,
                        "species_label": species_label,
                        "life_stage": age,
                        "genus": genus,
                        "__taxon_for_split__": species_label,
                    }
                    rows.append(row)
                    split_rows[split].append(row)
    frame = pd.DataFrame(rows)
    metadata_path = image_root / "global_metadata.csv"
    frame.to_csv(metadata_path, index=False)
    for split, values in split_rows.items():
        pd.DataFrame(values).to_csv(split_dir / f"{split}_split.csv", index=False)
    return data_root, split_root


def write_local_cluster(simulation_root: Path, data_root: Path) -> Path:
    cluster = yaml.safe_load(
        (ROOT / "configs/clusters/genome.yaml").read_text(encoding="utf-8")
    )
    cluster["slurm"]["paths"].update({
        "project_root": str(ROOT),
        "data_root": str(data_root),
        "metadata_csv": str(data_root / "01_Segmented/global_metadata.csv"),
        "results_root": str(simulation_root / "results"),
        "cache_root": str(simulation_root / "cache"),
    })
    cluster["slurm"]["scratch"]["root"] = "/tmp/devd/worm_species_local_smoke"
    cluster["slurm"]["logging"]["directory"] = str(simulation_root / "logs")
    path = simulation_root / "local_cluster.json"
    path.write_text(json.dumps(cluster, indent=2) + "\n", encoding="utf-8")
    return path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation-root",
        type=Path,
        default=ROOT / "local_slurm_simulation",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    simulation_root = args.simulation_root.resolve()
    simulation_root.mkdir(parents=True, exist_ok=True)
    data_root, split_root = build_synthetic_data(simulation_root)
    cluster = write_local_cluster(simulation_root, data_root)
    artifact_root = simulation_root / f"slurm_artifacts_{current_commit()}"
    if not artifact_root.exists():
        run([
            sys.executable, "-m", "worm_species.slurm", "render",
            "--config", str(CONFIG), "--cluster-config", str(cluster),
            "--artifacts-dir", str(artifact_root),
        ])
    if args.prepare_only:
        print(f"Prepared local SLURM simulation in {simulation_root}")
        return 0
    run([
        sys.executable, "-m", "worm_species.training",
        "--config", str(CONFIG),
        "--override",
        f"data.root_dir={data_root}",
        f"data.metadata_csv={data_root / '01_Segmented/global_metadata.csv'}",
        f"split.predefined_split_dir={split_root}",
        f"output.out_dir={simulation_root / 'results'}",
        "cache.enabled=false",
    ])
    print(f"Five-epoch local smoke results: {simulation_root / 'results'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
