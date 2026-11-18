"""Read the image metadata and restore the original paper splits."""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    p = Path(relative)
    if not relative or p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Expected a safe relative image path: {relative!r}")
    # A user can put the two year directories on another local disk via symlinks.
    return root / p


def metadata(dataset, *, verify_files=True, require_release=True):
    dataset = Path(dataset).resolve()
    path = dataset / "metadata.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Put the global metadata.csv in {dataset}")
    spec = json.loads((ROOT / "configs/release_inputs.json").read_text())
    if require_release and sha(path) != spec["metadata_sha256"]:
        raise ValueError("Global metadata differs from the frozen public release")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate image identities")
    worms = {}
    paths = set()
    for row in rows:
        if row["kind"] != "worm":
            raise ValueError("Calibration photos are not part of this release")
        identity = (row["taxon"], row["life_stage"])
        if worms.setdefault(row["individual_id"], identity) != identity:
            raise ValueError("Conflicting labels for one worm")
        for column in ["raw_path", "segmented_path"]:
            rel = row[column]
            if rel:
                p = safe_path(dataset, rel)
                if rel in paths:
                    raise ValueError(f"Duplicate photo path: {rel}")
                paths.add(rel)
                if verify_files and not p.is_file():
                    raise FileNotFoundError(f"Extract images.zip first: {p}")
        if bool(row["segmented_path"]) != (
            row["segmentation_status"] in ("reused", "segmented")
        ):
            raise ValueError("Segmentation status and available photo disagree")
    if require_release and (
        len(rows) != spec["images"]
        or sum(bool(r["segmented_path"]) for r in rows) != spec["segmented_images"]
    ):
        raise ValueError("Release image counts changed")
    return rows


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Refusing to write an empty runtime table")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, columns or list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def frozen_splits(rows):
    """Restore the original image order and keep each worm in one partition."""
    spec = json.loads((ROOT / "configs/reference_split_order.json").read_text())
    lookup = {r["image_id"]: r for r in rows}
    parts = {}
    worms = {}
    for split, ids in spec["splits"].items():
        if len(set(ids)) != len(ids):
            raise ValueError("Frozen split repeats an image")
        parts[split] = []
        for image_id in ids:
            r = dict(lookup[image_id])
            if (
                r["original_paper_split"] != split
                or r["year"] != "2025"
                or not r["segmented_path"]
            ):
                raise ValueError("Reference split membership or segmentation changed")
            # Masks are generated on demand. Classification does not require them.
            r.update(
                rel_path_raw=r["raw_path"],
                rel_path_seg=r["segmented_path"],
                rel_path_segmask="",
            )
            parts[split].append(r)
        worms[split] = {r["individual_id"] for r in parts[split]}
        if len(worms[split]) != spec["expected_individuals"][split]:
            raise ValueError("Frozen worm support changed")
    if any(
        worms[a] & worms[b]
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    ):
        raise ValueError("Biological train/validation/test leakage")
    frozen = {r["image_id"] for part in parts.values() for r in part}
    if frozen != {r["image_id"] for r in rows if r["original_paper_split"]}:
        raise ValueError("Extra reference image assignment")
    return parts


def prepare(dataset, output):
    """Create only small derived manifests and links; never copy image bytes."""
    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("Runtime output must be separate from the input dataset")
    rows = metadata(dataset)
    parts = frozen_splits(rows)
    identity = {
        "metadata_sha256": sha(dataset / "metadata.csv"),
        "dataset": str(dataset),
        "split_order_sha256": sha(ROOT / "configs/reference_split_order.json"),
    }
    receipt = output / "prepared.json"
    if output.exists():
        if (
            not receipt.is_file()
            or json.loads(receipt.read_text())["identity"] != identity
        ):
            raise ValueError(
                "Existing runtime inputs differ; choose a fresh output folder"
            )
        saved = json.loads(receipt.read_text())
        if any(
            not (output / rel).is_file() or sha(output / rel) != digest
            for rel, digest in saved["sha256"].items()
        ):
            raise ValueError(
                "A prepared runtime input changed; preserve it and choose a fresh output"
            )
        return saved
    output.mkdir(parents=True)
    view = output / "dataset"
    view.mkdir()
    for year in ["2025", "2026"]:
        (view / year).symlink_to(dataset / year, target_is_directory=True)
    # The analysis uses a few derived tables. Symlinks avoid duplicating the photos.
    runtime = []
    for r in rows:
        r = dict(r)
        r["rel_path_segmask"] = ""
        runtime.append(r)
    write_csv(view / "metadata/images.csv", runtime)
    split_root = output / "splits/split_csv"
    for name, part in parts.items():
        write_csv(split_root / f"{name}_split.csv", part)
    write_csv(
        output / "original/01_Segmented/global_metadata.csv",
        [r for part in parts.values() for r in part],
    )
    paths = {str(p.relative_to(output)): sha(p) for p in output.rglob("*.csv")}
    saved = {
        "identity": identity,
        "sha256": paths,
        "dataset_view": str(view),
        "splits": str(split_root),
        "reference_images": sum(map(len, parts.values())),
        "split_counts": {k: len(v) for k, v in parts.items()},
        "images_copied": False,
        "source_modified": False,
    }
    receipt.write_text(json.dumps(saved, indent=2) + "\n")
    return saved
