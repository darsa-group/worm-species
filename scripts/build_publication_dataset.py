#!/usr/bin/env python3
"""Combine capture years and segment worms; never train or alter source data."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SUCCESS = {"reused", "segmented"}


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def component(value):
    value = str(value)
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Invalid path component: {value!r}")
    return value


def source_path(root, relative):
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes source folder: {relative}")
    return path


def copy_file(source, destination):
    """Atomic copies; an existing different destination is never overwritten."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        a, b = source.stat(), destination.stat()
        if a.st_size == b.st_size and a.st_mtime_ns == b.st_mtime_ns:
            return
        if a.st_size == b.st_size and digest(source) == digest(destination):
            return
        raise ValueError(f"Destination differs from source: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name + ".copying")
    shutil.copy2(source, temp)
    temp.replace(destination)


def make_row(meta, source, year, camera, session, paper_split=""):
    barcode = component(meta.get("individual_barcode") or meta["barcode"])
    camera, session = component(camera), component(session)
    match = re.fullmatch(r"(.+)_(Adult|Juvenile)_\d+", barcode)
    age = meta.get("age") or (match[2] if match else "")
    taxon = meta.get("taxon") or (match[1] if match else "")
    kind = "worm" if match else "calibration" if age == "Test" else "unclassified"
    base = Path(str(year)) / camera
    relative = Path(barcode) / session / source.name
    raw_dir = "00_RawData" if kind == "worm" else kind
    image_id = hashlib.sha256(f"{year}/{camera}/{relative}".encode()).hexdigest()[:24]
    return {
        "image_id": image_id, "year": str(year), "camera": camera,
        "barcode": barcode, "individual_id": f"{year}:{barcode}",
        "capture_id": session, "location_code": meta.get("location_barcode", ""),
        "timestamp": meta.get("captured_at_utc") or meta.get("timestamp", ""),
        "taxon": taxon, "life_stage": age, "weight_g": meta.get("weight_g", ""),
        "index": meta.get("image_number") or meta.get("index", ""),
        "kind": kind, "original_paper_split": paper_split,
        "dataset_role": ("paper_" + paper_split if paper_split else
                         "external_test" if camera == "gphoto2" and kind == "worm" else
                         "reserved" if camera == "webcam" and kind == "worm" else
                         "historical_unassigned" if camera == "original_camera" else kind),
        "filename": source.name, "source_raw": str(source.resolve()),
        "source_sha256": meta.get("sha256", ""),
        "raw_path": str(base / raw_dir / relative),
        "segmented_path": str(base / "01_Segmented" / relative.with_suffix(".jpg")),
        "raw_mask_path": str(base / "masks" / "original" / relative.with_suffix(".png")),
        "crop_mask_path": str(base / "masks" / "segmented" / relative.with_suffix(".png")),
        "source_segmented": "", "source_mask": "",
    }


def inventory(old_root, new_root, split_root):
    old_root, new_root = old_root.resolve(), new_root.resolve()
    old_metadata = read_csv(old_root / "01_Segmented/global_metadata.csv")
    old_by_name = {}
    for row in old_metadata:
        key = (row["barcode"], row["filename"])
        if key in old_by_name:
            raise ValueError(f"Duplicate old image metadata: {key}")
        old_by_name[key] = row
    split_by_image = {}
    for split in ("train", "val", "test"):
        for row in read_csv(split_root / f"{split}_split.csv"):
            key = (row["barcode"], row["filename"])
            if key in split_by_image:
                raise ValueError(f"Image occurs in multiple paper splits: {key}")
            split_by_image[key] = split
    rows, found_old = [], set()
    for path in sorted((old_root / "00_RawData").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        barcode = path.relative_to(old_root / "00_RawData").parts[0]
        key = (barcode, path.name)
        meta = old_by_name.get(key)
        if meta is None:
            sidecar = path.with_suffix(".csv")
            candidates = read_csv(sidecar) if sidecar.exists() else []
            meta = next((r for r in candidates if r.get("filename") == path.name), None)
        if meta is None:
            raise ValueError(f"No capture metadata for historical image: {path}")
        year = meta["timestamp"][:4]
        if not re.fullmatch(r"20\d\d", year):
            raise ValueError(f"Invalid capture year: {path}")
        row = make_row(meta, path, year, "original_camera", "legacy", split_by_image.get(key, ""))
        if meta.get("rel_path_seg") and meta.get("rel_path_segmask"):
            rgb = source_path(old_root, meta["rel_path_seg"])
            mask = source_path(old_root, meta["rel_path_segmask"])
            if rgb.is_file() and mask.is_file():
                row.update(source_segmented=str(rgb), source_mask=str(mask))
        rows.append(row)
        found_old.add(key)
    if set(old_by_name) - found_old or set(split_by_image) - found_old:
        raise ValueError("Historical raw folder is missing images from metadata or paper splits")
    captured_paths = set()
    old_barcodes = {row["barcode"] for row in rows}
    for meta in read_csv(new_root / "captures.csv"):
        source = source_path(new_root, meta["image_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if source in captured_paths:
            raise ValueError(f"Duplicate capture row: {source}")
        captured_paths.add(source)
        camera = meta["storage_folder"]
        if camera not in {"gphoto2", "webcam"} or Path(meta["image_path"]).parts[0] != camera:
            raise ValueError(f"Unrecognized or inconsistent camera: {source}")
        timestamp = meta.get("captured_at_utc") or meta["timestamp"]
        year = timestamp[:4]
        if not re.fullmatch(r"20\d\d", year):
            raise ValueError(f"Invalid capture year: {source}")
        row = make_row(meta, source, year, camera, meta["capture_id"])
        if row["barcode"] in old_barcodes:
            raise ValueError(f"New and old specimen barcodes overlap; resolve identity first: {row['barcode']}")
        rows.append(row)
    disk_paths = {p.resolve() for camera in ("gphoto2", "webcam")
                  for p in (new_root / camera).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    if disk_paths != captured_paths:
        raise ValueError(f"Capture manifest and image folders disagree: {len(disk_paths ^ captured_paths)} files")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Image identity collision")
    return rows


def state_path(output, row):
    return output / "metadata/segmentation_records" / (row["image_id"] + ".json")


def read_state(output, row):
    path = state_path(output, row)
    return json.loads(path.read_text()) if path.exists() else {"status": "pending", "error": ""}


def summarize(rows):
    return {
        "images": len(rows),
        "groups": dict(sorted(Counter(f"{r['year']}/{r['camera']}/{r['kind']}" for r in rows).items())),
        "reuse_existing_segmentations": sum(bool(r["source_segmented"]) for r in rows),
        "worms_to_segment": sum(r["kind"] == "worm" and not r["source_segmented"] for r in rows),
        "raw_bytes": sum(Path(r["source_raw"]).stat().st_size for r in rows),
    }


def write_manifests(output, rows):
    all_rows, by_folder = [], {}
    for row in rows:
        state = read_state(output, row)
        record = {**row, "segmentation_status": state["status"], "segmentation_error": state.get("error", "")}
        record["rel_path_raw"] = row["raw_path"]
        record["rel_path_seg"] = row["segmented_path"] if state["status"] in SUCCESS else ""
        # Old masks are in original-image coordinates, never use them to crop segmented RGB.
        record["rel_path_rawmask"] = row["raw_mask_path"] if state["status"] in SUCCESS else ""
        record["rel_path_segmask"] = row["crop_mask_path"] if state["status"] == "segmented" else ""
        all_rows.append(record)
        base = Path(row["year"]) / row["camera"]
        category = "00_RawData" if row["kind"] == "worm" else row["kind"]
        folder = base / category
        by_folder.setdefault(folder, []).append({**record, "image_path": str(Path(row["raw_path"]).relative_to(folder))})
        if row["kind"] == "worm":
            folder = base / "01_Segmented"
            by_folder.setdefault(folder, [])
            if state["status"] in SUCCESS:
                by_folder[folder].append({**record, "image_path": str(Path(row["segmented_path"]).relative_to(folder))})
    fields = list(all_rows[0]) if all_rows else ["image_id"]
    write_csv(output / "metadata/images.csv", all_rows, fields)
    write_csv(output / "metadata/segmentation_status.csv", all_rows,
              ["image_id", "year", "camera", "barcode", "segmentation_status", "segmentation_error"])
    for folder, records in by_folder.items():
        write_csv(output / folder / "metadata.csv", records, fields + ["image_path"])
    summary = summarize(rows)
    summary["segmentation_status"] = dict(Counter(r["segmentation_status"] for r in all_rows))
    atomic_json(output / "metadata/summary.json", summary)
    return summary


@contextmanager
def dataset_lock(output):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another dataset build is running in {output}") from None
        yield


def prepare(output, rows, old_root, new_root, split_root):
    identity = {"schema": 1, "rows": rows}
    manifest = output / "metadata/inventory.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Sources differ from this dataset's inventory; choose a new DATASET_ROOT")
    (output / "metadata/prepared.json").unlink(missing_ok=True)
    atomic_json(manifest, identity)
    copy_file(old_root / "01_Segmented/global_metadata.csv", output / "metadata/sources/old_global_metadata.csv")
    copy_file(new_root / "captures.csv", output / "metadata/sources/new_captures.csv")
    for split in ("train", "val", "test"):
        copy_file(split_root / f"{split}_split.csv", output / f"metadata/original_paper_splits/{split}_split.csv")
    (output / "README.md").write_text(
        "# Publication dataset\n\n"
        "Images are copied, not moved or linked. Folders are year/camera/00_RawData, "
        "01_Segmented and masks. Every 00 and 01 folder has its own metadata.csv. "
        "Its image_path is relative to that folder; rel_path_* columns are relative "
        "to this dataset root. Read location_code as text to preserve leading zeros.\n\n"
        "Calibration and unclassified images are retained separately and not segmented. "
        "The original paper splits are archived unchanged and refer to the old dataset; "
        "original_paper_split records their membership in metadata/images.csv. "
        "New gphoto2 worms are reserved as external test; webcam worms are reserved "
        "for later evaluation. No training or splitting is performed.\n\n"
        "masks/original contains masks in EXIF-oriented raw-image coordinates for new "
        "segmentations, and unchanged original-coordinate masks for historical ones. "
        "masks/segmented contains masks aligned with new segmented RGB images. "
        "Historical segmented images and masks are reused unchanged.\n\n"
        "metadata/segmentation_status.csv lists every failure. Per-image records permit "
        "safe resume. Source CSVs are retained under metadata/sources.\n"
    )
    try:
        for index, row in enumerate(rows, 1):
            destination = output / row["raw_path"]
            is_new_copy = not destination.exists()
            copy_file(row["source_raw"], destination)
            if is_new_copy and row["source_sha256"] and digest(destination) != row["source_sha256"]:
                destination.unlink()
                raise ValueError(f"Capture checksum mismatch: {row['source_raw']}")
            if row["kind"] != "worm":
                atomic_json(state_path(output, row), {"status": "excluded_" + row["kind"], "error": ""})
            elif row["source_segmented"]:
                copy_file(row["source_segmented"], output / row["segmented_path"])
                copy_file(row["source_mask"], output / row["raw_mask_path"])
                atomic_json(state_path(output, row), {"status": "reused", "error": ""})
            if index % 100 == 0 or index == len(rows):
                print(f"Copying: {index}/{len(rows)} images", flush=True)
    finally:
        write_manifests(output, rows)
    atomic_json(output / "metadata/prepared.json", {"images": len(rows)})


def segment(output, rows, model_path, device, threads, limit=0):
    if not (output / "metadata/prepared.json").exists():
        raise ValueError("Dataset preparation is incomplete; run make dataset-prepare first")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    import torch
    from ultralytics import YOLO
    from PIL import Image
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from segment.inference_coco import process_file_two_stage_rotated_square_mask_vis_largest_cc

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Fix the NVIDIA driver or run make dataset DEVICE=cpu")
    torch.set_num_threads(threads)
    model = YOLO(str(model_path)).to(device)
    identity = {"model_sha256": digest(model_path), "method_sha256": digest(PROJECT / "segment/inference_coco.py"),
                "pad_px": 150, "square_margin": 300, "imgsz": 1024, "conf": 0.25,
                "method": "two_stage_rotated_square_largest_component"}
    config_path = output / "metadata/segmentation_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError("Segmentation method or weights changed; choose a new DATASET_ROOT")
    atomic_json(config_path, identity)

    # Explicit parameters prevent library defaults or saved training args from changing the method.
    def predict(image):
        return model.predict(image, device=device, imgsz=1024, conf=0.25, verbose=False)

    pending = []
    for row in rows:
        if row["kind"] != "worm" or row["source_segmented"]:
            continue
        state = read_state(output, row)
        products = [output / row[key] for key in ("segmented_path", "raw_mask_path", "crop_mask_path")]
        if state["status"] == "segmented" and all(p.is_file() and p.stat().st_size > 0 for p in products):
            continue
        pending.append(row)
    if limit:
        pending = pending[:limit]
    print(f"Segmentation: {len(pending)} pending images on {device}; threads={threads}", flush=True)
    start = time.monotonic()
    try:
        for index, row in enumerate(pending, 1):
            state = {"status": "failed", "error": "", "device": device}
            final = output / row["segmented_path"]
            temp = final.with_name(final.stem + ".partial.jpg")
            try:
                ok, info = process_file_two_stage_rotated_square_mask_vis_largest_cc(
                    str(output / row["raw_path"]), str(temp), predict, visualize=False)
                if not ok or info is None:
                    state["error"] = "No usable mask; inspect this image before retrying"
                else:
                    raw_mask = output / row["raw_mask_path"]
                    crop_mask = output / row["crop_mask_path"]
                    raw_mask.parent.mkdir(parents=True, exist_ok=True)
                    crop_mask.parent.mkdir(parents=True, exist_ok=True)
                    crop_temp = crop_mask.with_name(crop_mask.stem + ".partial.png")
                    Image.fromarray(info["mask_on_crop"]).save(crop_temp)
                    crop_temp.replace(crop_mask)
                    Path(str(temp) + ".png").replace(raw_mask)
                    temp.replace(final)
                    state.update(status="segmented", raw_width=info["width"], raw_height=info["height"])
            except (MemoryError, RuntimeError):
                # Hardware failures must stop the batch, not mark thousands of worms as failed.
                raise
            except Exception as error:
                state["error"] = f"{type(error).__name__}: {error}"
            finally:
                for partial in (temp, Path(str(temp) + ".png")):
                    partial.unlink(missing_ok=True)
            atomic_json(state_path(output, row), state)
            print(f"Segmented {index}/{len(pending)} [{row['camera']}] {state['status']} "
                  f"elapsed={time.monotonic() - start:.0f}s", flush=True)
            if index % 25 == 0:
                write_manifests(output, rows)
    finally:
        summary = write_manifests(output, rows)
        print(json.dumps(summary, indent=2), flush=True)
    return summary["segmentation_status"].get("failed", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "prepare", "segment", "all"], default="plan")
    parser.add_argument("--old-root", type=Path, default=Path("data/original"))
    parser.add_argument("--new-root", type=Path, default=Path("data/captures_2026"))
    parser.add_argument("--split-root", type=Path, default=PROJECT / "data/split_csv")
    parser.add_argument("--output", type=Path, default=Path("data/publication_dataset"))
    parser.add_argument("--model", type=Path, default=PROJECT / "data/checkpoints/segmentation.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many pending images (0 = all)")
    args = parser.parse_args()
    if args.threads < 1 or args.limit < 0:
        parser.error("threads must be positive and limit must be non-negative")
    output = args.output.resolve()
    for source in (args.old_root.resolve(), args.new_root.resolve()):
        if output.is_relative_to(source) or source.is_relative_to(output):
            parser.error("Output must be separate from both source folders")
    rows = inventory(args.old_root, args.new_root, args.split_root)
    print(json.dumps(summarize(rows), indent=2), flush=True)
    if args.mode == "plan":
        return 0
    with dataset_lock(output):
        if args.mode in {"prepare", "all"}:
            prepare(output, rows, args.old_root, args.new_root, args.split_root)
        else:
            saved = json.loads((output / "metadata/inventory.json").read_text())
            if saved != {"schema": 1, "rows": rows}:
                raise ValueError("Source inventory changed; use a new DATASET_ROOT")
        if args.mode in {"segment", "all"}:
            return 2 if segment(output, rows, args.model, args.device, args.threads, args.limit) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
