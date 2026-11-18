#!/usr/bin/env python3
"""Commands for the two public weight files and the global image metadata."""

from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from scripts import release_dataset as data


def models(root, task=None):
    """Verify the downloaded weights before loading either model."""
    root = Path(root).resolve()
    spec = json.loads((ROOT / "configs/release_inputs.json").read_text())
    for key in [task] if task else ["classification", "segmentation"]:
        path = root / spec[key]["relative_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Extract earthworm-models.zip first: {path}")
        if data.sha(path) != spec[key]["sha256"]:
            raise ValueError(f"Checkpoint hash differs from release: {path}")
    return root


def load_classifier(root, device):
    import torch
    from worm_species.models.multitask import build_multitask_model
    from worm_species.data.transforms import build_split_transform

    root = models(root, "classification")
    checkpoint = torch.load(
        root / "classification/convnext_base_seed2340.pt",
        map_location="cpu",
        weights_only=False,
    )
    # The checkpoint contains the learned weights; no pretrained download is needed.
    cfg = copy.deepcopy(checkpoint["cfg"])
    cfg["model"]["pretrained"] = False
    maps = checkpoint["label_to_index_by_task"]
    if maps != json.loads(
        (root / "classification/label_to_index_by_task.json").read_text()
    ):
        raise ValueError("Checkpoint and public class maps differ")
    model = build_multitask_model(
        cfg, {task: len(labels) for task, labels in maps.items()}
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    # Match the saved evaluation preprocessing, including image size and normalisation.
    transform = build_split_transform(
        split="test",
        preprocessing=cfg["preprocessing"],
        condition={"transform": "original"},
    )
    return model, maps, cfg, transform


def new_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"Preserve existing outputs; choose a fresh path: {path}")
    return path


def predict(args):
    """Classify segmented images and save probabilities for each output task."""
    import torch
    from PIL import Image

    torch.set_num_threads(args.threads)
    target = new_output(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    model, maps, cfg, transform = load_classifier(args.models, args.device)
    if args.input:
        input_path = Path(args.input).resolve()
        paths = (
            [input_path]
            if input_path.is_file()
            else sorted(
                p
                for p in input_path.rglob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
        )
        if not paths:
            raise ValueError("No input images found")
        records = [
            (
                {
                    "image_id": p.stem,
                    "individual_id": str(p.relative_to(input_path.parent)),
                    "genus": "",
                    "species_label": "",
                    "life_stage": "",
                    "taxon": "",
                },
                p,
            )
            for p in paths
        ]
    else:
        rows = data.metadata(args.dataset, verify_files=False)
        records = [
            (r, data.safe_path(args.dataset, r["segmented_path"]))
            for r in rows
            if r["segmented_path"]
        ]
        if args.image_id:
            records = [item for item in records if item[0]["image_id"] == args.image_id]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("No eligible segmented images selected")
    columns = [
        "task",
        "image_id",
        "individual_id",
        "true_index",
        "predicted_index",
        "predicted_label",
        "probabilities_json",
        "logits_json",
    ]
    temp = target.with_suffix(target.suffix + ".partial")
    if temp.exists():
        raise FileExistsError(
            f"Preserve the unfinished output and choose a fresh path: {temp}"
        )
    with temp.open("w", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            tensors = []
            for row, path in batch:
                with Image.open(path) as im:
                    tensors.append(transform(im.convert("RGB")))
            with torch.inference_mode():
                outputs = model(torch.stack(tensors).to(args.device))
            for task, logits in outputs.items():
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite classifier output")
                values = logits.float().cpu()
                probabilities = values.softmax(-1)
                names = {i: label for label, i in maps[task].items()}
                for i, (row, path) in enumerate(batch):
                    truth = row.get(
                        {
                            "age": "life_stage",
                            "genus": "genus",
                            "species": "species_label",
                        }[task],
                        "",
                    )
                    if task == "age":
                        truth = (
                            cfg["data"]["taxonomic_uncertainty"]
                            .get("life_stage_overrides", {})
                            .get(row.get("taxon", ""), truth)
                        )
                    predicted = int(probabilities[i].argmax())
                    writer.writerow(
                        dict(
                            task=task,
                            image_id=row["image_id"],
                            individual_id=row["individual_id"],
                            true_index=maps[task].get(truth, -1),
                            predicted_index=predicted,
                            predicted_label=names[predicted],
                            probabilities_json=json.dumps(probabilities[i].tolist()),
                            logits_json=json.dumps(values[i].tolist()),
                        )
                    )
    temp.rename(target)
    print(
        json.dumps(
            {
                "images": len(records),
                "prediction_rows": 3 * len(records),
                "output": str(target),
                "training": False,
            }
        )
    )


def segment(args):
    """Segment original photographs and save masks in both coordinate systems."""
    from PIL import Image
    from ultralytics import YOLO
    from segment.inference_coco import (
        process_file_two_stage_rotated_square_mask_vis_largest_cc,
    )
    import torch

    torch.set_num_threads(args.threads)
    model_root = models(args.models, "segmentation")
    target = new_output(args.output)
    target.mkdir(parents=True)
    yolo = YOLO(str(model_root / "segmentation/segmentation.pt"))
    settings = json.loads(
        (model_root / "segmentation/inference_config.json").read_text()
    )

    class FixedSettings:
        def __call__(self, image):
            return yolo.predict(
                image,
                imgsz=settings["imgsz"],
                conf=settings["conf"],
                device=args.device,
                verbose=False,
                save=False,
            )

    if args.input:
        inp = Path(args.input).resolve()
        paths = (
            [inp]
            if inp.is_file()
            else sorted(
                p
                for p in inp.rglob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
        )
        records = [(data.sha(p)[:24], p) for p in paths]
    else:
        rows = data.metadata(args.dataset, verify_files=False)
        if args.image_id:
            rows = [r for r in rows if r["image_id"] == args.image_id]
        records = [
            (r["image_id"], data.safe_path(args.dataset, r["raw_path"])) for r in rows
        ]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError("No raw images selected")
    successes = 0
    with (target / "segmentation_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            ["image_id", "status", "segmented_path", "raw_mask_path", "crop_mask_path"],
        )
        w.writeheader()
        for image_id, path in records:
            dest = target / "segmented" / f"{image_id}.jpg"
            ok, info = process_file_two_stage_rotated_square_mask_vis_largest_cc(
                str(path),
                str(dest),
                FixedSettings(),
                pad_px=settings["pad_px"],
                sq_margin=settings["square_margin"],
            )
            if ok:
                # The first mask aligns with the original photograph; the second
                # aligns with the rotated and cropped foreground image.
                rawmask = target / "masks/raw" / f"{image_id}.png"
                rawmask.parent.mkdir(parents=True, exist_ok=True)
                Path(str(dest) + ".png").rename(rawmask)
                cropmask = target / "masks/crop" / f"{image_id}.png"
                cropmask.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(info["mask_on_crop"]).save(cropmask)
                w.writerow(
                    dict(
                        image_id=image_id,
                        status="segmented",
                        segmented_path=str(dest.relative_to(target)),
                        raw_mask_path=str(rawmask.relative_to(target)),
                        crop_mask_path=str(cropmask.relative_to(target)),
                    )
                )
                successes += 1
            else:
                w.writerow(
                    dict(
                        image_id=image_id,
                        status="failed",
                        segmented_path="",
                        raw_mask_path="",
                        crop_mask_path="",
                    )
                )
    print(
        json.dumps(
            {
                "images": len(records),
                "successful": successes,
                "output": str(target),
                "source_modified": False,
            }
        )
    )


def evaluate(args):
    """Average repeated views so each worm contributes once to each score."""
    import numpy as np
    import pandas as pd
    from worm_species.deployment_audit import f1_fixed

    target = new_output(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.predictions, dtype={"individual_id": str, "image_id": str})
    if frame.duplicated(["task", "image_id"]).any():
        raise ValueError("Duplicate image/task predictions would reweight views")
    maps = json.loads(
        (Path(args.models) / "classification/label_to_index_by_task.json").read_text()
    )
    rows = []
    for task, g in frame.groupby("task"):
        truth = []
        pred = []
        for worm, views in g.groupby("individual_id"):
            if views.true_index.nunique() != 1:
                raise ValueError("Conflicting reference labels within a worm")
            label = int(views.true_index.iloc[0])
            # Unresolved species targets are excluded only from that task.
            if label < 0:
                continue
            if label >= len(maps[task]):
                raise ValueError(
                    "Reference label is outside the saved output vocabulary"
                )
            p = np.asarray(views.probabilities_json.map(json.loads).tolist())
            if p.shape[1] != len(maps[task]):
                raise ValueError("Output vocabulary differs from the saved model")
            if (
                not np.isfinite(p).all()
                or (p < 0).any()
                or not np.allclose(p.sum(1), 1, atol=1e-5)
            ):
                raise ValueError("Expected finite normalised probabilities")
            truth.append(label)
            pred.append(int(p.mean(0).argmax()))
        if not truth:
            continue
        # Keep the full trained vocabulary, even for a small selected sample.
        k = len(maps[task])
        cm = np.zeros((k, k), int)
        np.add.at(cm, (truth, pred), 1)
        present = cm.sum(1) > 0
        rows.append(
            dict(
                task=task,
                individuals=len(truth),
                output_classes=k,
                macro_f1_fixed_vocabulary=float(f1_fixed(truth, pred, range(k), k)),
                balanced_accuracy_present_classes=float(
                    np.mean(cm.diagonal()[present] / cm.sum(1)[present])
                ),
                uniform_accuracy_reference=1 / k,
                confusion_matrix=cm.tolist(),
            )
        )
    target.write_text(
        json.dumps(
            {
                "aggregation": "individual mean probabilities",
                "results": rows,
                "scope": "single supplied checkpoint; not the 30-seed paper comparison",
            },
            indent=2,
        )
        + "\n"
    )
    print(target)


def prepare(args):
    """Create the runtime tables and configuration without copying images."""
    import yaml

    receipt = data.prepare(args.dataset, args.runtime)
    cfg = yaml.safe_load((ROOT / "configs/paper.yaml").read_text())
    work = Path(args.work).resolve()
    cfg["paths"] = {
        k: str(v)
        for k, v in dict(
            dataset=receipt["dataset_view"],
            original_data=Path(args.runtime).resolve() / "original",
            splits=receipt["splits"],
            trained_results=work / "reference_inputs",
            predictions=work / "inference",
            features=work / "features",
            analysis=work / "analysis",
            adaptive_analysis=work / "adaptive_analysis",
            controls=work / "controls",
            received_controls=work / "controls/training/runs",
            control_summary=work / "controls/received_csv_summary/latest.json",
        ).items()
    }
    cfg["resources"].update(device=args.device, cpu_threads=args.threads)
    cfg["release_control_selections"] = str(
        ROOT / "configs/reference_control_selections.json"
    )
    config = Path(args.config).resolve()
    config.parent.mkdir(parents=True, exist_ok=True)
    contents = yaml.safe_dump(cfg, sort_keys=False)
    if config.exists() and config.read_text() != contents:
        raise ValueError(
            "Existing reproduction config differs; preserve it and choose another path"
        )
    config.write_text(contents)
    print(
        json.dumps(
            {
                "reference_images": receipt["reference_images"],
                "config": str(config),
                "training": False,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["check", "prepare", "predict", "segment", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--dataset", type=Path, default=ROOT / "data/earthworm-images")
        p.add_argument("--models", type=Path, default=ROOT / "data/earthworm-models")
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=4)
        if name in ["predict", "segment"]:
            p.add_argument(
                "--input",
                type=Path,
                help="A photo or directory; predict expects already segmented photos",
            )
            p.add_argument("--image-id")
            p.add_argument("--limit", type=int)
            p.add_argument("--output", type=Path, required=True)
        if name == "predict":
            p.add_argument("--batch-size", type=int, default=8)
        if name == "evaluate":
            p.add_argument("--predictions", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--runtime", type=Path, default=ROOT / "work/inputs")
            p.add_argument("--work", type=Path, default=ROOT / "work/results")
            p.add_argument(
                "--config", type=Path, default=ROOT / "work/reproduction.yaml"
            )
    args = parser.parse_args()
    if (
        args.threads < 1
        or getattr(args, "batch_size", 1) < 1
        or (getattr(args, "limit", None) is not None and args.limit < 1)
    ):
        parser.error("Threads, batch size and limit must be positive")
    if args.command == "check":
        rows = data.metadata(args.dataset)
        parts = data.frozen_splits(rows)
        models(args.models)
        print(
            json.dumps(
                {
                    "images": len(rows),
                    "reference_split_images": {k: len(v) for k, v in parts.items()},
                    "model_hashes_match": True,
                    "masks_required": False,
                    "training": False,
                }
            )
        )
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
