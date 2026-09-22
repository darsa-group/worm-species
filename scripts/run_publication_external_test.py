#!/usr/bin/env python3
"""Evaluate every publication baseline seed on the two new camera cohorts."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
for path in (PROJECT, PROJECT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

MODELS = ("convnext_base", "vit_b_16", "resnet50")
SEEDS = tuple(range(40, 2941, 100))
CAMERAS = ("gphoto2", "webcam")
TASKS = ("age", "genus", "species")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stamp(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


@contextmanager
def lock(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield


def protocol(cfg):
    """Only settings affecting evaluation; cluster/training paths are irrelevant."""
    return {"preprocessing": cfg["preprocessing"], "data": {
        key: cfg["data"].get(key) for key in (
            "crop_to_foreground", "crop_pad", "target_cols", "barcode_col",
            "taxonomic_uncertainty", "missing_label_values", "colour_retention")
    }, "use_amp": cfg["training"].get("use_amp", True)}


def discover_checkpoints(result_root):
    runs = {}
    for path in sorted((result_root / "runs/baseline").glob("**/config.json")):
        cfg = json.loads(path.read_text())
        model, seed = cfg["model"]["name"], cfg["seed"]
        if model not in MODELS or seed not in SEEDS:
            raise ValueError(f"Unexpected baseline model/seed in {path}: {model}, {seed}")
        key = (model, seed)
        if key in runs:
            raise ValueError(f"Multiple checkpoints for {key}; resolve duplicates before testing")
        condition = cfg.get("input_condition", {})
        if condition.get("transform", "original") != "original" or cfg.get("data_holdout", {}).get("enabled"):
            raise ValueError(f"Not a full-data original-image baseline: {path}")
        if cfg["data"].get("crop_to_foreground", True):
            raise ValueError(f"Expected the paper's no-recrop preprocessing: {path}")
        if cfg.get("multi_task", {}).get("hierarchy_loss", {}).get("enabled"):
            raise ValueError(f"Unexpected hierarchy-loss baseline: {path}")
        marker = path.parent.parent / "run_status.txt"
        if not marker.is_file() or marker.read_text().strip() != "0":
            raise ValueError(f"Baseline has no successful completion marker: {path}")
        checkpoint = path.parent / "best_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint (check uploaded symlink targets): {checkpoint}")
        mapping_path = path.parent / "label_to_index_by_task.json"
        mapping = json.loads(mapping_path.read_text())
        if set(mapping) != set(TASKS):
            raise ValueError(f"Unexpected model heads in {mapping_path}")
        for task, labels in mapping.items():
            if not labels or sorted(labels.values()) != list(range(len(labels))):
                raise ValueError(f"Non-contiguous class map: {mapping_path}/{task}")
        old_predictions = path.parent / "test_predictions_best.csv"
        if not old_predictions.is_file():
            raise FileNotFoundError(f"Original-test predictions required for comparison: {old_predictions}")
        runs[key] = {"model": model, "seed": seed, "checkpoint": str(checkpoint.absolute()),
                     "checkpoint_stamp": stamp(checkpoint), "config": cfg,
                     "label_maps": mapping, "original_predictions": str(old_predictions.absolute()),
                     "original_predictions_sha256": sha(old_predictions)}
    expected = {(m, s) for m in MODELS for s in SEEDS}
    if set(runs) != expected:
        raise ValueError(f"Need all 90 baselines. Missing: {sorted(expected - set(runs))}")
    ordered = [runs[(m, s)] for m in MODELS for s in SEEDS]
    first = ordered[0]
    if any(r["label_maps"] != first["label_maps"] or protocol(r["config"]) != protocol(first["config"]) for r in ordered):
        raise ValueError("Baseline class maps or preprocessing/taxonomy differ; cannot aggregate these seeds")
    return ordered


def camera_cohorts(dataset_root, year):
    manifest = dataset_root / "metadata/images.csv"
    rows = read_csv(manifest)
    selected = [r for r in rows if r["year"] == str(year) and r["camera"] in CAMERAS and r["kind"] == "worm"]
    if any(r["segmentation_status"] not in {"segmented", "reused", "failed"} for r in selected):
        raise ValueError("Segmentation is incomplete; finish it before freezing the test cohort")
    if len({r["image_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate image IDs in external test metadata")
    historical = {r["barcode"] for r in rows if r.get("original_paper_split")}
    overlap = historical & {r["barcode"] for r in selected}
    if overlap:
        raise ValueError(f"External and paper specimen barcodes overlap: {sorted(overlap)}")
    cohorts, excluded = {}, []
    for camera in CAMERAS:
        usable = []
        for row in sorted((r for r in selected if r["camera"] == camera), key=lambda r: r["image_id"]):
            if row["segmentation_status"] == "failed":
                excluded.append({**row, "exclusion_reason": "failed_segmentation"})
                continue
            relative = Path(row["rel_path_seg"])
            path = (dataset_root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(dataset_root.resolve()):
                raise ValueError(f"Invalid segmented image path: {relative}")
            if not path.is_file():
                raise FileNotFoundError(path)
            usable.append({**row, "file_bytes": str(path.stat().st_size), "file_mtime_ns": str(path.stat().st_mtime_ns)})
        if not usable:
            raise ValueError(f"No segmented worm images for {year}/{camera}")
        cohorts[camera] = usable
    return cohorts, excluded, sha(manifest)


def create_plan(args):
    root = args.output.resolve()
    dataset = args.dataset.resolve()
    results = args.publication_results.resolve()
    for source in (dataset, results):
        if root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError("External results must be separate from both dataset and original paper results")
    runs = discover_checkpoints(results)
    cohorts, excluded, manifest_hash = camera_cohorts(dataset, args.year)
    sources = [Path(__file__).resolve(), *sorted((PROJECT / "src/worm_species/models").glob("*.py")),
               *sorted((PROJECT / "src/worm_species/data").glob("*.py"))]
    plan = {"schema": 1, "dataset_root": str(dataset), "year": args.year,
            "manifest_sha256": manifest_hash, "cohorts": cohorts, "excluded": excluded,
            "runs": runs, "source_sha256": {str(p): sha(p) for p in sources},
            "batch_size": args.batch_size, "workers": args.workers, "cameras": list(CAMERAS)}
    plan["identity"] = identity(plan)
    with lock(root / ".plan.lock"):
        path = root / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError("This result folder belongs to another input snapshot. Choose a new EXTERNAL_RESULT")
        save_json(path, plan)
        for camera, records in cohorts.items():
            save_csv(root / "cohorts" / f"{camera}.csv", records)
        save_csv(root / "cohorts/excluded.csv", excluded, list(next(iter(cohorts.values()))[0]) + ["exclusion_reason"])
    print(f"90 checkpoints x 2 cameras; {len(cohorts['gphoto2'])} gphoto2, {len(cohorts['webcam'])} webcam images")
    print(f"Excluded failed segmentations: {len(excluded)}. Plan: {root / 'plan.json'}")
    return plan


def load_plan(root):
    plan = json.loads((root / "plan.json").read_text())
    expected = plan["identity"]
    if identity({k: v for k, v in plan.items() if k != "identity"}) != expected:
        raise ValueError("Plan identity mismatch")
    for path, expected_sha in plan["source_sha256"].items():
        if sha(path) != expected_sha:
            raise ValueError(f"Evaluation code changed after planning: {path}")
    return plan


def labeled_frame(rows, cfg):
    import pandas as pd
    from worm_species.data.taxonomy import parse_taxonomy_from_barcode, apply_taxonomic_uncertainty_rules
    frame = pd.DataFrame(rows).copy()
    frame["taxon_label"] = frame["taxon"]
    # Do not apply training rare-class filters to an external test cohort.
    frame = parse_taxonomy_from_barcode(frame, cfg)
    return apply_taxonomic_uncertainty_rules(frame, cfg)


def metric_record(truth, predicted, labels):
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score
    truth, predicted = np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)
    k = len(labels)
    if len(truth) != len(predicted) or ((truth < 0) | (truth >= k)).any() or ((predicted < 0) | (predicted >= k)).any():
        raise ValueError("Invalid or misaligned prediction indices")
    cm = (confusion_matrix(truth, predicted, labels=list(range(k)))
          if len(truth) else np.zeros((k, k), dtype=int))
    support = cm.sum(axis=1)
    present = support > 0
    metrics = {"n": len(truth), "classes_in_model": k, "classes_in_test": int(present.sum()),
               "uniform_chance": 1 / k,
               "accuracy": float((truth == predicted).mean()) if len(truth) else None,
               "balanced_accuracy": float((cm.diagonal()[present] / support[present]).mean()) if present.any() else None,
               "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)) if len(truth) else None,
               "majority_accuracy": float(support.max() / len(truth)) if len(truth) else None}
    return metrics, cm


def run_dir(root, run, camera):
    return root / camera / run["model"] / f"seed_{run['seed']}"


def complete(directory, plan):
    path = directory / "complete.json"
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return record.get("plan_identity") == plan["identity"] and all(
        (directory / name).is_file() and sha(directory / name) == checksum
        for name, checksum in record.get("files", {}).items()
    ) and set(record.get("files", {})) == {"predictions.csv", "metrics.json", *(f"confusion_{task}.csv" for task in TASKS)}


def cache_images(plan):
    """Resize once per node with the paper's PIL bilinear preprocessing, lossless PNG."""
    from PIL import Image
    base = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir())) / os.environ.get("USER", "user")
    cache = base / "publication_external_cache" / plan["identity"]
    size = plan["runs"][0]["config"]["preprocessing"]["image_size"]
    rows = [row for camera in CAMERAS for row in plan["cohorts"][camera]]
    with lock(cache / ".lock"):
        for row in rows:
            source = Path(plan["dataset_root"]) / row["rel_path_seg"]
            if stamp(source) != {"bytes": int(row["file_bytes"]), "mtime_ns": int(row["file_mtime_ns"])}:
                raise ValueError(f"Dataset image changed after planning: {source}")
            dest = cache / (row["image_id"] + ".png")
            if not dest.is_file():
                temp = dest.with_suffix(".partial.png")
                with Image.open(source) as image:
                    image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR).save(temp)
                temp.replace(dest)
    return cache


def evaluate_camera(model, cfg, maps, frame, camera, run, plan, root, device, checkpoint_hash, cache):
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from worm_species.data.datasets import MultiTaskWormImageDataset
    from worm_species.data.transforms import build_split_transform
    directory = run_dir(root, run, camera)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "complete.json").unlink(missing_ok=True)
    inputs = frame.copy()
    inputs["cached_image"] = inputs.image_id + ".png"
    transform = build_split_transform(split="test", preprocessing=cfg["preprocessing"],
                                      condition={"transform": "original"},
                                      original_colour_retention=cfg["data"].get("colour_retention", 1.0))
    dataset = MultiTaskWormImageDataset(inputs, root_dir=cache, image_col="cached_image", transform=transform,
                                      crop_to_foreground=False, target_cols=cfg["data"]["target_cols"],
                                      label_to_index_by_task=maps)
    loader = DataLoader(dataset, batch_size=plan["batch_size"], shuffle=False, num_workers=plan["workers"],
                        pin_memory=device.type == "cuda")
    names = {task: [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])] for task, mapping in maps.items()}
    rows, offset = [], 0
    with torch.inference_mode():
        for batch in loader:
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda" and cfg["training"].get("use_amp", True)):
                logits = model(batch["image"].to(device, non_blocking=True))
            for task in TASKS:
                probabilities = logits[task].float().softmax(dim=1).cpu().numpy()
                guesses = probabilities.argmax(axis=1)
                truth = batch["labels"][task].numpy()
                for j, (true, pred) in enumerate(zip(truth, guesses)):
                    source = frame.iloc[offset + j]
                    value = source[cfg["data"]["target_cols"][task]]
                    missing = pd.isna(value) or str(value).strip() == ""
                    rows.append({"camera": camera, "model": run["model"], "seed": run["seed"], "task": task,
                                 "image_id": source.image_id, "individual_id": source.individual_id,
                                 "barcode": source.barcode, "location_code": source.location_code,
                                 "rel_path_seg": source.rel_path_seg, "true_index": int(true),
                                 "predicted_index": int(pred), "true_label": "" if missing else str(value),
                                 "predicted_label": names[task][pred], "scored": int(true >= 0),
                                 "unscored_reason": "" if true >= 0 else "missing_or_uncertain_label" if missing else "outside_checkpoint_vocabulary",
                                 "predicted_probability": float(probabilities[j, pred]),
                                 "class_probabilities_json": json.dumps(dict(zip(names[task], map(float, probabilities[j]))))})
            offset += len(batch["image"])
    if offset != len(frame):
        raise ValueError("Incomplete image coverage")
    save_csv(directory / "predictions.csv", rows)
    predictions = pd.DataFrame(rows)
    metrics = {}
    for task in TASKS:
        selected = predictions.loc[predictions.task.eq(task) & predictions.scored.eq(1)]
        metrics[task], cm = metric_record(selected.true_index, selected.predicted_index, names[task])
        metrics[task]["unscored_images"] = len(frame) - len(selected)
        pd.DataFrame(cm, index=names[task], columns=names[task]).to_csv(directory / f"confusion_{task}.csv", index_label="true_label")
    save_json(directory / "metrics.json", metrics)
    files = ["predictions.csv", "metrics.json", *(f"confusion_{task}.csv" for task in TASKS)]
    save_json(directory / "complete.json", {"plan_identity": plan["identity"], "checkpoint_sha256": checkpoint_hash,
                                           "images": len(frame), "files": {name: sha(directory / name) for name in files}})
    print(f"Completed {run['model']} seed={run['seed']} camera={camera}: {len(frame)} images", flush=True)


def worker(root, index, device_name):
    import torch
    from worm_species.models.multitask import build_multitask_model
    plan = load_plan(root)
    if not 0 <= index < len(plan["runs"]):
        raise ValueError("Array task index is outside the plan")
    run = plan["runs"][index]
    with lock(root / ".locks" / f"task_{index}.lock"):
        if stamp(run["checkpoint"]) != run["checkpoint_stamp"]:
            raise ValueError("Checkpoint changed after planning")
        pending = [camera for camera in CAMERAS if not complete(run_dir(root, run, camera), plan)]
        if not pending:
            print("Both cameras already complete; skipping"); return
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU is available in this task")
        torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
        torch.manual_seed(run["seed"])
        checkpoint_hash = sha(run["checkpoint"])
        checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
        cfg = checkpoint["cfg"]
        if cfg["model"]["name"] != run["model"] or cfg["seed"] != run["seed"] or protocol(cfg) != protocol(run["config"]):
            raise ValueError("Checkpoint config does not match the planned baseline")
        maps = checkpoint["label_to_index_by_task"]
        if maps != run["label_maps"]:
            raise ValueError("Checkpoint class maps differ from saved sidecar maps")
        model_cfg = copy.deepcopy(cfg)
        model_cfg["model"]["pretrained"] = False  # Load the saved weights, never download ImageNet weights.
        model = build_multitask_model(model_cfg, {task: len(mapping) for task, mapping in maps.items()})
        model.load_state_dict(checkpoint["model_state"], strict=True)
        del checkpoint
        model.to(device).eval()
        cache = cache_images(plan)
        for camera in pending:  # gphoto2 first, webcam second for every checkpoint
            frame = labeled_frame(plan["cohorts"][camera], cfg)
            evaluate_camera(model, cfg, maps, frame, camera, run, plan, root, device, checkpoint_hash, cache)


def collect(root):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import t
    plan = load_plan(root)
    (root / "report_complete.json").unlink(missing_ok=True)
    incomplete = [f"{r['model']}/{r['seed']}/{c}" for r in plan["runs"] for c in CAMERAS
                  if not complete(run_dir(root, r, c), plan)]
    if incomplete:
        raise ValueError(f"Cannot report all 30 seeds: {len(incomplete)} evaluations incomplete; first: {incomplete[:5]}")
    report = root / "summary"
    report.mkdir(parents=True, exist_ok=True)
    metric_rows, matrices = [], {}
    for run in plan["runs"]:
        if sha(run["original_predictions"]) != run["original_predictions_sha256"]:
            raise ValueError("Original-test predictions changed after planning")
        old = pd.read_csv(run["original_predictions"])
        for camera in (*CAMERAS, "original_test"):
            for task in TASKS:
                names = [name for name, _ in sorted(run["label_maps"][task].items(), key=lambda kv: kv[1])]
                if camera == "original_test":
                    selected = old.loc[old.task.eq(task)]
                    metrics, cm = metric_record(selected.true_index, selected.predicted_index, names)
                else:
                    directory = run_dir(root, run, camera)
                    metrics = json.loads((directory / "metrics.json").read_text())[task]
                    frame = pd.read_csv(directory / f"confusion_{task}.csv", index_col=0)
                    if list(frame.index) != names or list(frame.columns) != names:
                        raise ValueError("Confusion matrix class ordering differs across seeds")
                    cm = frame.to_numpy()
                matrices.setdefault((run["model"], camera, task), []).append(cm)
                metric_rows.append({"model": run["model"], "seed": run["seed"], "camera": camera, "task": task, **metrics})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(report / "per_seed_metrics.csv", index=False)
    scores = ["accuracy", "balanced_accuracy", "macro_f1"]
    aggregate_rows, paired_rows = [], []
    for (model, camera, task), group in metrics.groupby(["model", "camera", "task"]):
        if set(group.seed) != set(SEEDS) or len(group) != 30:
            raise ValueError("Report requires each of the 30 unique seeds exactly once")
        for score in scores:
            values = group[score].dropna().to_numpy()
            mean = float(values.mean()) if len(values) else np.nan
            sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            half = float(t.ppf(.975, len(values)-1)*sd/np.sqrt(len(values))) if len(values)>1 else np.nan
            aggregate_rows.append({"model": model, "camera": camera, "task": task, "metric": score,
                                   "seeds": len(values), "mean": mean, "seed_sd": sd,
                                   "ci95_low": mean-half, "ci95_high": mean+half,
                                   "images_per_seed": int(group.n.iloc[0]), "uniform_chance": group.uniform_chance.iloc[0],
                                   "majority_accuracy": group.majority_accuracy.iloc[0]})
        if camera != "original_test":
            original = metrics.loc[metrics.model.eq(model) & metrics.camera.eq("original_test") & metrics.task.eq(task)].set_index("seed")
            external = group.set_index("seed")
            for score in scores:
                differences = (external[score] - original[score]).dropna()
                n = len(differences)
                mean = differences.mean()
                sd = differences.std(ddof=1)
                half = t.ppf(.975, n-1)*sd/np.sqrt(n) if n>1 else np.nan
                paired_rows.append({"model": model, "camera": camera, "task": task, "metric": score,
                                    "seeds": n, "mean_new_minus_original": mean,
                                    "ci95_low": mean-half, "ci95_high": mean+half})
    pd.DataFrame(aggregate_rows).to_csv(report / "seed_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(report / "new_vs_original.csv", index=False)
    cm_root = report / "confusion_matrices"
    cm_root.mkdir(exist_ok=True)
    for model in MODELS:
        maps = next(r["label_maps"] for r in plan["runs"] if r["model"] == model)
        for task in TASKS:
            names = [name for name, _ in sorted(maps[task].items(), key=lambda kv: kv[1])]
            fig, axes = plt.subplots(1, 2, figsize=(max(12, len(names)*2), max(5, len(names)*.8)), squeeze=False)
            for ax, camera in zip(axes.flat, CAMERAS):
                counts = np.stack(matrices[(model, camera, task)]).astype(float)
                support = counts.sum(axis=2, keepdims=True)
                fractions = np.divide(counts, support, out=np.zeros_like(counts), where=support>0)
                means, sd = fractions.mean(axis=0), fractions.std(axis=0, ddof=1)
                absent = support[0, :, 0] == 0
                means[absent] = np.nan
                sd[absent] = np.nan
                prefix = cm_root / f"{camera}_{model}_{task}"
                pd.DataFrame(counts.mean(axis=0), index=names, columns=names).to_csv(str(prefix) + "_mean_counts.csv", index_label="true_label")
                pd.DataFrame(means, index=names, columns=names).to_csv(str(prefix) + "_mean_row_fraction.csv", index_label="true_label")
                pd.DataFrame(sd, index=names, columns=names).to_csv(str(prefix) + "_row_fraction_seed_sd.csv", index_label="true_label")
                cmap = plt.colormaps["Blues"].copy(); cmap.set_bad("#eeeeee")
                image = ax.imshow(means, vmin=0, vmax=1, cmap=cmap)
                ax.set(xticks=range(len(names)), xticklabels=names, yticks=range(len(names)),
                       yticklabels=[f"{label} (n={int(n)})" for label, n in zip(names, support[0,:,0])],
                       title=f"{camera} — mean over 30 seeds", xlabel="Predicted label", ylabel="True label")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
                ax.tick_params(axis="y", labelsize=8)
                for (i,j), value in np.ndenumerate(means):
                    ax.text(j,i,"—" if np.isnan(value) else f"{100*value:.1f}%", ha="center", va="center",
                            color="white" if value>.5 else "black", fontsize=8)
                fig.colorbar(image, ax=ax, fraction=.035, pad=.02)
            fig.suptitle(f"{model} / {task} — external test, row-normalized confusion")
            fig.tight_layout()
            for extension in ("png", "svg", "pdf"):
                fig.savefig(cm_root / f"{model}_{task}.{extension}", dpi=180, bbox_inches="tight")
            plt.close(fig)
    (report / "README.md").write_text(
        "# External camera evaluation\n\n"
        "All 30 seeds of each baseline are evaluated separately on gphoto2 and webcam. "
        "Predictions include unscored uncertain/missing or out-of-vocabulary labels; "
        "metrics and confusion matrices use only scorable labels for each task. "
        "No rare-class threshold is applied to external test data.\n\n"
        "Confusion matrices show the mean of 30 row-normalized matrices, with the "
        "number of images per seed on each true-label row. Absent true classes are gray. "
        "The same images repeated across seeds are not counted as additional specimens. "
        "Mean counts, row fractions and seed SD are saved as CSVs.\n\n"
        "Metrics are image-level. Balanced accuracy averages recall over supported true "
        "classes; macro-F1 uses the union of observed true/predicted labels, as in the "
        "paper trainer. Uniform chance is 1/K for the checkpoint's class count; "
        "majority-class accuracy is reported separately.\n\n"
        "95% t intervals summarize variation across training seeds, not uncertainty "
        "across independent specimens. new_vs_original.csv pairs the same checkpoint "
        "seed across datasets; cohort/class composition can differ. Camera evaluations "
        "use all their successful images, so camera differences are not a controlled "
        "comparison of identical photographs. Failed segmentations are listed in "
        "../cohorts/excluded.csv.\n"
    )
    save_json(root / "report_complete.json", {"plan_identity": plan["identity"], "models": list(MODELS),
                                              "seeds_per_model": 30, "camera_evaluations": 180})
    print(f"Complete: {report}")


def pending_indices(root, plan):
    return [i for i, run in enumerate(plan["runs"])
            if any(not complete(run_dir(root, run, camera), plan) for camera in CAMERAS)]


def array_spec(indices):
    """Compress numeric indices while preserving their positions in the frozen plan."""
    groups = []
    for index in indices:
        if groups and index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return ",".join(str(g[0]) if len(g) == 1 else f"{g[0]}-{g[-1]}" for g in groups)


def render_jobs(root, plan, args):
    jobs, logs = root / "jobs", root / "logs"
    jobs.mkdir(parents=True, exist_ok=True); logs.mkdir(exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve())]
    header = "#!/usr/bin/env bash\nset -euo pipefail\n" + f"cd {shlex.quote(str(PROJECT))}\n"
    header += "export MPLCONFIGDIR=\"${TMPDIR:-/tmp}/worm-external-mpl-${USER}\"\n"
    array = jobs / "evaluate.sh"
    array.write_text(header + shlex.join(command + ["worker", "--output", str(root)]) + ' --index "$SLURM_ARRAY_TASK_ID" --device cuda:0\n')
    report = jobs / "report.sh"
    report.write_text(header + shlex.join(command + ["collect", "--output", str(root)]) + "\n")
    array_command = ["sbatch", "--parsable", "--job-name=worm-camera-test", f"--account={args.account}",
                     f"--partition={args.partition}", "--nodes=1", "--ntasks=1", "--gpus=1",
                     f"--cpus-per-task={args.cpus}", f"--mem={args.memory}", f"--time={args.time}",
                     f"--array={array_spec(pending_indices(root, plan))}%{args.max_active}",
                     f"--output={logs}/eval_%A_%a.out", f"--error={logs}/eval_%A_%a.err", str(array)]
    report_command = ["sbatch", "--parsable", "--job-name=worm-camera-report", f"--account={args.account}",
                      "--nodes=1", "--ntasks=1", f"--cpus-per-task={args.report_cpus}",
                      f"--mem={args.report_memory}", f"--time={args.report_time}",
                      f"--output={logs}/report_%j.out", f"--error={logs}/report_%j.err", str(report)]
    if args.report_partition:
        report_command.insert(-1, f"--partition={args.report_partition}")
    return array_command, report_command


def submit(args, plan):
    root = args.output.resolve()
    if args.mode == "plan":
        array, report = render_jobs(root, plan, args)
        if pending_indices(root, plan):
            print(shlex.join(array))
            report.insert(-1, "--dependency=afterok:<ARRAY_JOB_ID>")
        print(shlex.join(report))
        return
    with lock(root / ".submission.lock"):
        receipt_path = root / "submission.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("plan_identity") != plan["identity"]:
                raise ValueError("Submission receipt belongs to another plan")
            ids = [receipt[k] for k in ("array_job_id", "report_job_id") if receipt.get(k)]
            if ids:
                running = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i"], check=True, capture_output=True, text=True)
                if running.stdout.strip():
                    raise RuntimeError("Evaluation/report jobs are still active; see submission.json")
        array, report = render_jobs(root, plan, args)
        def job_id(command):
            value = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]
            if not value.isdigit():
                raise RuntimeError(f"Unexpected sbatch job ID: {value!r}")
            return value
        indices = pending_indices(root, plan)
        receipt = {"plan_identity": plan["identity"], "array_indices": indices}
        if indices:
            receipt["array_job_id"] = job_id(array)
            save_json(receipt_path, receipt)
            # sbatch options must precede the script path.
            report.insert(-1, "--dependency=afterok:" + receipt["array_job_id"])
        receipt["report_job_id"] = job_id(report)
        save_json(receipt_path, receipt)
        print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "submit", "worker", "collect", "status"])
    parser.add_argument("--dataset", type=Path, default=Path("/faststorage/project/worm-species/publication_dataset"))
    parser.add_argument("--publication-results", type=Path, default=PROJECT / "publication_30seed_result")
    parser.add_argument("--output", type=Path, default=PROJECT / "publication_external_2026")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--account", default="worm-species")
    parser.add_argument("--partition", default="gpu-short")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--memory", default="32G")
    parser.add_argument("--time", default="02:00:00")
    parser.add_argument("--max-active", type=int, default=12)
    parser.add_argument("--report-cpus", type=int, default=4)
    parser.add_argument("--report-memory", default="8G")
    parser.add_argument("--report-time", default="00:30:00")
    parser.add_argument("--report-partition", default="")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or min(args.cpus, args.report_cpus) < 1 or not 1 <= args.max_active <= 12:
        parser.error("Invalid resources: batch/cpus positive, workers nonnegative, max-active between 1 and 12")
    if args.mode in {"plan", "submit"}:
        plan = create_plan(args)
        submit(args, plan)
    elif args.mode == "worker":
        worker(args.output.resolve(), args.index, args.device)
    elif args.mode == "collect":
        with lock(args.output.resolve() / ".report.lock"):
            collect(args.output.resolve())
    else:
        root = args.output.resolve()
        plan = load_plan(root)
        counts = {camera: sum(complete(run_dir(root, run, camera), plan) for run in plan["runs"])
                  for camera in CAMERAS}
        print(json.dumps({"completed_per_camera": counts, "expected_per_camera": len(plan["runs"]),
                          "pending_array_indices": pending_indices(root, plan),
                          "report_complete": (root / "report_complete.json").is_file()}, indent=2))


if __name__ == "__main__":
    main()
