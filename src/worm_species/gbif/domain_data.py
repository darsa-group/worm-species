"""Deterministic two-domain manifests for the GBIF/Petri order experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .pipeline import resolve_manifest_image_path


TASK_COLUMNS = {"genus": "genus", "species": "species", "age": "age"}
SPLITS = ("train", "validation", "test")
DOMAINS = ("gbif", "petri")


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    return value


def load_domain_config(path: str | Path) -> dict:
    source = Path(path)
    config = _expand(yaml.safe_load(source.read_text(encoding="utf-8")))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("GBIF training config requires schema_version: 1")
    required_paths = {
        "project_root", "gbif_manifest", "petri_split_dir",
        "petri_data_root", "output_root", "conda_sh", "conda_env",
    }
    missing = required_paths.difference(config.get("paths", {}))
    if missing:
        raise ValueError(f"Missing GBIF training paths: {sorted(missing)}")
    slurm = config.get("slurm", {})
    if int(slurm.get("gpus_per_task", 0)) != 1:
        raise ValueError("Every GBIF training/inference task must request one GPU")
    if int(slurm.get("array_max_active", 0)) != 12:
        raise ValueError("slurm.array_max_active must be 12")
    if str(slurm.get("partition")) != "gpu-short,gpu-l40s,gpu-h200":
        raise ValueError(
            "slurm.partition must be gpu-short,gpu-l40s,gpu-h200"
        )
    if str(slurm.get("memory")) != "20G":
        raise ValueError("slurm.memory must be 20G")
    training = config.get("training", {})
    if int(training.get("num_workers", 0)) != 12:
        raise ValueError("training.num_workers must be 12")
    if int(slurm.get("cpus_per_task", 0)) != 16:
        raise ValueError("slurm.cpus_per_task must be 16")
    if int(training.get("batch_size", 0)) != 128:
        raise ValueError("training.batch_size must be 128")
    if int(training.get("prefetch_factor", 0)) != 4:
        raise ValueError("training.prefetch_factor must be 4")
    if not bool(training.get("persistent_workers", False)):
        raise ValueError("training.persistent_workers must be true")
    if bool((training.get("hierarchy_loss", {}) or {}).get("enabled", False)):
        raise ValueError("The approved GBIF experiment disables hierarchy loss")
    inference = config.get("inference", {})
    if int(inference.get("shards", 0)) != 12:
        raise ValueError("inference.shards must be 12")
    if int(inference.get("num_workers", 0)) != 12:
        raise ValueError("inference.num_workers must be 12")
    models = config.get("models", {})
    if models.get("primary") != ["vit_b_16", "resnet50", "convnext_base"]:
        raise ValueError("Approved primary models are ViT-B/16, ResNet-50, and ConvNeXt-Base")
    if models.get("primary_seeds") != [40, 140, 240, 340, 440]:
        raise ValueError("Approved primary seeds are 40, 140, 240, 340, and 440")
    if models.get("dino") != ["dinov3_vitb16"] or models.get("dino_seeds") != [40, 140, 240]:
        raise ValueError("Approved final DINOv3 seeds are 40, 140, and 240")
    if (
        int(training.get("batch_size", 0)) * int(training.get("steps_per_domain", 0))
        != int(training.get("mixed_batch_per_domain", 0)) * int(training.get("mixed_steps", 0))
    ):
        raise ValueError("Sequential and mixed domain image exposures must be equal")
    if bool((config.get("wandb", {}) or {}).get("log_model", True)):
        raise ValueError("W&B checkpoint/model uploads must remain disabled")
    return config


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _gbif_group_ids(frame: pd.DataFrame) -> pd.Series:
    """Join rows sharing an occurrence or exact image content."""
    sets = _DisjointSet()
    for row in frame.itertuples(index=False):
        occurrence = f"occ:{row.gbif_id}"
        sets.find(occurrence)
        digest = str(getattr(row, "sha256", "")).strip()
        if digest:
            sets.union(occurrence, f"sha:{digest}")
    roots = []
    for row in frame.itertuples(index=False):
        roots.append("gbif:" + sets.find(f"occ:{row.gbif_id}"))
    return pd.Series(roots, index=frame.index, dtype="string")


def _normalise_label(series: pd.Series) -> pd.Series:
    result = series.fillna("").astype(str).str.strip()
    return result.mask(result.str.lower().isin({
        "", "na", "n/a", "nan", "none", "null", "unknown",
        "unidentified", "missing", "not_available",
    }), "")


def _prepare_gbif(config: dict) -> pd.DataFrame:
    manifest = Path(config["paths"]["gbif_manifest"])
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    required = {"image_id", "gbif_id", "local_path", "genus", "species_label",
                "curation_label", "download_status", "sha256"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Curated GBIF manifest lacks columns: {sorted(missing)}")
    labels = set(config["data"].get("curation_labels", ["keep"]))
    frame = frame.loc[
        frame["curation_label"].isin(labels)
        & frame["download_status"].eq("downloaded")
    ].copy()
    if frame.empty:
        raise ValueError("No curated GBIF images match the configured labels")
    frame["sample_id"] = "gbif:" + frame["image_id"].astype(str)
    frame["group_id"] = _gbif_group_ids(frame)
    frame["image_path"] = frame["local_path"].map(
        lambda value: str(resolve_manifest_image_path(manifest, value))
    )
    frame["genus"] = _normalise_label(frame["genus"])
    frame["species"] = _normalise_label(frame["species_label"])
    frame["age"] = ""
    frame["domain"] = "gbif"
    return frame[[
        "sample_id", "group_id", "image_path", "domain", "genus",
        "species", "age", "gbif_id", "sha256",
    ]].reset_index(drop=True)


def _prepare_petri(config: dict) -> pd.DataFrame:
    split_root = Path(config["paths"]["petri_split_dir"])
    data_root = Path(config["paths"]["petri_data_root"])
    frames = []
    for split, filename in (
        ("train", "train_split.csv"),
        ("validation", "val_split.csv"),
        ("test", "test_split.csv"),
    ):
        source = split_root / filename
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
        required = {"barcode", "rel_path_seg", "genus", "species_label", "life_stage"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Petri split {source} lacks columns: {sorted(missing)}")
        frame["split"] = split
        frame["domain"] = "petri"
        frame["group_id"] = "petri:" + frame["barcode"].astype(str)
        frame["image_path"] = frame["rel_path_seg"].map(
            lambda value: str(data_root / value)
        )
        frame["genus"] = _normalise_label(frame["genus"])
        frame["species"] = _normalise_label(frame["species_label"])
        frame["age"] = _normalise_label(frame["life_stage"])
        identity = frame["rel_path_seg"].astype(str).map(
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
        )
        frame["sample_id"] = "petri:" + identity
        frame["gbif_id"] = ""
        frame["sha256"] = ""
        frames.append(frame[[
            "sample_id", "group_id", "image_path", "domain", "genus",
            "species", "age", "gbif_id", "sha256", "split",
        ]])
    result = pd.concat(frames, ignore_index=True)
    if result["sample_id"].duplicated().any():
        raise ValueError("Petri image paths do not produce unique sample IDs")
    return result


def _eligible_labels(
    frames: Iterable[pd.DataFrame], column: str, minimum_groups: int
) -> set[str]:
    combined = pd.concat([
        frame.loc[frame[column].ne(""), [column, "group_id"]]
        for frame in frames
    ], ignore_index=True)
    counts = combined.groupby(column)["group_id"].nunique()
    return set(counts[counts >= minimum_groups].index.astype(str))


def _mask_gbif_group_taxonomy_conflicts(frame: pd.DataFrame) -> dict[str, int]:
    """Do not train contradictory labels attached to the same visual group."""
    audit = {"conflicting_genus_groups": 0, "conflicting_species_groups": 0}
    for group_id, group in frame.groupby("group_id", sort=False):
        genera = set(group["genus"]) - {""}
        species = set(group["species"]) - {""}
        indices = group.index
        if len(genera) > 1:
            audit["conflicting_genus_groups"] += 1
            frame.loc[indices, ["genus", "species"]] = ""
        elif len(species) > 1:
            audit["conflicting_species_groups"] += 1
            frame.loc[indices, "species"] = ""
    return audit


def _assign_gbif_splits(frame: pd.DataFrame, config: dict) -> pd.Series:
    seed = int(config["data"]["seed"])
    proportions = config["data"]["gbif_split"]
    groups = frame.groupby("group_id", sort=True).agg(
        genus=("genus", lambda values: sorted(set(values) - {""})),
        species=("species", lambda values: sorted(set(values) - {""})),
    ).reset_index()
    conflicting = groups.loc[
        groups["genus"].map(len).gt(1) | groups["species"].map(len).gt(1)
    ]
    if not conflicting.empty:
        raise AssertionError("GBIF taxonomy conflicts must be masked before splitting")
    groups["stratum"] = groups.apply(
        lambda row: (
            "species:" + row["species"][0]
            if row["species"] else "genus:" + row["genus"][0]
        ),
        axis=1,
    )
    assignment: dict[str, str] = {}
    for stratum, stratum_frame in groups.groupby("stratum", sort=True):
        identifiers = stratum_frame["group_id"].tolist()
        identifiers.sort(key=lambda value: hashlib.sha256(
            f"{seed}|{stratum}|{value}".encode("utf-8")
        ).hexdigest())
        count = len(identifiers)
        if count < 3:
            for identifier in identifiers:
                assignment[identifier] = "train"
            continue
        test_n = max(1, int(round(count * float(proportions["test"]))))
        val_n = max(1, int(round(count * float(proportions["validation"]))))
        while test_n + val_n >= count:
            if test_n >= val_n and test_n > 1:
                test_n -= 1
            elif val_n > 1:
                val_n -= 1
            else:
                break
        for identifier in identifiers[:test_n]:
            assignment[identifier] = "test"
        for identifier in identifiers[test_n:test_n + val_n]:
            assignment[identifier] = "validation"
        for identifier in identifiers[test_n + val_n:]:
            assignment[identifier] = "train"
    return frame["group_id"].map(assignment)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_domain_manifests(config: dict) -> dict:
    gbif = _prepare_gbif(config)
    petri = _prepare_petri(config)
    taxonomy_conflicts = _mask_gbif_group_taxonomy_conflicts(gbif)
    minimum = int(config["data"]["minimum_independent_groups_per_class"])
    eligible = {
        task: _eligible_labels((gbif, petri), column, minimum)
        for task, column in TASK_COLUMNS.items()
    }
    for frame in (gbif, petri):
        for task, column in TASK_COLUMNS.items():
            frame.loc[~frame[column].isin(eligible[task]), column] = ""
        if frame[list(TASK_COLUMNS.values())].eq("").all(axis=1).any():
            frame.drop(
                index=frame.index[frame[list(TASK_COLUMNS.values())].eq("").all(axis=1)],
                inplace=True,
            )
            frame.reset_index(drop=True, inplace=True)

    gbif["split"] = _assign_gbif_splits(gbif, config)
    if gbif["split"].isna().any():
        raise AssertionError("Every GBIF group must receive a split")

    combined_train = pd.concat([
        gbif.loc[gbif["split"].eq("train")],
        petri.loc[petri["split"].eq("train")],
    ], ignore_index=True)
    label_maps = {}
    for task, column in TASK_COLUMNS.items():
        labels = sorted(eligible[task])
        absent = set(labels).difference(combined_train[column].loc[
            combined_train[column].ne("")
        ].unique())
        if absent:
            raise ValueError(
                f"Fixed {task} label map includes labels absent from training: {sorted(absent)}"
            )
        label_maps[task] = {label: index for index, label in enumerate(labels)}

    output_root = Path(config["paths"]["output_root"])
    prepared = output_root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_id", "group_id", "image_path", "domain", "genus",
        "species", "age", "gbif_id", "sha256", "split",
    ]
    artifact_paths = []
    for domain, frame in (("gbif", gbif), ("petri", petri)):
        for split in SPLITS:
            destination = prepared / f"{domain}_{split}.csv"
            frame.loc[frame["split"].eq(split), columns].to_csv(destination, index=False)
            artifact_paths.append(destination)
    label_maps_path = prepared / "label_maps.json"
    _atomic_json(label_maps_path, label_maps)
    artifact_paths.append(label_maps_path)

    summary = {
        "schema_version": 1,
        "inputs": {
            "gbif_manifest": str(Path(config["paths"]["gbif_manifest"]).resolve()),
            "gbif_manifest_sha256": file_sha256(config["paths"]["gbif_manifest"]),
            "petri_splits": {
                name: file_sha256(Path(config["paths"]["petri_split_dir"]) / filename)
                for name, filename in {
                    "train": "train_split.csv", "validation": "val_split.csv",
                    "test": "test_split.csv",
                }.items()
            },
        },
        "minimum_independent_groups_per_class": minimum,
        "masked_gbif_taxonomy_conflicts": taxonomy_conflicts,
        "label_counts": {task: len(mapping) for task, mapping in label_maps.items()},
        "rows": {
            domain: {
                split: int((frame["split"] == split).sum())
                for split in SPLITS
            }
            for domain, frame in (("gbif", gbif), ("petri", petri))
        },
        "groups": {
            domain: {
                split: int(frame.loc[frame["split"].eq(split), "group_id"].nunique())
                for split in SPLITS
            }
            for domain, frame in (("gbif", gbif), ("petri", petri))
        },
        "prepared_root": str(prepared.resolve()),
        "prepared_artifact_sha256": {
            path.name: file_sha256(path) for path in artifact_paths
        },
    }
    _atomic_json(prepared / "summary.json", summary)
    return summary


def prepared_paths(config: dict, domain: str, split: str) -> Path:
    if domain not in DOMAINS or split not in SPLITS:
        raise ValueError(f"Unknown prepared domain/split: {domain}/{split}")
    return Path(config["paths"]["output_root"]) / "prepared" / f"{domain}_{split}.csv"
