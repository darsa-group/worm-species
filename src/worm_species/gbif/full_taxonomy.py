"""Immutable full-taxonomy GBIF audit and split construction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from .pipeline import resolve_manifest_image_path


SPLITS = ("train", "validation", "test")
GBIF_TASKS = {"genus": "genus", "species": "species"}
PETRI_TASKS = {"genus": "genus", "species": "species", "age": "age"}
IDENTITY_COLUMNS = (
    "gbif_id", "occurrence_id", "sha256", "image_id", "local_path",
    "source_url", "media_reference",
)


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def load_full_taxonomy_config(path: str | Path) -> dict:
    config = _expand(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Full-taxonomy config requires schema_version: 1")
    if config.get("experiment_id") != "gbif_full_taxonomy_v1":
        raise ValueError("Refusing an unrecognised immutable experiment_id")
    required = {
        "project_root", "gbif_manifest", "petri_split_dir", "petri_data_root",
        "experiment_root", "conda_sh", "conda_env",
    }
    missing = required.difference(config.get("paths", {}))
    if missing:
        raise ValueError(f"Missing full-taxonomy paths: {sorted(missing)}")
    root = Path(config["paths"]["experiment_root"])
    if not root.is_absolute() or "gbif_full_taxonomy" not in root.name:
        raise ValueError("experiment_root must be a dedicated absolute full-taxonomy root")
    if config["models"]["backbones"] != ["convnext_base", "vit_b_16", "resnet50"]:
        raise ValueError("Full-taxonomy primary backbones changed")
    if config["models"]["seeds"] != [40, 140, 240]:
        raise ValueError("Full-taxonomy seeds must be 40, 140, and 240")
    if int(config["slurm"]["array_max_active"]) != 12:
        raise ValueError("Full-taxonomy array_max_active must be 12")
    if float(config["training"]["revised_hierarchy"]["weight"]) <= 0:
        raise ValueError("Revised hierarchy weight must be positive")
    if config["analysis"]["top_k"] != [1, 3, 5]:
        raise ValueError("Full-taxonomy top_k must be [1, 3, 5]")
    return config


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_commit(project_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
        capture_output=True, check=False,
    )
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(f"Could not resolve git commit: {result.stderr.strip()}")
    return result.stdout.strip()


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _clean(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return "" if text.lower() in {"", "nan", "none", "unknown", "unidentified"} else text


def canonical_taxonomy(genus: object, species: object) -> tuple[str, str, str]:
    """Return canonical genus/species labels and an explicit species status."""
    genus_text = _clean(genus).replace("_", " ").split(" ")[0]
    species_text = _clean(species).replace("_", " ")
    if genus_text:
        genus_text = genus_text[0].upper() + genus_text[1:]
    if not species_text:
        return genus_text, "", "missing_species_label"
    tokens = species_text.split()
    if len(tokens) < 2:
        return genus_text, "", "species_is_not_binomial"
    species_genus = tokens[0][0].upper() + tokens[0][1:]
    if not genus_text:
        return "", "", "missing_genus_label"
    if species_genus != genus_text:
        return genus_text, "", "species_genus_mismatch"
    canonical = "_".join([genus_text, *[token.lower() for token in tokens[1:]]])
    return genus_text, canonical, "valid"


def _build_connected_groups(frame: pd.DataFrame, connect_dhash: bool) -> pd.Series:
    sets = DisjointSet()
    columns = list(IDENTITY_COLUMNS) + (["dhash"] if connect_dhash else [])
    for index, row in frame.iterrows():
        row_node = f"row:{index}"
        sets.find(row_node)
        for column in columns:
            value = _clean(row.get(column, ""))
            if value:
                sets.union(row_node, f"{column}:{value}")
    return pd.Series(
        ["gbif-group:" + sets.find(f"row:{index}") for index in frame.index],
        index=frame.index, dtype="string",
    )


def _split_counts(group_count: int, proportions: dict) -> tuple[int, int, int]:
    if group_count == 1:
        return 1, 0, 0
    if group_count == 2:
        return 1, 0, 1
    validation = max(1, int(round(group_count * float(proportions["validation"]))))
    test = max(1, int(round(group_count * float(proportions["test"]))))
    while validation + test >= group_count:
        if test >= validation and test > 1:
            test -= 1
        elif validation > 1:
            validation -= 1
        else:
            break
    return group_count - validation - test, validation, test


def coverage_aware_split(frame: pd.DataFrame, config: dict) -> pd.Series:
    """Assign connected groups with species coverage before target proportions."""
    seed = int(config["data"]["split_seed"])
    proportions = config["data"]["target_proportions"]
    grouped = frame.groupby("group_id", sort=True).agg(
        species=("species", lambda values: sorted(set(values) - {""})),
        genus=("genus", lambda values: sorted(set(values) - {""})),
    ).reset_index()
    conflicts = grouped.loc[grouped["species"].map(len).gt(1) | grouped["genus"].map(len).gt(1)]
    if not conflicts.empty:
        raise ValueError(
            f"Fatal taxonomy conflict in {len(conflicts)} connected groups; see audit table"
        )
    grouped["stratum"] = grouped.apply(
        lambda row: (
            "species:" + row["species"][0] if row["species"]
            else "genus:" + row["genus"][0] if row["genus"]
            else "unlabelled"
        ), axis=1,
    )
    assignment: dict[str, str] = {}
    for stratum, subset in grouped.groupby("stratum", sort=True):
        identifiers = subset["group_id"].tolist()
        identifiers.sort(key=lambda value: hashlib.sha256(
            f"{seed}|{stratum}|{value}".encode("utf-8")
        ).hexdigest())
        train_n, validation_n, test_n = _split_counts(len(identifiers), proportions)
        # Train first is deliberate: every label in validation/test must exist in train.
        for identifier in identifiers[:train_n]:
            assignment[identifier] = "train"
        for identifier in identifiers[train_n:train_n + validation_n]:
            assignment[identifier] = "validation"
        for identifier in identifiers[train_n + validation_n:train_n + validation_n + test_n]:
            assignment[identifier] = "test"
    result = frame["group_id"].map(assignment)
    if result.isna().any():
        raise AssertionError("Every connected GBIF group must receive a split")
    return result


def _identity_leakage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in (*IDENTITY_COLUMNS, "dhash", "group_id"):
        if column not in frame:
            continue
        valid = frame.loc[frame[column].astype(str).str.strip().ne("")]
        for value, subset in valid.groupby(column, sort=False):
            splits = sorted(subset["split"].unique())
            if len(splits) > 1:
                rows.append({
                    "identity_type": column, "identity_value": value,
                    "splits": ",".join(splits), "images": len(subset),
                    "groups": subset["group_id"].nunique(),
                })
    return pd.DataFrame(rows, columns=(
        "identity_type", "identity_value", "splits", "images", "groups"
    ))


def _taxon_counts(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    valid = frame.loc[frame[task].ne("")]
    rows = []
    for label, group in valid.groupby(task, sort=True):
        row = {"task": task, "label": label}
        if task == "species":
            row["genus"] = group["genus"].iloc[0]
        for split in SPLITS:
            subset = group.loc[group["split"].eq(split)]
            row[f"{split}_images"] = len(subset)
            row[f"{split}_groups"] = subset["group_id"].nunique()
        row["total_images"] = len(group)
        row["total_groups"] = group["group_id"].nunique()
        row["trainable"] = row["train_groups"] > 0
        row["validation_evaluable"] = row["trainable"] and row["validation_groups"] > 0
        row["test_evaluable"] = row["trainable"] and row["test_groups"] > 0
        if not row["trainable"]:
            row["status_reason"] = "excluded_no_training_group"
        elif not row["test_evaluable"] and row["total_groups"] == 1:
            row["status_reason"] = "train_only_single_independent_group"
        elif not row["validation_evaluable"] and row["total_groups"] == 2:
            row["status_reason"] = "train_test_only_two_independent_groups"
        else:
            row["status_reason"] = "trainable_and_evaluable"
        rows.append(row)
    return pd.DataFrame(rows)


def _raw_species_audit(filtered: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw_labels = sorted(set(filtered["species_label"].astype(str)))
    for raw in raw_labels:
        subset = canonical.loc[canonical["raw_species"].eq(raw)]
        included = subset.loc[subset["included_in_split"]]
        statuses = sorted(set(subset["taxonomy_status"]))
        canonical_labels = sorted(set(included["species"]) - {""})
        exclusion_reasons = sorted(set(subset["row_exclusion_reason"]) - {""})
        rows.append({
            "raw_species": raw,
            "raw_genus_values": "|".join(sorted(set(subset["raw_genus"]) - {""})),
            "canonical_species": "|".join(canonical_labels),
            "taxonomy_status": "|".join(statuses),
            "images": len(subset), "groups": subset["group_id"].nunique(),
            "included_images": len(included),
            "excluded_images": int((~subset["included_in_split"]).sum()),
            "included_in_label_space": bool(canonical_labels),
            "exclusion_reason": (
                "" if canonical_labels else "|".join(exclusion_reasons or statuses)
            ),
        })
    return pd.DataFrame(rows)


def _prepare_petri(config: dict, destination: Path) -> dict:
    split_dir = Path(config["paths"]["petri_split_dir"])
    data_root = Path(config["paths"]["petri_data_root"])
    frames = []
    for split, filename in (("train", "train_split.csv"), ("validation", "val_split.csv"), ("test", "test_split.csv")):
        path = split_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        source = pd.read_csv(path, dtype=str, keep_default_na=False)
        rows = []
        for row in source.itertuples(index=False):
            genus, species, status = canonical_taxonomy(row.genus, row.species_label)
            rows.append({
                "sample_id": "petri:" + hashlib.sha256(str(row.rel_path_seg).encode()).hexdigest()[:24],
                "group_id": "petri:" + str(row.barcode),
                "image_path": str(data_root / row.rel_path_seg),
                "genus": genus, "species": species if status == "valid" else "",
                "age": _clean(row.life_stage), "split": split,
            })
        frame = pd.DataFrame(rows)
        frame.to_csv(destination / f"petri_{split}.csv", index=False)
        frames.append(frame)
    train = frames[0]
    maps = {
        task: {label: index for index, label in enumerate(sorted(train[column].loc[train[column].ne("")].unique()))}
        for task, column in PETRI_TASKS.items()
    }
    if any(not mapping for mapping in maps.values()):
        raise ValueError("Petri training label space is empty")
    atomic_json(destination / "petri_label_maps.json", maps)
    return {task: len(mapping) for task, mapping in maps.items()}


def run_full_taxonomy_audit(config: dict, config_path: Path) -> dict:
    root = Path(config["paths"]["experiment_root"])
    audit = root / "audit"
    prepared = root / "prepared"
    immutable = root / "immutable_inputs"
    marker = audit / "audit_manifest.json"
    source_manifest = Path(config["paths"]["gbif_manifest"])
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    source_hash = file_sha256(source_manifest)
    config_hash = file_sha256(config_path)
    commit = git_commit(config["paths"]["project_root"])
    if marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        identity = (source_hash, config_hash, commit)
        recorded = (
            existing.get("source_manifest_sha256"), existing.get("config_sha256"),
            existing.get("git_commit"),
        )
        if identity != recorded:
            raise RuntimeError(
                "Immutable full-taxonomy audit already exists with a different source, "
                "config, or git commit; choose a new experiment_id/root"
            )
        return {**existing, "status": "reused_immutable"}
    for directory in (audit, prepared, immutable):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_manifest, immutable / "gbif_manifest.csv")
    shutil.copy2(config_path, immutable / "gbif_full_taxonomy.yaml")
    raw = pd.read_csv(source_manifest, dtype=str, keep_default_na=False)
    required = {
        "image_id", "gbif_id", "occurrence_id", "local_path", "genus",
        "species_label", "download_status", "curation_label", "sha256", "dhash",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"GBIF manifest lacks required columns: {sorted(missing)}")
    labels = set(config["data"]["curation_labels"])
    filtered = raw.loc[
        raw["download_status"].eq("downloaded") & raw["curation_label"].isin(labels)
    ].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No downloaded curated GBIF rows remain")
    filtered["image_path"] = filtered["local_path"].map(
        lambda value: str(resolve_manifest_image_path(source_manifest, value))
    )
    missing_files = filtered.loc[~filtered["image_path"].map(lambda value: Path(value).is_file())]
    if not missing_files.empty:
        missing_files[["image_id", "local_path"]].to_csv(audit / "missing_source_files.csv", index=False)
        raise FileNotFoundError(f"{len(missing_files)} required GBIF source images are missing")
    filtered["group_id"] = _build_connected_groups(
        filtered, bool(config["data"]["connect_identical_perceptual_hashes"])
    )
    canonical_rows = []
    for row in filtered.itertuples(index=False):
        genus, species, status = canonical_taxonomy(row.genus, row.species_label)
        item = row._asdict()
        item.update(raw_genus=row.genus, raw_species=row.species_label, genus=genus, species=species, taxonomy_status=status)
        canonical_rows.append(item)
    canonical_all = pd.DataFrame(canonical_rows)
    conflict_rows = []
    for group_id, group in canonical_all.groupby("group_id", sort=True):
        genera = sorted(set(group["genus"]) - {""})
        species = sorted(set(group["species"]) - {""})
        if len(genera) > 1 or len(species) > 1:
            conflict_rows.append({
                "group_id": group_id, "genera": "|".join(genera),
                "species": "|".join(species), "images": len(group),
                "resolution": "quarantine_entire_connected_group",
                "reason": "contradictory_taxonomy_for_duplicate_or_occurrence_component",
            })
    conflicts = pd.DataFrame(conflict_rows, columns=(
        "group_id", "genera", "species", "images", "resolution", "reason",
    ))
    conflicts.to_csv(audit / "connected_group_taxonomy_conflicts.csv", index=False)
    conflict_ids = set(conflicts["group_id"]) if not conflicts.empty else set()
    canonical_all["included_in_split"] = ~canonical_all["group_id"].isin(conflict_ids)
    canonical_all["row_exclusion_reason"] = canonical_all["included_in_split"].map({
        True: "",
        False: "contradictory_taxonomy_for_duplicate_or_occurrence_component",
    })
    canonical_all.to_csv(audit / "canonical_gbif_rows_all_with_exclusions.csv", index=False)
    canonical = canonical_all.loc[canonical_all["included_in_split"]].copy()
    if canonical.empty:
        raise ValueError("All GBIF rows were excluded while resolving taxonomy conflicts")
    canonical["split"] = coverage_aware_split(canonical, config)
    leakage = _identity_leakage(canonical)
    leakage.to_csv(audit / "cross_split_identity_leakage.csv", index=False)
    if not leakage.empty:
        raise ValueError(f"Fatal unresolved cross-split identity leakage: {len(leakage)} rows")
    columns = [
        "image_id", "gbif_id", "occurrence_id", "group_id", "image_path",
        "raw_genus", "raw_species", "genus", "species", "taxonomy_status",
        "sha256", "dhash", "source_url", "media_reference", "split",
    ]
    canonical[columns].to_csv(audit / "canonical_gbif_rows.csv", index=False)
    for split in SPLITS:
        canonical.loc[canonical["split"].eq(split), columns].to_csv(
            prepared / f"gbif_{split}.csv", index=False
        )
    genus_counts = _taxon_counts(canonical, "genus")
    species_counts = _taxon_counts(canonical, "species")
    taxon_counts = pd.concat([genus_counts, species_counts], ignore_index=True)
    taxon_counts.to_csv(audit / "taxon_counts_by_split.csv", index=False)
    raw_audit = _raw_species_audit(filtered, canonical_all)
    raw_audit.to_csv(audit / "raw_species_inclusion_exclusion_audit.csv", index=False)
    species_counts.to_csv(audit / "species_inclusion_evaluability_audit.csv", index=False)
    genus_counts.to_csv(audit / "genus_inclusion_evaluability_audit.csv", index=False)
    train = canonical.loc[canonical["split"].eq("train")]
    gbif_maps = {
        task: {label: index for index, label in enumerate(sorted(train[column].loc[train[column].ne("")].unique()))}
        for task, column in GBIF_TASKS.items()
    }
    if not gbif_maps["genus"] or not gbif_maps["species"]:
        raise ValueError("Empty full-GBIF genus/species label space")
    species_parent = (
        train.loc[train["species"].ne(""), ["species", "genus"]].drop_duplicates()
    )
    if species_parent.groupby("species")["genus"].nunique().gt(1).any():
        raise ValueError("A canonical species maps to more than one genus")
    missing_parents = set(species_parent["genus"]).difference(gbif_maps["genus"])
    if missing_parents:
        raise ValueError(f"Species parent genera absent from label map: {sorted(missing_parents)}")
    atomic_json(prepared / "gbif_label_maps.json", gbif_maps)
    species_parent.to_csv(prepared / "species_to_genus.csv", index=False)
    petri_counts = _prepare_petri(config, prepared)
    petri_train = pd.read_csv(prepared / "petri_train.csv", dtype=str, keep_default_na=False)
    petri_species = set(petri_train["species"]) - {""}
    crosswalk = species_counts[["label", "genus"]].rename(columns={"label": "gbif_species"})
    crosswalk["petri_species"] = crosswalk["gbif_species"].where(crosswalk["gbif_species"].isin(petri_species), "")
    crosswalk["petri_seen"] = crosswalk["petri_species"].ne("")
    crosswalk["mapping_rule"] = crosswalk["petri_seen"].map({True: "exact_canonical_binomial", False: "absent_from_petri_training"})
    crosswalk.to_csv(audit / "gbif_petri_species_crosswalk.csv", index=False)
    semantic = canonical.loc[canonical["split"].eq("test"), [
        "image_id", "gbif_id", "group_id", "image_path", "genus", "species",
    ]].copy()
    semantic["annotation_status"] = "unannotated"
    semantic["blinded_model_condition"] = ""
    for column in (
        "worm_size_fraction", "partial_specimen", "occlusion",
        "background_clutter", "multiple_organisms_or_objects", "hands",
        "rulers", "containers", "annotator_id", "annotation_confidence",
    ):
        semantic[column] = ""
    semantic.to_csv(audit / "semantic_messiness_annotation_manifest.csv", index=False)
    (audit / "semantic_messiness_codebook.md").write_text(
        "# Blinded semantic-messiness annotation codebook\n\n"
        "Do not view model predictions while annotating. `worm_size_fraction` is the "
        "estimated fraction of image area occupied by visible worm tissue (0-1). "
        "Use 0/1/uncertain for partial specimen, occlusion, multiple organisms or "
        "objects, hands, rulers, and containers. Score background clutter on 0-3 "
        "with written adjudication notes added before annotation begins. Blank means "
        "not annotated, never a negative label.\n",
        encoding="utf-8",
    )
    frozen_split_hash = file_sha256(audit / "canonical_gbif_rows.csv")
    label_hash = file_sha256(prepared / "gbif_label_maps.json")
    raw_nonempty = filtered["species_label"].astype(str).str.strip().ne("")
    raw_species_count = filtered.loc[raw_nonempty, "species_label"].nunique()
    canonical_valid_count = canonical_all.loc[
        canonical_all["species"].ne(""), "species"
    ].nunique()
    discrepancy = {
        "expected_raw_species_minimum": int(config["data"]["expected_raw_species_minimum"]),
        "frozen_manifest_all_raw_species": int(raw.loc[raw["species_label"].ne(""), "species_label"].nunique()),
        "curated_downloaded_raw_species": int(raw_species_count),
        "canonical_valid_species": int(canonical_valid_count),
        "resolved_explanation": (
            "The >200 expectation is not present in the frozen manifest under the "
            "configured downloaded/curated filter; the exact source hash and filter "
            "counts identify this as an input-inventory discrepancy, not silent class masking."
            if raw_species_count < int(config["data"]["expected_raw_species_minimum"])
            else "The frozen manifest meets the expected raw-species minimum."
        ),
    }
    atomic_json(audit / "species_count_discrepancy.json", discrepancy)
    split_summary = {
        split: {
            "images": int(canonical["split"].eq(split).sum()),
            "groups": int(canonical.loc[canonical["split"].eq(split), "group_id"].nunique()),
            "genera": int(canonical.loc[canonical["split"].eq(split) & canonical["genus"].ne(""), "genus"].nunique()),
            "species": int(canonical.loc[canonical["split"].eq(split) & canonical["species"].ne(""), "species"].nunique()),
        }
        for split in SPLITS
    }
    result = {
        "schema_version": 1, "status": "complete",
        "experiment_id": config["experiment_id"],
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": source_hash,
        "frozen_manifest_sha256": file_sha256(immutable / "gbif_manifest.csv"),
        "config_sha256": config_hash, "git_commit": commit,
        "final_split_sha256": frozen_split_hash, "gbif_label_maps_sha256": label_hash,
        "filtered_images": int(len(canonical)),
        "pre_quarantine_filtered_images": int(len(canonical_all)),
        "quarantined_conflicting_groups": int(len(conflicts)),
        "quarantined_conflicting_images": int((~canonical_all["included_in_split"]).sum()),
        "connected_groups": int(canonical["group_id"].nunique()),
        "raw_species_count": int(raw_species_count),
        "canonical_valid_species_count": int(canonical_valid_count),
        "trainable_species_count": len(gbif_maps["species"]),
        "validation_evaluable_species_count": int(species_counts["validation_evaluable"].sum()),
        "test_evaluable_species_count": int(species_counts["test_evaluable"].sum()),
        "trainable_genus_count": len(gbif_maps["genus"]),
        "validation_evaluable_genus_count": int(genus_counts["validation_evaluable"].sum()),
        "test_evaluable_genus_count": int(genus_counts["test_evaluable"].sum()),
        "split_summary": split_summary, "petri_label_counts": petri_counts,
        "fatal_leakage_rows": 0,
    }
    atomic_json(marker, result)
    (audit / "audit_report.md").write_text(
        "# Full-GBIF immutable dataset audit\n\n"
        f"Frozen manifest SHA-256: `{source_hash}`. Git commit: `{commit}`.\n\n"
        f"Raw species: {raw_species_count}; canonical valid: {canonical_valid_count}; "
        f"trainable: {len(gbif_maps['species'])}; test-evaluable: "
        f"{int(species_counts['test_evaluable'].sum())}.\n\n"
        f"Split summary: `{json.dumps(split_summary, sort_keys=True)}`.\n\n"
        "Connected occurrence, exact hash, perceptual hash, image/source ID and path "
        "components have zero cross-split leakage. Contradictory connected components "
        f"quarantined before splitting: {len(conflicts)} groups / "
        f"{int((~canonical_all['included_in_split']).sum())} images. See the CSV audits "
        "for every excluded row and excluded or non-evaluable taxon.\n",
        encoding="utf-8",
    )
    return result
