from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import yaml
from PIL import Image


GBIF_API_BASE = "https://api.gbif.org/v1"
STILL_IMAGE_TOKENS = {
    "stillimage",
    "image",
    "http://purl.org/dc/dcmitype/stillimage",
}


def load_pipeline_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("GBIF pipeline config must be a schema-version 1 mapping")
    return config


def enabled_orders(config: dict) -> list[dict]:
    gbif = config["gbif"]
    orders = list(gbif.get("included_orders", []))
    orders.extend(
        order
        for order in gbif.get("optional_orders", [])
        if bool(order.get("enabled", False))
    )
    if not orders:
        raise ValueError("At least one GBIF order must be enabled")
    keys = [int(order["key"]) for order in orders]
    if len(keys) != len(set(keys)):
        raise ValueError("Enabled GBIF order keys must be unique")
    return orders


def build_download_request(config: dict, *, email: str) -> dict:
    if not email or "@" not in email:
        raise ValueError("A GBIF notification email is required")
    download = config["gbif"].get("download", {})
    predicates: list[dict] = [
        {
            "type": "in",
            "key": "ORDER_KEY",
            "values": [str(order["key"]) for order in enabled_orders(config)],
        }
    ]
    if download.get("require_still_image", True):
        predicates.append(
            {"type": "equals", "key": "MEDIA_TYPE", "value": "StillImage"}
        )
    if download.get("require_genus_key", True):
        predicates.append({"type": "isNotNull", "parameter": "GENUS_KEY"})
    dataset = config["gbif"].get("dataset")
    if dataset and dataset.get("key"):
        predicates.append(
            {"type": "equals", "key": "DATASET_KEY", "value": str(dataset["key"])}
        )
    return {
        "notificationAddresses": [email],
        "sendNotification": bool(download.get("send_notification", True)),
        "format": str(download.get("format", "DWCA")),
        "predicate": {"type": "and", "predicates": predicates},
    }


def audit_taxonomic_scope(config: dict) -> dict:
    """Resolve configured keys and current still-image counts using GBIF."""
    api_base = config["gbif"].get("api_base", GBIF_API_BASE).rstrip("/")
    configured = []
    for category, orders in (
        ("included", config["gbif"].get("included_orders", [])),
        ("excluded", config["gbif"].get("explicitly_excluded_orders", [])),
    ):
        for order in orders:
            taxonomy_response = requests.get(
                f"{api_base}/species/{int(order['key'])}", timeout=(15, 90)
            )
            taxonomy_response.raise_for_status()
            taxonomy = taxonomy_response.json()
            if (
                taxonomy.get("rank") != "ORDER"
                or taxonomy.get("canonicalName") != order["name"]
                or int(taxonomy.get("classKey", -1))
                != int(config["gbif"]["accepted_class_key"])
            ):
                raise RuntimeError(
                    f"GBIF key {order['key']} resolved to "
                    f"{taxonomy.get('rank')} {taxonomy.get('canonicalName')!r} "
                    f"under class {taxonomy.get('classKey')}, expected ORDER "
                    f"{order['name']!r} under class "
                    f"{config['gbif']['accepted_class_key']}"
                )
            count_response = requests.get(
                f"{api_base}/occurrence/search",
                params={
                    "order_key": int(order["key"]),
                    "media_type": "StillImage",
                    **(
                        {"dataset_key": config["gbif"]["dataset"]["key"]}
                        if config["gbif"].get("dataset", {}).get("key")
                        else {}
                    ),
                    "limit": 0,
                },
                timeout=(15, 90),
            )
            count_response.raise_for_status()
            configured.append(
                {
                    "category": category,
                    "key": int(order["key"]),
                    "configured_name": order["name"],
                    "enabled": category == "included"
                    or bool(order.get("enabled", False)),
                    "resolved_name": taxonomy["canonicalName"],
                    "class_key": taxonomy.get("classKey"),
                    "still_image_occurrence_count_before_genus_filter": int(
                        count_response.json()["count"]
                    ),
                }
            )
    match_response = requests.get(
        f"{api_base}/species/match",
        params={
            "name": config["gbif"]["requested_name"],
            "rank": config["gbif"]["requested_rank"],
        },
        timeout=(15, 90),
    )
    match_response.raise_for_status()
    match = match_response.json()
    return {
        "requested_taxon_match": match,
        "configured_orders": configured,
        "enabled_count_before_genus_filter": sum(
            item["still_image_occurrence_count_before_genus_filter"]
            for item in configured
            if item["enabled"]
        ),
        "note": (
            "Counts are current occurrence-search counts before the DWCA "
            "GENUS_KEY non-null predicate and can change as GBIF is reindexed."
        ),
    }


def request_download(config: dict, *, username: str, password: str, email: str) -> str:
    if not username or not password:
        raise ValueError("GBIF username and password are required")
    api_base = config["gbif"].get("api_base", GBIF_API_BASE).rstrip("/")
    response = requests.post(
        f"{api_base}/occurrence/download/request",
        auth=(username, password),
        json=build_download_request(config, email=email),
        timeout=(15, 90),
    )
    response.raise_for_status()
    key = response.text.strip().strip('"')
    if not key:
        raise RuntimeError("GBIF returned an empty download key")
    return key


def get_download_metadata(config: dict, download_key: str) -> dict:
    api_base = config["gbif"].get("api_base", GBIF_API_BASE).rstrip("/")
    response = requests.get(
        f"{api_base}/occurrence/download/{download_key}", timeout=(15, 90)
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError("GBIF download metadata was not a JSON object")
    return payload


def download_dwca(config: dict, download_key: str, output_path: str | Path) -> dict:
    metadata = get_download_metadata(config, download_key)
    if metadata.get("status") != "SUCCEEDED":
        raise RuntimeError(
            f"GBIF download {download_key} is {metadata.get('status')}, not SUCCEEDED"
        )
    url = metadata.get("downloadLink")
    if not url:
        raise RuntimeError("Succeeded GBIF metadata has no downloadLink")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    with requests.get(url, stream=True, timeout=(15, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    os.replace(partial, output)
    return metadata


def _find_dwca_table(root: Path, candidates: Iterable[str]) -> Path:
    names = {candidate.lower() for candidate in candidates}
    matches = [path for path in root.rglob("*") if path.is_file() and path.name.lower() in names]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one DWCA table named {sorted(names)}, found {len(matches)}"
        )
    return matches[0]


def _read_dwca_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_MINIMAL,
        low_memory=False,
    )


def _first(row: pd.Series, *names: str) -> str:
    for name in names:
        if name in row.index:
            value = str(row[name]).strip()
            if value:
                return value
    return ""


def _is_still_image(row: pd.Series) -> bool:
    media_type = _first(row, "type", "Type").lower()
    media_format = _first(row, "format", "Format").lower()
    identifier = _first(row, "identifier", "accessURI", "references").lower()
    if media_type in STILL_IMAGE_TOKENS or media_format.startswith("image/"):
        return True
    return urlparse(identifier).path.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
    )


def _normalise_species(value: str) -> str:
    return "_".join(value.strip().replace("_", " ").split())


def build_media_manifest(
    dwca_zip: str | Path,
    output_path: str | Path,
    *,
    dataset_key: str | None = None,
) -> dict:
    source = Path(dwca_zip)
    output = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with tempfile.TemporaryDirectory(prefix="worm-gbif-dwca-") as temp_dir:
        root = Path(temp_dir)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(root)
        occurrence_path = _find_dwca_table(root, ("occurrence.txt",))
        media_path = _find_dwca_table(
            root, ("multimedia.txt", "media.txt", "images.txt")
        )
        occurrences = _read_dwca_table(occurrence_path)
        media = _read_dwca_table(media_path)

    occurrence_id_column = next(
        (name for name in ("gbifID", "id", "coreid") if name in occurrences.columns),
        None,
    )
    media_id_column = next(
        (name for name in ("coreid", "gbifID", "id") if name in media.columns),
        None,
    )
    if occurrence_id_column is None or media_id_column is None:
        raise ValueError("Could not identify occurrence/media join columns in DWCA")

    occurrences = occurrences.drop_duplicates(occurrence_id_column, keep="first")
    occurrence_rows = {
        str(row[occurrence_id_column]).strip(): row
        for _, row in occurrences.iterrows()
    }
    records: list[dict] = []
    skipped = Counter()
    for _, media_row in media.iterrows():
        core_id = str(media_row[media_id_column]).strip()
        occurrence = occurrence_rows.get(core_id)
        if occurrence is None:
            skipped["media_without_occurrence"] += 1
            continue
        if dataset_key and _first(occurrence, "datasetKey") != dataset_key:
            skipped["outside_dataset"] += 1
            continue
        genus = _first(occurrence, "genus")
        genus_key = _first(occurrence, "genusKey")
        if not genus or not genus_key:
            skipped["missing_genus"] += 1
            continue
        if not _is_still_image(media_row):
            skipped["not_still_image"] += 1
            continue
        source_url = _first(media_row, "identifier", "accessURI", "references")
        if not source_url.startswith(("http://", "https://")):
            skipped["invalid_media_url"] += 1
            continue
        image_id = hashlib.sha256(f"{core_id}\0{source_url}".encode()).hexdigest()[:24]
        species = _first(occurrence, "species")
        records.append(
            {
                "image_id": image_id,
                "gbif_id": core_id,
                "occurrence_id": _first(occurrence, "occurrenceID"),
                "dataset_key": _first(occurrence, "datasetKey"),
                "basis_of_record": _first(occurrence, "basisOfRecord"),
                "dataset_name": _first(occurrence, "datasetName"),
                "publisher": _first(occurrence, "publisher"),
                "publishing_org_key": _first(occurrence, "publishingOrgKey"),
                "institution_code": _first(occurrence, "institutionCode"),
                "collection_code": _first(occurrence, "collectionCode"),
                "country": _first(occurrence, "country"),
                "country_code": _first(occurrence, "countryCode"),
                "state_province": _first(occurrence, "stateProvince"),
                "locality": _first(occurrence, "locality"),
                "decimal_latitude": _first(occurrence, "decimalLatitude"),
                "decimal_longitude": _first(occurrence, "decimalLongitude"),
                "coordinate_uncertainty_m": _first(occurrence, "coordinateUncertaintyInMeters"),
                "event_date": _first(occurrence, "eventDate"),
                "year": _first(occurrence, "year"),
                "month": _first(occurrence, "month"),
                "recorded_by": _first(occurrence, "recordedBy"),
                "occurrence_issues": _first(occurrence, "issue"),
                "scientific_name": _first(occurrence, "scientificName"),
                "taxon_key": _first(occurrence, "taxonKey"),
                "taxon_rank": _first(occurrence, "taxonRank"),
                "order": _first(occurrence, "order"),
                "order_key": _first(occurrence, "orderKey"),
                "family": _first(occurrence, "family"),
                "family_key": _first(occurrence, "familyKey"),
                "genus": genus,
                "genus_key": genus_key,
                "species": species,
                "species_label": _normalise_species(species),
                "species_key": _first(occurrence, "speciesKey"),
                "identified_by": _first(occurrence, "identifiedBy"),
                "identification_qualifier": _first(occurrence, "identificationQualifier"),
                "type_status": _first(occurrence, "typeStatus"),
                "source_url": source_url,
                "media_type": _first(media_row, "type", "Type"),
                "media_format": _first(media_row, "format", "Format"),
                "media_title": _first(media_row, "title", "Title"),
                "media_description": _first(media_row, "description", "Description"),
                "creator": _first(media_row, "creator", "Creator"),
                "media_reference": _first(media_row, "references", "Reference"),
                "license": _first(media_row, "license", "License")
                or _first(occurrence, "license"),
                "rights_holder": _first(media_row, "rightsHolder")
                or _first(occurrence, "rightsHolder"),
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("DWCA produced no genus-defined still-image records")
    duplicate_media_rows = int(frame.duplicated("image_id").sum())
    frame = frame.drop_duplicates("image_id", keep="first")
    frame = frame.sort_values(["gbif_id", "image_id"], kind="stable").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, output)
    summary = {
        "occurrence_rows": int(len(occurrences)),
        "media_rows": int(len(media)),
        "manifest_rows": int(len(frame)),
        "unique_occurrences": int(frame["gbif_id"].nunique()),
        "unique_genera": int(frame["genus_key"].nunique()),
        "unique_species": int(frame.loc[frame["species_key"] != "", "species_key"].nunique()),
        "duplicate_media_rows": duplicate_media_rows,
        "skipped": dict(sorted(skipped.items())),
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _image_extension(content_type: str, source_url: str, image_format: str) -> str:
    formats = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tif"}
    if image_format.upper() in formats:
        return formats[image_format.upper()]
    content_extensions = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
        "image/tiff": ".tif",
    }
    if content_type.split(";", 1)[0].lower() in content_extensions:
        return content_extensions[content_type.split(";", 1)[0].lower()]
    suffix = Path(urlparse(source_url).path).suffix.lower()
    return suffix if suffix in formats.values() else ".img"


def _dhash(image: Image.Image, size: int = 8) -> str:
    grayscale = image.convert("L").resize((size + 1, size))
    pixels = np.asarray(grayscale, dtype=np.uint8)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = sum(int(bit) << index for index, bit in enumerate(bits.flat))
    return f"{value:0{size * size // 4}x}"


def download_one_image(
    row: dict,
    image_dir: str | Path,
    *,
    max_bytes: int,
    connect_timeout: float,
    read_timeout: float,
    user_agent: str,
    attempts: int = 1,
    retry_backoff_seconds: float = 1.0,
) -> dict:
    image_root = Path(image_dir)
    image_root.mkdir(parents=True, exist_ok=True)
    result = dict(row)
    result.update(
        download_status="failed",
        local_path="",
        sha256="",
        dhash="",
        width="",
        height="",
        content_type="",
        bytes="",
        error="",
    )
    temporary = image_root / f".{row['image_id']}.part"
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(
                row["source_url"],
                stream=True,
                timeout=(connect_timeout, read_timeout),
                headers={"User-Agent": user_agent},
            ) as response:
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    if attempt < attempts:
                        retry_after = response.headers.get("Retry-After", "")
                        delay = (
                            float(retry_after)
                            if retry_after.isdigit()
                            else retry_backoff_seconds * (2 ** (attempt - 1))
                        )
                        time.sleep(min(delay, 120.0))
                        continue
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                declared = int(response.headers.get("Content-Length", 0) or 0)
                if declared > max_bytes:
                    raise ValueError(
                        f"declared size {declared} exceeds max_bytes {max_bytes}"
                    )
                digest = hashlib.sha256()
                total = 0
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(f"download exceeds max_bytes {max_bytes}")
                        digest.update(chunk)
                        handle.write(chunk)
            with Image.open(temporary) as image:
                image.load()
                width, height = image.size
                image_format = image.format or ""
                perceptual_hash = _dhash(image)
            extension = _image_extension(content_type, row["source_url"], image_format)
            destination = image_root / f"{row['image_id']}{extension}"
            os.replace(temporary, destination)
            result.update(
                download_status="downloaded",
                local_path=str(destination),
                sha256=digest.hexdigest(),
                dhash=perceptual_hash,
                width=width,
                height=height,
                content_type=content_type,
                bytes=total,
                error="",
            )
            break
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            result["error"] = f"{type(exc).__name__}: {exc}"
            if attempt < attempts and isinstance(
                exc, (requests.ConnectionError, requests.Timeout)
            ):
                time.sleep(min(retry_backoff_seconds * (2 ** (attempt - 1)), 120.0))
                continue
            break
    return result


def label_overlap_audit(
    manifest_path: str | Path,
    label_map_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    label_maps = json.loads(Path(label_map_path).read_text())
    known_genera = set(label_maps.get("genus", {}))
    known_species = set(label_maps.get("species", {}))
    if not known_genera or not known_species:
        raise ValueError("Checkpoint label map must contain genus and species tasks")
    frame["checkpoint_genus_overlap"] = frame["genus"].isin(known_genera)
    frame["checkpoint_species_overlap"] = frame["species_label"].isin(known_species)
    frame["checkpoint_scope"] = np.select(
        [
            frame["checkpoint_species_overlap"],
            frame["checkpoint_genus_overlap"],
        ],
        ["known_species", "known_genus_only"],
        default="unknown_genus",
    )
    summary = {
        "rows": int(len(frame)),
        "known_genera": sorted(known_genera),
        "known_species": sorted(known_species),
        "scope_counts": {
            str(key): int(value)
            for key, value in frame["checkpoint_scope"].value_counts().sort_index().items()
        },
        "overlapping_gbif_genera": sorted(set(frame["genus"]) & known_genera),
        "overlapping_gbif_species": sorted(set(frame["species_label"]) & known_species),
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        output.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    return summary


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_image_path(
    manifest_path: str | Path, local_path: str | Path
) -> Path:
    """Resolve a saved image path after moving the complete GBIF bundle.

    Downloader manifests historically store paths such as
    ``gbif_oligochaeta/images/<id>.jpg``.  On Genome the bundle lives under a
    data directory rather than the repository checkout, so callers must not
    assume that their current working directory is the original local repo.
    """
    value = Path(local_path)
    if value.is_absolute():
        return value
    if value.is_file():
        return value.resolve()
    bundle_root = Path(manifest_path).resolve().parent.parent
    parts = value.parts
    if parts and parts[0] == bundle_root.name:
        value = Path(*parts[1:])
    return bundle_root / value


def prune_missing_image_rows(
    manifest_path: str | Path,
    excluded_path: str | Path,
    *,
    apply: bool = False,
) -> dict:
    """Remove unusable rows from the active manifest while retaining an audit."""
    manifest = Path(manifest_path)
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    required = {"image_id", "download_status", "local_path", "sha256"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Downloaded manifest is missing columns: {missing_columns}")

    paths_exist = frame["local_path"].map(
        lambda value: bool(value)
        and resolve_manifest_image_path(manifest, value).is_file()
    )
    usable = (
        frame["download_status"].eq("downloaded")
        & frame["local_path"].ne("")
        & frame["sha256"].ne("")
        & paths_exist
    )
    retained = frame.loc[usable].copy()
    excluded = frame.loc[~usable].copy()
    excluded["exclusion_reason"] = np.select(
        [
            excluded["download_status"].ne("downloaded"),
            excluded["local_path"].eq(""),
            ~paths_exist.loc[excluded.index],
            excluded["sha256"].eq(""),
        ],
        [
            "download_failed",
            "blank_local_path",
            "local_file_missing",
            "blank_sha256",
        ],
        default="unusable",
    )
    summary = {
        "apply": bool(apply),
        "input_rows": int(len(frame)),
        "retained_rows": int(len(retained)),
        "excluded_rows": int(len(excluded)),
        "retained_occurrences": int(retained["gbif_id"].nunique())
        if "gbif_id" in retained
        else None,
        "retained_unique_files": int(retained["local_path"].nunique()),
        "exclusion_reasons": {
            str(key): int(value)
            for key, value in excluded["exclusion_reason"].value_counts().items()
        },
        "active_manifest": str(manifest),
        "excluded_manifest": str(excluded_path),
    }
    if not apply:
        return summary
    if excluded.empty:
        raise RuntimeError("No unusable image rows were found; nothing was changed")

    excluded_output = Path(excluded_path)
    excluded_output.parent.mkdir(parents=True, exist_ok=True)
    excluded_temporary = excluded_output.with_suffix(excluded_output.suffix + ".tmp")
    excluded.to_csv(excluded_temporary, index=False)
    os.replace(excluded_temporary, excluded_output)

    manifest_temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    retained.to_csv(manifest_temporary, index=False)
    os.replace(manifest_temporary, manifest)
    summary_path = manifest.with_suffix(".prune_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def filter_active_manifest_by_dataset(
    manifest_path: str | Path,
    dataset_key: str,
    dataset_name: str,
    excluded_path: str | Path,
    *,
    apply: bool = False,
) -> dict:
    """Keep one GBIF dataset active and audit every removed downloaded row."""
    manifest = Path(manifest_path)
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    required = {"image_id", "gbif_id", "dataset_key", "local_path"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Downloaded manifest is missing columns: {missing_columns}")
    retained = frame.loc[frame["dataset_key"] == dataset_key].copy()
    excluded = frame.loc[frame["dataset_key"] != dataset_key].copy()
    if retained.empty:
        raise RuntimeError(
            f"No active rows match GBIF dataset {dataset_name!r} ({dataset_key})"
        )
    excluded["exclusion_reason"] = "outside_selected_dataset"
    excluded["selected_dataset_key"] = dataset_key
    summary = {
        "apply": bool(apply),
        "selected_dataset_key": dataset_key,
        "selected_dataset_name": dataset_name,
        "input_rows": int(len(frame)),
        "retained_rows": int(len(retained)),
        "excluded_rows": int(len(excluded)),
        "retained_occurrences": int(retained["gbif_id"].nunique()),
        "retained_unique_files": int(retained["local_path"].nunique()),
        "active_manifest": str(manifest),
        "excluded_manifest": str(excluded_path),
    }
    if not apply:
        return summary
    if excluded.empty:
        return summary

    excluded_output = Path(excluded_path)
    excluded_output.parent.mkdir(parents=True, exist_ok=True)
    excluded_temporary = excluded_output.with_suffix(excluded_output.suffix + ".tmp")
    excluded.to_csv(excluded_temporary, index=False)
    os.replace(excluded_temporary, excluded_output)

    manifest_temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    retained.to_csv(manifest_temporary, index=False)
    os.replace(manifest_temporary, manifest)
    summary_path = manifest.with_suffix(".dataset_filter_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary
