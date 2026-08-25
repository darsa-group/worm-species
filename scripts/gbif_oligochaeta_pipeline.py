#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

import pandas as pd

from worm_species.gbif.embedding import cluster_embeddings
from worm_species.gbif.embedding import embed_manifest
from worm_species.gbif.inference import infer_existing_checkpoint
from worm_species.gbif.pipeline import build_download_request
from worm_species.gbif.pipeline import build_media_manifest
from worm_species.gbif.pipeline import audit_taxonomic_scope
from worm_species.gbif.pipeline import download_dwca
from worm_species.gbif.pipeline import download_one_image
from worm_species.gbif.pipeline import enabled_orders
from worm_species.gbif.pipeline import get_download_metadata
from worm_species.gbif.pipeline import label_overlap_audit
from worm_species.gbif.pipeline import load_pipeline_config
from worm_species.gbif.pipeline import request_download
from worm_species.gbif.pipeline import prune_missing_image_rows
from worm_species.gbif.pipeline import filter_active_manifest_by_dataset


def _workspace_path(config: dict, key: str) -> Path:
    return Path(config["workspace"][key])


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_scope(config: dict, args: argparse.Namespace) -> None:
    payload = {
        "requested_name": config["gbif"]["requested_name"],
        "accepted_class_name": config["gbif"]["accepted_class_name"],
        "accepted_class_key": config["gbif"]["accepted_class_key"],
        "enabled_orders": enabled_orders(config),
        "excluded_orders": config["gbif"].get("explicitly_excluded_orders", []),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        _write_json(Path(args.output), payload)


def command_audit_scope(config: dict, args: argparse.Namespace) -> None:
    payload = audit_taxonomic_scope(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        _write_json(Path(args.output), payload)


def command_request(config: dict, args: argparse.Namespace) -> None:
    username = os.environ.get("GBIF_USERNAME", "")
    password = os.environ.get("GBIF_PASSWORD", "")
    email = os.environ.get("GBIF_EMAIL", "")
    payload = build_download_request(config, email=email)
    key = request_download(
        config, username=username, password=password, email=email
    )
    audit = Path(args.audit or "gbif_oligochaeta/download/request.json")
    _write_json(audit, {"download_key": key, "request": payload})
    print(key)


def command_status(config: dict, args: argparse.Namespace) -> None:
    metadata = get_download_metadata(config, args.download_key)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    if args.output:
        _write_json(Path(args.output), metadata)


def command_dwca(config: dict, args: argparse.Namespace) -> None:
    output = Path(args.output) if args.output else _workspace_path(config, "dwca_zip")
    metadata = download_dwca(config, args.download_key, output)
    _write_json(output.with_suffix(".metadata.json"), metadata)
    print(output)


def command_manifest(config: dict, args: argparse.Namespace) -> None:
    source = Path(args.dwca) if args.dwca else _workspace_path(config, "dwca_zip")
    output = Path(args.output) if args.output else _workspace_path(config, "manifest")
    dataset = config["gbif"].get("dataset", {})
    print(json.dumps(
        build_media_manifest(source, output, dataset_key=dataset.get("key")),
        indent=2,
        sort_keys=True,
    ))


def command_images(config: dict, args: argparse.Namespace) -> None:
    source = Path(args.manifest) if args.manifest else _workspace_path(config, "manifest")
    output = Path(args.output) if args.output else _workspace_path(config, "downloaded_manifest")
    image_dir = Path(args.image_dir) if args.image_dir else _workspace_path(config, "image_dir")
    settings = config["images"]
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    prior = pd.DataFrame()
    if output.is_file():
        prior = pd.read_csv(output, dtype=str, keep_default_na=False)
    prior_by_url = {
        row["source_url"]: row
        for row in prior.to_dict(orient="records")
        if row.get("download_status") == "downloaded"
        and Path(row.get("local_path", "")).is_file()
    }
    source_rows = frame.to_dict(orient="records")
    representatives_by_url: dict[str, dict] = {}
    for row in source_rows:
        representatives_by_url.setdefault(row["source_url"], row)
    pending = [
        row for url, row in representatives_by_url.items() if url not in prior_by_url
    ]
    workers = args.workers or int(settings.get("workers", 4))
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    print(
        f"media rows={len(source_rows):,}; unique URLs={len(representatives_by_url):,}; "
        f"reusing={len(prior_by_url):,}; pending={len(pending):,}; workers={workers}",
        flush=True,
    )

    def download(row: dict) -> dict:
        return download_one_image(
            row,
            image_dir,
            max_bytes=int(settings["max_bytes"]),
            connect_timeout=float(settings["connect_timeout_seconds"]),
            read_timeout=float(settings["read_timeout_seconds"]),
            user_agent=str(settings["user_agent"]),
            attempts=int(settings.get("attempts", 1)),
            retry_backoff_seconds=float(settings.get("retry_backoff_seconds", 1.0)),
        )

    completed_by_url: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for completed in executor.map(download, pending):
            completed_by_url[completed["source_url"]] = completed
            if len(completed_by_url) % 250 == 0:
                available = {**prior_by_url, **completed_by_url}
                partial_records = []
                for source_row in source_rows:
                    asset = available.get(source_row["source_url"])
                    if asset is not None:
                        partial_records.append({**source_row, **{
                            column: asset.get(column, "")
                            for column in (
                                "download_status", "local_path", "sha256", "dhash",
                                "width", "height", "content_type", "bytes", "error",
                            )
                        }})
                output.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(partial_records).to_csv(output, index=False)
                completed_successes = sum(
                    row.get("download_status") == "downloaded"
                    for row in completed_by_url.values()
                )
                print(
                    f"processed {len(completed_by_url):,}/{len(pending):,} pending URLs; "
                    f"downloaded={completed_successes:,}; "
                    f"failed={len(completed_by_url) - completed_successes:,}",
                    flush=True,
                )

    download_columns = [
        "download_status", "local_path", "sha256", "dhash", "width", "height",
        "content_type", "bytes", "error",
    ]
    records = []
    for source_row in source_rows:
        asset = (
            prior_by_url.get(source_row["source_url"])
            or completed_by_url[source_row["source_url"]]
        )
        records.append({
            **source_row,
            **{column: asset.get(column, "") for column in download_columns},
        })
    if not records:
        raise ValueError("The source image manifest is empty")
    if len(completed_by_url):
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(output, index=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    result["exact_duplicate_of"] = ""
    downloaded = result["download_status"] == "downloaded"
    first_for_hash: dict[str, str] = {}
    for index, row in result.loc[downloaded].iterrows():
        digest = row["sha256"]
        if digest in first_for_hash:
            result.at[index, "exact_duplicate_of"] = first_for_hash[digest]
        else:
            first_for_hash[digest] = row["image_id"]
    result.to_csv(output, index=False)
    if len(result) != len(frame) or result["image_id"].nunique() != len(frame):
        raise RuntimeError("Download result does not cover every manifest image row")
    per_occurrence = result.groupby("gbif_id")["download_status"].agg(
        expected_images="size",
        downloaded_images=lambda values: int((values == "downloaded").sum()),
        failed_images=lambda values: int((values != "downloaded").sum()),
    ).reset_index()
    incomplete = per_occurrence.loc[per_occurrence["failed_images"] > 0]
    incomplete_path = output.with_suffix(".incomplete_occurrences.csv")
    incomplete.to_csv(incomplete_path, index=False)
    status_counts = result["download_status"].value_counts(dropna=False)
    summary = {
        "manifest_image_rows": int(len(frame)),
        "result_image_rows": int(len(result)),
        "occurrences": int(result["gbif_id"].nunique()),
        "complete_occurrences": int(len(per_occurrence) - len(incomplete)),
        "incomplete_occurrences": int(len(incomplete)),
        "downloaded_images": int(status_counts.get("downloaded", 0)),
        "failed_images": int(len(result) - status_counts.get("downloaded", 0)),
        "exact_duplicate_images": int((result["exact_duplicate_of"] != "").sum()),
    }
    _write_json(output.with_suffix(".summary.json"), summary)
    print(result["download_status"].value_counts(dropna=False).to_string())
    if bool(settings.get("require_complete", True)) and not incomplete.empty:
        raise RuntimeError(
            f"{len(incomplete)} occurrences still have failed images; rerun the "
            f"resumable command after reviewing {incomplete_path}"
        )


def command_prune_missing_images(config: dict, args: argparse.Namespace) -> None:
    manifest = Path(args.manifest) if args.manifest else _workspace_path(
        config, "downloaded_manifest"
    )
    excluded = Path(args.excluded) if args.excluded else (
        manifest.parent / "excluded_missing_images.csv"
    )
    print(json.dumps(
        prune_missing_image_rows(
            manifest,
            excluded,
            apply=bool(args.apply),
        ),
        indent=2,
        sort_keys=True,
    ))


def command_filter_dataset(config: dict, args: argparse.Namespace) -> None:
    dataset = config["gbif"].get("dataset", {})
    if not dataset.get("key") or not dataset.get("name"):
        raise ValueError("gbif.dataset.key and gbif.dataset.name are required")
    manifest = Path(args.manifest) if args.manifest else _workspace_path(
        config, "downloaded_manifest"
    )
    excluded = Path(args.excluded) if args.excluded else (
        manifest.parent / "excluded_outside_dataset.csv"
    )
    print(json.dumps(
        filter_active_manifest_by_dataset(
            manifest,
            str(dataset["key"]),
            str(dataset["name"]),
            excluded,
            apply=bool(args.apply),
        ),
        indent=2,
        sort_keys=True,
    ))


def command_embed(config: dict, args: argparse.Namespace) -> None:
    settings = config["embedding"]
    print(json.dumps(embed_manifest(
        args.manifest or _workspace_path(config, "downloaded_manifest"),
        args.embeddings or _workspace_path(config, "embeddings"),
        args.index or _workspace_path(config, "embedding_index"),
        model_name=args.model or settings["model"],
        batch_size=args.batch_size or int(settings["batch_size"]),
        num_workers=args.workers if args.workers is not None else int(settings["num_workers"]),
        device_name=args.device or settings["device"],
        l2_normalize=bool(settings.get("l2_normalize", True)),
    ), indent=2, sort_keys=True))


def command_cluster(config: dict, args: argparse.Namespace) -> None:
    settings = config["clustering"]
    print(json.dumps(cluster_embeddings(
        args.embeddings or _workspace_path(config, "embeddings"),
        args.index or _workspace_path(config, "embedding_index"),
        args.output or _workspace_path(config, "clusters"),
        seed=int(settings["seed"]),
        pca_dimensions=int(settings["pca_dimensions"]),
        projection=str(settings["projection"]),
        min_cluster_size=int(settings["min_cluster_size"]),
        min_samples=int(settings["min_samples"]),
    ), indent=2, sort_keys=True))


def command_overlap(config: dict, args: argparse.Namespace) -> None:
    print(json.dumps(label_overlap_audit(
        args.manifest,
        args.label_map,
        args.output,
    ), indent=2, sort_keys=True))


def command_infer_existing(config: dict, args: argparse.Namespace) -> None:
    print(json.dumps(infer_existing_checkpoint(
        args.manifest,
        args.checkpoint,
        args.output,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device_name=args.device,
        curation_labels=tuple(args.curation_label),
    ), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gbif_oligochaeta.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scope = subparsers.add_parser("scope")
    scope.add_argument("--output")
    scope.set_defaults(function=command_scope)

    audit_scope = subparsers.add_parser("audit-scope")
    audit_scope.add_argument("--output")
    audit_scope.set_defaults(function=command_audit_scope)

    request = subparsers.add_parser("request-download")
    request.add_argument("--audit")
    request.set_defaults(function=command_request)

    status = subparsers.add_parser("download-status")
    status.add_argument("download_key")
    status.add_argument("--output")
    status.set_defaults(function=command_status)

    dwca = subparsers.add_parser("download-dwca")
    dwca.add_argument("download_key")
    dwca.add_argument("--output")
    dwca.set_defaults(function=command_dwca)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--dwca")
    manifest.add_argument("--output")
    manifest.set_defaults(function=command_manifest)

    images = subparsers.add_parser("download-images")
    images.add_argument("--manifest")
    images.add_argument("--output")
    images.add_argument("--image-dir")
    images.add_argument("--workers", type=int)
    images.set_defaults(function=command_images)

    prune = subparsers.add_parser(
        "prune-missing-images",
        help="Remove rows without usable files and retain them in an exclusion audit.",
    )
    prune.add_argument("--manifest")
    prune.add_argument("--excluded")
    prune.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite the active manifest; without this flag only report counts.",
    )
    prune.set_defaults(function=command_prune_missing_images)

    dataset_filter = subparsers.add_parser(
        "filter-dataset",
        help="Keep only the configured GBIF dataset in the active manifest.",
    )
    dataset_filter.add_argument("--manifest")
    dataset_filter.add_argument("--excluded")
    dataset_filter.add_argument("--apply", action="store_true")
    dataset_filter.set_defaults(function=command_filter_dataset)

    embed = subparsers.add_parser("embed")
    embed.add_argument("--manifest")
    embed.add_argument("--embeddings")
    embed.add_argument("--index")
    embed.add_argument("--model")
    embed.add_argument("--batch-size", type=int)
    embed.add_argument("--workers", type=int)
    embed.add_argument("--device")
    embed.set_defaults(function=command_embed)

    cluster = subparsers.add_parser("cluster")
    cluster.add_argument("--embeddings")
    cluster.add_argument("--index")
    cluster.add_argument("--output")
    cluster.set_defaults(function=command_cluster)

    overlap = subparsers.add_parser("audit-overlap")
    overlap.add_argument("--manifest", required=True)
    overlap.add_argument("--label-map", required=True)
    overlap.add_argument("--output")
    overlap.set_defaults(function=command_overlap)

    inference = subparsers.add_parser("infer-existing")
    inference.add_argument("--manifest", required=True)
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--batch-size", type=int, default=64)
    inference.add_argument("--workers", type=int, default=4)
    inference.add_argument("--device", default="auto")
    inference.add_argument("--curation-label", action="append", default=["keep"])
    inference.set_defaults(function=command_infer_existing)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_pipeline_config(args.config)
    args.function(config, args)


if __name__ == "__main__":
    main()
