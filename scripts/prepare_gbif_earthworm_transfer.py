#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from worm_species.gbif.transfer import prepare_transfer_bundle
from worm_species.gbif.transfer import validate_transfer_bundle
from worm_species.gbif.transfer import write_transfer_file_list
from worm_species.gbif.pipeline import load_pipeline_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or prepare the local GBIF earthworm transfer bundle."
    )
    parser.add_argument("--bundle-root", default="gbif_oligochaeta")
    parser.add_argument("--config", default="configs/gbif_oligochaeta.yaml")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Write transfer/SHA256SUMS and ready.json after validation.",
    )
    parser.add_argument(
        "--verify-image-hashes",
        action="store_true",
        help="Recompute each unique image SHA-256 (slower, used before transfer).",
    )
    parser.add_argument(
        "--write-file-list",
        action="store_true",
        help="Write an rsync file list without hashing image contents.",
    )
    args = parser.parse_args()
    try:
        config = load_pipeline_config(args.config)
        required_dataset_key = config["gbif"].get("dataset", {}).get("key")
        if args.prepare and args.write_file_list:
            raise ValueError("--prepare and --write-file-list are mutually exclusive")
        if args.write_file_list:
            result = write_transfer_file_list(
                args.bundle_root,
                required_dataset_key=required_dataset_key,
            )
        elif args.prepare:
            result = prepare_transfer_bundle(
                args.bundle_root,
                verify_image_hashes=False,
                show_progress=True,
            )
        else:
            result = validate_transfer_bundle(
                args.bundle_root,
                verify_image_hashes=False,
                required_dataset_key=required_dataset_key,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
