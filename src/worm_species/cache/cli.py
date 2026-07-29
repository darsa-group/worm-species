"""Command line interface for persistent image-cache maintenance."""

from __future__ import annotations

import argparse
import json
import sys

from .condition_variants import DEFAULT_TRANSFORMS
from .condition_variants import build_condition_cache
from .condition_variants import resolved_condition_cache_directory
from .condition_variants import verify_condition_cache
from .maintenance import CacheMaintenanceError
from .maintenance import build_persistent_cache, verify_persistent_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m worm_species.cache")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="build or reuse a persistent cache")
    build.add_argument("--config", default="config.yaml")
    build.add_argument("--data-root", required=True)
    build.add_argument("--metadata-csv", required=True)
    build.add_argument("--cache-dir", required=True)
    build.add_argument("--image-col", default="rel_path_seg")
    build.add_argument("--force", "--force-rebuild", action="store_true")
    build.add_argument("--json", action="store_true", dest="json_output")
    verify = subparsers.add_parser("verify", help="verify marker and manifest")
    verify.add_argument("--cache-dir", required=True)
    verify.add_argument("--json", action="store_true", dest="json_output")
    condition_build = subparsers.add_parser(
        "build-conditions",
        help="build one indexed deterministic condition cache",
    )
    condition_build.add_argument("--config", required=True)
    condition_build.add_argument("--data-root", required=True)
    condition_build.add_argument("--metadata-csv", required=True)
    condition_build.add_argument("--base-cache-dir", required=True)
    condition_build.add_argument("--condition-cache-dir", required=True)
    condition_build.add_argument("--condition-index", type=int, required=True)
    condition_build.add_argument(
        "--transforms",
        nargs="+",
        default=sorted(DEFAULT_TRANSFORMS),
    )
    condition_build.add_argument("--image-col", default="rel_path_seg")
    condition_build.add_argument("--num-workers", type=int, default=8)
    condition_build.add_argument(
        "--force", "--force-rebuild", action="store_true"
    )
    condition_build.add_argument(
        "--json", action="store_true", dest="json_output"
    )
    condition_verify = subparsers.add_parser(
        "verify-condition", help="verify one condition-cache directory"
    )
    condition_verify.add_argument("--cache-dir", required=True)
    condition_verify.add_argument(
        "--json", action="store_true", dest="json_output"
    )
    condition_path = subparsers.add_parser(
        "condition-path",
        help="print the condition-cache directory for one resolved run",
    )
    condition_path.add_argument("--config", required=True)
    condition_path.add_argument("--condition-cache-dir", required=True)
    condition_path.add_argument(
        "--if-cacheable",
        action="store_true",
        help="print nothing and succeed for an uncached condition",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_persistent_cache(
                args.config,
                data_root=args.data_root,
                metadata_csv=args.metadata_csv,
                cache_dir=args.cache_dir,
                image_col=args.image_col,
                force=args.force,
            )
        elif args.command == "verify":
            result = verify_persistent_cache(args.cache_dir)
        elif args.command == "build-conditions":
            result = build_condition_cache(
                args.config,
                data_root=args.data_root,
                metadata_csv=args.metadata_csv,
                base_cache_dir=args.base_cache_dir,
                condition_cache_root=args.condition_cache_dir,
                condition_index=args.condition_index,
                transforms_to_cache=args.transforms,
                image_col=args.image_col,
                num_workers=args.num_workers,
                force=args.force,
            )
        elif args.command == "verify-condition":
            result = verify_condition_cache(args.cache_dir)
        else:
            condition_directory = resolved_condition_cache_directory(
                args.config,
                args.condition_cache_dir,
                require_cacheable=not args.if_cacheable,
            )
            if condition_directory is not None:
                print(condition_directory)
            return 0
    except (OSError, ValueError) as exc:
        print(f"configuration/path error: {exc}", file=sys.stderr)
        return 2
    except CacheMaintenanceError as exc:
        print(f"cache error: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"{result.status}: {result.cache_dir}")
    return 0
