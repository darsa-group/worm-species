"""Check the exact source inventory and persistent links before tagging a release."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(require_links=False, require_committed=False):
    errors = []
    manifest = ROOT / "SOURCE_MANIFEST.csv"
    if not manifest.is_file():
        return ["Build the source archive to create SOURCE_MANIFEST.csv first"]
    rows = list(csv.DictReader(manifest.open()))
    expected = set((ROOT / "configs/release_source_paths.txt").read_text().splitlines())
    if {r["path"] for r in rows} != expected:
        errors.append("Source manifest and release file list differ")
    for row in rows:
        path = ROOT / row["path"]
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
        ):
            errors.append(f'Rebuild the source manifest after editing {row["path"]}')
    links = json.loads((ROOT / "configs/release_inputs.json").read_text())
    if require_links:
        for name in ("dataset_doi", "model_doi"):
            if not re.fullmatch(r"10\.\d{4,9}/\S+", links.get(name) or ""):
                errors.append(
                    f"Enter the reserved {name} in configs/release_inputs.json"
                )
        repository = links.get("github_repository_url", "")
        if (
            not repository.startswith("https://github.com/")
            or links.get("github_release_url") != repository + "/releases/tag/v1.0.1"
        ):
            errors.append("Confirm the versioned GitHub release URL for v1.0.1")
    if require_committed:
        paths = (ROOT / "configs/release_commit_paths.txt").read_text().splitlines()
        for name in paths:
            result = subprocess.run(
                ["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True
            )
            path = ROOT / name
            if (
                result.returncode
                or not path.is_file()
                or result.stdout != path.read_bytes()
            ):
                errors.append(f"Release file has not been committed at HEAD: {name}")
        for name in ("SOURCE_MANIFEST.csv", "SHA256SUMS"):
            result = subprocess.run(
                ["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True
            )
            if result.returncode or result.stdout != (ROOT / name).read_bytes():
                errors.append(f"Commit the current {name}")
    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-links", action="store_true")
    parser.add_argument("--require-committed", action="store_true")
    args = parser.parse_args()
    errors = check(args.require_links, args.require_committed)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    print("Release source checks passed")
