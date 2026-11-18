"""Build a source archive without datasets, checkpoints or generated outputs."""

import argparse
import csv
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ROOT / "configs/release_source_paths.txt"


def sources():
    paths = []
    for name in SOURCE_PATHS.read_text().splitlines():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Source paths must stay within the repository")
        if path.parts[0] in (
            "data",
            "work",
            "outputs",
            "zenodo_release_2026-11-18",
            "manuscript",
            "slurm",
        ):
            raise ValueError(
                f"Generated or private directory in public archive: {name}"
            )
        if not (ROOT / path).is_file():
            raise FileNotFoundError(path)
        paths.append(ROOT / path)
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate source paths")
    return sorted(paths)


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh source archive path")
    paths = sources()
    inventory = []
    for path in paths:
        contents = path.read_bytes()
        inventory.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    manifest = ROOT / "SOURCE_MANIFEST.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, ["path", "bytes", "sha256"], lineterminator="\n")
        w.writeheader()
        w.writerows(inventory)
    paths.append(manifest)
    checksum = ROOT / "SHA256SUMS"
    checksum.write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT)}\n"
            for p in paths
        )
    )
    paths.append(checksum)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as z:
        for path in paths:
            info = zipfile.ZipInfo(
                "earthworm-image-classification/" + str(path.relative_to(ROOT)),
                date_time=(2026, 11, 18, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, path.read_bytes(), compresslevel=9)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError("Source archive CRC verification failed")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n"
    )
    return {
        "archive": str(output),
        "files": len(paths),
        "bytes": output.stat().st_size,
        "sha256": digest,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.output))
