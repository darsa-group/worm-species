#!/bin/bash
#SBATCH --account=worm-species
#SBATCH --job-name=worm_build_cache
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=worm_build_cache_%j.out
#SBATCH --error=worm_build_cache_%j.err

set -euo pipefail

# ~/worm-species may itself be a symbolic link. Resolve it once so that all
# persistent paths refer to the real target rather than creating another link.
PROJECT_ENTRY="${PROJECT_ROOT:-${HOME}/worm-species}"

[[ -e "$PROJECT_ENTRY" ]] || {
    echo "ERROR: project path does not exist: $PROJECT_ENTRY" >&2
    exit 1
}

PROJECT_ROOT="$(readlink -f "$PROJECT_ENTRY")"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/source}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"

BASE_CONFIG="${BASE_CONFIG:-config.yaml}"
CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-wormspecies}"

METADATA_CSV="${METADATA_CSV:-${DATA_ROOT}/01_Segmented/global_metadata.csv}"
IMAGE_COL="${IMAGE_COL:-rel_path_seg}"

# Persistent cache directory. This is a normal directory, not another symlink.
CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/image_cache}"
READY_MARKER="${CACHE_DIR}/CACHE_READY"
LOCK_FILE="${CACHE_DIR}/CACHE_BUILD.lock"

# Set FORCE_REBUILD=1 when the images, metadata, preprocessing, or cache format
# have changed and the cache must be recreated.
FORCE_REBUILD="${FORCE_REBUILD:-0}"

echo "------------------------------------------------------------"
echo "Persistent image-cache build"
echo "Project entry:  $PROJECT_ENTRY"
echo "Resolved root:  $PROJECT_ROOT"
echo "Source root:    $SOURCE_ROOT"
echo "Data root:      $DATA_ROOT"
echo "Metadata:       $METADATA_CSV"
echo "Cache:          $CACHE_DIR"
echo "Host:           $(hostname)"
echo "------------------------------------------------------------"

[[ -d "$SOURCE_ROOT" ]] || {
    echo "ERROR: source directory not found: $SOURCE_ROOT" >&2
    exit 1
}

[[ -d "$DATA_ROOT" ]] || {
    echo "ERROR: data directory not found: $DATA_ROOT" >&2
    exit 1
}

[[ -f "${SOURCE_ROOT}/${BASE_CONFIG}" ]] || {
    echo "ERROR: config not found: ${SOURCE_ROOT}/${BASE_CONFIG}" >&2
    exit 1
}

[[ -f "$METADATA_CSV" ]] || {
    echo "ERROR: metadata file not found: $METADATA_CSV" >&2
    exit 1
}

[[ -f "$CONDA_SH" ]] || {
    echo "ERROR: conda initialisation script not found: $CONDA_SH" >&2
    exit 1
}

mkdir -p "$CACHE_DIR"

# Prevent concurrent jobs from building the same persistent cache.
exec 200>"$LOCK_FILE"
flock -x 200

if [[ "$FORCE_REBUILD" == "1" ]]; then
    echo "FORCE_REBUILD=1: removing the previous cache contents."

    find "$CACHE_DIR" -mindepth 1 -maxdepth 1 \
        ! -name "$(basename "$LOCK_FILE")" \
        -exec rm -rf {} +
fi

if [[ -f "$READY_MARKER" ]]; then
    echo "Cache is already complete:"
    echo "  $READY_MARKER"
    echo "Use FORCE_REBUILD=1 to rebuild it."
    exit 0
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$SOURCE_ROOT"

echo "Python: $(command -v python)"
python --version

python - \
    "${SOURCE_ROOT}/${BASE_CONFIG}" \
    "$DATA_ROOT" \
    "$METADATA_CSV" \
    "$IMAGE_COL" \
    "$CACHE_DIR" <<'PY'
from __future__ import annotations

import hashlib
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.cache import build_image_cache
from src.dataset_multitask import prepare_metadata
from src.utils import load_config
print(f"Python: {sys.executable}")
config_path = Path(sys.argv[1]).resolve()
data_root = Path(sys.argv[2]).resolve()
metadata_csv = Path(sys.argv[3]).resolve()
image_col = sys.argv[4]
cache_dir = Path(sys.argv[5]).resolve()

cfg = load_config(config_path)

cfg.setdefault("data", {})
cfg["data"]["root_dir"] = str(data_root)
cfg["data"]["metadata_csv"] = str(metadata_csv)
cfg["data"]["image_col"] = image_col

# Supply all cache-directory names used by the current project variants.
cfg.setdefault("cache", {})
cfg["cache"]["enabled"] = True
cfg["cache"]["cache_dir"] = str(cache_dir)
cfg["cache"]["dir"] = str(cache_dir)
cfg["cache"]["root_dir"] = str(cache_dir)
cfg["cache"]["root_dir_cache"] = str(cache_dir)

print("Preparing metadata...")
df = prepare_metadata(cfg)
print(f"Metadata rows: {len(df)}")

if len(df) == 0:
    raise RuntimeError("The prepared metadata contains no rows.")

print("Building persistent image cache...")
df_cached = build_image_cache(cfg, df)

if df_cached is None:
    raise RuntimeError("build_image_cache returned None.")

cache_column = "_cached_image_path"
if cache_column not in df_cached.columns:
    raise RuntimeError(
        f"Expected column {cache_column!r} was not produced. "
        f"Available columns: {list(df_cached.columns)}"
    )

n_total = len(df_cached)
n_cached = int(df_cached[cache_column].notna().sum())

print(f"Cached rows: {n_cached}/{n_total}")

if n_cached != n_total:
    raise RuntimeError(
        f"Cache is incomplete: {n_total - n_cached} of {n_total} rows "
        "have no cached image path."
    )

manifest_path = cache_dir / "cache_manifest.txt"
manifest_path.write_text(
    "\n".join(
        [
            f"created_utc={datetime.now(timezone.utc).isoformat()}",
            f"host={socket.gethostname()}",
            f"config={config_path}",
            f"config_sha256={hashlib.sha256(config_path.read_bytes()).hexdigest()}",
            f"data_root={data_root}",
            f"metadata_csv={metadata_csv}",
            f"image_col={image_col}",
            f"rows={n_total}",
            f"cached_rows={n_cached}",
        ]
    )
    + "\n"
)

print(f"Manifest: {manifest_path}")
PY

touch "$READY_MARKER"

echo "------------------------------------------------------------"
echo "Persistent cache is ready."
echo "Cache directory: $CACHE_DIR"
du -sh "$CACHE_DIR"
echo "------------------------------------------------------------"
