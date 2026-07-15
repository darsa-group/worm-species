#!/usr/bin/env bash
set -Eeuo pipefail

# Copy a project to an external drive, then replace selected local files with
# symbolic links to the verified copies on that drive.
#
# The default is a dry run. Add --execute to make changes.
#
# Usage:
#   ./archive_project_to_drive.sh SOURCE_DIR DRIVE_PROJECT_DIR [--execute]
#
# Example:
#   ./archive_project_to_drive.sh \
#       "$HOME/worm-species" \
#       "/media/$USER/ResearchDrive/worm-species" \
#       --execute

usage() {
    printf 'Usage: %s SOURCE_DIR DRIVE_PROJECT_DIR [--execute]\n' "$0" >&2
    exit 2
}

[[ $# -eq 2 || $# -eq 3 ]] || usage

SOURCE_INPUT=$1
DEST_INPUT=$2
MODE=${3:---dry-run}
[[ "$MODE" == "--dry-run" || "$MODE" == "--execute" ]] || usage

command -v rsync >/dev/null || { echo "Error: rsync is not installed." >&2; exit 1; }
command -v realpath >/dev/null || { echo "Error: realpath is not installed." >&2; exit 1; }

[[ -d "$SOURCE_INPUT" ]] || { echo "Error: source directory does not exist: $SOURCE_INPUT" >&2; exit 1; }

SOURCE=$(realpath -e -- "$SOURCE_INPUT")
DEST_PARENT=$(dirname -- "$DEST_INPUT")
DEST_NAME=$(basename -- "$DEST_INPUT")
[[ -d "$DEST_PARENT" ]] || { echo "Error: destination parent is not mounted or does not exist: $DEST_PARENT" >&2; exit 1; }
DEST=$(realpath -e -- "$DEST_PARENT")/$DEST_NAME

[[ "$SOURCE" != "$DEST" ]] || { echo "Error: source and destination are identical." >&2; exit 1; }
case "$DEST/" in "$SOURCE/"*) echo "Error: destination cannot be inside the source." >&2; exit 1;; esac
case "$SOURCE/" in "$DEST/"*) echo "Error: source cannot be inside the destination." >&2; exit 1;; esac

# Edit these patterns to control which files are removed locally and symlinked.
# They are matched case-insensitively against filenames.
LINK_NAME_PATTERNS=(
    '*.pt'
    '*.pth'
    '*.ckpt'
    '*.onnx'
    '*.safetensors'
    '*.out'
    '*.err'
    '*.csv'
)

# Files in these directories are also symlinked, regardless of extension.
# Directory names are matched anywhere below SOURCE.
LINK_DIRECTORY_NAMES=(
    'checkpoints'
    'weights'
)

RSYNC_OPTIONS=(
    --archive
    --human-readable
    --info=progress2
    --partial
    --safe-links
)

if [[ "$MODE" == "--dry-run" ]]; then
    RSYNC_OPTIONS+=(--dry-run)
    echo "DRY RUN: no files will be copied, removed, or linked."
fi

echo "Source:      $SOURCE"
echo "Destination: $DEST"

if [[ "$MODE" == "--execute" ]]; then
    mkdir -p -- "$DEST"
fi

# Copy the complete project. The trailing slashes copy the contents of SOURCE.
rsync "${RSYNC_OPTIONS[@]}" -- "$SOURCE/" "$DEST/"

find_args=("$SOURCE" -type f '(')
first=true
for pattern in "${LINK_NAME_PATTERNS[@]}"; do
    $first || find_args+=(-o)
    find_args+=(-iname "$pattern")
    first=false
done
for directory in "${LINK_DIRECTORY_NAMES[@]}"; do
    $first || find_args+=(-o)
    find_args+=(-ipath "*/$directory/*")
    first=false
done
find_args+=(')' -print0)

selected=0
linked=0
while IFS= read -r -d '' source_file; do
    ((selected += 1))
    relative_path=${source_file#"$SOURCE/"}
    destination_file="$DEST/$relative_path"

    if [[ "$MODE" == "--dry-run" ]]; then
        printf 'Would replace with symlink: %s -> %s\n' "$source_file" "$destination_file"
        continue
    fi

    [[ -f "$destination_file" ]] || {
        echo "Error: copied file is missing; local file retained: $destination_file" >&2
        exit 1
    }

    # Byte-for-byte verification is required before deleting the local copy.
    if ! cmp -s -- "$source_file" "$destination_file"; then
        echo "Error: verification failed; local file retained: $source_file" >&2
        exit 1
    fi

    link_target=$(realpath --relative-to="$(dirname -- "$source_file")" -- "$destination_file")
    rm -- "$source_file"
    ln -s -- "$link_target" "$source_file"
    ((linked += 1))
done < <(find "${find_args[@]}")

if [[ "$MODE" == "--dry-run" ]]; then
    echo "Dry run complete. Selected $selected file(s). Re-run with --execute after reviewing the output."
else
    echo "Complete. Copied the project and replaced $linked verified local file(s) with symbolic links."
fi
