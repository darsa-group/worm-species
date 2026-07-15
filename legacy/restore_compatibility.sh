#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SOURCE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
MANIFEST="${SCRIPT_DIR}/compatibility.map"
TARGET_ROOT="${SOURCE_ROOT}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: legacy/restore_compatibility.sh [--root PATH] [--dry-run]

Restore archived compatibility paths without overwriting existing files.
The target must be a repository root containing the legacy archive when
relative SLURM and configuration links are expected to resolve.
EOF
}

while (($#)); do
    case "$1" in
        --root)
            [[ $# -ge 2 ]] || { echo "ERROR: --root requires a path" >&2; exit 2; }
            TARGET_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest not found: $MANIFEST" >&2; exit 1; }
[[ -d "$TARGET_ROOT" ]] || { echo "ERROR: target root is not a directory: $TARGET_ROOT" >&2; exit 1; }
TARGET_ROOT="$(cd -- "$TARGET_ROOT" && pwd -P)"

conflicts=0

record_conflict() {
    echo "CONFLICT: $1" >&2
    conflicts=$((conflicts + 1))
}

validate_relative_path() {
    local value="$1"
    case "$value" in
        ""|/*|..|../*|*/../*|*/..)
            return 1
            ;;
    esac
}

destination_matches() {
    local destination="$1" kind="$2" source="$3" link_target="$4" mode="$5"
    if [[ "$kind" == "copy" ]]; then
        [[ -f "$destination" && ! -L "$destination" ]] || return 1
        cmp -s -- "$source" "$destination" || return 1
        [[ "$(stat -c '%a' -- "$destination")" == "$mode" ]]
    else
        [[ -L "$destination" ]] || return 1
        [[ "$(readlink -- "$destination")" == "$link_target" ]]
    fi
}

# Validate every source, destination, and parent before changing the target.
while IFS=$'\t' read -r active archive kind link_target mode expected_hash replacement; do
    [[ "$active" == "active_path" ]] && continue
    [[ -n "$active" ]] || continue

    if ! validate_relative_path "$active" || ! validate_relative_path "$archive"; then
        record_conflict "unsafe manifest path: $active -> $archive"
        continue
    fi
    if [[ "$kind" != "copy" && "$kind" != "symlink" ]]; then
        record_conflict "unsupported restore kind '$kind' for $active"
        continue
    fi

    source="${SOURCE_ROOT}/${archive}"
    destination="${TARGET_ROOT}/${active}"
    if [[ ! -f "$source" ]]; then
        record_conflict "archive source is missing: $archive"
        continue
    fi
    actual_hash="$(sha256sum -- "$source")"
    actual_hash="${actual_hash%% *}"
    if [[ "$actual_hash" != "$expected_hash" ]]; then
        record_conflict "archive hash mismatch: $archive"
        continue
    fi

    parent="$(dirname -- "$destination")"
    probe="$parent"
    while [[ "$probe" != "$TARGET_ROOT" && "$probe" != "/" ]]; do
        if [[ -L "$probe" ]]; then
            record_conflict "destination parent is a symlink: $probe"
            break
        fi
        if [[ -e "$probe" && ! -d "$probe" ]]; then
            record_conflict "destination parent is not a directory: $probe"
            break
        fi
        probe="$(dirname -- "$probe")"
    done

    if [[ -e "$destination" || -L "$destination" ]]; then
        if ! destination_matches "$destination" "$kind" "$source" "$link_target" "$mode"; then
            record_conflict "refusing to overwrite: $active"
        fi
    fi
done < "$MANIFEST"

if ((conflicts)); then
    echo "ERROR: $conflicts conflict(s); no compatibility paths were restored." >&2
    exit 1
fi

while IFS=$'\t' read -r active archive kind link_target mode expected_hash replacement; do
    [[ "$active" == "active_path" ]] && continue
    [[ -n "$active" ]] || continue
    source="${SOURCE_ROOT}/${archive}"
    destination="${TARGET_ROOT}/${active}"

    if [[ -e "$destination" || -L "$destination" ]]; then
        echo "ALREADY RESTORED: $active"
        continue
    fi
    if ((DRY_RUN)); then
        echo "WOULD RESTORE: $active"
        continue
    fi

    mkdir -p -- "$(dirname -- "$destination")"
    if [[ "$kind" == "copy" ]]; then
        install -m "$mode" -- "$source" "$destination"
    else
        ln -s -- "$link_target" "$destination"
    fi
    echo "RESTORED: $active"
done < "$MANIFEST"
