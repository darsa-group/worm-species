#!/usr/bin/env bash
# Validate, transfer, and verify the complete local GBIF earthworm data bundle.

set -euo pipefail

mode="check"
bundle_root="gbif_oligochaeta"
remote="devd@login.genome.au.dk"
remote_path="/home/devd/worm-species/data/gbif_oligochaeta"
python_bin="${PYTHON:-/home/devd/miniconda3/envs/wormspecies/bin/python}"

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "" \
        "Options:" \
        "  --mode MODE          check, dry-run, transfer, verify, pull-results," \
        "                       or push-curation (default: check)" \
        "  --bundle-root PATH   Local bundle (default: ${bundle_root})" \
        "  --remote HOST        SSH destination (default: ${remote})" \
        "  --remote-path PATH   Genome data directory (default: ${remote_path})" \
        "  -h, --help           Show this help" \
        "" \
        "check and dry-run are local-only and never open an SSH connection."
}

while (($#)); do
    case "$1" in
        --mode)
            mode=${2:?"--mode requires a value"}
            shift 2
            ;;
        --bundle-root)
            bundle_root=${2:?"--bundle-root requires a value"}
            shift 2
            ;;
        --remote)
            remote=${2:?"--remote requires a value"}
            shift 2
            ;;
        --remote-path)
            remote_path=${2:?"--remote-path requires a value"}
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$mode" in
    check|dry-run|transfer|verify|pull-results|push-curation) ;;
    *)
        printf 'Error: unsupported --mode: %s\n' "$mode" >&2
        exit 2
        ;;
esac

quoted_remote_path=$(printf '%q' "$remote_path")

local_check() {
    PYTHONPATH=.:src "$python_bin" scripts/prepare_gbif_earthworm_transfer.py \
        --bundle-root "$bundle_root"
}

if [[ "$mode" == "check" ]]; then
    local_check
    exit 0
fi

if [[ "$mode" == "dry-run" ]]; then
    local_check
    printf '\nNo connection was opened. Once ready, transfer will run:\n'
    printf '  ssh %q mkdir -p %q\n' "$remote" "$remote_path"
    printf '  rsync -a --partial --info=progress2 --files-from=%q %q %q\n' \
        "${bundle_root%/}/transfer/FILES.txt" "${bundle_root%/}/" \
        "$remote:${remote_path%/}/"
    exit 0
fi

for command_name in ssh; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Error: required command is unavailable: %s\n' "$command_name" >&2
        exit 127
    fi
done

if ! command -v rsync >/dev/null 2>&1; then
    printf 'Error: required command is unavailable: rsync\n' >&2
    exit 127
fi

if [[ "$mode" == "verify" ]]; then
    PYTHONPATH=.:src "$python_bin" scripts/prepare_gbif_earthworm_transfer.py \
        --bundle-root "$bundle_root" --write-file-list
    transfer_file_list="$bundle_root/transfer/FILES.txt"
    comparison_file=$(mktemp "${TMPDIR:-/tmp}/gbif-rsync-verify.XXXXXX")
    trap 'rm -f "$comparison_file"' EXIT
    rsync -ani --protect-args --itemize-changes \
        --files-from="$transfer_file_list" \
        "${bundle_root%/}/" "$remote:${remote_path%/}/" > "$comparison_file"
    if [[ -s "$comparison_file" ]]; then
        printf 'Transfer differs from the local bundle:\n' >&2
        sed -n '1,100p' "$comparison_file" >&2
        exit 1
    fi
    printf 'Size and modification-time comparison passed (no content hashing).\n'
    exit 0
fi

if [[ "$mode" == "pull-results" ]]; then
    mkdir -p "$bundle_root/embeddings" "$bundle_root/curation"
    rsync -a --partial --protect-args \
        "$remote:${remote_path%/}/embeddings/" "$bundle_root/embeddings/"
    rsync -a --partial --protect-args \
        --include='clusters.csv' --include='clusters.summary.json' --exclude='*' \
        "$remote:${remote_path%/}/curation/" "$bundle_root/curation/"
    printf 'Pulled Genome embeddings and clusters into %s\n' "$bundle_root"
    exit 0
fi

if [[ "$mode" == "push-curation" ]]; then
    curated="$bundle_root/curation/curated_manifest.csv"
    if [[ ! -f "$curated" ]]; then
        printf 'Error: interactive export is missing: %s\n' "$curated" >&2
        exit 2
    fi
    ssh "$remote" "mkdir -p $quoted_remote_path/curation"
    curation_files=("$curated")
    if [[ -f "$bundle_root/curation/decisions.csv" ]]; then
        curation_files+=("$bundle_root/curation/decisions.csv")
    fi
    rsync -a --partial --protect-args "${curation_files[@]}" \
        "$remote:${remote_path%/}/curation/"
    printf 'Pushed the curated manifest and decisions to Genome.\n'
    exit 0
fi

# Build a list of active-dataset images plus provenance; no content hashing.
PYTHONPATH=.:src "$python_bin" scripts/prepare_gbif_earthworm_transfer.py \
    --bundle-root "$bundle_root" --write-file-list
transfer_file_list="$bundle_root/transfer/FILES.txt"

ssh "$remote" "mkdir -p $quoted_remote_path"
rsync \
    -a \
    --partial \
    --human-readable \
    --info=progress2 \
    --protect-args \
    --files-from="$transfer_file_list" \
    "${bundle_root%/}/" \
    "$remote:${remote_path%/}/"

printf 'Transfer complete without content hashing: %s:%s\n' \
    "$remote" "$remote_path"
