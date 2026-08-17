#!/usr/bin/env bash
# Download scientific publication results with several resumable rsync workers.
# W&B runtime directories are excluded; they are not publication artifacts and
# may contain conflicting latest-run symlinks.

set -euo pipefail

remote="devd@login.genome.au.dk"
remote_path="/home/devd/worm-species/source/publication_30seed_result"
destination="./publication_30seed_result"
workers=6

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "" \
        "Options:" \
        "  --remote HOST          SSH host (default: ${remote})" \
        "  --remote-path PATH     Remote result directory" \
        "  --destination PATH     Local directory (default: ${destination})" \
        "  --workers N            Concurrent rsync processes (default: ${workers})" \
        "  -h, --help             Show this help"
}

while (($#)); do
    case "$1" in
        --remote)
            remote=${2:?"--remote requires a value"}
            shift 2
            ;;
        --remote-path)
            remote_path=${2:?"--remote-path requires a value"}
            shift 2
            ;;
        --destination)
            destination=${2:?"--destination requires a value"}
            shift 2
            ;;
        --workers)
            workers=${2:?"--workers requires a value"}
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

if ! [[ "$workers" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Error: --workers must be a positive integer.\n' >&2
    exit 2
fi

for command_name in ssh rsync mktemp; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Error: required command is unavailable: %s\n' "$command_name" >&2
        exit 127
    fi
done

mkdir -p "$destination"
transfer_tmp=$(mktemp -d "${TMPDIR:-/tmp}/worm-results-transfer.XXXXXX")
control_path="$transfer_tmp/ssh-control"
ssh_options=(
    -o ControlMaster=auto
    -o ControlPersist=10m
    -o "ControlPath=$control_path"
)
rsync_ssh="ssh -o ControlMaster=auto -o ControlPersist=10m -o ControlPath=$control_path"

cleanup() {
    ssh "${ssh_options[@]}" -O exit "$remote" >/dev/null 2>&1 || true
    rm -rf "$transfer_tmp"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

all_files="$transfer_tmp/all-files"
quoted_remote_path=$(printf '%q' "$remote_path")
printf 'Reading remote file list from %s:%s ...\n' "$remote" "$remote_path"
ssh "${ssh_options[@]}" "$remote" \
    "cd $quoted_remote_path && find . -type d -name wandb -prune -o -type f -print0" \
    > "$all_files"

file_count=0
for ((index = 0; index < workers; index++)); do
    : > "$transfer_tmp/shard-$index"
done
while IFS= read -r -d '' relative_path; do
    relative_path=${relative_path#./}
    printf '%s\0' "$relative_path" \
        >> "$transfer_tmp/shard-$((file_count % workers))"
    ((file_count += 1))
done < "$all_files"

if ((file_count == 0)); then
    printf 'No files found under the remote result directory.\n'
    exit 0
fi

active_workers=$((file_count < workers ? file_count : workers))
printf 'Transferring %d files with %d workers into %s\n' \
    "$file_count" "$active_workers" "$destination"

pids=()
worker_numbers=()
for ((index = 0; index < active_workers; index++)); do
    printf 'Starting worker %d/%d\n' "$((index + 1))" "$active_workers"
    rsync \
        -avh \
        --partial \
        --human-readable \
        --info=progress2 \
        --exclude='wandb/***' \
        --from0 \
        --files-from="$transfer_tmp/shard-$index" \
        -e "$rsync_ssh" \
        "$remote:$remote_path/" \
        "$destination/" &
    pids+=("$!")
    worker_numbers+=("$((index + 1))")
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'Worker %d finished.\n' "${worker_numbers[$index]}"
    else
        printf 'Worker %d failed. Rerun the same command to resume.\n' \
            "${worker_numbers[$index]}" >&2
        failed=1
    fi
done

if ((failed)); then
    exit 1
fi

printf 'Transfer complete: %s\n' "$destination"
