#!/usr/bin/env bash
# Keep the full publication tree on the SSD and a lightweight local mirror.

set -euo pipefail

local_root="/home/devd/worm-species/publication_30seed_result"
ssd_root="/mnt/extssd/Earthworms/worm-species/publication_30seed_result"
mode="migrate"
large_size="100M"

usage() {
    printf '%s\n' \
        "Usage: $0 [options]" \
        "" \
        "Options:" \
        "  --mode migrate         Copy local results to SSD, verify, then link (default)" \
        "  --mode refresh-links   Copy small SSD files locally and create/update links" \
        "  --local-root PATH      Lightweight local result tree" \
        "  --ssd-root PATH        Canonical full result tree on SSD" \
        "  --large-size SIZE      Link files at least this large (default: 100M)" \
        "  -h, --help             Show this help"
}

while (($#)); do
    case "$1" in
        --mode)
            mode=${2:?"--mode requires a value"}
            shift 2
            ;;
        --local-root)
            local_root=${2:?"--local-root requires a value"}
            shift 2
            ;;
        --ssd-root)
            ssd_root=${2:?"--ssd-root requires a value"}
            shift 2
            ;;
        --large-size)
            large_size=${2:?"--large-size requires a value"}
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

if [[ "$mode" != "migrate" && "$mode" != "refresh-links" ]]; then
    printf 'Error: --mode must be migrate or refresh-links.\n' >&2
    exit 2
fi
case "$local_root" in
    */publication_30seed_result) ;;
    *)
        printf 'Error: --local-root must end in publication_30seed_result.\n' >&2
        exit 2
        ;;
esac
case "$ssd_root" in
    */publication_30seed_result) ;;
    *)
        printf 'Error: --ssd-root must end in publication_30seed_result.\n' >&2
        exit 2
        ;;
esac
if [[ "$local_root" == "$ssd_root" ]]; then
    printf 'Error: local and SSD roots must be different.\n' >&2
    exit 2
fi
for command_name in rsync find cmp ln readlink; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Error: required command is unavailable: %s\n' "$command_name" >&2
        exit 127
    fi
done
if [[ ! -d "$local_root" && "$mode" == "migrate" ]]; then
    printf 'Error: local result directory does not exist: %s\n' "$local_root" >&2
    exit 1
fi

mkdir -p "$ssd_root" "$local_root"
if [[ ! -w "$ssd_root" ]]; then
    printf 'Error: SSD result directory is not writable: %s\n' "$ssd_root" >&2
    exit 1
fi

if [[ "$mode" == "migrate" ]]; then
    printf 'Copying local results to SSD (local files remain in place for verification)...\n'
    # Scientific result files are regular files. W&B creates thousands of
    # transient latest-run/log symlinks, many of which are already broken; skip
    # those links so they cannot block verification or pollute the SSD copy.
    rsync -aH --no-links --exclude='wandb/***' --partial --info=progress2 \
        "$local_root/" "$ssd_root/"

    printf 'Checksum-verifying every copied local file...\n'
    verification=$(mktemp "${TMPDIR:-/tmp}/worm-results-verify.XXXXXX")
    trap 'rm -f "$verification"' EXIT
    rsync -aHnci --no-links --exclude='wandb/***' \
        "$local_root/" "$ssd_root/" > "$verification"
    if [[ -s "$verification" ]]; then
        printf 'Error: SSD verification found differences; local files were not replaced.\n' >&2
        sed -n '1,40p' "$verification" >&2
        exit 1
    fi
    printf 'SSD copy verified.\n'
else
    printf 'Refreshing the lightweight local metadata tree from SSD...\n'
    rsync -aH \
        --max-size="$large_size" \
        --exclude='wandb/***' \
        --exclude='*.pt' \
        --exclude='*.pth' \
        --exclude='*.ckpt' \
        --exclude='*.bin' \
        --exclude='*.safetensors' \
        "$ssd_root/" "$local_root/"
fi

printf 'Creating local links for checkpoints and files >= %s...\n' "$large_size"
linked=0
while IFS= read -r -d '' ssd_file; do
    relative=${ssd_file#"$ssd_root"/}
    local_file="$local_root/$relative"
    mkdir -p "$(dirname "$local_file")"

    if [[ -L "$local_file" ]]; then
        current_target=$(readlink "$local_file")
        if [[ "$current_target" == "$ssd_file" ]]; then
            continue
        fi
        ln -sfn "$ssd_file" "$local_file"
        ((linked += 1))
        continue
    fi
    if [[ -e "$local_file" ]]; then
        if ! cmp -s "$local_file" "$ssd_file"; then
            printf 'Error: refusing to replace differing local file: %s\n' \
                "$local_file" >&2
            exit 1
        fi
        rm -f "$local_file"
    fi
    ln -s "$ssd_file" "$local_file"
    ((linked += 1))
done < <(
    find "$ssd_root" -type f \
        \( -size "+$large_size" \
        -o -name '*.pt' \
        -o -name '*.pth' \
        -o -name '*.ckpt' \
        -o -name '*.bin' \
        -o -name '*.safetensors' \) \
        -print0
)

printf 'Linked %d files. Canonical results: %s\n' "$linked" "$ssd_root"
printf 'Lightweight local view: %s\n' "$local_root"
