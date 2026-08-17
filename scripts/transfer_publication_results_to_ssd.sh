#!/usr/bin/env bash
# One-command local migration, remote download, and local link refresh.

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ssd_root="/mnt/extssd/Earthworms/worm-species/publication_30seed_result"
local_root="$project_root/publication_30seed_result"
workers="${WORKERS:-8}"

"$project_root/scripts/store_publication_results_on_ssd.sh" \
    --mode migrate \
    --local-root "$local_root" \
    --ssd-root "$ssd_root"

"$project_root/scripts/download_publication_results.sh" \
    --remote devd@login.genome.au.dk \
    --remote-path /home/devd/worm-species/source/publication_30seed_result \
    --destination "$ssd_root" \
    --workers "$workers"

"$project_root/scripts/store_publication_results_on_ssd.sh" \
    --mode refresh-links \
    --local-root "$local_root" \
    --ssd-root "$ssd_root"

printf 'SSD transfer and local link refresh completed.\n'
