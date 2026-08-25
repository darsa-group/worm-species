#!/usr/bin/env bash
# Render or explicitly submit the Genome DINOv3 + UMAP/HDBSCAN job.

set -euo pipefail

mode="dry-run"
remote="devd@login.genome.au.dk"
project_root="/home/devd/worm-species/wormsource2"
bundle_root="/home/devd/worm-species/data/gbif_oligochaeta"
conda_env="wormspecies-gbif"

usage() {
    printf '%s\n' \
        "Usage: $0 [--mode dry-run|submit] [options]" \
        "" \
        "  --remote HOST         SSH destination (default: ${remote})" \
        "  --project-root PATH   Genome checkout (default: ${project_root})" \
        "  --bundle-root PATH    Transferred data bundle (default: ${bundle_root})" \
        "  --conda-env NAME      Genome conda environment (default: ${conda_env})" \
        "  -h, --help            Show this help"
}

while (($#)); do
    case "$1" in
        --mode) mode=${2:?"--mode requires a value"}; shift 2 ;;
        --remote) remote=${2:?"--remote requires a value"}; shift 2 ;;
        --project-root) project_root=${2:?"--project-root requires a value"}; shift 2 ;;
        --bundle-root) bundle_root=${2:?"--bundle-root requires a value"}; shift 2 ;;
        --conda-env) conda_env=${2:?"--conda-env requires a value"}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$mode" != "dry-run" && "$mode" != "submit" ]]; then
    printf 'Error: --mode must be dry-run or submit.\n' >&2
    exit 2
fi

export_list="ALL,GBIF_PROJECT_ROOT=$project_root,GBIF_BUNDLE_ROOT=$bundle_root,GBIF_CONDA_ENV=$conda_env"
remote_command="cd $(printf '%q' "$project_root") && mkdir -p logs && sbatch --export=$(printf '%q' "$export_list") slurm/gbif_earthworms_dinov3.sbatch"

if [[ "$mode" == "dry-run" ]]; then
    printf 'Dry-run only; no SSH connection or Slurm submission was made.\n'
    printf 'Remote: %s\nProject: %s\nBundle: %s\nConda env: %s\n' \
        "$remote" "$project_root" "$bundle_root" "$conda_env"
    printf 'Submission command:\n  ssh %q %q\n' "$remote" "$remote_command"
    printf 'Job stages: DINOv3 embeddings -> PCA -> UMAP -> HDBSCAN clusters.\n'
    exit 0
fi

ssh "$remote" "$remote_command"
