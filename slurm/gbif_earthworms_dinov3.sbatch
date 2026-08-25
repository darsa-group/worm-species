#!/usr/bin/env bash
#SBATCH --job-name=gbif-earthworm-dinov3
#SBATCH --account=worm-species
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60384M
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/gbif-earthworm-dinov3-%j.out
#SBATCH --error=logs/gbif-earthworm-dinov3-%j.err

set -euo pipefail

project_root="${GBIF_PROJECT_ROOT:-/home/devd/worm-species/wormsource2}"
bundle_root="${GBIF_BUNDLE_ROOT:-/home/devd/worm-species/data/gbif_oligochaeta}"
conda_sh="${GBIF_CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
conda_env="${GBIF_CONDA_ENV:-wormspecies-gbif}"

manifest="$bundle_root/manifests/downloaded_manifest.csv"
embeddings="$bundle_root/embeddings/dinov3_embeddings.npy"
index="$bundle_root/embeddings/dinov3_embedding_index.csv"
clusters="$bundle_root/curation/clusters.csv"

test -d "$project_root"
test -f "$manifest"
test -f "$conda_sh"

cd "$project_root"
source "$conda_sh"
conda activate "$conda_env"
mkdir -p "$bundle_root/embeddings" "$bundle_root/curation"

PYTHONPATH=.:src python scripts/prepare_gbif_earthworm_transfer.py \
    --bundle-root "$bundle_root"
python -c 'import timm, torch, umap; from sklearn.cluster import HDBSCAN; assert torch.cuda.is_available()'
nvidia-smi

PYTHONPATH=.:src python scripts/gbif_oligochaeta_pipeline.py embed \
    --manifest "$manifest" \
    --embeddings "$embeddings" \
    --index "$index" \
    --device cuda

PYTHONPATH=.:src python scripts/gbif_oligochaeta_pipeline.py cluster \
    --embeddings "$embeddings" \
    --index "$index" \
    --output "$clusters"

printf 'DINOv3 embeddings and UMAP/HDBSCAN clusters complete: %s\n' "$clusters"
