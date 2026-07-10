#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1

set -euo pipefail

: "${PROJECT_SRC:?}"
: "${DATA_SRC:?}"
: "${SCRATCH_ROOT:?}"
: "${SCRATCH_PROJECT:?}"
: "${SCRATCH_DATA:?}"
: "${SCRATCH_OUTPUTS:?}"

echo "------------------------------------------------------------"
echo "NODE-LOCAL SETUP"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "SCRATCH_ROOT: $SCRATCH_ROOT"
echo "------------------------------------------------------------"

rm -rf "$SCRATCH_ROOT"
mkdir -p "$SCRATCH_PROJECT" "$SCRATCH_DATA" "$SCRATCH_OUTPUTS"

echo "Copying project to node-local scratch..."
rsync -a \
    --exclude ".git" \
    --exclude "__pycache__" \
    --exclude ".ipynb_checkpoints" \
    --exclude "outputs" \
    --exclude "outputs_slurm" \
    "${PROJECT_SRC}/" "${SCRATCH_PROJECT}/"

echo "Copying only global_metadata.csv and *_seg.jpg files..."

mkdir -p "$SCRATCH_DATA/01_Segmented"

rsync -a --info=progress2 \
    --include='*/' \
    --include='global_metadata.csv' \
    --include='*_seg.jpg' \
    --exclude='*' \
    "${DATA_SRC}/01_Segmented/" \
    "$SCRATCH_DATA/01_Segmented/"

touch "${SCRATCH_ROOT}/READY"


echo "Hardware Info on node $(hostname):"
lscpu


echo "Setup complete on node $(hostname)."
echo "Scratch ready: $SCRATCH_ROOT"
