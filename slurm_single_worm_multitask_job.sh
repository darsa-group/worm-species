#!/bin/bash
#--------------------------------------------------------------------------#
# Single independent 1-GPU SLURM job for one worm multi-task run
#
# This script is submitted by submit_worm_multitask_sweep_jobs.sh.
# It should live in the project root.
#--------------------------------------------------------------------------#

#SBATCH -p ghpc_gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00

set -euo pipefail

# These variables are passed by the launcher using sbatch --export.
: "${RUN_INDEX:?RUN_INDEX is required}"
: "${RUN_NAME:?RUN_NAME is required}"
: "${RUN_SPEC_FILE:?RUN_SPEC_FILE is required}"
: "${PROJECT_SRC:?PROJECT_SRC is required}"
: "${DATA_SRC:?DATA_SRC is required}"
: "${RESULTS_ROOT:?RESULTS_ROOT is required}"
: "${BASE_CONFIG:?BASE_CONFIG is required}"
: "${TRAIN_SCRIPT:?TRAIN_SCRIPT is required}"
: "${CONDA_SH:?CONDA_SH is required}"
: "${CONDA_ENV:?CONDA_ENV is required}"

#--------------------------------------------------------------------------#
# Scratch paths
#--------------------------------------------------------------------------#

TMPDIR="/scratch/${USER}/${SLURM_JOB_ID}_${RUN_NAME}"
export TMPDIR

SCRATCH_PROJECT="${TMPDIR}/worm-species"
SCRATCH_DATA="${TMPDIR}/data"
SCRATCH_OUTPUTS="${TMPDIR}/outputs"
RUN_BACK_OUT="${RESULTS_ROOT}/${RUN_NAME}"

mkdir -p "$SCRATCH_PROJECT" "$SCRATCH_DATA" "$SCRATCH_OUTPUTS" "$RUN_BACK_OUT"

cleanup() {
    echo "Cleaning scratch: $TMPDIR"
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

echo "------------------------------------------------------------"
echo "Run name: $RUN_NAME"
echo "Run index: $RUN_INDEX"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "TMPDIR: $TMPDIR"
echo "PROJECT_SRC: $PROJECT_SRC"
echo "DATA_SRC: $DATA_SRC"
echo "RUN_BACK_OUT: $RUN_BACK_OUT"
echo "RUN_SPEC_FILE: $RUN_SPEC_FILE"
echo "------------------------------------------------------------"

cp "$RUN_SPEC_FILE" "${RUN_BACK_OUT}/run_overrides.args"

# Read override arguments from file.
# This assumes values do not contain spaces, which is true for typical
# key=value config overrides such as model.name=resnet18.
read -r -a OVERRIDE_ARGS < "$RUN_SPEC_FILE"

echo "Overrides:"
printf '  %q\n' "${OVERRIDE_ARGS[@]}"

#--------------------------------------------------------------------------#
# Copy project and data to node-local scratch
#--------------------------------------------------------------------------#

echo "Copying project to scratch..."
rsync -a \
    --exclude ".git" \
    --exclude "__pycache__" \
    --exclude ".ipynb_checkpoints" \
    --exclude "outputs" \
    --exclude "outputs_slurm" \
    "${PROJECT_SRC}/" "${SCRATCH_PROJECT}/"

echo "Copying data to scratch..."
rsync -a --info=progress2 "${DATA_SRC}/" "${SCRATCH_DATA}/"

#--------------------------------------------------------------------------#
# Activate environment
#--------------------------------------------------------------------------#

source "$CONDA_SH"
conda activate "$CONDA_ENV"

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"

python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch version:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY

#--------------------------------------------------------------------------#
# Run training
#--------------------------------------------------------------------------#

cd "$SCRATCH_PROJECT"

status=0

srun python "$TRAIN_SCRIPT" \
    --config "$BASE_CONFIG" \
    --override \
        data.root_dir="$SCRATCH_DATA" \
        output.out_dir="$SCRATCH_OUTPUTS" \
        "${OVERRIDE_ARGS[@]}" \
    || status=$?

echo "$status" > "${SCRATCH_OUTPUTS}/run_status.txt"

#--------------------------------------------------------------------------#
# Copy results back even if training failed
#--------------------------------------------------------------------------#

echo "Copying results back to: $RUN_BACK_OUT"
rsync -a "${SCRATCH_OUTPUTS}/" "${RUN_BACK_OUT}/"

echo "Finished ${RUN_NAME} with status ${status}"
exit "$status"
