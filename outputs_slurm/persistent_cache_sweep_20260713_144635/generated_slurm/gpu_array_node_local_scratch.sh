#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1

set -euo pipefail

: "${PROJECT_SRC:?}"
: "${DATA_SRC:?}"
: "${RUN_SPECS_DIR:?}"
: "${SCRATCH_ROOT:?}"
: "${SCRATCH_PROJECT:?}"
: "${SCRATCH_DATA:?}"
: "${SCRATCH_OUTPUTS:?}"
: "${RESULTS_ROOT:?}"
: "${BASE_CONFIG:?}"
: "${TRAIN_SCRIPT:?}"
: "${CONDA_SH:?}"
: "${CONDA_ENV:?}"
: "${WANDB_ENABLED:?}"
: "${WANDB_PROJECT:?}"
: "${WANDB_MODE:?}"
: "${WANDB_RUN_GROUP:?}"

RUN_INDEX="${SLURM_ARRAY_TASK_ID}"
RUN_NAME=$(
    awk -F '\t' -v idx="$RUN_INDEX" 'NR > 1 && $1 == idx {print $2; exit}' \
        "${RESULTS_ROOT}/sweep_plan.tsv"
)

if [[ -z "$RUN_NAME" ]]; then
    echo "ERROR: could not resolve run name for array index $RUN_INDEX" >&2
    exit 1
fi

RUN_SPEC_FILE="${RUN_SPECS_DIR}/${RUN_NAME}.args"

RUN_SCRATCH_OUT="${SCRATCH_OUTPUTS}/${RUN_NAME}"
RUN_BACK_OUT="${RESULTS_ROOT}/${RUN_NAME}"
JOB_TMP="${TMPDIR:-${TMP:-/tmp/${SLURM_JOB_ID}}}"
RUN_TMP_ROOT="${JOB_TMP}/worm_species"
RUN_TMP_OUT="${RUN_TMP_ROOT}/${RUN_NAME}"
LOCAL_CACHE="${RUN_TMP_ROOT}/image_cache"


mkdir -p "$RUN_TMP_OUT" "$RUN_BACK_OUT"

copy_results_back() {
    local status=$?

    echo "$status" > "${RUN_TMP_OUT}/run_status.txt"

    echo "Copying results back to persistent storage:"
    echo "  $RUN_BACK_OUT"

    rsync -a "${RUN_TMP_OUT}/" "${RUN_BACK_OUT}/" || {
        echo "ERROR: result copy-back failed." >&2
        return 90
    }

    return "$status"
}
trap copy_results_back EXIT

[[ -f "$RUN_SPEC_FILE" ]] || {
    echo "ERROR: run specification not found: $RUN_SPEC_FILE" >&2
    exit 1
}

[[ -d "$CACHE_DIR" && -f "${CACHE_DIR}/CACHE_READY" ]] || {
    echo "ERROR: persistent cache is not ready: $CACHE_DIR" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Decide whether the cache can fit in this job's temporary filesystem.
# ---------------------------------------------------------------------------
CACHE_BYTES=$(du -sb "$CACHE_DIR" | awk '{print $1}')
TMP_AVAILABLE_BYTES=$(df -PB1 "$JOB_TMP" | awk 'NR == 2 {print $4}')
RESERVE_BYTES=$((TMP_RESERVE_GB * 1024 * 1024 * 1024))
REQUIRED_BYTES=$((CACHE_BYTES + RESERVE_BYTES))

echo "------------------------------------------------------------"
echo "GPU sweep task"
echo "Run:                  $RUN_NAME"
echo "SLURM job ID:         $SLURM_JOB_ID"
echo "Array task ID:        $SLURM_ARRAY_TASK_ID"
echo "Host:                 $(hostname)"
echo "TMPDIR:               $JOB_TMP"
echo "Persistent cache:     $CACHE_DIR"
echo "Cache size:           $(du -sh "$CACHE_DIR" | cut -f1)"
echo "TMP available bytes:  $TMP_AVAILABLE_BYTES"
echo "Required bytes:       $REQUIRED_BYTES"
echo "Copy mode:            $COPY_CACHE_TO_TMP"
echo "Temporary output:     $RUN_TMP_OUT"
echo "Persistent output:    $RUN_BACK_OUT"
echo "------------------------------------------------------------"

USE_LOCAL_CACHE=1

case "$COPY_CACHE_TO_TMP" in
    0)
        USE_LOCAL_CACHE=0
        ;;
    1)
        if ((TMP_AVAILABLE_BYTES < REQUIRED_BYTES)); then
            echo "ERROR: insufficient temporary space for the cache." >&2
            echo "Use COPY_CACHE_TO_TMP=0 or increase the available scratch space." >&2
            exit 1
        fi
        USE_LOCAL_CACHE=1
        ;;
    auto)
        if ((TMP_AVAILABLE_BYTES >= REQUIRED_BYTES)); then
            USE_LOCAL_CACHE=1
        else
            echo "Cache does not fit in temporary storage with the requested reserve."
            echo "Falling back to the persistent cache."
            USE_LOCAL_CACHE=0
        fi
        ;;
esac

if ((USE_LOCAL_CACHE == 1)); then
    echo "Copying cache to job-local temporary storage..."
    mkdir -p "$LOCAL_CACHE"

    rsync -a --info=progress2 \
        "${CACHE_DIR}/" \
        "${LOCAL_CACHE}/"

    [[ -f "${LOCAL_CACHE}/CACHE_READY" ]] || {
        echo "ERROR: copied cache has no CACHE_READY marker." >&2
        exit 1
    }

    CACHE_INPUT="$LOCAL_CACHE"
else
    echo "Reading the cache directly from persistent storage."
    CACHE_INPUT="$CACHE_DIR"
fi

echo "Cache used for training: $CACHE_INPUT"

cp "$RUN_SPEC_FILE" "${RUN_TMP_OUT}/run_overrides.args"
mapfile -t OVERRIDE_ARGS < "$RUN_SPEC_FILE"

echo "Overrides:"
if ((${#OVERRIDE_ARGS[@]})); then
    printf '  %q\n' "${OVERRIDE_ARGS[@]}"
else
    echo "  <none>"
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$SOURCE_ROOT"
export CUDA_LAUNCH_BLOCKING=1

echo "------------------------------------------------------------"
echo "GPU ARRAY TASK"
echo "Run: $RUN_NAME"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Node-local SCRATCH_ROOT: $SCRATCH_ROOT"
echo "RUN_SPEC_FILE: $RUN_SPEC_FILE"
echo "------------------------------------------------------------"


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
if os.getenv("WANDB_ENABLED", "true").lower() in {"1", "true", "yes", "on"}:
    import wandb
    print("Weights & Biases:", wandb.__version__)
else:
    print("Weights & Biases: disabled")
PY

echo "Running training script: $TRAIN_SCRIPT"
echo "Scratch Data: $SCRATCH_DATA"
srun python "$TRAIN_SCRIPT" \
    --config "$BASE_CONFIG" \
    --override \
        "${OVERRIDE_ARGS[@]}" \
        data.root_dir="$SCRATCH_DATA" \
        data.metadata_csv="$SCRATCH_DATA/01_Segmented/global_metadata.csv" \
        output.out_dir="$RUN_SCRATCH_OUT" \
        sweep.enabled=false \
        colour_ablation.enabled=false \
        wandb.enabled="$WANDB_ENABLED" \
        wandb.project="$WANDB_PROJECT" \
        wandb.entity="$WANDB_ENTITY" \
        wandb.group="$WANDB_RUN_GROUP" \
        wandb.name="$RUN_NAME" \
        wandb.mode="$WANDB_MODE" \
        cache.root_dir_cache="$SCRATCH_ROOT" \
        cache.dir="$CACHE_ROOT" \
        split.predefined_split_dir="$SCRATCH_PROJECT" \
    || status=$?

echo "$status" > "${RUN_SCRATCH_OUT}/run_status.txt"

echo "Copying result back to: $RUN_BACK_OUT"
rsync -a "${RUN_SCRATCH_OUT}/" "${RUN_BACK_OUT}/"

echo "Finished ${RUN_NAME} with status ${status}"
exit "$status"
