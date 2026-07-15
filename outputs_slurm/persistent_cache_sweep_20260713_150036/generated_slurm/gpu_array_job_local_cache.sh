#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1

set -euo pipefail

: "${SOURCE_ROOT:?}"
: "${DATA_ROOT:?}"
: "${BASE_CONFIG:?}"
: "${TRAIN_SCRIPT:?}"
: "${RUN_SPECS_DIR:?}"
: "${METADATA_CSV:?}"
: "${CACHE_DIR:?}"
: "${COPY_CACHE_TO_TMP:?}"
: "${TMP_RESERVE_GB:?}"
: "${RESULTS_ROOT:?}"
: "${CONDA_SH:?}"
: "${CONDA_ENV:?}"
: "${WANDB_ENABLED:?}"
: "${WANDB_PROJECT:?}"
: "${WANDB_MODE:?}"
: "${WANDB_RUN_GROUP:?}"

# WANDB_ENTITY may intentionally be empty.
WANDB_ENTITY="${WANDB_ENTITY:-}"

RUN_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
RUN_NAME=$(
    awk -F '\t' -v idx="$RUN_INDEX" \
        'NR > 1 && $1 == idx {print $2; exit}' \
        "${RESULTS_ROOT}/sweep_plan.tsv"
)

if [[ -z "$RUN_NAME" ]]; then
    echo "ERROR: could not resolve run name for array index $RUN_INDEX" >&2
    exit 1
fi

RUN_SPEC_FILE="${RUN_SPECS_DIR}/${RUN_NAME}.args"
RUN_BACK_OUT="${RESULTS_ROOT}/${RUN_NAME}"

JOB_TMP="${TMPDIR:-${TMP:-/tmp/${SLURM_JOB_ID:?}}}"
RUN_TMP_ROOT="${JOB_TMP}/worm_species"
RUN_TMP_OUT="${RUN_TMP_ROOT}/outputs/${RUN_NAME}"
LOCAL_CACHE="${RUN_TMP_ROOT}/image_cache"

mkdir -p "$JOB_TMP" "$RUN_TMP_OUT" "$RUN_BACK_OUT"

copy_results_back() {
    local status=$?
    local copy_status=0

    # Prevent recursion when this function exits explicitly.
    trap - EXIT
    set +e

    printf '%s\n' "$status" > "${RUN_TMP_OUT}/run_status.txt"

    echo "Copying results back to persistent storage:"
    echo "  ${RUN_TMP_OUT}/"
    echo "  -> ${RUN_BACK_OUT}/"

    rsync -a "${RUN_TMP_OUT}/" "${RUN_BACK_OUT}/"
    copy_status=$?

    if ((copy_status != 0)); then
        echo "ERROR: result copy-back failed with status $copy_status." >&2
        if ((status == 0)); then
            status=90
        fi
    fi

    exit "$status"
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

cp "$RUN_SPEC_FILE" "${RUN_TMP_OUT}/run_overrides.args"
mapfile -t OVERRIDE_ARGS < "$RUN_SPEC_FILE"

# Remove empty lines from the run-spec file while preserving each argument as
# one array element.
FILTERED_OVERRIDE_ARGS=()
for arg in "${OVERRIDE_ARGS[@]}"; do
    [[ -n "$arg" ]] && FILTERED_OVERRIDE_ARGS+=("$arg")
done
OVERRIDE_ARGS=("${FILTERED_OVERRIDE_ARGS[@]}")

# ---------------------------------------------------------------------------
# Decide whether the cache fits in this task's temporary filesystem.
# ---------------------------------------------------------------------------
CACHE_BYTES=$(du -sb "$CACHE_DIR" | awk '{print $1}')
TMP_AVAILABLE_BYTES=$(df -PB1 "$JOB_TMP" | awk 'NR == 2 {print $4}')
RESERVE_BYTES=$((TMP_RESERVE_GB * 1024 * 1024 * 1024))
REQUIRED_BYTES=$((CACHE_BYTES + RESERVE_BYTES))

[[ "$CACHE_BYTES" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not determine cache size for $CACHE_DIR" >&2
    exit 1
}

[[ "$TMP_AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not determine available temporary space for $JOB_TMP" >&2
    exit 1
}

echo "------------------------------------------------------------"
echo "GPU sweep task"
echo "Run:                  $RUN_NAME"
echo "SLURM job ID:         ${SLURM_JOB_ID:-unknown}"
echo "Array task ID:        $RUN_INDEX"
echo "Host:                 $(hostname)"
echo "Temporary root:       $JOB_TMP"
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
            echo "Use COPY_CACHE_TO_TMP=0 or increase available temporary space." >&2
            exit 1
        fi
        USE_LOCAL_CACHE=1
        ;;
    auto)
        if ((TMP_AVAILABLE_BYTES >= REQUIRED_BYTES)); then
            USE_LOCAL_CACHE=1
        else
            echo "Cache does not fit with the requested reserve."
            echo "Falling back to the persistent cache."
            USE_LOCAL_CACHE=0
        fi
        ;;
esac

if ((USE_LOCAL_CACHE == 1)); then
    echo "Copying cache to task-local temporary storage..."
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

CACHE_PARENT="$(dirname "$CACHE_INPUT")"

echo "Cache parent passed to cache.root_dir_cache: $CACHE_PARENT"
echo "Cache directory passed to cache.dir:         $CACHE_INPUT"

echo "Overrides:"
if ((${#OVERRIDE_ARGS[@]})); then
    printf '  %q\n' "${OVERRIDE_ARGS[@]}"
else
    echo "  <none>"
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

cd "$SOURCE_ROOT"

# Keep this enabled while diagnosing asynchronous CUDA failures. Set to 0
# externally or remove it when debugging is complete for better performance.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

python - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch version:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this array task.")

print("GPU name:", torch.cuda.get_device_name(0))

if os.getenv("WANDB_ENABLED", "true").lower() in {"1", "true", "yes", "on"}:
    import wandb
    print("Weights & Biases:", wandb.__version__)
else:
    print("Weights & Biases: disabled")
PY

status=0

echo "Running training script: $TRAIN_SCRIPT"
echo "Shared data root:       $DATA_ROOT"
echo "Metadata CSV:           $METADATA_CSV"
echo "Training output:        $RUN_TMP_OUT"

srun python "$TRAIN_SCRIPT" \
    --config "$BASE_CONFIG" \
    --override \
        "${OVERRIDE_ARGS[@]}" \
        data.root_dir="$DATA_ROOT" \
        data.metadata_csv="$METADATA_CSV" \
        output.out_dir="$RUN_TMP_OUT" \
        sweep.enabled=false \
        colour_ablation.enabled=false \
        wandb.enabled="$WANDB_ENABLED" \
        wandb.project="$WANDB_PROJECT" \
        wandb.entity="$WANDB_ENTITY" \
        wandb.group="$WANDB_RUN_GROUP" \
        wandb.name="$RUN_NAME" \
        wandb.mode="$WANDB_MODE" \
        cache.root_dir_cache="$CACHE_PARENT" \
        cache.dir="$CACHE_INPUT" \
        split.predefined_split_dir="$SOURCE_ROOT" \
    || status=$?

echo "Finished ${RUN_NAME} with status ${status}"
exit "$status"
