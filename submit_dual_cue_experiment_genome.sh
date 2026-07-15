#!/bin/bash
#--------------------------------------------------------------------------#
# Model sweep with test-time cue suppression and job-local cache copying.
#
# Workflow:
#   1. Read sweep.parameters from config.yaml.
#   2. Write one run-spec file per sweep combination.
#   3. Run one configuration per SLURM GPU-array task.
#   4. Optionally copy the persistent image cache to the task's temporary
#      filesystem. SLURM/the operating system manages that temporary space;
#      no separate cleanup jobs are submitted.
#   5. Write training outputs to temporary storage and copy them back to the
#      shared RESULTS_ROOT when the task exits.
#   6. Submit one dependent result-collection job after the array finishes.
#
# Run from the project root, for example:
#
#   cd ~/worm-species
#   chmod +x submit_worm_node_local_scratch_sweep_fixed.sh
#   bash submit_worm_node_local_scratch_sweep_fixed.sh
#
# Optional examples:
#
#   COPY_CACHE_TO_TMP=auto MAX_ACTIVE=4 \
#     bash submit_worm_node_local_scratch_sweep_fixed.sh
#
#   COPY_CACHE_TO_TMP=0 \
#     bash submit_worm_node_local_scratch_sweep_fixed.sh
#--------------------------------------------------------------------------#

set -euo pipefail

#==========================================================================#
# USER SETTINGS
#==========================================================================#

PROJECT_ENTRY="${PROJECT_ROOT:-${HOME}/worm-species}"

[[ -e "$PROJECT_ENTRY" ]] || {
    echo "ERROR: project path does not exist: $PROJECT_ENTRY" >&2
    exit 1
}

PROJECT_ROOT="$(readlink -f "$PROJECT_ENTRY")"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/source}"
PROJECT_SRC="$SOURCE_ROOT"

# Shared dataset and persistent image cache.
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
DATA_SRC="$DATA_ROOT"

BASE_CONFIG="${BASE_CONFIG:-config.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_multitask_cue_suppression.py}"
RUN_SPEC_GENERATOR="${RUN_SPEC_GENERATOR:-generate_dual_cue_run_specs.py}"
RESULT_COLLECTOR="${RESULT_COLLECTOR:-collect_dual_cue_results.py}"

METADATA_CSV="${METADATA_CSV:-${DATA_ROOT}/01_Segmented/global_metadata.csv}"
CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/image_cache}"
CACHE_READY="${CACHE_DIR}/CACHE_READY"

CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-wormspecies}"

# At most this many one-GPU array tasks run simultaneously.
MAX_ACTIVE="${MAX_ACTIVE:-12}"

# GPU-array resources.
GPU_ACCOUNT="${GPU_ACCOUNT:-worm-species}"
GPU_PARTITION="${GPU_PARTITION:-gpu-short,gpu-h200,gpu-l40s}"
GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-12}"
GPU_MEM="${GPU_MEM:-12384}"
GPU_TIME="${GPU_TIME:-01:30:00}"

# Result-collection resources. COLLECT_PARTITION is optional; when empty,
# SLURM uses the account/user's default partition.
COLLECT_PARTITION="${COLLECT_PARTITION:-}"
COLLECT_CPUS_PER_TASK="${COLLECT_CPUS_PER_TASK:-1}"
COLLECT_MEM="${COLLECT_MEM:-4096}"
COLLECT_TIME="${COLLECT_TIME:-00:20:00}"

RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_SRC}/outputs_slurm/persistent_cache_sweep_$(date +%Y%m%d_%H%M%S)}"

WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-worm-species-cues}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$(basename "$RESULTS_ROOT")}"

# Cache-copy modes:
#   1    Always copy the persistent cache to the task-local temporary space.
#   0    Read directly from the persistent cache.
#   auto Copy only when sufficient temporary space is available.
COPY_CACHE_TO_TMP="${COPY_CACHE_TO_TMP:-1}"

# Free temporary space retained for outputs and other temporary files.
TMP_RESERVE_GB="${TMP_RESERVE_GB:-5}"

RUN_SPECS_DIR="${RESULTS_ROOT}/run_specs"
SLURM_LOG_DIR="${RESULTS_ROOT}/slurm_logs"
GENERATED_DIR="${RESULTS_ROOT}/generated_slurm"

GPU_ARRAY_SCRIPT="${GENERATED_DIR}/gpu_array_job_local_cache.sh"
COLLECT_SCRIPT="${GENERATED_DIR}/collect_test_cue_suppression_results.sh"

# Optional additional sbatch arguments.
GPU_EXTRA_SBATCH_ARGS="${GPU_EXTRA_SBATCH_ARGS:-}"
COLLECT_EXTRA_SBATCH_ARGS="${COLLECT_EXTRA_SBATCH_ARGS:-}"

#==========================================================================#
# VALIDATION
#==========================================================================#

echo "------------------------------------------------------------"
echo "Sweep using a persistent master cache"
echo "Project entry:      $PROJECT_ENTRY"
echo "Resolved root:      $PROJECT_ROOT"
echo "Source root:        $SOURCE_ROOT"
echo "Data root:          $DATA_ROOT"
echo "Persistent cache:   $CACHE_DIR"
echo "Copy mode:          $COPY_CACHE_TO_TMP"
echo "TMP reserve:        ${TMP_RESERVE_GB} GiB"
echo "Results:            $RESULTS_ROOT"
echo "W&B enabled:        $WANDB_ENABLED"
echo "W&B project:        $WANDB_PROJECT"
echo "W&B group:          $WANDB_RUN_GROUP"
echo "W&B mode:           $WANDB_MODE"
echo "------------------------------------------------------------"

case "$COPY_CACHE_TO_TMP" in
    auto|0|1) ;;
    *)
        echo "ERROR: COPY_CACHE_TO_TMP must be auto, 0, or 1." >&2
        exit 1
        ;;
esac

[[ "$MAX_ACTIVE" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: MAX_ACTIVE must be a positive integer: $MAX_ACTIVE" >&2
    exit 1
}

[[ "$TMP_RESERVE_GB" =~ ^[0-9]+$ ]] || {
    echo "ERROR: TMP_RESERVE_GB must be a non-negative integer: $TMP_RESERVE_GB" >&2
    exit 1
}

[[ -d "$SOURCE_ROOT" ]] || {
    echo "ERROR: source directory not found: $SOURCE_ROOT" >&2
    exit 1
}

[[ -d "$DATA_ROOT" ]] || {
    echo "ERROR: data directory not found: $DATA_ROOT" >&2
    exit 1
}

[[ -f "${SOURCE_ROOT}/${BASE_CONFIG}" ]] || {
    echo "ERROR: config not found: ${SOURCE_ROOT}/${BASE_CONFIG}" >&2
    exit 1
}

[[ -f "${SOURCE_ROOT}/${TRAIN_SCRIPT}" ]] || {
    echo "ERROR: training script not found: ${SOURCE_ROOT}/${TRAIN_SCRIPT}" >&2
    exit 1
}

[[ -f "${SOURCE_ROOT}/${RUN_SPEC_GENERATOR}" ]] || {
    echo "ERROR: run-spec generator not found: ${SOURCE_ROOT}/${RUN_SPEC_GENERATOR}" >&2
    exit 1
}

[[ -f "${SOURCE_ROOT}/${RESULT_COLLECTOR}" ]] || {
    echo "ERROR: result collector not found: ${SOURCE_ROOT}/${RESULT_COLLECTOR}" >&2
    exit 1
}

[[ -f "$METADATA_CSV" ]] || {
    echo "ERROR: metadata file not found: $METADATA_CSV" >&2
    exit 1
}

[[ -d "$CACHE_DIR" && -f "$CACHE_READY" ]] || {
    echo "ERROR: persistent cache is not ready: $CACHE_DIR" >&2
    echo "Expected marker: $CACHE_READY" >&2
    echo "Run 01_build_persistent_cache_resolved.sh first." >&2
    exit 1
}

[[ -f "$CONDA_SH" ]] || {
    echo "ERROR: conda initialisation script not found: $CONDA_SH" >&2
    exit 1
}

command -v sbatch >/dev/null 2>&1 || {
    echo "ERROR: sbatch is not available in PATH." >&2
    exit 1
}

mkdir -p "$RESULTS_ROOT" "$RUN_SPECS_DIR" "$SLURM_LOG_DIR" "$GENERATED_DIR"

cat > "${RESULTS_ROOT}/launcher_settings.txt" <<EOF
PROJECT_ENTRY=${PROJECT_ENTRY}
PROJECT_ROOT=${PROJECT_ROOT}
SOURCE_ROOT=${SOURCE_ROOT}
DATA_ROOT=${DATA_ROOT}
BASE_CONFIG=${BASE_CONFIG}
TRAIN_SCRIPT=${TRAIN_SCRIPT}
RUN_SPEC_GENERATOR=${RUN_SPEC_GENERATOR}
RESULT_COLLECTOR=${RESULT_COLLECTOR}
METADATA_CSV=${METADATA_CSV}
CACHE_DIR=${CACHE_DIR}
COPY_CACHE_TO_TMP=${COPY_CACHE_TO_TMP}
TMP_RESERVE_GB=${TMP_RESERVE_GB}
RESULTS_ROOT=${RESULTS_ROOT}
GPU_ACCOUNT=${GPU_ACCOUNT}
GPU_PARTITION=${GPU_PARTITION}
GPU_CPUS_PER_TASK=${GPU_CPUS_PER_TASK}
GPU_MEM=${GPU_MEM}
GPU_TIME=${GPU_TIME}
MAX_ACTIVE=${MAX_ACTIVE}
COLLECT_PARTITION=${COLLECT_PARTITION}
COLLECT_CPUS_PER_TASK=${COLLECT_CPUS_PER_TASK}
COLLECT_MEM=${COLLECT_MEM}
COLLECT_TIME=${COLLECT_TIME}
CONDA_ENV=${CONDA_ENV}
WANDB_ENABLED=${WANDB_ENABLED}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_ENTITY=${WANDB_ENTITY}
WANDB_MODE=${WANDB_MODE}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP}
EOF

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

#==========================================================================#
# GENERATE RUN SPECIFICATIONS
#==========================================================================#

echo "Generating dual-cue experiment specs from ${PROJECT_SRC}/${BASE_CONFIG}"

N_RUNS=$(
    python "${PROJECT_SRC}/${RUN_SPEC_GENERATOR}" \
        "${PROJECT_SRC}/${BASE_CONFIG}" \
        "$RUN_SPECS_DIR" \
        "${RESULTS_ROOT}/sweep_plan.tsv"
)

if [[ ! "$N_RUNS" =~ ^[0-9]+$ ]] || ((N_RUNS < 1)); then
    echo "ERROR: invalid number of generated runs: $N_RUNS" >&2
    exit 1
fi

echo "Number of matched-condition training runs: $N_RUNS"
echo "Sweep plan: ${RESULTS_ROOT}/sweep_plan.tsv"
echo "Experiment plan: ${RESULTS_ROOT}/dual_cue_experiment_plan.json"

ARRAY_MAX=$((N_RUNS - 1))

#==========================================================================#
# WRITE GPU ARRAY SCRIPT
#==========================================================================#

cat > "$GPU_ARRAY_SCRIPT" <<'GPUJOB'
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
GPUJOB

chmod +x "$GPU_ARRAY_SCRIPT"

#==========================================================================#
# WRITE RESULT-COLLECTION SCRIPT
#==========================================================================#

cat > "$COLLECT_SCRIPT" <<'COLLECT'
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1

set -euo pipefail

: "${RESULTS_ROOT:?}"
: "${PROJECT_SRC:?}"
: "${RESULT_COLLECTOR:?}"
: "${CONDA_SH:?}"
: "${CONDA_ENV:?}"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

python "${PROJECT_SRC}/${RESULT_COLLECTOR}" "$RESULTS_ROOT"
COLLECT

chmod +x "$COLLECT_SCRIPT"

#==========================================================================#
# SUBMIT GPU ARRAY
#==========================================================================#

GPU_EXPORTS="ALL"
GPU_EXPORTS+=",SOURCE_ROOT=${SOURCE_ROOT}"
GPU_EXPORTS+=",DATA_ROOT=${DATA_ROOT}"
GPU_EXPORTS+=",BASE_CONFIG=${BASE_CONFIG}"
GPU_EXPORTS+=",TRAIN_SCRIPT=${TRAIN_SCRIPT}"
GPU_EXPORTS+=",RUN_SPECS_DIR=${RUN_SPECS_DIR}"
GPU_EXPORTS+=",METADATA_CSV=${METADATA_CSV}"
GPU_EXPORTS+=",CACHE_DIR=${CACHE_DIR}"
GPU_EXPORTS+=",COPY_CACHE_TO_TMP=${COPY_CACHE_TO_TMP}"
GPU_EXPORTS+=",TMP_RESERVE_GB=${TMP_RESERVE_GB}"
GPU_EXPORTS+=",RESULTS_ROOT=${RESULTS_ROOT}"
GPU_EXPORTS+=",CONDA_SH=${CONDA_SH}"
GPU_EXPORTS+=",CONDA_ENV=${CONDA_ENV}"
GPU_EXPORTS+=",WANDB_ENABLED=${WANDB_ENABLED}"
GPU_EXPORTS+=",WANDB_PROJECT=${WANDB_PROJECT}"
GPU_EXPORTS+=",WANDB_ENTITY=${WANDB_ENTITY}"
GPU_EXPORTS+=",WANDB_MODE=${WANDB_MODE}"
GPU_EXPORTS+=",WANDB_RUN_GROUP=${WANDB_RUN_GROUP}"

GPU_SBATCH_ARGS=(
    --parsable
    --account="$GPU_ACCOUNT"
    -p "$GPU_PARTITION"
    --cpus-per-task="$GPU_CPUS_PER_TASK"
    --mem="$GPU_MEM"
    -t "$GPU_TIME"
    --array="0-${ARRAY_MAX}%${MAX_ACTIVE}"
    --job-name="worm_sweep"
    --output="${SLURM_LOG_DIR}/gpu_%A_%a.out"
    --error="${SLURM_LOG_DIR}/gpu_%A_%a.err"
    --export="$GPU_EXPORTS"
)

# Intentionally allow word splitting so a user can supply multiple additional
# sbatch options in GPU_EXTRA_SBATCH_ARGS.
# shellcheck disable=SC2206
GPU_EXTRA_ARGS_ARRAY=($GPU_EXTRA_SBATCH_ARGS)
GPU_SBATCH_ARGS+=("${GPU_EXTRA_ARGS_ARRAY[@]}")
GPU_SBATCH_ARGS+=("$GPU_ARRAY_SCRIPT")

ARRAY_JOB_ID=$(sbatch "${GPU_SBATCH_ARGS[@]}")
echo "GPU array job ID: $ARRAY_JOB_ID"

#==========================================================================#
# SUBMIT RESULT COLLECTOR
#==========================================================================#

echo "Submitting result-collection job..."

COLLECT_EXPORTS="ALL"
COLLECT_EXPORTS+=",RESULTS_ROOT=${RESULTS_ROOT}"
COLLECT_EXPORTS+=",PROJECT_SRC=${PROJECT_SRC}"
COLLECT_EXPORTS+=",RESULT_COLLECTOR=${RESULT_COLLECTOR}"
COLLECT_EXPORTS+=",CONDA_SH=${CONDA_SH}"
COLLECT_EXPORTS+=",CONDA_ENV=${CONDA_ENV}"

COLLECT_SBATCH_ARGS=(
    --parsable
    --cpus-per-task="$COLLECT_CPUS_PER_TASK"
    --mem="$COLLECT_MEM"
    -t "$COLLECT_TIME"
    --dependency="afterany:${ARRAY_JOB_ID}"
    --job-name="worm_cue_collect"
    --output="${SLURM_LOG_DIR}/collect_%j.out"
    --error="${SLURM_LOG_DIR}/collect_%j.err"
    --export="$COLLECT_EXPORTS"
)

if [[ -n "$COLLECT_PARTITION" ]]; then
    COLLECT_SBATCH_ARGS+=(-p "$COLLECT_PARTITION")
fi

# shellcheck disable=SC2206
COLLECT_EXTRA_ARGS_ARRAY=($COLLECT_EXTRA_SBATCH_ARGS)
COLLECT_SBATCH_ARGS+=("${COLLECT_EXTRA_ARGS_ARRAY[@]}")
COLLECT_SBATCH_ARGS+=("$COLLECT_SCRIPT")

COLLECT_JOB_ID=$(sbatch "${COLLECT_SBATCH_ARGS[@]}")
echo "Result-collection job ID: $COLLECT_JOB_ID"

echo "------------------------------------------------------------"
echo "Submitted:"
echo "  GPU array:        ${ARRAY_JOB_ID}"
echo "  result collector: ${COLLECT_JOB_ID}"
echo
echo "Array range:"
echo "  0-${ARRAY_MAX}%${MAX_ACTIVE}"
echo
echo "Results:"
echo "  ${RESULTS_ROOT}"
echo
echo "Monitor:"
echo "  squeue -j ${ARRAY_JOB_ID},${COLLECT_JOB_ID}"
echo
echo "Logs:"
echo "  ${SLURM_LOG_DIR}"
echo "------------------------------------------------------------"