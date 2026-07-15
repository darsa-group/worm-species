#!/bin/bash

set -euo pipefail

# ~/worm-species may itself be a symbolic link. Resolve it to its persistent
# target; do not create another cache symlink.
PROJECT_ENTRY="${PROJECT_ROOT:-${HOME}/worm-species}"

[[ -e "$PROJECT_ENTRY" ]] || {
    echo "ERROR: project path does not exist: $PROJECT_ENTRY" >&2
    exit 1
}

PROJECT_ROOT="$(readlink -f "$PROJECT_ENTRY")"
SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}/source}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"

BASE_CONFIG="${BASE_CONFIG:-config.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_multitask_masked_hloss_wandb.py}"

CONDA_SH="${CONDA_SH:-${HOME}/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-wormspecies}"

METADATA_CSV="${METADATA_CSV:-${DATA_ROOT}/01_Segmented/global_metadata.csv}"
CACHE_DIR="${CACHE_DIR:-${DATA_ROOT}/image_cache}"
CACHE_READY="${CACHE_DIR}/CACHE_READY"

# ---------------------------------------------------------------------------
# Cache-copy behaviour
# ---------------------------------------------------------------------------
# auto: copy to $TMPDIR when sufficient space is available; otherwise read the
#       persistent cache directly.
# 1:    require a copy to $TMPDIR and fail when space is insufficient.
# 0:    always read the persistent cache directly.
COPY_CACHE_TO_TMP="${COPY_CACHE_TO_TMP:-1}"

# Leave this much free space for outputs and other temporary files.
TMP_RESERVE_GB="${TMP_RESERVE_GB:-5}"

# ---------------------------------------------------------------------------
# SLURM settings
# ---------------------------------------------------------------------------
GPU_ACCOUNT="${GPU_ACCOUNT:-worm-species}"
GPU_PARTITION="${GPU_PARTITION:-gpu-short,gpu-h200}"
GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-8}"
GPU_MEM="${GPU_MEM:-12384}"
GPU_TIME="${GPU_TIME:-01:30:00}"
MAX_ACTIVE="${MAX_ACTIVE:-12}"
GPU_EXTRA_SBATCH_ARGS="${GPU_EXTRA_SBATCH_ARGS:-}"

RESULTS_ROOT="${RESULTS_ROOT:-${SOURCE_ROOT}/outputs_slurm/persistent_cache_sweep_$(date +%Y%m%d_%H%M%S)}"

# W&B CHANGE 1: these can be overridden when launching the sweep, for example:
# WANDB_PROJECT=worm-species WANDB_MODE=offline bash this_script.sh
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-worm-species}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$(basename "$RESULTS_ROOT")}"

RUN_SPECS_DIR="${RESULTS_ROOT}/run_specs"
SLURM_LOG_DIR="${RESULTS_ROOT}/slurm_logs"
GENERATED_DIR="${RESULTS_ROOT}/generated_slurm"
GPU_ARRAY_SCRIPT="${GENERATED_DIR}/gpu_array_from_persistent_cache.sh"

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

[[ -f "$METADATA_CSV" ]] || {
    echo "ERROR: metadata file not found: $METADATA_CSV" >&2
    exit 1
}

[[ -d "$CACHE_DIR" && -f "$CACHE_READY" ]] || {
    echo "ERROR: persistent cache is not ready: $CACHE_DIR" >&2
    echo "Run 01_build_persistent_cache_resolved.sh first." >&2
    exit 1
}

[[ -f "$CONDA_SH" ]] || {
    echo "ERROR: conda initialisation script not found: $CONDA_SH" >&2
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
CONDA_ENV=${CONDA_ENV}
WANDB_ENABLED=${WANDB_ENABLED}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_ENTITY=${WANDB_ENTITY}
WANDB_MODE=${WANDB_MODE}
WANDB_RUN_GROUP=${WANDB_RUN_GROUP}
EOF

source "$CONDA_SH"
conda activate "$CONDA_ENV"

# ---------------------------------------------------------------------------
# Generate one run specification for every sweep combination.
# ---------------------------------------------------------------------------
N_RUNS=$(
python - "${SOURCE_ROOT}/${BASE_CONFIG}" "$RUN_SPECS_DIR" "${RESULTS_ROOT}/sweep_plan.tsv" <<'PY'
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import yaml

config_path = Path(sys.argv[1])
run_specs_dir = Path(sys.argv[2])
sweep_plan_path = Path(sys.argv[3])

with config_path.open("r") as handle:
    cfg = yaml.safe_load(handle)

sweep_cfg = cfg.get("sweep", {}) or {}
enabled = bool(sweep_cfg.get("enabled", False))
params = sweep_cfg.get("parameters", {}) or {}

run_specs_dir.mkdir(parents=True, exist_ok=True)


def format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


if not enabled or not params:
    (run_specs_dir / "run_000.args").write_text("")
    sweep_plan_path.write_text(
        "run_index\trun_name\toverrides\n"
        "0\trun_000\t<no sweep enabled>\n"
    )
    print(1)
    raise SystemExit(0)

if not isinstance(params, dict):
    raise TypeError("sweep.parameters must be a dictionary.")

keys = list(params)
value_lists = []

for key in keys:
    values = params[key]
    if not isinstance(values, list):
        raise TypeError(f"sweep.parameters.{key} must be a list.")
    if not values:
        raise ValueError(f"sweep.parameters.{key} is empty.")
    value_lists.append(values)

plan_lines = ["run_index\trun_name\toverrides"]

for index, combination in enumerate(itertools.product(*value_lists)):
    run_name = f"run_{index:03d}"
    overrides = [
        f"{key}={format_value(value)}"
        for key, value in zip(keys, combination)
    ]

    (run_specs_dir / f"{run_name}.args").write_text(
        "\n".join(overrides) + "\n"
    )
    plan_lines.append(
        f"{index}\t{run_name}\t" + " ".join(overrides)
    )

sweep_plan_path.write_text("\n".join(plan_lines) + "\n")
print(len(plan_lines) - 1)
PY
)

ARRAY_MAX=$((N_RUNS - 1))

echo "Number of runs: $N_RUNS"
echo "Sweep plan: ${RESULTS_ROOT}/sweep_plan.tsv"

# ---------------------------------------------------------------------------
# Generate the array task.
# ---------------------------------------------------------------------------
cat > "$GPU_ARRAY_SCRIPT" <<'GPUJOB'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

set -euo pipefail

: "${SOURCE_ROOT:?}"
: "${DATA_ROOT:?}"
: "${BASE_CONFIG:?}"
: "${TRAIN_SCRIPT:?}"
: "${CONDA_SH:?}"
: "${CONDA_ENV:?}"
: "${METADATA_CSV:?}"
: "${CACHE_DIR:?}"
: "${COPY_CACHE_TO_TMP:?}"
: "${TMP_RESERVE_GB:?}"
: "${RUN_SPECS_DIR:?}"
: "${RESULTS_ROOT:?}"
: "${WANDB_ENABLED:?}"
: "${WANDB_PROJECT:?}"
: "${WANDB_MODE:?}"
: "${WANDB_RUN_GROUP:?}"

RUN_INDEX="${SLURM_ARRAY_TASK_ID}"
RUN_NAME=$(printf "run_%03d" "$RUN_INDEX")
RUN_SPEC_FILE="${RUN_SPECS_DIR}/${RUN_NAME}.args"
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

# W&B CHANGE 2: give each array element a clear name while grouping the
# complete array as one sweep-like experiment in the W&B interface.
export WANDB_PROJECT WANDB_ENTITY WANDB_MODE WANDB_RUN_GROUP
export WANDB_NAME="$RUN_NAME"

echo "Python: $(command -v python)"
python --version
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PY'
import os
import torch

print("PyTorch:", torch.__version__)
if os.getenv("WANDB_ENABLED", "true").lower() in {"1", "true", "yes", "on"}:
    import wandb
    print("Weights & Biases:", wandb.__version__)
else:
    print("Weights & Biases: disabled")
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this array task.")

print("GPU:", torch.cuda.get_device_name(0))
PY

srun python "$TRAIN_SCRIPT" \
    --config "$BASE_CONFIG" \
    --override \
        "${OVERRIDE_ARGS[@]}" \
        data.root_dir="$DATA_ROOT" \
        data.metadata_csv="$METADATA_CSV" \
        output.out_dir="$RUN_TMP_OUT" \
        sweep.enabled=false \
        wandb.enabled="$WANDB_ENABLED" \
        wandb.project="$WANDB_PROJECT" \
        wandb.entity="$WANDB_ENTITY" \
        wandb.group="$WANDB_RUN_GROUP" \
        wandb.name="$RUN_NAME" \
        wandb.mode="$WANDB_MODE" \
        cache.enabled=true \
        cache.cache_dir="$CACHE_INPUT" \
        cache.dir="$CACHE_INPUT" \
        cache.root_dir="$CACHE_INPUT" \
        cache.root_dir_cache="$CACHE_INPUT" \
        split.predefined_split_dir="$SOURCE_ROOT"
GPUJOB

chmod +x "$GPU_ARRAY_SCRIPT"

echo "Submitting array: 0-${ARRAY_MAX}%${MAX_ACTIVE}"

# shellcheck disable=SC2086
ARRAY_JOB_ID=$(
    sbatch --parsable \
        --account="$GPU_ACCOUNT" \
        --partition="$GPU_PARTITION" \
        --exclude=gn-1002 \
        --cpus-per-task="$GPU_CPUS_PER_TASK" \
        --mem="$GPU_MEM" \
        --time="$GPU_TIME" \
        --array="0-${ARRAY_MAX}%${MAX_ACTIVE}" \
        --job-name="worm_sweep" \
        --output="${SLURM_LOG_DIR}/gpu_%A_%a.out" \
        --error="${SLURM_LOG_DIR}/gpu_%A_%a.err" \
        --export=ALL,SOURCE_ROOT="$SOURCE_ROOT",DATA_ROOT="$DATA_ROOT",BASE_CONFIG="$BASE_CONFIG",TRAIN_SCRIPT="$TRAIN_SCRIPT",CONDA_SH="$CONDA_SH",CONDA_ENV="$CONDA_ENV",METADATA_CSV="$METADATA_CSV",CACHE_DIR="$CACHE_DIR",COPY_CACHE_TO_TMP="$COPY_CACHE_TO_TMP",TMP_RESERVE_GB="$TMP_RESERVE_GB",RUN_SPECS_DIR="$RUN_SPECS_DIR",RESULTS_ROOT="$RESULTS_ROOT",WANDB_ENABLED="$WANDB_ENABLED",WANDB_PROJECT="$WANDB_PROJECT",WANDB_ENTITY="$WANDB_ENTITY",WANDB_MODE="$WANDB_MODE",WANDB_RUN_GROUP="$WANDB_RUN_GROUP" \
        $GPU_EXTRA_SBATCH_ARGS \
        "$GPU_ARRAY_SCRIPT"
)

cat > "${RESULTS_ROOT}/submitted_jobs.tsv" <<EOF
name	job_id
gpu_array	${ARRAY_JOB_ID}
EOF

echo "------------------------------------------------------------"
echo "Submitted GPU array: $ARRAY_JOB_ID"
echo "Results:             $RESULTS_ROOT"
echo "Logs:                $SLURM_LOG_DIR"
echo
echo "Monitor with:"
echo "  squeue -j $ARRAY_JOB_ID"
echo "------------------------------------------------------------"
