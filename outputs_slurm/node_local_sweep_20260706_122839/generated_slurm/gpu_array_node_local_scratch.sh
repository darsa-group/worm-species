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

RUN_INDEX="${SLURM_ARRAY_TASK_ID}"
RUN_NAME=$(printf "run_%03d" "$RUN_INDEX")
RUN_SPEC_FILE="${RUN_SPECS_DIR}/${RUN_NAME}.args"

RUN_SCRATCH_OUT="${SCRATCH_OUTPUTS}/${RUN_NAME}"
RUN_BACK_OUT="${RESULTS_ROOT}/${RUN_NAME}"

echo "------------------------------------------------------------"
echo "GPU ARRAY TASK"
echo "Run: $RUN_NAME"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Node-local SCRATCH_ROOT: $SCRATCH_ROOT"
echo "RUN_SPEC_FILE: $RUN_SPEC_FILE"
echo "------------------------------------------------------------"

# -------------------------------------------------------------------------#
# Node-local lazy scratch setup
# -------------------------------------------------------------------------#
# Since /scratch is node-local, this check happens independently on each node.
# The first task that lands on a given node copies project+data.
# Later tasks on the same node reuse the READY copy.

if [[ ! -f "${SCRATCH_ROOT}/READY" ]]; then
    echo "ERROR: node-local scratch is not ready on node $(hostname)." >&2
    echo "Expected marker: ${SCRATCH_ROOT}/READY" >&2
    echo "The setup job may have failed or this array task landed on an unexpected node." >&2
    exit 1
fi

echo "Using prepared node-local scratch on node $(hostname): $SCRATCH_ROOT"

mkdir -p "$RUN_SCRATCH_OUT" "$RUN_BACK_OUT"

if [[ ! -f "$RUN_SPEC_FILE" ]]; then
    echo "ERROR: missing run spec: $RUN_SPEC_FILE" >&2
    exit 1
fi

cp "$RUN_SPEC_FILE" "${RUN_BACK_OUT}/run_overrides.args"

mapfile -t OVERRIDE_ARGS < "$RUN_SPEC_FILE"

echo "Overrides:"
printf '  %q\n' "${OVERRIDE_ARGS[@]}"

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

cd "$SCRATCH_PROJECT"

status=0

PROFILE_DIR="${RUN_SCRATCH_OUT}/profiling"
mkdir -p "$PROFILE_DIR"

echo "Starting GPU/CPU profiling logs in $PROFILE_DIR"

nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    -l 5 > "${PROFILE_DIR}/gpu_usage.csv" &

GPU_MONITOR_PID=$!

(
    while true; do
        echo "===== $(date) ====="
        ps -u "$USER" \
            -o pid,ppid,pcpu,pmem,rss,vsz,comm,args \
            --sort=-%cpu | head -n 30
        sleep 5
    done
) > "${PROFILE_DIR}/cpu_usage.txt" &

CPU_MONITOR_PID=$!

trap 'kill "$GPU_MONITOR_PID" "$CPU_MONITOR_PID" 2>/dev/null || true' EXIT
# ------------------------------------------------------------
# Build image cache once per node-local scratch directory
# ------------------------------------------------------------

CACHE_ROOT="${SCRATCH_ROOT}/image_cache"
CACHE_READY="${SCRATCH_ROOT}/IMAGE_CACHE_READY"
CACHE_LOCK="${SCRATCH_ROOT}/IMAGE_CACHE.lock"

mkdir -p "$CACHE_ROOT"

(
    flock -x 200

    if [[ ! -f "$CACHE_READY" ]]; then
        echo "[$(date)] Image cache not ready. Building it once on $(hostname)..."
        echo "Cache root: $CACHE_ROOT"

        python - "$BASE_CONFIG" "$SCRATCH_DATA" "$CACHE_ROOT" "${OVERRIDE_ARGS[@]}" <<'PY'
import sys
from pathlib import Path

from src.utils import load_config, apply_overrides
from src.dataset_multitask import prepare_metadata
from src.cache import build_image_cache

base_config = sys.argv[1]
scratch_data = sys.argv[2]
cache_root = sys.argv[3]
override_args = sys.argv[4:]

cfg = load_config(base_config)
cfg = apply_overrides(cfg, override_args)

# Force paths for node-local scratch.
cfg["data"]["root_dir"] = scratch_data
cfg["data"]["metadata_csv"] = f"{scratch_data}/01_Segmented/global_metadata.csv"
cfg["data"]["image_col"] = "rel_path_seg"

# Force cache on node-local scratch.
cfg.setdefault("cache", {})
cfg["cache"]["enabled"] = True

# Set several common names so this works with most cache.py variants.
cfg["cache"]["cache_dir"] = cache_root
cfg["cache"]["dir"] = cache_root
cfg["cache"]["root_dir"] = cache_root

print("Preparing metadata...")
df = prepare_metadata(cfg)

print("Building image cache...")
df_cached = build_image_cache(cfg, df)

n_total = len(df_cached)
n_cached = int(df_cached["_cached_image_path"].notna().sum())

print(f"Cached rows: {n_cached}/{n_total}")

if n_cached == 0:
    raise RuntimeError("Image cache was built but no cached images were produced.")
PY

        touch "$CACHE_READY"
        echo "[$(date)] Image cache ready."
    else
        echo "[$(date)] Image cache already ready. Skipping cache build."
    fi

) 200>"$CACHE_LOCK"

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
        cache.root_dir_cache="$SCRATCH_ROOT" \
        cache.dir="$CACHE_ROOT" \
    || status=$?

echo "$status" > "${RUN_SCRATCH_OUT}/run_status.txt"

echo "Copying result back to: $RUN_BACK_OUT"
rsync -a "${RUN_SCRATCH_OUT}/" "${RUN_BACK_OUT}/"

echo "Finished ${RUN_NAME} with status ${status}"
exit "$status"
