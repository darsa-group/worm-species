#!/bin/bash
#--------------------------------------------------------------------------#
# Config-based sweep with NODE-LOCAL scratch caching
#
# Use this when /scratch is NOT shared between nodes.
#
# Workflow:
#   1. Launcher reads sweep.parameters from config.yaml.
#   2. Launcher writes one run-spec file per sweep combination.
#   3. A SLURM GPU job array runs one configuration per task.
#   4. Each task checks the local /scratch on the node it lands on.
#      If READY is missing, it copies project + data to that node.
#      If READY exists, it reuses the node-local copy.
#   5. Each task copies its own result back to RESULTS_ROOT.
#   6. After the array finishes, cleanup jobs are submitted to the two GPU
#      nodes and remove their local scratch copies.
#
# Copy to project root and run:
#
#   cd ~/worm-species
#   chmod +x submit_worm_node_local_scratch_sweep.sh
#   GPU_NODES="nodeA nodeB" bash submit_worm_node_local_scratch_sweep.sh
#
# Find node names with:
#   sinfo -N -h -p ghpc_gpu -o "%N"
#--------------------------------------------------------------------------#

set -euo pipefail

#==========================================================================#
# USER SETTINGS
#==========================================================================#

PROJECT_SRC="${PROJECT_SRC:-$(pwd)}"

# Dataset on shared filesystem. This is copied to node-local /scratch by
# the first job that lands on each node.
DATA_SRC="${DATA_SRC:-/usr/home/qgg/mehrot/petridish-worm-images}"

BASE_CONFIG="${BASE_CONFIG:-config.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_multitask_masked.py}"

CONDA_SH="${CONDA_SH:-/usr/home/qgg/mehrot/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-wormspecies}"

# At most this many 1-GPU array tasks run at once.
<<<<<<< HEAD
MAX_ACTIVE="${MAX_ACTIVE:-2}"
=======
MAX_ACTIVE="${MAX_ACTIVE:-10}"
>>>>>>> refs/remotes/origin/main


# GPU job resources.
GPU_PARTITION="${GPU_PARTITION:-ghpc_gpu}"
<<<<<<< HEAD
GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-16}"
=======
GPU_CPUS_PER_TASK="${GPU_CPUS_PER_TASK:-8}"
>>>>>>> refs/remotes/origin/main
GPU_MEM="${GPU_MEM:-16384}"
GPU_TIME="${GPU_TIME:-04:00:00}"

# Cleanup jobs run after the GPU array. They need to run on the same GPU nodes
# because /scratch is node-local.
CLEANUP_PARTITION="${CLEANUP_PARTITION:-$GPU_PARTITION}"
CLEANUP_CPUS_PER_TASK="${CLEANUP_CPUS_PER_TASK:-1}"
CLEANUP_MEM="${CLEANUP_MEM:-2048}"
CLEANUP_TIME="${CLEANUP_TIME:-00:30:00}"

# Required: the two GPU nodes whose /scratch may contain cached data.
# Example:
#   GPU_NODES="gpu001 gpu002" bash submit_worm_node_local_scratch_sweep.sh
if [[ -n "${GPU_NODES:-}" ]]; then
    read -r -a CLEANUP_NODES <<< "$GPU_NODES"
else
    CLEANUP_NODES=("CHANGE_ME_GPU_NODE_1" "CHANGE_ME_GPU_NODE_2")
fi

GPU_NODELIST=$(IFS=,; echo "${CLEANUP_NODES[*]}")

SETUP_CPUS_PER_TASK="${SETUP_CPUS_PER_TASK:-4}"
SETUP_MEM="${SETUP_MEM:-8192}"
SETUP_TIME="${SETUP_TIME:-01:00:00}"
# Results location on shared filesystem.
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_SRC}/outputs_slurm/node_local_sweep_$(date +%Y%m%d_%H%M%S)}"

# Same path string on every node, but because /scratch is node-local this is
# physically a separate directory per node.
SCRATCH_ID="${SCRATCH_ID:-worm_node_local_sweep_$(date +%Y%m%d_%H%M%S)_$$}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}/${SCRATCH_ID}}"

SCRATCH_PROJECT="${SCRATCH_ROOT}/project"
SCRATCH_DATA="${SCRATCH_ROOT}/data"
SCRATCH_OUTPUTS="${SCRATCH_ROOT}/outputs"

RUN_SPECS_DIR="${RESULTS_ROOT}/run_specs"
SLURM_LOG_DIR="${RESULTS_ROOT}/slurm_logs"
GENERATED_DIR="${RESULTS_ROOT}/generated_slurm"

GPU_ARRAY_SCRIPT="${GENERATED_DIR}/gpu_array_node_local_scratch.sh"
CLEANUP_SCRIPT="${GENERATED_DIR}/cleanup_node_local_scratch.sh"

# Extra sbatch arguments if needed.
GPU_EXTRA_SBATCH_ARGS="${GPU_EXTRA_SBATCH_ARGS:-}"
CLEANUP_EXTRA_SBATCH_ARGS="${CLEANUP_EXTRA_SBATCH_ARGS:-}"

#==========================================================================#
# CHECKS
#==========================================================================#

if [[ ! -d "$PROJECT_SRC" ]]; then
    echo "ERROR: PROJECT_SRC does not exist: $PROJECT_SRC" >&2
    exit 1
fi

if [[ ! -d "$DATA_SRC" ]]; then
    echo "ERROR: DATA_SRC does not exist: $DATA_SRC" >&2
    echo "Set DATA_SRC before running, for example:" >&2
    echo "  DATA_SRC=/path/to/data bash $0" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_SRC}/${BASE_CONFIG}" ]]; then
    echo "ERROR: Config not found: ${PROJECT_SRC}/${BASE_CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${PROJECT_SRC}/${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training script not found: ${PROJECT_SRC}/${TRAIN_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: CONDA_SH not found: $CONDA_SH" >&2
    exit 1
fi

for node in "${CLEANUP_NODES[@]}"; do
    if [[ "$node" == CHANGE_ME* ]]; then
        echo "ERROR: Set GPU_NODES before running." >&2
        echo "Find node names with:" >&2
        echo "  sinfo -N -h -p ${GPU_PARTITION} -o \"%N\"" >&2
        echo "Then run, for example:" >&2
        echo "  GPU_NODES=\"gpu001 gpu002\" bash $0" >&2
        exit 1
    fi
done

mkdir -p "$RESULTS_ROOT" "$RUN_SPECS_DIR" "$SLURM_LOG_DIR" "$GENERATED_DIR"

echo "------------------------------------------------------------"
echo "PROJECT_SRC: $PROJECT_SRC"
echo "DATA_SRC: $DATA_SRC"
echo "BASE_CONFIG: $BASE_CONFIG"
echo "TRAIN_SCRIPT: $TRAIN_SCRIPT"
echo "RESULTS_ROOT: $RESULTS_ROOT"
echo "SCRATCH_ROOT: $SCRATCH_ROOT"
echo "MAX_ACTIVE: $MAX_ACTIVE"
echo "GPU_PARTITION: $GPU_PARTITION"
echo "CLEANUP_NODES: ${CLEANUP_NODES[*]}"
echo "CONDA_ENV: $CONDA_ENV"
echo "------------------------------------------------------------"

cat > "${RESULTS_ROOT}/launcher_settings.txt" <<EOF
PROJECT_SRC=${PROJECT_SRC}
DATA_SRC=${DATA_SRC}
BASE_CONFIG=${BASE_CONFIG}
TRAIN_SCRIPT=${TRAIN_SCRIPT}
RESULTS_ROOT=${RESULTS_ROOT}
SCRATCH_ID=${SCRATCH_ID}
SCRATCH_ROOT=${SCRATCH_ROOT}
SCRATCH_PROJECT=${SCRATCH_PROJECT}
SCRATCH_DATA=${SCRATCH_DATA}
SCRATCH_OUTPUTS=${SCRATCH_OUTPUTS}
MAX_ACTIVE=${MAX_ACTIVE}
GPU_PARTITION=${GPU_PARTITION}
CLEANUP_PARTITION=${CLEANUP_PARTITION}
CLEANUP_NODES=${CLEANUP_NODES[*]}
CONDA_SH=${CONDA_SH}
CONDA_ENV=${CONDA_ENV}
EOF

#==========================================================================#
# GENERATE RUN SPECS FROM config.yaml
#==========================================================================#

source "$CONDA_SH"
conda activate "$CONDA_ENV"

echo "Generating sweep specs from ${PROJECT_SRC}/${BASE_CONFIG}"

N_RUNS=$(
python - "${PROJECT_SRC}/${BASE_CONFIG}" "$RUN_SPECS_DIR" "${RESULTS_ROOT}/sweep_plan.tsv" <<'PY'
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise SystemExit(
        "PyYAML is required. Install it in the conda environment, e.g. conda install pyyaml"
    ) from e

config_path = Path(sys.argv[1])
run_specs_dir = Path(sys.argv[2])
sweep_plan_path = Path(sys.argv[3])

with config_path.open("r") as f:
    cfg = yaml.safe_load(f)

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

if not enabled or len(params) == 0:
    (run_specs_dir / "run_000.args").write_text("")
    sweep_plan_path.write_text(
        "run_index\trun_name\toverrides\n"
        "0\trun_000\t<no sweep enabled>\n"
    )
    print(1)
    raise SystemExit(0)

if not isinstance(params, dict):
    raise TypeError("sweep.parameters must be a dictionary.")

keys = list(params.keys())
values = []

for key in keys:
    vals = params[key]
    if not isinstance(vals, list):
        raise TypeError(f"sweep.parameters.{key} must be a list.")
    if len(vals) == 0:
        raise ValueError(f"sweep.parameters.{key} is empty.")
    values.append(vals)

plan_lines = ["run_index\trun_name\toverrides"]

n = 0
for n, combo in enumerate(itertools.product(*values)):
    run_name = f"run_{n:03d}"
    override_lines = [
        f"{key}={format_value(value)}"
        for key, value in zip(keys, combo)
    ]

    # One override per line avoids quoting problems.
    (run_specs_dir / f"{run_name}.args").write_text(
        "\n".join(override_lines) + "\n"
    )

    plan_lines.append(
        f"{n}\t{run_name}\t" + " ".join(override_lines)
    )

sweep_plan_path.write_text("\n".join(plan_lines) + "\n")
print(n + 1)
PY
)

echo "Number of runs: $N_RUNS"
echo "Sweep plan: ${RESULTS_ROOT}/sweep_plan.tsv"

ARRAY_MAX=$((N_RUNS - 1))


#==========================================================================#
# WRITE SETUP SCRIPT
#==========================================================================#

SETUP_SCRIPT="${GENERATED_DIR}/setup_node_local_scratch.sh"

cat > "$SETUP_SCRIPT" <<'SETUP'
#!/bin/bash
<<<<<<< HEAD
=======
#SBATCH -account worm-species
>>>>>>> refs/remotes/origin/main
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
SETUP

chmod +x "$SETUP_SCRIPT"

#==========================================================================#
# WRITE GPU ARRAY SCRIPT
#==========================================================================#

cat > "$GPU_ARRAY_SCRIPT" <<'GPUJOB'
#!/bin/bash
<<<<<<< HEAD
=======
#SBATCH -account worm-species
>>>>>>> refs/remotes/origin/main
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
        split.predefined_split_dir="$SCRATCH_PROJECT" \
    || status=$?

echo "$status" > "${RUN_SCRATCH_OUT}/run_status.txt"

echo "Copying result back to: $RUN_BACK_OUT"
rsync -a "${RUN_SCRATCH_OUT}/" "${RUN_BACK_OUT}/"

echo "Finished ${RUN_NAME} with status ${status}"
exit "$status"
GPUJOB

chmod +x "$GPU_ARRAY_SCRIPT"

#==========================================================================#
# WRITE CLEANUP SCRIPT
#==========================================================================#

cat > "$CLEANUP_SCRIPT" <<'CLEANUP'
#!/bin/bash
<<<<<<< HEAD
=======
#SBATCH -account worm-species
>>>>>>> refs/remotes/origin/main
#SBATCH -N 1
#SBATCH -n 1

set -euo pipefail

: "${SCRATCH_ROOT:?}"

echo "------------------------------------------------------------"
echo "NODE-LOCAL CLEANUP"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Removing local scratch: $SCRATCH_ROOT"
echo "------------------------------------------------------------"

rm -rf "$SCRATCH_ROOT"

echo "Cleanup complete on node $(hostname)."
CLEANUP

chmod +x "$CLEANUP_SCRIPT"

#==========================================================================#
# SUBMIT SETUP JOBS, ONE PER GPU NODE
#==========================================================================#


echo "Submitting setup jobs, one per GPU node..."

setup_job_ids=()

for node in "${CLEANUP_NODES[@]}"; do
    echo "Submitting setup job for node: $node"

    setup_job_id=$(
        sbatch --parsable \
            -p "$GPU_PARTITION" \
            --nodelist="$node" \
            --cpus-per-task="$SETUP_CPUS_PER_TASK" \
            --mem="$SETUP_MEM" \
            -t "$SETUP_TIME" \
            --job-name="worm_setup_${node}" \
            --output="${SLURM_LOG_DIR}/setup_${node}_%j.out" \
            --error="${SLURM_LOG_DIR}/setup_${node}_%j.err" \
            --export=ALL,PROJECT_SRC="$PROJECT_SRC",DATA_SRC="$DATA_SRC",SCRATCH_ROOT="$SCRATCH_ROOT",SCRATCH_PROJECT="$SCRATCH_PROJECT",SCRATCH_DATA="$SCRATCH_DATA",SCRATCH_OUTPUTS="$SCRATCH_OUTPUTS" \
            "$SETUP_SCRIPT"
    )

    setup_job_ids+=("$setup_job_id")
done

SETUP_DEPENDENCY=$(IFS=:; echo "${setup_job_ids[*]}")

echo "Setup job IDs: ${setup_job_ids[*]}"
echo "GPU array will depend on: afterok:${SETUP_DEPENDENCY}"

#==========================================================================#
# SUBMIT GPU ARRAY
#==========================================================================#
GPU_NODELIST=$(IFS=,; echo "${CLEANUP_NODES[*]}")
echo "Submitting GPU job array..."
echo "ARRAY_SPEC=0-${ARRAY_MAX}%${MAX_ACTIVE}"
# shellcheck disable=SC2086
ARRAY_JOB_ID=$(
    sbatch --parsable \
        -p "$GPU_PARTITION" \
        --nodelist="$GPU_NODELIST" \
        --cpus-per-task="$GPU_CPUS_PER_TASK" \
        --mem="$GPU_MEM" \
        -t "$GPU_TIME" \
        --array="0-${ARRAY_MAX}%${MAX_ACTIVE}" \
        --dependency="afterok:${SETUP_DEPENDENCY}" \
        --job-name="worm_sweep" \
        --output="${SLURM_LOG_DIR}/gpu_%A_%a.out" \
        --error="${SLURM_LOG_DIR}/gpu_%A_%a.err" \
        --export=ALL,PROJECT_SRC="$PROJECT_SRC",DATA_SRC="$DATA_SRC",RUN_SPECS_DIR="$RUN_SPECS_DIR",SCRATCH_ROOT="$SCRATCH_ROOT",SCRATCH_PROJECT="$SCRATCH_PROJECT",SCRATCH_DATA="$SCRATCH_DATA",SCRATCH_OUTPUTS="$SCRATCH_OUTPUTS",RESULTS_ROOT="$RESULTS_ROOT",BASE_CONFIG="$BASE_CONFIG",TRAIN_SCRIPT="$TRAIN_SCRIPT",CONDA_SH="$CONDA_SH",CONDA_ENV="$CONDA_ENV" \
        $GPU_EXTRA_SBATCH_ARGS \
        "$GPU_ARRAY_SCRIPT"
)

echo "GPU array job ID: $ARRAY_JOB_ID"

#==========================================================================#
# SUBMIT ONE CLEANUP JOB PER GPU NODE
#==========================================================================#

echo "Submitting cleanup jobs, one per GPU node, after GPU array finishes..."

: > "${RESULTS_ROOT}/submitted_jobs.tsv"
echo -e "name\tjob_id\tnode" >> "${RESULTS_ROOT}/submitted_jobs.tsv"
echo -e "gpu_array\t${ARRAY_JOB_ID}\tNA" >> "${RESULTS_ROOT}/submitted_jobs.tsv"

cleanup_job_ids=()

for node in "${CLEANUP_NODES[@]}"; do
    echo "Submitting cleanup job for node: $node"

    # shellcheck disable=SC2086
    cleanup_job_id=$(
        sbatch --parsable \
            -p "$CLEANUP_PARTITION" \
            --nodelist="$node" \
            --cpus-per-task="$CLEANUP_CPUS_PER_TASK" \
            --mem="$CLEANUP_MEM" \
            -t "$CLEANUP_TIME" \
            --dependency="afterany:${ARRAY_JOB_ID}" \
            --job-name="worm_clean_${node}" \
            --output="${SLURM_LOG_DIR}/cleanup_${node}_%j.out" \
            --error="${SLURM_LOG_DIR}/cleanup_${node}_%j.err" \
            --export=ALL,SCRATCH_ROOT="$SCRATCH_ROOT" \
            $CLEANUP_EXTRA_SBATCH_ARGS \
            "$CLEANUP_SCRIPT"
    )

    cleanup_job_ids+=("$cleanup_job_id")
    echo -e "cleanup\t${cleanup_job_id}\t${node}" >> "${RESULTS_ROOT}/submitted_jobs.tsv"
done

echo "------------------------------------------------------------"
echo "Submitted:"
echo "  GPU array: ${ARRAY_JOB_ID}"
echo "  cleanup:   ${cleanup_job_ids[*]}"
echo
echo "Array range:"
echo "  0-${ARRAY_MAX}%${MAX_ACTIVE}"
echo
echo "Results:"
echo "  ${RESULTS_ROOT}"
echo
echo "Scratch path used independently on each node:"
echo "  ${SCRATCH_ROOT}"
echo
echo "Monitor:"
echo "  squeue -j ${ARRAY_JOB_ID},$(IFS=,; echo "${cleanup_job_ids[*]}")"
echo
echo "Logs:"
echo "  ${SLURM_LOG_DIR}"
echo "------------------------------------------------------------"
