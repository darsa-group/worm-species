#!/bin/bash
#--------------------------------------------------------------------------#
# Launcher for independent 1-GPU SLURM sweep jobs
#
# Run this from your project root, for example:
#   cd ~/worm-species
#   bash submit_worm_multitask_sweep_jobs.sh
#
# This script:
#   1. Creates one run-spec file per sweep entry.
#   2. Submits one independent SLURM job per run.
#   3. Requests 1 GPU per job.
#   4. Keeps at most MAX_ACTIVE jobs active at once.
#   5. Lets each job copy data to scratch, train, copy results back, and clean scratch.
#--------------------------------------------------------------------------#

set -euo pipefail

#--------------------------------------------------------------------------#
# USER SETTINGS
#--------------------------------------------------------------------------#

# Run this launcher from the project root. Usually this is enough:
PROJECT_SRC="${PROJECT_SRC:-$(pwd)}"

# Dataset location on the shared filesystem.
# This directory should contain the image folders and metadata CSV used by config.yaml.
DATA_SRC="${DATA_SRC:-/usr/home/qgg/mehrot/worm-species/data}"

# Config and training script, relative to PROJECT_SRC.
BASE_CONFIG="${BASE_CONFIG:-config.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_multitask_masked.py}"

# Conda setup used by each child job.
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-wormspecies}"

# Keep only this many SLURM jobs active at once.
# Active means pending or running.
MAX_ACTIVE="${MAX_ACTIVE:-2}"

# Results are copied back here.
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_SRC}/outputs_slurm/individual_jobs_$(date +%Y%m%d_%H%M%S)}"

# Single-run job script. This must exist.
SINGLE_JOB_SCRIPT="${SINGLE_JOB_SCRIPT:-${PROJECT_SRC}/slurm_single_worm_multitask_job.sh}"

# Polling interval while waiting for a free slot.
POLL_SECONDS="${POLL_SECONDS:-30}"

#--------------------------------------------------------------------------#
# SWEEP DEFINITIONS
#
# Each line is one independent run.
# These are passed to:
#   python train_multitask_masked.py --config config.yaml --override ...
#
# Edit these entries for your real sweep.
#--------------------------------------------------------------------------#

SWEEP_OVERRIDES=(
"model.name=efficientnet_b0 data.image_col=rel_path_seg data.split_target_col=__taxon_for_split__ training.lr=0.0001"
"model.name=efficientnet_b0 data.image_col=rel_path_raw data.split_target_col=__taxon_for_split__ training.lr=0.0001"
"model.name=resnet18 data.image_col=rel_path_seg data.split_target_col=__taxon_for_split__ training.lr=0.0001"
"model.name=resnet18 data.image_col=rel_path_raw data.split_target_col=__taxon_for_split__ training.lr=0.0001"
)

#--------------------------------------------------------------------------#
# Checks
#--------------------------------------------------------------------------#

if [[ ! -d "$PROJECT_SRC" ]]; then
    echo "ERROR: PROJECT_SRC does not exist: $PROJECT_SRC" >&2
    exit 1
fi

if [[ ! -d "$DATA_SRC" ]]; then
    echo "ERROR: DATA_SRC does not exist: $DATA_SRC" >&2
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

if [[ ! -f "$SINGLE_JOB_SCRIPT" ]]; then
    echo "ERROR: Single-job script not found: $SINGLE_JOB_SCRIPT" >&2
    echo "Copy slurm_single_worm_multitask_job.sh into your project root first." >&2
    exit 1
fi

mkdir -p "$RESULTS_ROOT/run_specs" "$RESULTS_ROOT/slurm_logs"

echo "PROJECT_SRC: $PROJECT_SRC"
echo "DATA_SRC: $DATA_SRC"
echo "RESULTS_ROOT: $RESULTS_ROOT"
echo "MAX_ACTIVE: $MAX_ACTIVE"
echo "SINGLE_JOB_SCRIPT: $SINGLE_JOB_SCRIPT"

# Save sweep plan.
{
    echo -e "run_index\trun_name\toverrides"
    for i in "${!SWEEP_OVERRIDES[@]}"; do
        run_name=$(printf "run_%03d" "$i")
        echo -e "${i}\t${run_name}\t${SWEEP_OVERRIDES[$i]}"
    done
} > "${RESULTS_ROOT}/sweep_plan.tsv"

# Create one argument file per run.
for i in "${!SWEEP_OVERRIDES[@]}"; do
    run_name=$(printf "run_%03d" "$i")
    printf "%s\n" "${SWEEP_OVERRIDES[$i]}" > "${RESULTS_ROOT}/run_specs/${run_name}.args"
done

submitted_jobs=()

active_count() {
    local count=0

    for jobid in "${submitted_jobs[@]:-}"; do
        # If squeue returns a line, the job is still pending/running/configuring/etc.
        if squeue -h -j "$jobid" | grep -q .; then
            count=$((count + 1))
        fi
    done

    echo "$count"
}

wait_for_slot() {
    while [[ "$(active_count)" -ge "$MAX_ACTIVE" ]]; do
        echo "Currently active jobs: $(active_count). Waiting ${POLL_SECONDS}s for a free slot..."
        sleep "$POLL_SECONDS"
    done
}

#--------------------------------------------------------------------------#
# Submit jobs
#--------------------------------------------------------------------------#

for i in "${!SWEEP_OVERRIDES[@]}"; do
    wait_for_slot

    run_name=$(printf "run_%03d" "$i")
    run_spec_file="${RESULTS_ROOT}/run_specs/${run_name}.args"

    echo "Submitting ${run_name}"
    echo "  spec: ${run_spec_file}"

    jobid=$(
        sbatch --parsable \
            --job-name="worm_${run_name}" \
            --output="${RESULTS_ROOT}/slurm_logs/${run_name}_%j.out" \
            --error="${RESULTS_ROOT}/slurm_logs/${run_name}_%j.err" \
            --export=ALL,RUN_INDEX="$i",RUN_NAME="$run_name",RUN_SPEC_FILE="$run_spec_file",PROJECT_SRC="$PROJECT_SRC",DATA_SRC="$DATA_SRC",RESULTS_ROOT="$RESULTS_ROOT",BASE_CONFIG="$BASE_CONFIG",TRAIN_SCRIPT="$TRAIN_SCRIPT",CONDA_SH="$CONDA_SH",CONDA_ENV="$CONDA_ENV" \
            "$SINGLE_JOB_SCRIPT"
    )

    submitted_jobs+=("$jobid")
    echo -e "${run_name}\t${jobid}" >> "${RESULTS_ROOT}/submitted_jobs.tsv"
    echo "  job id: ${jobid}"
done

#--------------------------------------------------------------------------#
# Monitor until all submitted jobs leave the queue
#--------------------------------------------------------------------------#

echo "All jobs submitted."
echo "Submitted job IDs: ${submitted_jobs[*]}"
echo "Waiting until all submitted jobs have finished..."

while [[ "$(active_count)" -gt 0 ]]; do
    echo "Active jobs: $(active_count)"
    squeue -h -j "$(IFS=,; echo "${submitted_jobs[*]}")" -o "%.18i %.9P %.32j %.8u %.2t %.10M %.6D %R" || true
    sleep "$POLL_SECONDS"
done

echo "All submitted jobs have left the SLURM queue."
echo "Results should be in: $RESULTS_ROOT"

# Basic final status summary.
echo
echo "Run status files:"
find "$RESULTS_ROOT" -name run_status.txt -print -exec cat {} \; || true
