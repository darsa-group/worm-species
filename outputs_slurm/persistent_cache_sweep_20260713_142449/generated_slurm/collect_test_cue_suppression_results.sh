#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1

set -euo pipefail

: "${RESULTS_ROOT:?}"
: "${PROJECT_SRC:?}"
: "${RESULT_COLLECTOR:?}"
: "${CONDA_SH:?}"
: "${CONDA_ENV:?}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

python "${PROJECT_SRC}/${RESULT_COLLECTOR}" "$RESULTS_ROOT"
