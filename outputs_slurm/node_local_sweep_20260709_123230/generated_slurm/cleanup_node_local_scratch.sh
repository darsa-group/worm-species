#!/bin/bash
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
