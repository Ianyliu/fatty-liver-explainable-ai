#!/usr/bin/env bash
#SBATCH --job-name=xai-replay-ridge
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
project_root="${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$project_root"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$project_root/outputs/.matplotlib"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-1}"
# Validation only by default; --execute must be supplied explicitly.
.venv/bin/python scripts/reproduce_patient.py "$@"
