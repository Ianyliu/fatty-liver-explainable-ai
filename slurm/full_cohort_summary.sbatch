#!/usr/bin/env bash
#SBATCH --job-name=xai-p0-full-summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/outputs/.matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python -u scripts/full_cohort_study.py summarize --plan "${XAI_FULL_PLAN:?}"
