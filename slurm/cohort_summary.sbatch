#!/usr/bin/env bash
#SBATCH --job-name=xai-p0-summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
project_root="${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
cd "$project_root"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$project_root/outputs/.matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python -u scripts/cohort_study.py summarize \
  --cohort "${XAI_COHORT_PLAN:?Set the frozen cohort_plan.json}"
