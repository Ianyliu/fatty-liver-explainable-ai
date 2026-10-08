#!/usr/bin/env bash
#SBATCH --job-name=xai-p0-full
#SBATCH --array=0-374%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/outputs/.matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python -u scripts/full_cohort_study.py run \
  --plan "${XAI_FULL_PLAN:?Set the prepared full_cohort_plan.json}" \
  --task "${SLURM_ARRAY_TASK_ID:?}" --device "${XAI_DEVICE:-cuda:0}" --threads "$OMP_NUM_THREADS"
