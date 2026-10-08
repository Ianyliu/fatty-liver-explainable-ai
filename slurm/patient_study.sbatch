#!/usr/bin/env bash
#SBATCH --job-name=xai-n1-study
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
project_root="${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
cd "$project_root"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$project_root/outputs/.matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python -u scripts/patient_study.py run \
  --plan "${XAI_STUDY_PLAN:?Set the prepared private plan.json path}" \
  --seed "${SLURM_ARRAY_TASK_ID:?}" --device "${XAI_DEVICE:-cuda:0}" \
  --threads "$OMP_NUM_THREADS"
