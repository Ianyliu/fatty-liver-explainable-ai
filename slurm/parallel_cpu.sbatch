#!/usr/bin/env bash
#SBATCH --job-name=xai-parallel-cpu
#SBATCH --array=0-29%4
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/outputs/.matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python -u scripts/parallel_experiments.py cpu \
  --plan "${XAI_PARALLEL_PLAN:?}" --task "${SLURM_ARRAY_TASK_ID:?}" --threads "$OMP_NUM_THREADS"
