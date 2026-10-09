#!/usr/bin/env bash
#SBATCH --job-name=xai-loo-recovery
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python -u scripts/full_secondary_recovery.py loo --plan "${XAI_RECOVERY_PLAN:?}" --task "${SLURM_ARRAY_TASK_ID:?}"
