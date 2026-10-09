#!/usr/bin/env bash
#SBATCH --job-name=xai-recovery-stage
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/outputs/.matplotlib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
case "${XAI_RECOVERY_STAGE:?}" in
  summarize|report)
    .venv/bin/python -u scripts/full_secondary_recovery.py "$XAI_RECOVERY_STAGE" --plan "${XAI_RECOVERY_PLAN:?}" ;;
  integrate)
    .venv/bin/python -u scripts/integrate_full_secondary_recovery.py --plan "${XAI_RECOVERY_PLAN:?}" --manuscript "${XAI_MANUSCRIPT_TARGET:?}" ;;
  *) exit 2 ;;
esac
