#!/usr/bin/env bash
#SBATCH --job-name=xai-jsm-review
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/outputs/proceedings_2026/.matplotlib"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec .venv/bin/python -u scripts/run_proceedings_review.py --manifest "${XAI_PROCEEDINGS_MANIFEST:?}"
