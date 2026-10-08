#!/usr/bin/env bash
#SBATCH --job-name=xai-smoke
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
project_root="${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$project_root"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$project_root/outputs/.matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
# CPU by default. For GPU, request one GPU using your partition's documented
# sbatch options and export XAI_DEVICE=cuda:0 (Slurm remaps visible devices).
.venv/bin/python -u scripts/smoke_patient.py \
    --device "${XAI_DEVICE:-cpu}" --threads "$OMP_NUM_THREADS" \
    --output "outputs/smoke/slurm-${SLURM_JOB_ID:?}" "$@"
