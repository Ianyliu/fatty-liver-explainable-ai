#!/usr/bin/env bash
#SBATCH --job-name=xai-pilot
#SBATCH --array=0-9%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
project_root="${XAI_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$project_root"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$project_root/outputs/.matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Manifest is the private candidate_patients.csv produced by the fresh audit.
# Every task verifies the entire ten-patient input gate before loading weights.
.venv/bin/python -u - "${XAI_PATIENT_MANIFEST:?Set the audited candidate_patients.csv path}" <<'PY'
import csv
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path.cwd() / "scripts"))
import project_paths as paths
from smoke_patient import patient_record

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
patients = [row["MI_ID"] for row in rows[:10]]
if len(patients) != 10 or len(set(patients)) != 10:
    raise SystemExit("Pilot requires ten distinct complete patients; rerun the input audit after upload")
for patient in patients:
    record, images = patient_record(patient, paths)
    if float(record["liver_fatty"]) <= 0 or len(images) < 20:
        raise SystemExit("Manifest contains a patient outside the positive / >=20-image cohort")
index = int(os.environ["SLURM_ARRAY_TASK_ID"])
if not 0 <= index < 10:
    raise SystemExit("Array index must be 0–9")
output = f"outputs/pilot/slurm-{os.environ['SLURM_ARRAY_JOB_ID']}/task-{index}"
subprocess.run([
    sys.executable, "-u", "scripts/smoke_patient.py", "--patient", patients[index],
    "--device", os.environ.get("XAI_DEVICE", "cpu"), "--threads", os.environ["OMP_NUM_THREADS"],
    "--output", output,
], check=True)
PY
