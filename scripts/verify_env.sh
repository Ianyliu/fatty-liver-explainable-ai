#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
uv_bin="${UV_BIN:-$project_root/.uv/bin/uv}"
if [[ ! -x "$uv_bin" ]]; then
    uv_bin="$(command -v uv)"
fi
export MPLBACKEND=Agg
export MPLCONFIGDIR="$project_root/outputs/.matplotlib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$uv_bin" lock --check --offline
"$uv_bin" sync --check --offline
"$uv_bin" pip check --python .venv/bin/python
# Direct invocation also works in restricted process sandboxes where uv run
# cannot reap its finished child. The environment was checked just above.
.venv/bin/python scripts/check_environment.py --numerics
