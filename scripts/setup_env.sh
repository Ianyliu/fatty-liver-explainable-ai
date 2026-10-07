#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
uv_bin="${UV_BIN:-$project_root/.uv/bin/uv}"
if [[ ! -x "$uv_bin" ]]; then
    uv_bin="$(command -v uv || true)"
fi
if [[ -z "$uv_bin" ]]; then
    echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH:-/tmp}/fatty-liver-uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$UV_CACHE_DIR/python}"
export UV_LINK_MODE=copy
"$uv_bin" python install 3.9.25
"$uv_bin" venv --python 3.9.25 --allow-existing .venv
# glmnet 2.2.1 imports numpy.distutils before declaring build requirements.
"$uv_bin" pip install --python .venv/bin/python 'numpy==1.22.4' 'setuptools==59.8.0' 'wheel==0.45.1'
"$uv_bin" sync --frozen
if [[ ! -e .env ]]; then
    cp .env.example .env
fi
mkdir -p data checkpoints outputs logs
