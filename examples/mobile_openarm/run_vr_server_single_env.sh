#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-$(command -v python)}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the tested environment first: conda activate openarm-runtime" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec env LD_LIBRARY_PATH="$CONDA_PREFIX/lib" "$PYTHON_BIN" "$SCRIPT_DIR/vr_server.py" "$@"
