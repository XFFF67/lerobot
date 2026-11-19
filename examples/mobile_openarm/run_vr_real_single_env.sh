#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON:-$(command -v python)}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Activate the tested environment first: conda activate openarm-runtime" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec env LD_LIBRARY_PATH="$CONDA_PREFIX/lib" "$PYTHON_BIN" -m lerobot.scripts.openarm_vr \
    --ik-python "$PYTHON_BIN" "$@"
