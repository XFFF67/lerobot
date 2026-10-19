#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTRY="$1"
shift
ENV_NAME="${MOBILE_OPENARM_CONDA_ENV:-hei-rebot-vr}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -z "${CONDA_PREFIX:-}" || "$(basename "$CONDA_PREFIX")" != "$ENV_NAME" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
      CONDA_BASE="$("$CONDA_EXE" info --base)"
    elif command -v conda >/dev/null 2>&1; then
      CONDA_BASE="$(conda info --base)"
    else
      echo "Conda not found. Set PYTHON=/path/to/hei-rebot-vr/bin/python." >&2
      exit 1
    fi
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
  fi
  PYTHON=python
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec env -u LD_LIBRARY_PATH "$PYTHON" "$SCRIPT_DIR/$ENTRY" "$@"
