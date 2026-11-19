#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_NAME="${OPENARM_RUNTIME_ENV:-openarm-runtime}"

if [[ $# -gt 0 ]]; then
    if [[ "$1" != "--name" || $# -ne 2 ]]; then
        echo "Usage: $0 [--name ENV_NAME]" >&2
        exit 2
    fi
    ENV_NAME="$2"
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda not found. Install Miniconda/Anaconda first." >&2
    exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Refusing to modify existing Conda environment: $ENV_NAME" >&2
    echo "Choose another name with --name, or remove it manually after inspection." >&2
    exit 1
fi

echo "[1/5] Creating isolated environment: $ENV_NAME"
conda create -y -n "$ENV_NAME" \
    --override-channels \
    -c conda-forge \
    --strict-channel-priority \
    'python=3.12' \
    'numpy=2.2.6' \
    'pinocchio=3.9.0' \
    'casadi=3.7.0' \
    'pip=26.1.2' \
    'setuptools=80.10.2' \
    'wheel=0.47.0'
PYTHON_BIN="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
ENV_PREFIX="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"

echo "[2/5] Installing CPU-only PyTorch"
"$PYTHON_BIN" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.11.0+cpu' \
    'torchvision==0.26.0+cpu'

echo "[3/5] Installing pinned runtime packages"
"$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/runtime_requirements.txt"

echo "[4/5] Installing this exact LeRobot checkout and OpenArm dependencies"
"$PYTHON_BIN" -m pip install \
    -c "$SCRIPT_DIR/runtime_requirements.txt" \
    -e "$REPO_ROOT[openarms,pyserial-dep,pyzmq-dep]"

echo "[5/5] Checking the merged runtime"
env LD_LIBRARY_PATH="$ENV_PREFIX/lib" PYTHONPATH="$REPO_ROOT/src" \
    "$PYTHON_BIN" "$SCRIPT_DIR/check_openarm_runtime.py"

echo
echo "Environment ready. Activate it with:"
echo "  conda activate $ENV_NAME"
echo "Then run the *_single_env.sh launchers during validation."
