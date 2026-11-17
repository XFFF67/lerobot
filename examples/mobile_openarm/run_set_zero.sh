#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MOTOR_PYTHON="${OPENARM_MOTOR_PYTHON:-$HOME/anaconda3/envs/lerobot312/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec env -u LD_LIBRARY_PATH "$MOTOR_PYTHON" -m lerobot.scripts.openarm_zero "$@"
