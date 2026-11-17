#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE_PYTHON="${OPENARM_MOBILE_PYTHON:-$HOME/anaconda3/envs/hei-rebot-vr/bin/python}"
exec env -u LD_LIBRARY_PATH "$MOBILE_PYTHON" "$SCRIPT_DIR/vr_mobile_bridge.py" "$@"
