#!/usr/bin/env bash
# setup_env.sh — create the Python environment for run_chain.py.
#
# Idempotent: creates .venv (if missing) and installs requirements.txt.
# Run from the example directory on the target machine:
#
#     ./setup_env.sh
#
# Requires python3 >= 3.10 (override with PYTHON=/path/to/python3 ./setup_env.sh).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: $PYTHON not found (need Python >= 3.10)" >&2
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    echo ">> creating .venv with $PYTHON"
    "$PYTHON" -m venv .venv
fi

echo ">> installing requirements into .venv"
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo ">> environment ready."
.venv/bin/python -c "import numpy, scipy, ase; \
print('   numpy', numpy.__version__, '| scipy', scipy.__version__, '| ase', ase.__version__)"
