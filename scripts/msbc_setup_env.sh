#!/bin/bash -l
# Set up a user-owned Python environment on MSBC.
#
# Run from the repository root:
#   bash scripts/msbc_setup_env.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/llm-from-scratch}"

cd "$PROJECT_DIR"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate pytorch
else
    echo "conda was not found on PATH. Run this after logging into MSBC." >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python scripts/check_cuda_env.py
python -m unittest discover -v tests
