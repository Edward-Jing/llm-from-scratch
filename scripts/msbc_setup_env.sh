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

VENV_PYTHON="$VENV_DIR/bin/python"

"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install "tokenizers>=0.19" "transformers>=4.44" "pytest>=8.0"
"$VENV_PYTHON" -m pip install --no-deps -e .

"$VENV_PYTHON" scripts/check_cuda_env.py
"$VENV_PYTHON" -m unittest discover -v tests
