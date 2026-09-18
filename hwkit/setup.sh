#!/usr/bin/env bash
# Sets up a conda environment with IBM aihwkit (CPU build, macOS arm64).
set -euo pipefail

ENV_NAME="aihwkit"
PYTHON_VERSION="3.11"
CONDA_BASE="/opt/homebrew/Caskroom/miniforge/base"

if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Install it from https://brew.sh first." >&2
    exit 1
fi

if [ ! -d "$CONDA_BASE" ]; then
    echo "Installing Miniforge via Homebrew..."
    brew install miniforge
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment '${ENV_NAME}' (python ${PYTHON_VERSION})..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

echo "Installing aihwkit..."
pip install -r requirements.txt

# aihwkit and PyTorch both bundle their own OpenMP runtime, which crashes on
# import on macOS unless this is set. See: https://github.com/IBM/aihwkit issues.
conda env config vars set KMP_DUPLICATE_LIB_OK=TRUE -n "$ENV_NAME"

echo
echo "Done. Reactivate the environment to pick up the env var, then verify with:"
echo "  conda activate ${ENV_NAME}"
echo "  python -c \"import aihwkit; print(aihwkit.__version__)\""
