# HWKit GUI Setup

Installs [IBM aihwkit](https://github.com/IBM/aihwkit) (CPU build) into a dedicated conda
environment named `aihwkit`, using Miniforge (native arm64 conda distribution).

## Usage

```bash
./setup.sh
```

This will:
1. Install Miniforge via Homebrew if not already present.
2. Create a conda environment `aihwkit` with Python 3.11.
3. `pip install aihwkit` (installs PyTorch as a dependency).
4. Set `KMP_DUPLICATE_LIB_OK=TRUE` on the environment — required because aihwkit
   and PyTorch each bundle their own OpenMP runtime, which otherwise crashes on
   import on macOS.

## Verify

```bash
conda activate aihwkit
python -c "import aihwkit; print(aihwkit.__version__)"
```

If conda commands aren't recognized in a terminal, open a new terminal window/tab
(conda init writes to `~/.zshrc`, which only takes effect in new shells).
