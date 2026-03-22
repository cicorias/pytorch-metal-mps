# metalcheck

PyTorch GPU benchmarks for macOS Metal (MPS) and AMD ROCm acceleration.

## Overview

This repo contains benchmark notebooks and utilities for comparing GPU vs CPU performance with PyTorch on two platforms:

### Apple Silicon (MPS / Metal)

The MPS notebooks were run on a MacBook Neo 8GB — essentially an iPhone chip:

- Apple A18 Pro
- macOS 26.3.2

```text
  Model Name:	MacBook Neo
  Model Identifier:	Mac17,5
  Model Number:	MHFG4LL/A
  Chip:	Apple A18 Pro
  Total Number of Cores:	6 (2 Performance and 4 Efficiency)
  Memory:	8 GB
  System Firmware Version:	13822.81.10
  OS Loader Version:	13822.81.10

  System Version:	macOS 26.3.2 (25D2140)
  Kernel Version:	Darwin 25.3.0
```

### AMD Ryzen AI HX 370 (ROCm)

The ROCm notebooks run inside a **devcontainer** on a Framework Laptop 13:

- AMD Ryzen AI 9 HX 370 (8+16) @ 5.16 GHz
- AMD Radeon 890M Graphics (Integrated, RDNA 3.5)
- Omarchy 3.4.2 / Linux 6.19.8-arch1-1
- ROCm 7.1 (via `rocm/pytorch` Docker image)

```text
  PC:  Laptop 13 (AMD Ryzen AI 300 Series) (A9)
  CPU: AMD Ryzen AI 9 HX 370 (8+16) @ 5.16 GHz
  GPU: AMD Radeon 890M Graphics [Integrated]
  RAM: 62.08 GiB
  OS:  Omarchy 3.4.2 / Linux 6.19.8-arch1-1
```

## Prerequisites

### macOS (MPS)

- macOS 12.3+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Linux / AMD (ROCm via Devcontainer)

- Docker with GPU support
- AMD ROCm drivers on the host system (`rocm-core` package)
- VS Code with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- AMD GPU with ROCm support (e.g., Radeon 890M, RX 7000 series)

## Quick Start

### macOS (MPS)

```bash
cd metalcheck

# Create venv and install all dependencies
uv venv --seed
uv sync --all-groups

# Register the Jupyter kernel (for VS Code notebook support)
uv run python -m ipykernel install --user --name metalcheck --display-name "Python 3 (metalcheck)"

# Verify MPS/Metal is working
uv run check-mps
```

### Linux / AMD (ROCm via Devcontainer)

1. Open this folder in VS Code
2. When prompted, click **"Reopen in Container"** (or use Command Palette: `Dev Containers: Reopen in Container`)
3. The container builds with PyTorch 2.8.0 + ROCm 7.1 pre-installed
4. Dependencies install automatically via `pip install -e '.[dev]'`
5. Select the Jupyter kernel at `/opt/venv/bin/python`

```bash
# Verify ROCm is working (inside the container)
check-rocm
```

> **Note:** The devcontainer sets `HSA_OVERRIDE_GFX_VERSION=11.0.0` automatically for Radeon 890M (gfx1150) compatibility.

## Project Structure

```text
metalcheck/
├── .devcontainer/        # ROCm devcontainer config
│   └── devcontainer.json
├── src/metalcheck/       # Python package
│   ├── __init__.py       # Package exports
│   ├── device.py         # MPS / ROCm / CUDA device detection
│   └── utils.py          # Timing helpers, benchmarks, system info
├── notebooks/            # Jupyter notebooks
│   ├── 01_mps_quickstart.ipynb
│   ├── 01_mps_quickstart_ryzen_ai_hx370.ipynb
│   ├── 02_neural_network_benchmark.ipynb
│   ├── 02_neural_network_benchmark_ryzen_ai_hx370.ipynb
│   ├── 03_convolution_benchmark.ipynb
│   ├── 03_convolution_benchmark_ryzen_ai_hx370.ipynb
│   ├── 04_elementwise_benchmark.ipynb
│   └── 04_elementwise_benchmark_ryzen_ai_hx370.ipynb
├── scripts/              # Standalone scripts
│   ├── check_mps.py
│   └── check_rocm.py
├── tests/                # pytest tests
│   └── test_device.py
├── .vscode/              # VS Code settings & recommended extensions
├── pyproject.toml        # Project config & dependencies
└── .env                  # Sets UV_VENV_SEED=true
```

## Usage

### CLI

```bash
# macOS: Run the MPS check
uv run check-mps

# Linux (in devcontainer): Run the ROCm check
check-rocm

# Run benchmark scripts
uv run python scripts/check_mps.py    # macOS
python scripts/check_rocm.py           # devcontainer
```

### In Python

```python
from metalcheck import get_device, is_mps_available, is_rocm_available, device_info
from metalcheck.utils import benchmark_matmul

device = get_device()          # Returns best GPU (MPS > CUDA/ROCm > CPU)
info = device_info()           # Dict of device/system info
results = benchmark_matmul()   # Benchmark matmul on best device
```

### Notebooks

- **MPS notebooks** (`01_mps_quickstart.ipynb`, etc.): Run on macOS with the `metalcheck` kernel
- **ROCm notebooks** (`*_ryzen_ai_hx370.ipynb`): Run inside the devcontainer with the `/opt/venv/bin/python` kernel

## VS Code Setup

When you open this project, VS Code will recommend installing:

- **Python** + **Pylance** — language support & type checking
- **Jupyter** — notebook support (renderers + keymap)
- **Ruff** — fast linter & formatter

The workspace settings (`.vscode/settings.json`) are pre-configured to:

- Use the `.venv` interpreter
- Enable Ruff format-on-save
- Enable pytest discovery

## Running Tests

```bash
# macOS
uv run pytest

# Devcontainer
pytest
```

Tests auto-skip based on GPU availability: MPS tests skip on Linux, ROCm tests skip on macOS.

## Dependencies

| Package | Purpose |
|---------|---------|
| torch | PyTorch with MPS / ROCm backend |
| torchvision | Vision models & transforms |
| torchaudio | Audio processing |
| numpy | Numerical computing |
| matplotlib | Plotting |
| seaborn | Statistical visualization |
| pandas | Data manipulation |
| scikit-learn | Machine learning utilities |
| tqdm | Progress bars |
| jupyter + ipykernel | Notebook support (dev) |
| pytest | Testing (dev) |
| ruff | Linting & formatting (dev) |
