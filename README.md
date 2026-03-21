# metalcheck

PyTorch environment for macOS Metal (MPS) GPU acceleration on Apple Silicon.

## Prerequisites

- macOS 12.3+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# Clone and enter the project
cd metalcheck

# Create venv (seeded with pip) and install all dependencies
uv venv --seed
uv sync --all-groups

# Register the Jupyter kernel (for VS Code notebook support)
uv run python -m ipykernel install --user --name metalcheck --display-name "Python 3 (metalcheck)"

# Verify MPS/Metal is working
uv run check-mps
```

## Project Structure

```
metalcheck/
├── src/metalcheck/       # Python package
│   ├── __init__.py       # Package exports
│   ├── device.py         # MPS device detection & utilities
│   └── utils.py          # Timing helpers, benchmarks, system info
├── notebooks/            # Jupyter notebooks
│   └── 01_mps_quickstart.ipynb
├── scripts/              # Standalone scripts
│   └── check_mps.py
├── tests/                # pytest tests
│   └── test_device.py
├── .vscode/              # VS Code settings & recommended extensions
├── pyproject.toml        # Project config & dependencies
└── .env                  # Sets UV_VENV_SEED=true
```

## Usage

### CLI

```bash
# Run the MPS check (installed as entry point)
uv run check-mps

# Run the benchmark script
uv run python scripts/check_mps.py
```

### In Python

```python
from metalcheck import get_device, is_mps_available, device_info
from metalcheck.utils import benchmark_matmul

device = get_device()          # Returns MPS device if available, else CPU
info = device_info()           # Dict of device/system info
results = benchmark_matmul()   # Benchmark matmul on best device
```

### Notebooks

Open `notebooks/01_mps_quickstart.ipynb` in VS Code. Select the **"Python 3 (metalcheck)"** kernel when prompted.

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
uv run pytest
```

## Dependencies

| Package | Purpose |
|---------|---------|
| torch | PyTorch with MPS backend |
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
