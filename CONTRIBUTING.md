# Contributing to PRIMED-AI

## Local Setup

### With pip
1. Clone the repository.
2. Run `pip install -e ".[dev]"` or `make install`.
3. Ensure Python 3.10+ and PyTorch with CUDA are installed.

### With uv (recommended — faster, reproducible)
1. Clone the repository.
2. Run `uv pip install -e ".[dev]"` or `make install-uv`.
3. Ensure Python 3.10+ is active in your uv environment.

## Running tests
```bash
make test        # runs pytest tests/
```

## Linting
```bash
make lint        # runs pre-commit on all files (ruff + formatting)
```

## Data Access
* **MIMIC Data Paths:** Ensure you have ORCD cluster access and signed DUAs for MIMIC-IV, MIMIC-IV-Echo, and MIMIC-IV-ECG.
* **NEVER commit raw data.** (Check `.gitignore` for ignored directories like `data/`, `logs/`, etc.)
