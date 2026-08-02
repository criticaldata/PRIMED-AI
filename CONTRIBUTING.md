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

## Reproducibility

`data/`, `results/`, `probes/`, and `logs/` are gitignored, so the cohort, embeddings, checkpoints,
and metrics behind the reported numbers are **not** in this repository. Only code and configs are
versioned here. To rerun anything end to end you need all of:

| Requirement | Where it comes from |
|---|---|
| PhysioNet credentialing + signed DUAs | [MIMIC-IV-Echo](https://physionet.org/content/mimic-iv-echo/0.1/), [MIMIC-IV-ECG](https://physionet.org/content/mimic-iv-ecg/1.0/), [MIMIC-IV](https://physionet.org/content/mimic-iv/3.1/) |
| Echo embeddings | [MITCriticalData/mimic-iv-echo-jepa-embeddings](https://huggingface.co/datasets/MITCriticalData/mimic-iv-echo-jepa-embeddings) — gated, PhysioNet credentials required. See [docs/embeddings.md](docs/embeddings.md) |
| ECG embeddings | HuBERT-ECG Parquet on the ORCD pool, or re-extract with [`scripts/extract_ecg_embeddings.py`](scripts/extract_ecg_embeddings.py) |
| GPU + storage | ORCD cluster (H200 or equivalent for extraction; probe training is minutes on CPU/GPU) |

Runs are seeded (`--seed`, default 42) and every evaluation script writes its input paths, seed, and
model dimensions alongside the metrics, so a result JSON records what produced it.

**When you report a number**, state whether it came from a run you actually executed and on what
data. Never present an estimate, a partial run, or a synthetic-demo output as a measured result —
`src/primed_ai/failure/demo.py` generates synthetic data for harness validation only, and anything
it produces must be labelled as such.
