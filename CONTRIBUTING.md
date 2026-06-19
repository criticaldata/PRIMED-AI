# Contributing to PRIMED-AI

## Local Setup
1. Clone the repository.
2. Run `pip install -e .` or `make install`.
3. Ensure Python 3.10+ and PyTorch with CUDA are installed.

## Data Access
* **MIMIC Data Paths:** Ensure you have ORCD cluster access and signed DUAs for MIMIC-IV, MIMIC-IV-Echo, and MIMIC-IV-ECG. 
* **NEVER commit raw data.** (Check `.gitignore` for ignored directories like `data/`, `logs/`, etc.)