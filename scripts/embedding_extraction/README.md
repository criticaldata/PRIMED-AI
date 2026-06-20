# Embedding Extraction Scripts

Full pipeline for extracting frozen EchoJEPA / V-JEPA2 embeddings from MIMIC-IV-Echo.

For ORCD paths, loading examples, and output format details see [`docs/embeddings.md`](../../docs/embeddings.md).

## Files

| Script | Stage | Description |
|--------|-------|-------------|
| `convert_dicom.py` | Stage 1 | DICOM → MP4 conversion (multi-frame echo DICOMs, 256×256) |
| `extract_embeddings.py` | Stage 2 | Frozen V-JEPA2 / EchoJEPA forward pass over MP4 files; saves `.pt` per folder |
| `extract_slurm.sh` | Stage 2 | SLURM array script for V-JEPA2 natural models (`vitl`, `vith`, `vitg`, `vitg-384`) |
| `extract_echo_slurm.sh` | Stage 2 | SLURM array script for EchoJEPA fine-tuned models (`echo-vitl-*`, `echo-vitb-*`) |
| `merge_embeddings.py` | Stage 2b | Merge per-folder `.pt` files into a single `{model}_embeddings_all.pt` |
| `to_parquet.py` | Stage 3 | Convert merged `.pt` to sharded Parquet with MIMIC metadata joined |

## Quick commands (ORCD cluster)

```bash
module load miniforge/24.3.0-0
conda activate vjepa2-312

# Stage 1 — convert DICOMs to MP4 (run once)
python scripts/embedding_extraction/convert_dicom.py

# Stage 2 — submit SLURM array (10 folders × 1 L40S GPU)
sbatch scripts/embedding_extraction/extract_echo_slurm.sh echo-vitl-mimic117

# Stage 2b — merge per-folder outputs
python scripts/embedding_extraction/merge_embeddings.py --model echo-vitl-mimic117

# Stage 3 — convert to Parquet with MIMIC metadata
python scripts/embedding_extraction/to_parquet.py --model echo-vitl-mimic117
```

## Available models

| Alias | Weights file | embed_dim | Notes |
|-------|-------------|-----------|-------|
| `vitl` | `vitl.pt` | 1024 | V-JEPA2 natural pretrain (Meta AI) |
| `vith` | `vith.pt` | 1280 | V-JEPA2 natural pretrain |
| `vitg` | `vitg.pt` | 1408 | V-JEPA2 natural pretrain |
| `vitg-384` | `vitg-384.pt` | 1408 | V-JEPA2 natural pretrain, 384px |
| `echo-vitl-scratch` | `vitl-scratch-pt-210-c25.pt` | 1024 | EchoJEPA ViT-L, trained from scratch |
| `echo-vitl-mimic100` | `vjepa21_vitl_mimic_pt100.pt` | 1024 | V-JEPA2.1 ViT-L, MIMIC 100-epoch FT |
| `echo-vitl-mimic117` | `vjepa21_vitl_mimic_pt117.pt` | 1024 | V-JEPA2.1 ViT-L, MIMIC 117-epoch FT **(recommended)** |
| `echo-vitl-vmix22m` | `vitl-vmix22m-pt220-c55.pt` | 1024 | ViT-L VMix-22M pretrain + MIMIC FT |
| `echo-vitb-mimic169` | `vjepa2_1_vitb_mimic_pt169_c60.pt` | 768 | V-JEPA2.1 ViT-B, MIMIC 169-epoch FT |

All weights: `/orcd/pool/006/lceli_shared/weights/`

## Cluster setup

| Field | Value |
|-------|-------|
| Partition | `mit_preemptable` |
| GPU | `gpu:l40s:1` per task |
| Array | `0–9` (folders `p10`–`p19`) |
| Max concurrent | 4 tasks (QOSMaxGRESPerUser cap) |
| Job time limit | 2 days (preemptable; jobs resume via embedding cache) |
