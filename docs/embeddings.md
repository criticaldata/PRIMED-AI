# Embedding Extraction — EchoJEPA & ECG-FM

How to access pre-extracted embeddings on the MIT ORCD cluster, load them in your pipeline, and reproduce extraction from scratch.

**Extraction code lives in a separate repo:** [`sebasmos/EchoJEPA-VE`](https://github.com/sebasmos/EchoJEPA-VE). Do not duplicate it here. This document covers paths, formats, and loading only.

See [TECHNICAL.md](../TECHNICAL.md) for the full pipeline context and [OVERVIEW.md](../OVERVIEW.md) for scope decisions.

---

## Table of Contents

1. [Pre-extracted embeddings (use these first)](#1-pre-extracted-embeddings-use-these-first)
2. [Model weights](#2-model-weights)
3. [Loading embeddings in Python](#3-loading-embeddings-in-python)
4. [Encoder configs](#4-encoder-configs)
5. [Reproducing extraction from scratch](#5-reproducing-extraction-from-scratch)
6. [Output format reference](#6-output-format-reference)

---

## 1. Pre-extracted embeddings (use these first)

All embeddings are stored on the shared ORCD pool under:

```
/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo/
```

### Available extraction runs

| Run | Model | Weights | embed_dim | Format | Path |
|-----|-------|---------|-----------|--------|------|
| V-JEPA2 ViT-L (natural) | `vitl` | `vitl.pt` (Meta AI) | 1024 | `.pt` per folder + merged | `mimic-iv-jepa-embedding-pt/` |
| V-JEPA2 ViT-L (natural) | `vitl` | `vitl.pt` (Meta AI) | 1024 | Parquet (HF-ready) | `mimic-iv-echo-jepa-embeddings/jepa-l-embeddings/` |

EchoJEPA fine-tuned runs are **in progress** (SLURM array jobs submitted Jun 20, 2026). Paths will follow the same pattern with the model alias as prefix (e.g. `echo-vitl-mimic117_embeddings_all.pt`).

### Expected EchoJEPA embedding locations (once jobs complete)

```
/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo/
├── echo-vitl-scratch_embeddings_p10.pt ... p19.pt
├── echo-vitl-scratch_embeddings_all.pt
├── echo-vitl-mimic100_embeddings_p10.pt ... p19.pt   # V-JEPA2.1 ViT-L, 100-epoch MIMIC FT
├── echo-vitl-mimic100_embeddings_all.pt
├── echo-vitl-mimic117_embeddings_p10.pt ... p19.pt   # V-JEPA2.1 ViT-L, 117-epoch MIMIC FT
├── echo-vitl-mimic117_embeddings_all.pt
├── echo-vitl-vmix22m_embeddings_p10.pt  ... p19.pt   # ViT-L, VMix-22M pretrain + MIMIC FT
├── echo-vitl-vmix22m_embeddings_all.pt
├── echo-vitb-mimic169_embeddings_p10.pt ... p19.pt   # V-JEPA2.1 ViT-B, 169-epoch MIMIC FT
└── echo-vitb-mimic169_embeddings_all.pt
```

---

## 2. Model weights

All checkpoints live at:

```
/orcd/pool/006/lceli_shared/weights/
```

### V-JEPA2 — natural image pretrain (Meta AI)

| File | Model alias | embed_dim | Source |
|------|-------------|-----------|--------|
| `vitl.pt` | `vitl` | 1024 | `dl.fbaipublicfiles.com/vjepa2/vitl.pt` |
| `vith.pt` | `vith` | 1280 | `dl.fbaipublicfiles.com/vjepa2/vith.pt` |
| `vitg.pt` | `vitg` | 1408 | `dl.fbaipublicfiles.com/vjepa2/vitg.pt` |
| `vitg-384.pt` | `vitg-384` | 1408 | `dl.fbaipublicfiles.com/vjepa2/vitg-384.pt` |

### EchoJEPA — echo-finetuned (Alif Munim, MIT Critical Data)

| File | Model alias | embed_dim | Description |
|------|-------------|-----------|-------------|
| `vitl-scratch-pt-210-c25.pt` | `echo-vitl-scratch` | 1024 | ViT-L trained from scratch on echo, 210 epochs |
| `vjepa21_vitl_mimic_pt100.pt` | `echo-vitl-mimic100` | 1024 | V-JEPA2.1 ViT-L fine-tuned on MIMIC-IV-Echo, 100 epochs |
| `vjepa21_vitl_mimic_pt117.pt` | `echo-vitl-mimic117` | 1024 | V-JEPA2.1 ViT-L fine-tuned on MIMIC-IV-Echo, 117 epochs **(recommended)** |
| `vitl-vmix22m-pt220-c55.pt` | `echo-vitl-vmix22m` | 1024 | ViT-L pretrained on VMix-22M (220 epochs) + MIMIC fine-tune |
| `vjepa2_1_vitb_mimic_pt169_c60.pt` | `echo-vitb-mimic169` | 768 | V-JEPA2.1 ViT-B fine-tuned on MIMIC-IV-Echo, 169 epochs |

> All 5 EchoJEPA checkpoints are on ORCD as of Jun 20, 2026. Access requires ORCD cluster membership — contact the PRIMED-AI leads.

---

## 3. Loading embeddings in Python

### PyTorch `.pt` (fastest for local jobs)

```python
import torch

EMBED_DIR = "/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo/mimic-iv-jepa-embedding-pt"

# Load one folder (p10 = patients p10xxxxx)
embeddings = torch.load(f"{EMBED_DIR}/vitl_embeddings_p10.pt", map_location="cpu")

# Load all folders merged
embeddings = torch.load(f"{EMBED_DIR}/vitl_embeddings_all.pt", map_location="cpu")

# Iterate
for file_path, emb in embeddings.items():
    print(file_path, emb.shape)  # e.g. "p10036337/s91664836/file.mp4" → torch.Size([1024])

# Stack into matrix for probe training
X = torch.stack(list(embeddings.values()))  # [N, embed_dim]
```

Keys are relative file paths of the form `p{subject_id}/{study_id}/filename.mp4`.

### Parquet / HuggingFace Datasets (recommended for PRIMED-AI probe training)

```python
from datasets import load_dataset
import pandas as pd

PARQUET_DIR = (
    "/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo"
    "/mimic-iv-echo-jepa-embeddings/jepa-l-embeddings"
)

# Via HuggingFace datasets
ds = load_dataset("parquet", data_dir=PARQUET_DIR)
print(ds["train"][0])

# Via pandas (if you prefer)
import glob
dfs = [pd.read_parquet(p) for p in sorted(glob.glob(f"{PARQUET_DIR}/*.parquet"))]
df = pd.concat(dfs, ignore_index=True)
```

### Schema (Parquet)

| Column | dtype | Source |
|--------|-------|--------|
| `subject_id` | int64 | echo-record-list.csv |
| `study_id` | int64 | echo-record-list.csv |
| `dicom_id` | str | filename |
| `file_path` | str | embedding key (relative MP4 path) |
| `acquisition_datetime` | str | echo-record-list.csv |
| `study_datetime` | str | echo-study-list.csv |
| `note_id` | str (nullable) | echo-study-list.csv |
| `note_seq` | str (nullable) | echo-study-list.csv |
| `note_charttime` | str (nullable) | echo-study-list.csv |
| `embedding` | list[float32] | mean-pooled ViT embedding |

Join on `study_id` or `subject_id` to align with the paired cohort from `build_cohort.py`.

---

## 4. Encoder configs

Hydra configs for the encoders live in [`configs/encoder/`](../configs/encoder/).

### `configs/encoder/echojepa.yaml`

```yaml
name: echojepa
checkpoint_path: /orcd/pool/006/lceli_shared/weights/vjepa21_vitl_mimic_pt117.pt
embed_dim: 1024
img_size: 256
num_frames: 16
frozen: true
pooling: mean
```

> Use `echo-vitl-mimic117` as the primary model for PRIMED-AI probe experiments. It is the highest-epoch MIMIC fine-tune of ViT-L and covers the full MIMIC-IV-Echo corpus.

---

## 5. Reproducing extraction from scratch

Extraction is handled by [`sebasmos/EchoJEPA-VE`](https://github.com/sebasmos/EchoJEPA-VE). Follow that repo's [`readme-embeddings.md`](https://github.com/sebasmos/EchoJEPA-VE/blob/main/readme-embeddings.md) for the full pipeline.

**Summary of the 3-stage pipeline:**

```
MIMIC-IV-Echo DICOMs
        │
        ▼  [Stage 1] data/convert_dicom.py
/orcd/pool/006/lceli_shared/mimic-iv-echo-mp4/   (~525K MP4 files, 256×256)
        │
        ▼  [Stage 2] scripts/extract-embeddings/extract_embeddings.py
             (SLURM array: 10 folders × 1 L40S GPU, mit_preemptable partition)
/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo/  (.pt per folder)
        │
        ▼  [Stage 3] scripts/extract-embeddings/to_parquet.py
             (joins MIMIC metadata, writes sharded Parquet)
/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo/  (Parquet shards)
```

### Submitting a new extraction run (ORCD)

```bash
# Clone EchoJEPA-VE
git clone https://github.com/sebasmos/EchoJEPA-VE
cd EchoJEPA-VE

# Load environment
module load miniforge/24.3.0-0
conda activate vjepa2-312

# Submit SLURM array for an EchoJEPA model
sbatch scripts/extract-embeddings/extract_echo_slurm.sh echo-vitl-mimic117

# Merge per-folder outputs once all tasks complete
python scripts/extract-embeddings/merge_embeddings.py --model echo-vitl-mimic117
```

Available model aliases: `vitl`, `vith`, `vitg`, `vitg-384`, `echo-vitl-scratch`, `echo-vitl-mimic100`, `echo-vitl-mimic117`, `echo-vitl-vmix22m`, `echo-vitb-mimic169`.

### Cluster setup

| Field | Value |
|-------|-------|
| Partition | `mit_preemptable` |
| GPU | `gpu:l40s:1` per task |
| Array | `0–9` (folders `p10`–`p19`) |
| Max concurrent | 4 tasks (QOSMaxGRESPerUser cap) |
| Job time limit | 2 days (preemptable; jobs auto-resume via cache) |

---

## 6. Output format reference

### Per-folder `.pt` files

```python
# Dict[str, torch.Tensor]  — key = relative MP4 path, value = shape [embed_dim]
{
  "p10036337/s91664836/00007c75-d5e9d39c-5e7d5b47-a67fb01e-b073dd60.mp4": tensor([...]),
  ...
}
```

### Merged `.pt` file

Same dict, all 10 folders concatenated (~525K entries for a full run).

| Model | Shape per video | Merged file size (approx) |
|-------|----------------|--------------------------|
| `vitl` / `echo-vitl-*` | `[1024]` | ~2.1 GB |
| `vith` | `[1280]` | ~2.6 GB |
| `vitg` / `vitg-384` | `[1408]` | ~2.9 GB |
| `echo-vitb-mimic169` | `[768]` | ~1.6 GB |
