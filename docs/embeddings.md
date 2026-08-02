# Embedding Extraction — EchoJEPA & ECG

Where the pre-extracted embeddings live, how to load them, and how to re-run extraction from
scratch.

See [TECHNICAL.md](../TECHNICAL.md) for the full pipeline context.

---

## Table of Contents

1. [Echo embeddings (use these first)](#1-echo-embeddings-use-these-first)
2. [ECG embeddings](#2-ecg-embeddings)
3. [Model weights](#3-model-weights)
4. [Encoder configs](#4-encoder-configs)
5. [Reproducing extraction from scratch](#5-reproducing-extraction-from-scratch)

---

## 1. Echo embeddings (use these first)

All EchoJEPA / V-JEPA2 embeddings for MIMIC-IV-Echo (~525K clips) are published as a gated
HuggingFace dataset:

**[MITCriticalData/mimic-iv-echo-jepa-embeddings](https://huggingface.co/datasets/MITCriticalData/mimic-iv-echo-jepa-embeddings)**

The dataset card is the source of truth for available checkpoint variants, the Parquet schema, and
access requirements — it is not duplicated here. Access is gated behind an active PhysioNet
credentialed account with a signed MIMIC-IV-Echo DUA.

```python
from datasets import load_dataset

ds = load_dataset(
    "MITCriticalData/mimic-iv-echo-jepa-embeddings",
    data_dir="vjepa2.1-vitl-mimic-pt-100",   # variant used for the reported results
)
```

Each row carries `subject_id`, `study_id`, `dicom_id`, `file_path`, acquisition/study timestamps,
and a mean-pooled `embedding` (1024-d for ViT-L, 768-d for ViT-B). Join on `study_id` or
`subject_id` to align with the paired cohort from `build_cohort.py`.

> Use `vjepa2.1-vitl-mimic-pt-100` for PRIMED-AI probe experiments — it is the variant behind every
> reported echo number. `echo-vitl-mimic117` (117-epoch fine-tune) is the highest-epoch ViT-L
> checkpoint if you are starting a new sweep.

The same Parquet shards are mirrored on the MIT ORCD pool under
`jepa-embeddings-mimiciv-echo/mimic-iv-echo-jepa-embeddings/` for jobs running on the cluster, plus
raw per-folder `.pt` dumps keyed by relative MP4 path (`p{subject_id}/{study_id}/{dicom_id}.mp4`):

```python
import torch

# Dict[str, torch.Tensor], one entry per clip; p10–p19 folders, or *_all.pt merged (~2.1 GB for ViT-L)
embeddings = torch.load("vitl_embeddings_p10.pt", map_location="cpu")
X = torch.stack(list(embeddings.values()))  # [N, embed_dim]
```

---

## 2. ECG embeddings

The reported ECG numbers come from pre-pooled **HuBERT-ECG** embeddings (768-d), read directly from
Parquet — path in [`configs/encoder/ecg_fm.yaml`](../configs/encoder/ecg_fm.yaml) under
`hubert_ecg_parquet`. The model is
[`Edoardo-BS/hubert-ecg-base`](https://huggingface.co/Edoardo-BS/hubert-ecg-base); its model card
covers loading and the pretraining corpus.

[`encoders/ecg_fm.py`](../src/primed_ai/encoders/ecg_fm.py) wraps
[**ECG-FM**](https://huggingface.co/wanglab/ecg-fm) as an alternative extraction path, driven by
[`scripts/extract_ecg_embeddings.py`](../scripts/extract_ecg_embeddings.py). It downloads weights via
`huggingface_hub` and needs the `fairseq_signals` stack. It is implemented and unit-tested but has
not produced any reported number. Both paths are 768-d after pooling, so probes accept either.

---

## 3. Model weights

Echo checkpoints live on ORCD under `weights/`. Access requires ORCD cluster membership — contact
the PRIMED-AI leads.

### V-JEPA2 — natural video pretrain (Meta AI)

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
| `vjepa21_vitl_mimic_pt117.pt` | `echo-vitl-mimic117` | 1024 | V-JEPA2.1 ViT-L fine-tuned on MIMIC-IV-Echo, 117 epochs |
| `vitl-vmix22m-pt220-c55.pt` | `echo-vitl-vmix22m` | 1024 | ViT-L pretrained on VMix-22M (220 epochs) + MIMIC fine-tune |
| `vjepa2_1_vitb_mimic_pt169_c60.pt` | `echo-vitb-mimic169` | 768 | V-JEPA2.1 ViT-B fine-tuned on MIMIC-IV-Echo, 169 epochs |

---

## 4. Encoder configs

Hydra configs live in [`configs/encoder/`](../configs/encoder/): `echojepa.yaml` (checkpoint path,
`embed_dim`, frame count, pooling) and `ecg_fm.yaml` (ECG-FM wrapper settings plus the
`hubert_ecg_parquet` path the probes actually read). Both set `frozen: true` — foundation-model
weights are never fine-tuned.

---

## 5. Reproducing extraction from scratch

Only needed for a checkpoint variant not already published. Scripts live in
[`scripts/embedding_extraction/`](../scripts/embedding_extraction/).

**3-stage pipeline:**

| Stage | Script | Output |
|---|---|---|
| 1 | `convert_dicom.py` | ~525K MP4 files, 256×256 |
| 2 | `extract_embeddings.py` (SLURM array via `extract_echo_slurm.sh`) | `.pt` per folder |
| 2b | `merge_embeddings.py` | merged `*_all.pt` |
| 3 | `to_parquet.py` | sharded Parquet with MIMIC metadata joined |

```bash
module load miniforge/24.3.0-0
conda activate vjepa2-312

sbatch scripts/embedding_extraction/extract_echo_slurm.sh echo-vitl-mimic117
python scripts/embedding_extraction/merge_embeddings.py --model echo-vitl-mimic117
python scripts/embedding_extraction/to_parquet.py --model echo-vitl-mimic117
```

Model aliases: `vitl`, `vith`, `vitg`, `vitg-384`, `echo-vitl-scratch`, `echo-vitl-mimic100`,
`echo-vitl-mimic117`, `echo-vitl-vmix22m`, `echo-vitb-mimic169`.

### Cluster setup

| Field | Value |
|-------|-------|
| Partition | `mit_preemptable` |
| GPU | `gpu:l40s:1` per task |
| Array | `0–9` (folders `p10`–`p19`) |
| Max concurrent | 4 tasks (QOSMaxGRESPerUser cap) |
| Job time limit | 2 days (preemptable; jobs auto-resume via cache) |
