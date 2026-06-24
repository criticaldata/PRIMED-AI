# Technical Pipeline — EchoJEPA + ECG-FM LVEF Estimation

> **Note (Jun 24, 2026):** this pipeline is the **substrate** for the submitted paper, whose
> contribution is a model-agnostic **modality-failure analysis framework**
> (`src/primed_ai/failure/`; see [`paper/`](./paper/)). The pipeline description below remains
> accurate and is what the framework is instantiated on.

Frozen-embedding, multimodal pipeline for estimating left ventricular ejection fraction (LVEF) from paired echocardiogram and ECG data. Foundation models are used as fixed feature extractors; all task-specific learning happens in lightweight probes on top of cached embeddings.

See [OVERVIEW.md](./OVERVIEW.md) for submission strategy, scope decisions, and timeline.

---

## 1. Architecture overview

```
MIMIC-IV-Echo ──┐
                ├──► Paired cohort ──► Frozen embedding extraction ──► Probes ──► Evaluation
MIMIC-IV-ECG  ──┤         ▲
                │         │
MIMIC-IV      ──┘    LVEF labels
  (Clinical)

Foundation models (frozen, no fine-tuning):
  EchoJEPA-L  → echo video embeddings
  ECG-FM      → 12-lead ECG embeddings

Probes (trainable):
  ECG-only    → linear / MLP on pooled ECG-FM embedding
  Echo-only   → attentive probe on EchoJEPA embedding
  Fused       → cross-attention over echo ⊕ ECG (headline)
  Quick win   → concat(pooled echo, pooled ECG) → MLP
```

**Design principle:** separate expensive forward passes (run once, cache to disk) from cheap probe training (minutes, fully reproducible). Do not fine-tune foundation-model weights.

---

## 2. Data sources

| Dataset | Version | Role |
|---|---|---|
| MIMIC-IV-Echo | 0.1 | Echo DICOM studies + structured LVEF labels |
| MIMIC-IV-ECG | 1.0 | 12-lead ECG waveforms |
| MIMIC-IV | 3.1 | Demographics (sex, age, race) for fairness stratification |

All three require PhysioNet credentialing and signed Data Use Agreements (DUAs) before access.

**Out of scope:** Tasks B (valvular hemodynamic disease) and C (HFpEF phenotyping) are excluded due to label availability blockers. This pipeline targets **Task A — ejection fraction only**.

---

## 3. Cohort construction

Build the paired cohort **before** running any embedding extraction. Do not process the full ~525K echo corpus.

### 3.1 Pairing protocol

For each echo study:

1. Match on `subject_id` to the nearest ECG record within a **24–48 hour** temporal window (using MIMIC timestamps).
2. Join to the structured **LVEF** label from MIMIC-IV-Echo.

Expected cohort size: a few thousand to tens of thousands of paired rows (exact count depends on window strictness and label availability).

### 3.2 Train / test split

- Hold out a fixed test split partitioned by `subject_id`.
- No patient may appear in both train and test sets (prevents leakage across splits).

### 3.3 Cohort flow (for paper diagram)

```
All MIMIC-IV-Echo studies (~525K)
  → filter: has structured LVEF label
  → filter: has ECG within 24–48h on same subject_id
  → deduplicate / resolve multi-match edge cases
  → final paired cohort
  → subject_id-level train / val / test split
```

Document exclusion counts at each step for the cohort flow diagram.

---

## 4. Foundation models

Both models run in inference mode only. Weights are frozen; gradients do not flow back into them.

### 4.1 EchoJEPA-L

| Property | Detail |
|---|---|
| Architecture | Video JEPA (Joint Embedding Predictive Architecture) |
| Input | Echo video (DICOM → MP4) |
| Output | 1024-dim mean-pooled embedding per study (ViT-L) |
| Weights | **Available** on ORCD — see [`docs/embeddings.md §2`](docs/embeddings.md#2-model-weights) |
| Primary checkpoint | `vjepa21_vitl_mimic_pt117.pt` (ViT-L, 117-epoch MIMIC fine-tune) |
| Config | [`configs/encoder/echojepa.yaml`](configs/encoder/echojepa.yaml) |
| Reference | arXiv:2602.02603 |

Five EchoJEPA checkpoints are available (ViT-L and ViT-B variants, various MIMIC fine-tune epochs). Pre-extracted embeddings for `vitl` (natural pretrain) are already on ORCD; EchoJEPA fine-tuned runs are in progress. See [`docs/embeddings.md`](docs/embeddings.md) for paths, loading code, and [`scripts/embedding_extraction/`](scripts/embedding_extraction/) for the full pipeline.

EchoJEPA embeddings preserve spatial structure across frames. This matters for probe design — a simple linear head is insufficient; an **attentive probe** is required (see §5.2).

### 4.2 ECG-FM

| Property | Detail |
|---|---|
| Architecture | wav2vec2-style signal foundation model |
| Input | 12-lead ECG waveform |
| Output | Token-level or sequence embeddings (pool before probing) |
| Weights | Publicly available on Hugging Face |
| Reference | arXiv:2408.05178 |

**Open decision:** pooling strategy for ECG-FM embeddings — mean pooling vs. attentive pooling. Resolve with a quick empirical check early in probe development.

---

## 5. Embedding extraction

Run once over the paired cohort only; cache all outputs to disk.

### 5.1 EchoJEPA forward pass

```
Input:  echo DICOM video (paired cohort rows)
Model:  EchoJEPA-L (frozen)
Output: per-study embedding tensor → cache to disk
```

### 5.2 ECG-FM forward pass

```
Input:  12-lead ECG waveform (same paired cohort rows)
Model:  ECG-FM (frozen)
Output: per-record embedding tensor → cache to disk
```

### 5.3 Caching

- Store embeddings keyed by study/record identifier aligned to the paired cohort index.
- Probes read from cache only — no re-running foundation models during probe iteration.
- Confirm GPU and storage budget on compute cluster (ORCD) before extraction.

---

## 6. Probes

All probes are trained on cached embeddings. Recommended build order: quick concat baseline first, then refined architectures.

### 6.1 ECG-only baseline

```
ECG-FM embedding → pool (mean or attentive) → linear / MLP → LVEF prediction
```

Simplest unimodal baseline. ECG-FM embeddings are lower-dimensional and more amenable to linear probing than EchoJEPA outputs.

### 6.2 Echo-only baseline

```
EchoJEPA embedding → attentive pooling → probe head → LVEF prediction
```

EchoJEPA produces high-dimensional, spatial-preserving embeddings. **Attentive pooling is required** — a plain linear probe on raw embeddings is expected to underperform.

### 6.3 Quick-win fused probe (Day 2 target)

```
pool(echo embedding) ⊕ pool(ECG embedding) → MLP → LVEF prediction
```

Concatenation + MLP. Fast to implement; provides an early end-to-end number before the cross-attention fusion probe is ready.

### 6.4 Fused probe — headline model (Day 3 target)

```
EchoJEPA embedding ──┐
                     ├──► cross-attention fusion → probe head → LVEF prediction
ECG-FM embedding   ──┘
```

Cross-attention over echo and ECG representations. This is the model used for the missing-modality deployment analysis (§7.1).

### 6.5 Prediction targets

| Target | Type | Use |
|---|---|---|
| Continuous LVEF | Regression | Primary metric: MAE |
| EF ≤ 40% | Binary classification | Clinical gate: AUROC (HFrEF threshold) |

Train probes to predict continuous LVEF. Derive the EF≤40% binary gate from the continuous output or via a separate classification head.

---

## 7. Deployment analyses

These analyses are the primary scientific contribution — not fusion accuracy alone.

### 7.1 Missing-modality robustness (centerpiece)

Train the fused probe on full (echo + ECG) data. At evaluation time, simulate inference-time modality availability:

| Condition | Echo | ECG | Simulates |
|---|---|---|---|
| Full | ✓ | ✓ | Ideal: both modalities available |
| Echo dropped | ✗ | ✓ | ED, rural clinic, overnight — ECG only |
| ECG dropped | ✓ | ✗ | Echo-only deployment |

For each condition, report:

- Continuous LVEF **MAE**
- **EF≤40% AUROC**

Plot a **degradation curve** across conditions. The key question: when echo is unavailable at inference, does the model degrade gracefully or fail silently?

**Implementation note:** at echo-dropped evaluation, zero out or mask the echo branch of the fused probe rather than retraining a separate ECG-only model. This tests the actual deployed fused model under missing input.

### 7.2 Fairness audit

Post-hoc stratification of probe predictions — no additional training required.

Stratify by:

- **Sex**
- **Age band**
- **Race**

Report per-stratum:

- EF prediction error (MAE or residual distribution)
- EF≤40% AUROC

Flag known MIMIC gender-curation bias in documentation. This analysis is cheap and directly addresses the equity evaluation pillar.

### 7.3 Calibration (optional stretch)

If time permits, assess calibration of the EF≤40% binary gate (e.g., reliability diagram, expected calibration error). A deployable risk score should produce well-calibrated probabilities, not just high AUROC.

---

## 8. Metrics

| Metric | Target | Baseline references |
|---|---|---|
| LVEF MAE | Continuous regression | EchoJEPA: 5.97 MAE |
| EF≤40% AUROC | Binary clinical gate | ECG-FM: 0.929 AUROC |
| Missing-modality degradation | Δ MAE / Δ AUROC across conditions | No external baseline — this is the novel result |
| Fairness gap | Δ MAE / Δ AUROC across demographic strata | No external baseline |

**Pre-flight check:** confirm EF≤40% prevalence in the paired cohort is high enough for stable AUROC estimation before locking results.

---

## 9. Compute and infrastructure

| Resource | Requirement |
|---|---|
| GPU | H200 or equivalent; reserve early on ORCD |
| Storage | Sufficient for cached embedding tensors across full paired cohort |
| Runtime | Subsetting to paired cohort keeps extraction under ~1 hour (vs. hours for full 525K echo corpus) |
| Reproducibility | Fixed random seeds; logged hyperparameters; versioned embedding cache |

Probe training after caching is CPU/GPU-light and completes in minutes.

---

## 10. Fallback pipelines

If EchoJEPA-L weights are unavailable by end of Day 2:

### Fallback A — ECG-FM only

```
Paired cohort (ECG + LVEF only) → ECG-FM embeddings → probe → deployment analyses
```

Loses fusion novelty. Retains missing-modality framing (ECG-only is the realistic deployment scenario) and fairness audit.

### Fallback B — Case Report (no embeddings)

Document data-access barriers, pairing protocol design, and lessons learned. Requires no model weights or GPU time.

---

## 11. Related work positioning

| Work | Relationship |
|---|---|
| EchoJEPA (arXiv:2602.02603) | Echo foundation model used in this pipeline |
| ECG-FM (arXiv:2408.05178) | ECG foundation model used in this pipeline |
| EchoingECG (arXiv:2509.25791) | Closest prior cross-modal echo+ECG work — differentiate on frozen-embedding fusion + deployment-risk framing |

Read EchoingECG before writing methods. Explicitly state how this work differs: frozen embeddings (no fine-tuning), missing-modality evaluation at inference, and fairness audit as primary contributions.

---

## 12. Open technical decisions

- [ ] ECG-FM pooling: mean vs. attentive — empirical check on validation set
- [ ] Echo↔ECG pairing window: 24h vs. 48h — sensitivity analysis or fixed choice with justification
- [ ] Multi-match resolution: when multiple ECGs fall within the window, take nearest timestamp
- [ ] EF≤40% prevalence in paired cohort — confirm before reporting AUROC
- [ ] GPU + storage budget for embedding cache on ORCD
- [ ] Missing-modality masking strategy: zero-out vs. learned null token vs. branch dropout at eval only

---

## 13. Expected artifacts

| Artifact | Description |
|---|---|
| `cohort/` | Paired cohort table with subject_id, echo_id, ecg_id, LVEF, demographics, split assignment |
| `embeddings/echo/` | Cached EchoJEPA-L embeddings per study |
| `embeddings/ecg/` | Cached ECG-FM embeddings per record |
| `probes/` | Trained probe checkpoints (ECG-only, echo-only, concat-MLP, cross-attention fused) |
| `results/` | Metrics tables, degradation curve, fairness stratification, calibration plots |
| `logs/` | Extraction and training logs with hyperparameters and seeds |

---

*References: EchoJEPA arXiv:2602.02603 · ECG-FM arXiv:2408.05178 · EchoingECG arXiv:2509.25791 · MIMIC-IV-Echo 0.1 · MIMIC-IV-ECG 1.0 · MIMIC-IV 3.1*
