# Task B: Valvular Hemodynamic Disease Framework

This document outlines the complete implementation, clinical validation, and deployment evaluation protocol for **Task B: Valvular Hemodynamic Disease** (Aortic Stenosis, Mitral Regurgitation, Tricuspid Regurgitation) in PRIMED-AI.

---

## 1. Overview & Objectives

Task B extends the PRIMED-AI frozen-embedding probe framework to multi-target valvular disease phenotyping:

- **Aortic Stenosis (AS):** Aortic valve narrowing causing left ventricular outflow obstruction.
- **Mitral Regurgitation (MR):** Retrograde flow from the left ventricle into the left atrium during systole.
- **Tricuspid Regurgitation (TR):** Retrograde flow from the right ventricle into the right atrium, reflecting right-sided hemodynamic overload and pulmonary hypertension.

The core scientific question:
> *When expensive echocardiography is unavailable at deployment (e.g. in the Emergency Department or rural triage), how effectively can ubiquitous 12-lead ECG foundation model embeddings screen for severe valvular disease, and does the multimodal system fail loudly (near decision boundary) or silently (confidently wrong)?*

---

## 2. Pipeline Components

### 2.1 Phase 1 (P1): MedSpacy Clinical NER & ConText Label Extraction
- **Script:** `scripts/extract_valvular_labels.py`
- Extracts severity grades ($0=\text{None}$, $1=\text{Mild}$, $2=\text{Moderate}$, $3=\text{Severe}$) and clinical gates from MIMIC-IV discharge summaries.
- Includes **ConText assertion filtering** (excludes `FAMILY` history, `POSSIBLE_EXISTENCE` rule-outs) and **quantitative Doppler rules** ($\text{AVA} \le 1.0\text{ cm}^2$, $\text{Mean Gradient} \ge 40\text{ mmHg}$).
- Subject-level split isolation ($70\%$ Train, $10\%$ Val, $20\%$ Held-out Test) via `scripts/make_splits.py`.

### 2.2 Phase 2 (P2): Multimodal PyTorch Data Module & Manifest Synchronization
- **Module:** `src/primed_ai/data/valvular_dataset.py`
- **Builder:** `scripts/build_valvular_manifest.py`
- Provides `ValvularMultimodalDataset` and `collate_valvular_batch` supporting runtime missing-modality simulation (`mask_modality='echo'` or `mask_modality='ecg'`).

### 2.3 Phase 3 (P3): Neural Multimodal Probes & Cross-Attention
- **Architectures:** `src/primed_ai/probes/neural_valvular.py`
  - `CrossAttentionValvularProbe`: Bidirectional cross-attention between spatial Echo clip tokens ($T \times 1024$) and ECG tokens ($1 \times 768$).
  - `ConcatMLPValvularProbe`: Fused concatenation baseline.
  - `MultiTaskValvularHead`: Shared trunk predicting binary clinical gates and ordinal severity grades.
- **Training Script:** `scripts/train_valvular_neural_probes.py`

---

## 3. Cohort Summary & Prevalence

From 10,000 candidate clinical records with matched 12-lead ECGs:
- **Total Labeled Studies:** 8,732
- **Unique Patients:** 7,766

| Condition | ICD-10 Code | Total Positive Cases | Prevalence (%) |
|---|---|---|---|
| **Aortic Stenosis (AS Mod/Sev)** | `I35.0` | 1,322 | 15.1% |
| **Mitral Regurgitation (MR Mod/Sev)** | `I34.0` | 2,309 | 26.4% |
| **Tricuspid Regurgitation (TR Mod/Sev)** | `I07.1` | 1,217 | 13.9% |

### Concordance with Billing Records:
- **ICD-10 Recall:** 100.0% across all confirmed diagnostic codes.
- **Subclinical Detection:** Discovered additional moderate/severe valvular findings documented in echo reports that were not listed as primary hospital billing discharge codes.

---

## 4. Modality-Failure & Deployment Results (Held-Out Test Set, $n=1,777$)

| Condition | Multimodal Full AUROC [95% CI] | Drop-ECG (Echo Only) | Drop-Echo (ECG Only) |
|---|---|---|---|
| **Aortic Stenosis (AS Mod/Sev)** | **0.818** [0.791, 0.847] | 0.803 | 0.631 |
| **Mitral Regurgitation (MR Mod/Sev)** | **0.814** [0.790, 0.836] | 0.810 | 0.617 |
| **Tricuspid Regurgitation (TR Mod/Sev)** | **0.759** [0.726, 0.790] | 0.781 | 0.560 |

### Loud vs. Silent Missing-Modality Failure Profile (Echo Dropped):
- **Mitral Regurgitation:** 169 induced critical misses $\rightarrow$ 147 Silent ($87.0\%$), 22 Loud ($13.0\%$).
- **Aortic Stenosis:** 23 induced critical misses $\rightarrow$ 23 Silent ($100.0\%$).
- **Tricuspid Regurgitation:** 13 induced critical misses $\rightarrow$ 13 Silent ($100.0\%$).

---

## 5. Quickstart & Reproducibility

### Run Clinical NER Extraction:
```bash
python scripts/extract_valvular_labels.py --limit 10000
```

### Generate Subject-Level Splits:
```bash
python scripts/make_splits.py \
  --input cohort/valvular_cohort.parquet \
  --output cohort/valvular_cohort_with_splits.parquet \
  --subject-splits cohort/valvular_subject_splits.csv \
  --manifest logs/valvular_splits.json
```

### Train Cross-Attention Neural Probes & Evaluate MFA:
```bash
python scripts/train_valvular_neural_probes.py --model cross_attn --epochs 25
```

### Run Unit Tests:
```bash
pytest tests/test_valvular_probes.py tests/test_valvular_dataset.py tests/test_neural_valvular_probes.py
```
