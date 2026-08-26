# Technical Pipeline — EchoJEPA + HuBERT-ECG LVEF Estimation

> **Note (Jun 24, 2026):** this pipeline is the **substrate** for a model-agnostic
> **modality-failure analysis framework** (`src/primed_ai/failure/`), which is the project's main
> contribution. The pipeline description below remains accurate and is what the framework is
> instantiated on.

Frozen-embedding, multimodal pipeline for estimating left ventricular ejection fraction (LVEF) from paired echocardiogram and ECG data. Foundation models are used as fixed feature extractors; all task-specific learning happens in lightweight probes on top of cached embeddings.

See [README.md](./README.md) for project scope and status.

---

## 1. Architecture overview

Paired cohort → frozen embedding extraction → probes → evaluation. MIMIC-IV-Echo and MIMIC-IV-ECG
supply the two modalities; MIMIC-IV Clinical supplies LVEF labels and demographics.

| Component | Trainable | Role |
|---|---|---|
| EchoJEPA-L | no | Echo video → 1024-d embedding per study |
| HuBERT-ECG | no | 12-lead ECG → 768-d pooled embedding per record |
| ECG-only probe | yes | Linear / MLP on pooled ECG embedding |
| Echo-only probe | yes | Attentive probe on EchoJEPA embedding |
| Concat probe | yes | `concat(pooled echo, pooled ECG)` → MLP — quick baseline |
| Fused probe | yes | Cross-attention over echo ⊕ ECG — headline model |

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

1. Match on `subject_id` to the nearest ECG record within a symmetric **±24 hour** window
   (`build_cohort.py --window-hours 24`, the canonical build; the built manifest's
   max |delta_hours| is 23.98).
2. Join to the structured **LVEF** label from MIMIC-IV-Echo.

Current synchronized manifest size is 1,208 paired rows. Without extracting more
EchoJEPA studies, the hard ceiling is 6,617 echo studies from subjects who also have an
ECG; larger cohorts require additional echo embedding coverage, not just a looser join.

Window sensitivity, measured on the built manifest: tightening to ±12h keeps 767 of the
1,208 rows and ±6h keeps 515. Widening beyond ±24h cannot be measured from the manifest —
it needs a cohort-database rebuild (`scripts/run_cohort_sensitivity.py` wraps the
24h/48h/admission comparison into one command for whoever has BigQuery access).

### 3.2 Train / test split

- Hold out a fixed test split partitioned by `subject_id`.
- No patient may appear in both train and test sets (prevents leakage across splits).

### 3.3 Cohort flow

| Step | Filter |
|---|---|
| 1 | All MIMIC-IV-Echo studies (~525K) |
| 2 | Has a structured LVEF label |
| 3 | Has an ECG within 24–48h on the same `subject_id` |
| 4 | Deduplicate / resolve multi-match edge cases |
| 5 | Final paired cohort |
| 6 | `subject_id`-level train / val / test split |

Record exclusion counts at each step — they are what makes the cohort auditable.

---

## 4. Foundation models

Both models run in inference mode only. Weights are frozen; gradients do not flow back into them.

### 4.1 EchoJEPA-L

| Property | Detail |
|---|---|
| Architecture | Video JEPA (Joint Embedding Predictive Architecture) |
| Input | Echo video (DICOM → MP4) |
| Output | 1024-dim mean-pooled embedding per study (ViT-L) |
| Weights | **Available** on ORCD — see [`docs/embeddings.md §3`](docs/embeddings.md#3-model-weights) |
| Primary checkpoint | `vjepa21_vitl_mimic_pt117.pt` (ViT-L, 117-epoch MIMIC fine-tune) |
| Config | [`configs/encoder/echojepa.yaml`](configs/encoder/echojepa.yaml) |
| Reference | arXiv:2602.02603 |

Five EchoJEPA checkpoints are available (ViT-L and ViT-B variants, various MIMIC fine-tune epochs). Pre-extracted embeddings for every variant are published on HuggingFace as [`MITCriticalData/mimic-iv-echo-jepa-embeddings`](https://huggingface.co/datasets/MITCriticalData/mimic-iv-echo-jepa-embeddings) (gated, PhysioNet credentials required) and mirrored on ORCD. See [`docs/embeddings.md`](docs/embeddings.md) for loading code and [`scripts/embedding_extraction/`](scripts/embedding_extraction/) for the extraction pipeline.

EchoJEPA embeddings preserve spatial structure across frames. This matters for probe design — a simple linear head is insufficient; an **attentive probe** is required (see §6.2).

### 4.2 ECG encoder

Two ECG paths exist in this repo, and they are not interchangeable:

| Path | Status |
|---|---|
| **HuBERT-ECG** — pre-extracted pooled embeddings (768-d), consumed directly from Parquet | **Produced every reported ECG result.** Path in `configs/encoder/ecg_fm.yaml: hubert_ecg_parquet` |
| **ECG-FM** — frozen wrapper (`encoders/ecg_fm.py`), wav2vec2-style, loads via `fairseq_signals` from `wanglab/ecg-fm` (arXiv:2408.05178) | Implemented and unit-tested; **not** used for any reported number |

Both are 768-d after pooling, so the probes accept either. Anything downstream that says "ECG
embedding" below refers to the HuBERT-ECG Parquet unless stated otherwise.

The current HuBERT-ECG Parquet stores one pooled vector per record. Mean-vs-attentive
ECG pooling is therefore not a runnable ablation yet: attentive pooling over tiled copies
of that vector is mathematically identical to mean pooling. Token-level ECG needs a
re-extraction pass before this decision can be revisited.

---

## 5. Embedding extraction

Run once over the paired cohort only; cache all outputs to disk.

| Pass | Input | Model | Output |
|---|---|---|---|
| Echo | Echo DICOM video, paired cohort rows | EchoJEPA-L (frozen) | Per-study embedding tensor |
| ECG | 12-lead waveform, same cohort rows | HuBERT-ECG (frozen; ECG-FM wrapper is the alternative path) | Per-record embedding tensor |

### 5.1 Caching

- Store embeddings keyed by study/record identifier aligned to the paired cohort index.
- Probes read from cache only — no re-running foundation models during probe iteration.
- Confirm GPU and storage budget on compute cluster (ORCD) before extraction.

---

## 6. Probes

All probes are trained on cached embeddings. Recommended build order: quick concat baseline first, then refined architectures.

| Probe | Head | Notes |
|---|---|---|
| **6.1** ECG-only | pool (mean or attentive) → linear / MLP | Simplest unimodal baseline. ECG embeddings are lower-dimensional and more amenable to linear probing than EchoJEPA outputs. |
| **6.2** Echo-only | attentive pooling → probe head | EchoJEPA embeddings are high-dimensional and spatial-preserving, so **attentive pooling is required** — a plain linear probe on raw embeddings underperforms. |
| **6.3** Concat fused | `pool(echo) ⊕ pool(ECG)` → MLP | Fast to implement; gives an early end-to-end number before cross-attention is ready. |
| **6.4** Cross-attention fused | cross-attention over echo + ECG → probe head | **Headline model.** The one used for the missing-modality deployment analysis (§7.1). |

All four predict continuous LVEF; the EF≤40% gate is derived from that output.

**What the cross-attention block can actually attend to (#72).** With the current inputs
the fusion is one-directional at best. The HuBERT-ECG side arrives as one pooled vector
tiled into identical tokens, so the `echo_to_ecg` attention output is provably independent
of the echo query — identical keys force uniform softmax weights, and identical values make
the weighted sum equal that value. That direction reduces to a fixed linear image of the
pooled ECG vector (pinned by `test_echo_to_ecg_attention_is_degenerate_on_tiled_ecg_tokens`).
`ecg_to_echo` attends genuinely only when clip-level echo tokens are retained
(`build_echo_study_embeddings --max-clips`); on the pooled manifest both directions
degenerate and the fused probe is equivalent to a concat model with per-modality linear
pre-maps. Reported pooled-manifest numbers should be read accordingly. Decision: keep the
cross-attention architecture — it fuses for real once clip-level echo tokens are in the
manifest — and treat token-level ECG (which needs a re-extraction pass from raw waveforms;
the parquet stores pooled vectors only) as the gate for restoring the second direction.

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

Plot a **degradation curve** across conditions. The key question: when a modality is unavailable
at inference, does the model degrade gracefully or fail silently?

On the canonical pooled run both answers show up in one table. Dropping echo degrades loudly —
MAE nearly doubles (20.55 vs 10.42) while AUROC only slips to 0.689, so the damage is visible in
the output. Dropping ECG does the opposite: MAE stays flat (11.22) while AUROC falls to 0.383 with
a 95% interval of 0.279–0.493 that excludes 0.5, i.e. the gate ranking inverts. A monitor watching
regression error would not catch it. Report AUROC per condition, never MAE alone.

**Implementation note:** at dropped-modality evaluation, mask the selected fused-probe
branch at inference time (`mask_echo` / `mask_ecg`) rather than retraining a unimodal
model. This tests the deployed fused checkpoint under missing input. Learned null tokens
and branch dropout are deferred unless the missing-modality rerun shows masking is
unstable.

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

### 7.3 Calibration

Assess calibration of the EF≤40% binary gate (reliability diagram, expected calibration
error). A deployable risk score should produce well-calibrated probabilities, not just high
AUROC. The Platt scaler is fit on val predictions only and applied to test
(`scripts/evaluate_calibration.py`); measured numbers are in the
[README results](README.md#results). Caveat when reading dropped-condition ECE: a model
whose predictions collapse toward the training mean can look well calibrated after Platt
scaling while discriminating *below* chance. That is not hypothetical here — `ecg_dropped`
posts the lowest ECE of the three conditions (0.015) on an AUROC of 0.383 whose interval
excludes 0.5, and its bins show why: 234 of 245 cases land in one 0.1–0.2 bin whose mean
confidence (0.168) and observed rate (0.171) differ by 0.003. A near-constant score is
trivially calibrated. Report ECE alongside AUROC and the bin occupancy, never ECE alone.

### 7.4 Modality-failure views

The three per-example views — failure taxonomy, complementarity matrix, loud-vs-silent
dropout profile — are what `src/primed_ai/failure/` contributes, and
`scripts/run_failure_analysis.py` can produce them from two different predictors:

| Route | Predictor | Use |
|---|---|---|
| `--predictions results/missing_modality.json` | the fused checkpoint's own per-example predictions | the reported model; what the paper figures should be built from (#79) |
| `--manifest` / `--cohort` | a Ridge on concatenated embeddings, absent modalities zeroed | harness validation on real embeddings, independent of probe training |
| `--demo` | the same Ridge on planted synthetic structure | recovers known ground truth; no PHI |

The first route fits nothing. With two modalities the harness only ever asks for the full
set and each singleton, and the missing-modality eval has already scored exactly those
three conditions, so the views are a re-reading of the reported predictions rather than a
second model. The routes disagree substantially — the ridge harness puts drop-echo MAE at
117.5 where the checkpoint gives 20.55 — so every report carries a `provenance` block
naming its producer, and the exported bundle names the file after it
(`failure_report.fused.json` vs `failure_report.ridge.json`).

---

## 8. Metrics

| Metric | Target | Baseline references |
|---|---|---|
| LVEF MAE | Continuous regression | EchoJEPA: 5.97 MAE (published, different cohort) · in-cohort echo-only probe: 11.09 |
| EF≤40% AUROC | Binary clinical gate | ECG-FM: 0.929 AUROC (published, different cohort) · in-cohort ECG-only probe: 0.671 |
| Missing-modality degradation | Δ MAE / Δ AUROC across conditions | No external baseline — this is the novel result |
| Fairness gap | Δ MAE / Δ AUROC across demographic strata | No external baseline |

The published solo numbers are **not** like-for-like with this cohort. On identical splits
(seed 42, 245-row test frame, `scripts/diagnose_baseline_gap.py`): fused 10.42 MAE / 0.771
AUROC vs echo-only 11.09 / 0.771, ECG-only 11.60 / 0.671, concat 10.93 / 0.716. Paired
bootstrap deltas: fused beats ECG-only on both metrics (ΔMAE −1.17, 95% CI [−1.99, −0.33];
ΔAUROC +0.100, CI [0.03, 0.17]) and is never behind either solo probe. The gap to the
published baselines is therefore a property of the cohort and label regime, not a fusion
failure — even the in-cohort echo-only probe (same encoder as the published 5.97) lands at
11.09. Contributing factors, measured: 821 training rows after the non-finite drop (829 in
the split); heterogeneous LVEF label sources (the `lvef_upper` fallback contributes 6 test
rows at 43.0 MAE — 18 of its 24 cohort rows carry a physiologically implausible 100.0 —
versus 5.2 MAE on `lvef_3d` rows); and the mean-pooled echo regime (§6.4).

**Pre-flight check:** confirm EF≤40% prevalence in the paired cohort is high enough for stable AUROC estimation before locking results (`scripts/check_ef40_prevalence.py`).

Measured missing-modality numbers are in the [README results table](README.md#results); reproducing them is covered in [CONTRIBUTING.md](CONTRIBUTING.md#reproducibility).

---

## 9. Compute and infrastructure

| Resource | Requirement |
|---|---|
| GPU | Not needed for cached-vector probe training; needed only for future ECG token re-extraction or new EchoJEPA extraction |
| Storage | Pooled manifest 14 MB, one four-probe checkpoint set 7 MB, full results tree ~1 MB — everything fits under `data/`, `probes/`, `results/` in the repo working copy; `--max-clips` echo manifests scale linearly with retained clips |
| Runtime | All CPU, measured on an Apple M1 Pro (16 GB): four-probe M10 training 47.7s wall-clock, missing-modality eval with 1,000 bootstrap resamples 11.8s, fairness stratification 8.3s, per-condition calibration ~5s. A full train + evaluate cycle is about 90 seconds |
| Reproducibility | Fixed random seeds; logged hyperparameters; versioned embedding cache |

Probe training after caching is CPU/GPU-light and completes in minutes.

---

## 10. Related work positioning

| Work | Relationship |
|---|---|
| EchoJEPA (arXiv:2602.02603) | Echo foundation model used in this pipeline |
| HuBERT-ECG | ECG foundation model that produced the reported ECG embeddings |
| ECG-FM (arXiv:2408.05178) | Alternative ECG foundation model — wrapper shipped, not used for reported results |
| EchoingECG (arXiv:2509.25791) | Closest prior cross-modal echo+ECG work — differentiate on frozen-embedding fusion + deployment-risk framing |

This work differs from EchoingECG on three axes: frozen embeddings (no fine-tuning), missing-modality evaluation at inference, and fairness audit as primary contributions.

---

## 11. Open technical decisions

- [x] ECG pooling: mean — the parquet stores one pooled vector per record, and attentive
      pooling over tiled copies of it is mathematically identical to mean pooling, so this
      is the only runnable choice. Revisit only after a token-level ECG re-extraction (#72).
- [x] Echo↔ECG pairing window: fixed at ±24h — this is what the canonical cohort was built
      with (max |delta_hours| 23.98). Within-manifest tightening loses rows fast (767 at
      ±12h, 515 at ±6h); widening to 48h needs a cohort-database rebuild and is scoped
      under D07 (#62) via `scripts/run_cohort_sensitivity.py`.
- [x] Multi-match resolution: when multiple ECGs fall within the window, take nearest timestamp
- [x] EF≤40% prevalence in paired cohort: train/test are reportable; validation has only 26 EF<=40 positives, so validation AUROC should be treated as unstable
- [x] GPU + storage budget for cached probe training: no GPU needed; current pooled manifest and checkpoints are laptop-scale
- [x] Missing-modality masking strategy: use inference-time branch masks; no learned null token or branch dropout for the canonical rerun

---

## 12. Expected artifacts

| Artifact | Description |
|---|---|
| `cohort/` | Paired cohort table with subject_id, echo_id, ecg_id, LVEF, demographics, split assignment |
| `embeddings/echo/` | Cached EchoJEPA-L embeddings per study |
| `embeddings/ecg/` | Cached ECG embeddings per record |
| `probes/` | Trained probe checkpoints (ECG-only, echo-only, concat-MLP, cross-attention fused) |
| `results/` | Metrics tables, degradation curve, fairness stratification, calibration plots |
| `logs/` | Extraction and training logs with hyperparameters and seeds |

---

*References: EchoJEPA arXiv:2602.02603 · HuBERT-ECG · ECG-FM arXiv:2408.05178 · EchoingECG arXiv:2509.25791 · MIMIC-IV-Echo 0.1 · MIMIC-IV-ECG 1.0 · MIMIC-IV 3.1*
