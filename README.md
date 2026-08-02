<p align="center">
  <h1 align="center">PRIMED-AI</h1>
  <p align="center">
    <strong>EchoJEPA + HuBERT-ECG: frozen-embedding, deployment-risk study of multimodal LVEF estimation</strong>
  </p>
  <p align="center">
    <a href="https://github.com/criticaldata/PRIMED-AI">criticaldata/PRIMED-AI</a>
  </p>
</p>

---

## Overview

**PRIMED-AI** studies whether fusing two cardiac foundation models — **EchoJEPA-L** (echocardiogram video) and **HuBERT-ECG** (12-lead ECG) — can estimate left ventricular ejection fraction (LVEF) in a way that is ready for real-world deployment, not just benchmark accuracy.

The pipeline uses **frozen embeddings only**: foundation models act as fixed feature extractors; all task-specific learning happens in lightweight probes on top of cached representations.

> *ECG is ubiquitous and cheap; echo requires a sonographer and a cart. If we train a fused model but only ECG is available at inference, how much accuracy do we lose — and does the model degrade gracefully or fail silently?*

---

## Motivation

LVEF is a core measure of heart pump function, used to diagnose heart failure and guide treatment. Echocardiography is accurate but resource-intensive; ECG is cheap, fast, and available almost everywhere.

Most multimodal ML papers ask whether fusion beats unimodal baselines on a leaderboard. This project asks the **deployment-and-risk** questions instead:

| Question | What we evaluate |
|----------|------------------|
| **Does it work?** | Missing-modality robustness when echo or ECG is unavailable at inference |
| **Is it equitable?** | Fairness audit of EF error across sex, age, and race |
| **Is it safe?** | Whether accuracy degrades gracefully or fails silently under shift |

Fusion vs. unimodal accuracy is **supporting evidence**, not the thesis.

---

## Pipeline

| Stage | Input | Output |
|-------|-------|--------|
| **Cohort construction** | MIMIC-IV-Echo studies, MIMIC-IV-ECG records, MIMIC-IV demographics | Paired rows on `subject_id` within 24–48h, joined to structured LVEF, split by subject |
| **Embedding extraction** | Paired cohort only (~few K–tens of K studies, not all 525K echos) | Frozen EchoJEPA-L (1024-d) and HuBERT-ECG (768-d) vectors, cached to Parquet |
| **Probes** | Cached embeddings | ECG-only, echo-only (attentive), concat-MLP, and cross-attention fusion heads |
| **Deployment analyses** | Trained fused checkpoint + held-out test split | Missing-modality degradation, fairness stratification, optional calibration |

Expensive forward passes run once and are cached; probe training reads only from cache and completes in minutes. Foundation-model weights are never fine-tuned.

See [TECHNICAL.md](./TECHNICAL.md) for full pipeline details.

---

## Results

Held-out test split, n = 245. One fused cross-attention checkpoint (M09) trained on both modalities, scored under three inference-time conditions — no separate unimodal models are trained for the dropped conditions.

| Condition | Inference input | LVEF MAE | EF≤40% AUROC |
|-----------|-----------------|---------:|-------------:|
| `full` | Echo + ECG present | **10.28** | **0.766** |
| `ecg_dropped` | ECG branch masked, echo present | 15.13 | 0.750 |
| `echo_dropped` | Echo branch masked, ECG present | 18.57 | 0.693 |

Dropping echo costs more than dropping ECG on both metrics. AUROC holds up better than MAE under either drop, which is the graceful-degradation pattern rather than silent failure.

**Provenance:** real run on cached EchoJEPA (`vjepa2.1-vitl-mimic-pt-100`) + HuBERT-ECG embeddings, point estimates transcribed from the E02 evaluation into `results/missing_modality.json`. Bootstrap confidence intervals need per-example predictions and are pending a canonical rerun. `results/` is gitignored, so that JSON is not in this repository — see [CONTRIBUTING.md](./CONTRIBUTING.md#reproducibility) for what reproducing these numbers takes.

Reproduce with:

```bash
python scripts/evaluate_missing_modality.py
```

Defaults assume the standard cohort/embedding/checkpoint layout; `--help` lists the paths. `--embed-dim`, `--echo-dim`, and `--ecg-dim` must match the checkpoint architecture or the `state_dict` load fails on shape.

---

## Models & data

| Component | Source |
|-----------|--------|
| **EchoJEPA-L** | Video JEPA for echo · [arXiv:2602.02603](https://arxiv.org/abs/2602.02603) · [bowang-lab/EchoJEPA](https://github.com/bowang-lab/EchoJEPA) |
| **Echo embeddings** | Pre-extracted, all 6 checkpoint variants · [MITCriticalData/mimic-iv-echo-jepa-embeddings](https://huggingface.co/datasets/MITCriticalData/mimic-iv-echo-jepa-embeddings) (gated) |
| **HuBERT-ECG** | Self-supervised 12-lead ECG model · [Edoardo-BS/hubert-ecg-base](https://huggingface.co/Edoardo-BS/hubert-ecg-base) — **produced the reported ECG embeddings** |
| **ECG-FM** | Alternative ECG foundation model · [arXiv:2408.05178](https://arxiv.org/abs/2408.05178) · [wanglab/ecg-fm](https://huggingface.co/wanglab/ecg-fm) — wrapper shipped, not used for reported results |
| **MIMIC-IV-Echo** 0.1 | Echo DICOM + structured LVEF · [PhysioNet](https://physionet.org/content/mimic-iv-echo/0.1/) |
| **MIMIC-IV-ECG** 1.0 | 12-lead waveforms · [PhysioNet](https://physionet.org/content/mimic-iv-ecg/1.0/) |
| **MIMIC-IV** 3.1 | Demographics for fairness audit · [PhysioNet](https://physionet.org/content/mimic-iv/3.1/) |

All MIMIC datasets and the embedding dataset require PhysioNet credentialing and signed Data Use Agreements. [docs/embeddings.md](./docs/embeddings.md) covers loading and re-extraction.

> **Which ECG encoder?** The reported results use pre-extracted **HuBERT-ECG** embeddings. The repo
> also ships a frozen **ECG-FM** wrapper ([`encoders/ecg_fm.py`](./src/primed_ai/encoders/ecg_fm.py),
> [`scripts/extract_ecg_embeddings.py`](./scripts/extract_ecg_embeddings.py)) as an alternative
> extraction path — it is implemented and unit-tested but did **not** produce any reported number.

**Scope:** Task A — continuous LVEF regression + EF≤40% binary clinical gate (HFrEF threshold). Valvular disease and HFpEF phenotyping are out of scope.

---

## Metrics

| Metric | Purpose |
|--------|---------|
| **LVEF MAE** | Continuous regression quality |
| **EF≤40% AUROC** | Clinical gate for reduced ejection fraction |
| **Missing-modality degradation** | Δ MAE / Δ AUROC when echo or ECG is dropped at inference |
| **Fairness gap** | Per-stratum error across sex, age band, and race |

---

## Repository guide

| Document | Contents |
|----------|----------|
| [TECHNICAL.md](./TECHNICAL.md) | Pipeline architecture, probes, deployment analyses, expected artifacts |
| [docs/embeddings.md](./docs/embeddings.md) | Embedding sources, loading code, model registry, re-extraction |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | Local setup, tests, linting, data access, reproducibility |

Open work lives on the [issue tracker](https://github.com/criticaldata/PRIMED-AI/issues).

---

## Status

The work pivoted from a fusion benchmark to a model-agnostic **modality-failure analysis
framework** — failure taxonomy, complementarity matrix, and loud-vs-silent dropout profile. That
framework is the main contribution and lives in
[src/primed_ai/failure/](./src/primed_ai/failure/); the fusion pipeline described above is the
substrate it is instantiated on.

Still open: PhysioNet credentialing and reserved GPU/storage, both needed for the canonical
real-data rerun.

---

## Related work

| Work | Relationship |
|------|--------------|
| [EchoJEPA](https://arxiv.org/abs/2602.02603) | Echo foundation model used in this pipeline |
| [HuBERT-ECG](https://huggingface.co/Edoardo-BS/hubert-ecg-base) | ECG foundation model behind the reported ECG embeddings |
| [ECG-FM](https://arxiv.org/abs/2408.05178) | Alternative ECG foundation model — wrapper shipped, not used for reported results |
| [EchoingECG](https://arxiv.org/abs/2509.25791) | Closest prior cross-modal echo+ECG work — differentiated on frozen-embedding fusion + deployment-risk framing |

---

## Citation

If you use this work, please cite the repository:

```bibtex
@misc{primedai2026,
  title        = {PRIMED-AI: Deployment-Risk Study of Multimodal LVEF Estimation with EchoJEPA and HuBERT-ECG},
  author       = {Critical Data},
  year         = {2026},
  howpublished = {\url{https://github.com/criticaldata/PRIMED-AI}}
}
```

---

## License

MIT — see [LICENSE](./LICENSE).

`scripts/embedding_extraction/src/` is vendored V-JEPA code from Meta Platforms and stays under its
own MIT license (see [scripts/embedding_extraction/src/LICENSE](./scripts/embedding_extraction/src/LICENSE)).

---

<p align="center">
  <sub>PRIMED-AI · Critical Data</sub>
</p>
