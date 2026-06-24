# When the Cheap Modality Is the Only One: A Deployment Case Report on Frozen-Embedding Multimodal LVEF Estimation

> **DAIH @ COLM 2026 — Deployment Case Report (≤4 pp excl. references).**
> Track floor / fallback per project plan. Double-blind: **no author names, institutions, repo names, usernames, or cluster paths.** Cite EchoJEPA / ECG-FM / EchoingECG in the third person.
> `[FILL: …]` = needs a real number (Tier 1 = BigQuery cohort, Tier 2 = ECG-only probe). Prose below is written to read correctly whether or not those numbers arrive.

---

## Abstract

Echocardiography is the clinical reference for left ventricular ejection fraction (LVEF) but requires a sonographer, a cart, and time; the electrocardiogram (ECG) is ubiquitous, cheap, and acquired everywhere from the emergency department to the rural clinic. A multimodal model trained on paired echo + ECG may quietly assume both modalities are present at inference — an assumption that breaks exactly where deployment is hardest. We report a work-in-progress study designed around the deployment question rather than a benchmark delta: *if a fused echo + ECG model is trained on both modalities but only the ECG is available at inference, how much accuracy is lost, and does the model degrade gracefully or fail silently?* We describe a frozen-embedding pipeline pairing a video JEPA echo encoder (EchoJEPA-L) with a wav2vec2-style ECG foundation model (ECG-FM) over MIMIC-IV, a missing-modality evaluation protocol, and a post-hoc fairness audit across sex, age, and race. We report what was achievable within a one-week sprint and, candidly, the data-access and weight-availability barriers that constrained full results — barriers we argue are themselves a deployment finding for cardiac foundation models. `[FILL: one-sentence headline result if Tier 1/2 numbers available; otherwise frame as protocol + lessons.]`

## 1. Introduction

The dominant evaluation of medical AI asks whether a model is *accurate*. Deployment asks three harder questions — is it *safe*, is it *equitable*, does it *work where it is actually used*. These can diverge sharply for multimodal models. Fusing echo and ECG can improve LVEF estimation on a held-out test set where both modalities are present, yet the realistic deployment regime is asymmetric: the ECG is acquired almost everywhere, while echo requires equipment and expertise concentrated in well-resourced settings. A fused model that has learned to lean on the echo branch may therefore perform well in evaluation and poorly precisely in the low-resource settings that most need a cheap screen.

This case report takes that asymmetry as the central object of study. We do not present fusion-beats-unimodal as the contribution; we present (i) a **missing-modality robustness** protocol that measures degradation when echo is withheld at inference, and (ii) a **fairness audit** of EF error across demographic strata — both framed as deployment-readiness checks. We use frozen foundation-model embeddings throughout (no fine-tuning), which makes the pipeline cheap, reproducible, and a realistic stand-in for how teams without large compute budgets would actually build such a system. We report the pipeline, the protocols, the results obtained, and — in the spirit of the venue's explicit welcome of work-in-progress and lessons-learned — the access barriers that bounded the study.

**Contributions.** (1) A deployment-framed problem statement for multimodal cardiac estimation under inference-time modality availability. (2) A frozen-embedding echo + ECG pipeline over MIMIC-IV with a missing-modality evaluation protocol and a fairness audit. (3) A candid account of the data-access, model-weight, and compute barriers to building deployable cardiac foundation-model systems, offered as a deployment finding in its own right.

## 2. Related work and positioning

Foundation models now exist for both modalities: a video Joint-Embedding Predictive Architecture for echocardiography (EchoJEPA) and a wav2vec2-style self-supervised model for the 12-lead ECG (ECG-FM). The closest prior work, EchoingECG, learns a cross-modal relationship between echo and ECG. **We differ on three axes that matter for deployment rather than representation quality:** (i) we use *frozen* embeddings with lightweight probes — no fine-tuning of either backbone — isolating what the released representations already encode; (ii) our evaluation centers on *inference-time missing-modality* degradation, not paired-modality performance; and (iii) we add an *equity audit* as a first-class result. The point is not a new architecture but a deployment lens on existing ones. `[FILL: 1–2 sentences after reading EchoingECG (arXiv:2509.25791) — W01 — pin the exact methodological boundary.]`

## 3. Data and cohort design

**Sources.** MIMIC-IV-Echo (v0.1; echo DICOM studies + structured LVEF), MIMIC-IV-ECG (v1.0; 12-lead waveforms), and MIMIC-IV (v3.1; demographics). All three require PhysioNet credentialing and signed DUAs. Cohort construction runs server-side on BigQuery so the full ~525K-study echo corpus is never materialized locally.

**Pairing protocol.** For each echo study with a structured LVEF label, we match the nearest ECG on the same `subject_id` within a **24–48 h** window; multi-match cases resolve to the nearest timestamp. We derive a continuous LVEF target and the clinically meaningful **EF ≤ 40%** (HFrEF) binary gate.

**Splitting.** We partition by `subject_id` (70/10/20 train/val/test) with an explicit zero-leakage check, so no patient appears in more than one split.

**Cohort statistics.** `[FILL Tier 1: total echos → with LVEF → with paired ECG → final cohort N; per-split N; LVEF mean/SD; EF≤40% prevalence overall + per split; demographic coverage for sex/age-band/race. Insert CONSORT-style flow as Figure 1.]` If unavailable, state explicitly that the cohort builder is implemented and validated but was not executable within the sprint due to the access barriers in §6.

**Caveat.** MIMIC is single-center; estimates may not transfer across institutions, and we note known gender-curation limitations in the demographic fields, relevant to the fairness audit (§5.2).

## 4. Method

**Frozen encoders.** EchoJEPA-L (ViT-L; 1024-dim, spatial-structure-preserving echo-video embeddings) and ECG-FM (wav2vec2-style; token/sequence embeddings over the 12-lead waveform). Both run in inference mode only; gradients never flow into the backbones. Embeddings are extracted once over the paired cohort and cached to disk, so probe training is reproducible and completes in minutes.

**Probes.** (a) *ECG-only* — pooled ECG-FM embedding → linear/MLP head. (b) *Echo-only* — **attentive** pooling over EchoJEPA tokens → head (a linear probe on the raw high-dimensional embedding is expected to underperform). (c) *Concat-MLP* — pooled echo ⊕ pooled ECG → MLP (quick fused baseline). (d) *Cross-attention fusion* — the headline model used for the missing-modality analysis. All probes predict continuous LVEF; the EF≤40% gate is derived from the continuous output. Primary metrics: **MAE** (regression) and **EF≤40% AUROC** (gate). Reference points from the source models: EchoJEPA ≈ 5.97 MAE; ECG-FM ≈ 0.929 AUROC for the EF≤40% gate.

**Missing-modality protocol (centerpiece).** Train the fused probe on full data; evaluate the *same deployed model* on the held-out test set under three conditions — **full** (echo+ECG), **echo-dropped** (mask the echo branch — ECG-only deployment, the common case), and **ECG-dropped** (echo-only deployment) — by masking/zeroing a branch at inference rather than retraining a unimodal model. Report MAE and AUROC per condition and plot the degradation curve.

**Fairness protocol.** Post-hoc, no extra training: stratify test predictions by sex, age band, and race; report per-stratum MAE and EF≤40% AUROC; flag small strata and the gender-curation caveat.

## 5. Status and results

We report the implementation and protocol as the primary deliverable of this case report, with empirical results to the extent the sprint allowed.

**Implemented and tested (code-complete).** Frozen EchoJEPA-L and ECG-FM encoder wrappers; the BigQuery cohort builder (pairing, LVEF join, demographics, subject-level split, EF≤40% prevalence check, CONSORT flow export); the on-disk embedding cache (atomic writes, resume, manifest); ECG-only, echo-only attentive, concat-MLP, and cross-attention probes; and the missing-modality and fairness evaluation code.

**Empirical results.** `[FILL Tier 2: ECG-only probe val/test MAE + EF≤40% AUROC with bootstrap CIs, vs. naive mean-LVEF baseline. If the fused model + missing-modality curve were obtained, insert the degradation table (full / echo-dropped / ECG-dropped) and Figure 2, and the fairness table. If not obtained, state plainly that the probes and evaluation are implemented and unit-tested but were not run end-to-end within the sprint window for the reasons in §6 — and that the degradation/fairness numbers are therefore deferred.]`

## 6. Lessons learned: the access path *is* a deployment finding

The gap between a code-complete pipeline and a fully-evaluated one was, in this sprint, almost entirely an *access* gap — and that gap is informative for anyone trying to deploy cardiac foundation models.

- **Credentialing latency across a team.** Every contributor who touches data needs PhysioNet credentialing, three signed DUAs, and per-dataset BigQuery access requests. This is serial and slow, and it gates the entire data stage for the whole team, not just one person.
- **The headline echo weights are not openly available.** The strongest echo encoder's weights were pending public release and had to be obtained privately. A pipeline whose central modality depends on a private weight transfer is, by definition, not yet deployable — and not independently reproducible by reviewers.
- **Compute reservation is a real-time dependency.** Embedding extraction needs a reserved GPU; the reservation, not the compute, was on the critical path.
- **Pairing tooling had to be built from scratch.** There is no off-the-shelf echo↔ECG temporal join; the 24–48 h windowing, multi-match resolution, and leakage-safe splitting were all bespoke.
- **The deployment asymmetry mirrors the access asymmetry.** The modality that is hard to *acquire at inference* (echo) is also the one that was hard to *access for training* (private weights, DICOM volume). The cheap, ubiquitous modality (ECG) was the easy one on both axes. This is the paper's thesis restated as a lived constraint: building for the data-rich setting is easy; the deployment-relevant, ECG-only setting is where both the engineering and the science get hard.

We argue these are not incidental sprint hiccups but structural features of the path from a published cardiac foundation model to a deployable screening tool, and that reporting them is squarely within this venue's remit.

## 7. Discussion and future work

The deployment question reframes a multimodal model's value: not "how much does fusion add when both modalities are present" but "how safely does it behave when the expensive modality is absent." Our protocol operationalizes that as a degradation curve and a graceful-vs-silent-failure question; our fairness audit asks whether any degradation is borne unequally. Future work: complete the missing-modality and fairness measurements at scale; extend to Tasks B (valvular hemodynamics) and C (HFpEF phenotyping), both currently blocked by label availability; add EF≤40% calibration (a deployable risk score must be calibrated, not merely discriminative); and pursue external, multi-center validation to test the single-center caveat directly.

## References

`[FILL: COLM 2026 .bib — EchoJEPA arXiv:2602.02603; ECG-FM arXiv:2408.05178; EchoingECG arXiv:2509.25791; MIMIC-IV / -Echo / -ECG PhysioNet citations. References excluded from the 4-page limit.]`
