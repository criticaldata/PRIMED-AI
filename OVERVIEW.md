# DAIH @ COLM 2026 — Submission Plan
### EchoJEPA + ECG-FM: a frozen-embedding, deployment-risk study of multimodal LVEF estimation

> **FINAL OUTCOME (Jun 24, 2026):** the submission pivoted from "fusion vs. unimodal" to a
> model-agnostic **modality-failure evaluation framework** (failure taxonomy · complementarity
> matrix · loud-vs-silent dropout) — see [`src/primed_ai/failure/`](./src/primed_ai/failure/).
> The paper source is no longer tracked in this repository. The strategy below is the original
> plan, kept for provenance.

**Target venue:** DAIH (Deploying AI in Healthcare), COLM 2026 Workshop — non-archival
**Submission deadline:** **June 23, 2026 (AoE)** · Notification: July 24 · Workshop: Oct 9, San Francisco
**Platform:** OpenReview · **Format:** COLM 2026 LaTeX template (mandatory) · **Review:** double-blind
**Time remaining as of this plan (June 17):** ~6 working days
**Approach:** frozen foundation-model embeddings only (no fine-tuning), probing + fusion

---

## 0. TL;DR

1. **Reframe the work.** DAIH is a *deployment-and-risk* workshop, not a benchmark venue. A "fusion beats unimodal on EF MAE" paper is a weak fit. Lead instead with **missing-modality robustness** and a **fairness audit**; use the fusion result as supporting evidence, not the thesis.
2. **Scope to Task A (ejection fraction) only.** Cut Tasks B and C — both have high-severity label blockers that are not solvable in a week.
3. **Critical path = EchoJEPA-L weight access.** If not in hand by end of Day 2, pivot (see §6).
4. **Write toward the 8-page Research Paper, keep the 4-page Case Report as a guaranteed fallback.** Hard go/no-go on Day 3.
5. **Build the paired cohort *before* extracting embeddings.** Subset, don't process all 525K echos.

---

## 1. Why framing is the whole game

DAIH organizes around three questions — *is it safe, is it equitable, does it work* — and is explicitly skeptical that benchmark accuracy implies real-world readiness. The CFP's welcomed topics include evaluation beyond benchmark accuracy, robustness under temporal/institutional/demographic shift, fairness auditing, and multimodal clinical decision support.

The roadmap's current framing — *"does fusing EchoJEPA + ECG-FM beat either alone on EF"* — is a pure benchmark-gain story, which is close to the thing this workshop pushes back against. Same experiments, but the **paper's question** must change.

**Note on fit (be honest with yourselves):** EchoJEPA is a video/vision JEPA and ECG-FM is a wav2vec2-style signal model — neither is an LLM or VLM in the strict sense. The CFP says "language and vision-language models." The work fits the *multimodal clinical applications / robustness / equity* topics, but the framing has to do the work of making the fit obvious to a reviewer. Do not assume the fit is self-evident.

---

## 2. The reframe — choose the spine

| Angle | DAIH pillar hit | Cost | Verdict |
|---|---|---|---|
| **Missing-modality robustness at inference** | "Does it work" / robustness under shift | Low (already an ablation in roadmap) | **Spine** |
| **Fairness audit** (sex, age, race) across EF error | "Is it equitable" | Very low (post-hoc on existing predictions) | **Second pillar** |
| Fusion vs. unimodal | multimodal modeling | Already required | Supporting evidence, not the point |
| Shortcut learning / "discordant experiments" | safety / failure characterization | High (slow) | Future work — do not attempt this week |

**Headline question to put in the abstract:**
> ECG is ubiquitous and cheap; echo requires a sonographer and a cart. If we train a fused EF model but only ECG is available at inference (ED, overnight, rural clinic), how much accuracy do we lose — and does the model degrade gracefully or fail silently?

This converts a benchmark result into a deployment-readiness result.

---

## 3. Track decision + go/no-go

| Track | Length | Outcome | Fit |
|---|---|---|---|
| **Research Paper** | ≤ 8 pp (excl. refs) | Spotlight or poster | Natural home for multimodal + robustness + fairness |
| **Deployment Case Report** | ≤ 4 pp (excl. refs) | Poster | Explicitly welcomes work-in-progress, negative results, lessons learned |

**Strategy:** write toward the **Research Paper**, but the **Case Report is a strict subset** (same intro, data, blocker narrative; fewer experiments) and is the guaranteed floor.

**Go/no-go checkpoint — end of Day 3:**
- ✅ Clean fused EF result **and** the missing-modality degradation curve in hand → finish **Research Paper**.
- ❌ Otherwise → collapse to **Case Report**, whose "Uncertainties and Blockers" narrative is already written in the roadmap.

You must pick one track at submission. This structure means the deadline can never catch you empty-handed.

---

## 4. Scope: Task A (ejection fraction) only

| Task | Label availability | Decision |
|---|---|---|
| **A — EF detection** | Continuous LVEF already structured in MIMIC-IV-Echo; EF≤40% binary gate | **DO** |
| B — Valvular hemodynamic disease | Doppler measures buried in free-text reports (**high-severity blocker**) | Cut → future work |
| C — HF phenotyping (HFpEF) | No clean ground truth; needs echo params + labs + ICD (**high-severity blocker**) | Cut → future work |

Per the roadmap's own assessment, Task A has **no high-severity blockers once weights are released**; everything else needed is available today. Mention B/C in one paragraph as future directions.

**Metrics to report for Task A:**
- Continuous LVEF: **MAE** (baseline references: EchoJEPA 5.97 MAE; ECG-FM 0.929 AUROC for EF≤40%)
- Clinical gate: **EF≤40% AUROC** (identifies HFrEF candidates — clinically meaningful threshold)
- Optionally: calibration of the EF≤40% gate (a deployable risk score must be calibrated)

---

## 5. Blockers & risk register

| Issue | Severity | Mitigation |
|---|---|---|
| **EchoJEPA-L weights pending PhysioNet approval** | **Critical / blocks everything** | Request checkpoint access from the EchoJEPA authors. Chase Day 1. |
| Echo↔ECG temporal pairing (no existing tooling) | Medium | Join on `subject_id` + 24–48h time window using MIMIC timestamps. Build this Day 1. |
| GPU availability on ORCD | Medium | Reserve H200 (or any capable GPU) Day 1. Subsetting makes a smaller GPU viable. |
| Credentialing / DUA for all team members | Medium | Verify PhysioNet credentialing + signed DUAs on all three datasets before touching data. |
| Novelty vs. EchoingECG (arXiv 2509.25791) | Medium | Read it before writing the intro; pin down how frozen-embedding fusion + deployment lens differs from their cross-modal work. **Do not let a reviewer draw this line for you.** |
| Doppler / HFpEF labels | High | Out of scope — cut with Tasks B/C. |
| No external paired Echo+ECG+label dataset | Low | Acknowledge single-center (MIMIC) generalizability limit as a stated deployment caveat. |

**Fallbacks if weights don't arrive by end of Day 2:**
1. **ECG-FM-only EF deployment study** — loses fusion novelty, keeps robustness + fairness story.
2. **Pure lessons-learned Case Report** on data-access barriers to deployable cardiac FMs — needs zero echo embeddings; strong venue fit.

---

## 6. Technical pipeline (frozen embeddings only)

### 6.1 Build the paired cohort *first* (Day 1)
- Do **not** process all 525K echos. Construct the cohort first:
  - echo study ↔ nearest ECG on `subject_id` within a **24–48h window**
  - join to structured **LVEF** label
- Result: a few-thousand-to-tens-of-thousands-row cohort. This turns "a few hours of extraction" into well under an hour and de-risks the week.
- Hold out a fixed test split by `subject_id` (no patient leakage across train/test).

### 6.2 Extract embeddings (Day 2) — frozen forward pass
- **EchoJEPA-L** over paired echo videos (DICOM) → cache embeddings to disk.
- **ECG-FM** over paired 12-lead ECGs → cache embeddings to disk.
- Cache everything; probes then train in minutes and are fully reproducible.

### 6.3 Probes (Days 2–3)
| Step | Probe | Note |
|---|---|---|
| ECG only | linear/MLP on pooled ECG-FM embedding | baseline |
| Echo only | **attentive probe** on EchoJEPA embedding | JEPA's high-dim, spatial-preserving embeddings need attentive pooling, not linear |
| Fused | **cross-attention** over (echo ⊕ ECG) | headline model |
| Quick win first | concat(pooled echo, pooled ECG) → MLP | get a number on the board Day 2 before the elegant version |

### 6.4 Deployment analyses (Days 3–4)
- **Missing-modality at inference (the centerpiece):** train fused; evaluate (a) full, (b) echo dropped, (c) ECG dropped. Plot the degradation curve. Report both MAE and EF≤40% AUROC.
- **Fairness:** stratify EF error / AUROC by **sex, age band, race**. Cheap, post-hoc, hits the equity pillar. (Meeting notes already flag MIMIC gender-curation bias.)
- **Optional stretch:** calibration of the EF≤40% gate.

---

## 7. Day-by-day

**Day 1 (Tue, June 17 — today)**
- ☐ Request EchoJEPA-L weights from the model authors — *single most important action.*
- ☐ Pull ECG-FM weights (open on HF).
- ☐ Confirm PhysioNet credentialing + DUAs for all members.
- ☐ Reserve GPU on ORCD.
- ☐ Build paired cohort (echo↔ECG↔LVEF) and the held-out split.

**Day 2 (Wed)**
- ☐ Frozen forward passes on the *paired cohort only*; cache embeddings.
- ☐ Crude result on the board: concat → MLP probe + two unimodal baselines.

**Day 3 (Thu)**
- ☐ Attentive echo probe + cross-attention fusion probe.
- ☐ Missing-modality ablation (full / echo-dropped / ECG-dropped), MAE + EF≤40% AUROC.
- ☐ **GO/NO-GO:** Research Paper vs. Case Report.

**Day 4 (Fri)**
- ☐ Fairness stratification (sex, age, race).
- ☐ Optional: calibration of EF≤40% gate.
- ☐ Freeze results; lock the two money figures (degradation curve + fairness table).

**Day 5 (Sat)**
- ☐ Write in the COLM 2026 template from the start. Roadmap = most of methods/motivation; meeting notes = framing.

**Day 6 + buffer (Sun–Mon)**
- ☐ Internal review; check every number against its log.
- ☐ Anonymization pass (see §9).
- ☐ Submit **early** on OpenReview — do not aim for the AoE wire.

---

## 8. Paper outline

### 8-page Research Paper
1. **Abstract** — deployment question (missing-modality), not the fusion delta.
2. **Introduction** — ECG ubiquity vs. echo cost; why inference-time modality availability is the real deployment constraint; contributions.
3. **Related work** — foundation models for echo/ECG; **explicit positioning vs. EchoingECG**; cite own arXiv papers in third person.
4. **Data** — MIMIC-IV-Echo/ECG/Clinical; pairing protocol; cohort flow diagram; single-center caveat.
5. **Method** — frozen embeddings; attentive echo probe; cross-attention fusion; missing-modality protocol.
6. **Results** —
   - 6.1 Fusion vs. unimodal (supporting)
   - 6.2 **Missing-modality degradation** (centerpiece, money figure 1)
   - 6.3 **Fairness** (money figure 2)
   - 6.4 (opt.) calibration
7. **Discussion** — deployment implications; what fails silently; limitations.
8. **Future work** — Tasks B/C, shortcut-learning experiments, external validation.

### 4-page Case Report fallback
Intro (compressed) → Data + pairing → Fusion + missing-modality result → **Lessons learned / blockers** (lift directly from the roadmap's "Uncertainties and Blockers") → brief discussion. Negative/partial results are explicitly welcome.

---

## 9. Submission checklist (each item can desk-reject)

- ☐ **Anonymize.** Scrub deanonymizing details from all artifacts: source repo names, PhysioNet usernames, lab and institution names, shared cluster paths. Cite own related arXiv papers in **third person**.
- ☐ **Template:** COLM 2026 LaTeX template specifically (not NeurIPS/generic).
- ☐ **Page limits:** ≤8 pp research / ≤4 pp case report, excluding references.
- ☐ **Non-archival + dual-submission allowed** → this does **not** burn the future journal submission. A workshop-shaped slice now is free.
- ☐ **LLM-usage policy:** follow COLM 2026 policy.
- ☐ **Novelty positioning vs. EchoingECG** stated in the intro.
- ☐ Submit on OpenReview before the AoE wire.

---

## 10. Open items to resolve before Day 5
- [ ] Read EchoingECG (arXiv 2509.25791) and write the one-paragraph differentiation.
- [ ] Confirm the EF≤40% prevalence in the paired cohort is high enough for a stable AUROC.
- [ ] Decide pooling strategy for ECG-FM embeddings (mean vs. attentive) — quick empirical check.
- [ ] Confirm GPU + storage budget for cached embeddings on ORCD.

---

*References (from roadmap): EchoJEPA arXiv:2602.02603 · ECG-FM arXiv:2408.05178 · EchoingECG arXiv:2509.25791 · MIMIC-IV-Echo 0.1 · MIMIC-IV-ECG 1.0 · MIMIC-IV 3.1 (PhysioNet, credentialed + DUA).*