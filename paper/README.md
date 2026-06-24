# Paper — DAIH @ COLM 2026 (submitted)

**"Loud or Silent? A Reusable Framework for Per-Modality Failure Analysis in Multimodal Clinical
AI."** An 8-page research paper proposing a model-agnostic **modality-failure framework** (failure
taxonomy · complementarity matrix · loud-vs-silent dropout profile), validated on synthetic data
(multi-seed + 3-modality) and demonstrated on a real echo+ECG LVEF cohort. The harness lives in
`src/primed_ai/failure/`. **Status: submitted to DAIH @ COLM 2026.**

## Layout

```
paper/
├── case_report.tex      # main document (compile this)
├── references.bib       # bibliography
├── latexmkrc            # TEXINPUTS → template/; output → build/
├── template/            # vendored COLM 2026 style (do not edit)
├── figures/             # figures for \includegraphics{...}
├── fragments/           # \input{} snippets (results_section is integrated)
├── drafts/              # markdown working notes
└── build/               # PDF + LaTeX intermediates (gitignored)
```

## Build

```bash
cd paper
latexmk -pdf case_report.tex
open build/case_report.pdf
```

Requires TeX Live / MacTeX with `tikz`, `microtype`, and `latexmk`.

## Results & figures

`fragments/results_section.tex` (the **Results** section) is integrated via
`\input{fragments/results_section}`. Figures: `figures/taxonomy.pdf` + `figures/complementarity3.pdf`
(synthetic validation + 3-modality generality, from `scripts/validate_harness.py` and
`scripts/run_failure_analysis.py --demo`) and `figures/degradation_curve.pdf` (real-cohort demo,
from `scripts/plot_degradation_curve.py`).

## Submission checklist (done)

- [x] **Page limit:** 8 pages (body) under the COLM 2026 template, references excluded.
- [x] **Anonymization:** `[submission]` mode; no author names, cluster paths, or usernames in the PDF.
- [x] **References:** `references.bib` complete (author lists from arXiv / PhysioNet).
- [x] **LLM-usage statement** included per COLM 2026 policy.
- [x] **Submitted** to DAIH @ COLM 2026 on OpenReview.

> Caveat: if the code is ever attached as supplementary, scrub the `PRIMED-AI` repo name and commit
> metadata to preserve double-blind.

## Optional upgrades (same draft)

- **Tier 1 (BigQuery):** cohort N, EF≤40% prevalence, Figure 1 counts → §3 / §5.
- **Tier 2 (GPU):** ECG-only probe MAE/AUROC → §5 or `fragments/results_section.tex`.
