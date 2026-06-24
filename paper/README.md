# Paper — DAIH @ COLM 2026 (Case Report track)

Frozen-embedding multimodal LVEF estimation, written as a **4-page deployment case report**
(the project's designed fallback track). Tier-0: no end-to-end experimental run; the contribution
is the released pipeline + protocols + the data-access lessons.

## Files
- `case_report.tex` — the paper (compilable, self-contained preamble).
- `references.bib` — bibliography (author lists/MIMIC cites have `TODO` placeholders).
- `case_report_draft.md` — the prose source/working draft (kept for reference).

## Build
```bash
cd paper
latexmk -pdf case_report.tex
# or: pdflatex case_report && bibtex case_report && pdflatex case_report && pdflatex case_report
```
Produces `case_report.pdf`. Requires a TeX distribution (TeX Live / MacTeX) with `tikz` + `natbib`.

## Before submission — checklist (each item can desk-reject)
- [ ] **COLM 2026 template.** Drop the official `colm2026_conference.sty`/`.cls` next to the `.tex`
      and replace the portable preamble (marked `TEMPLATE NOTE` at the top). The current preamble
      is a stand-in only — **wrong template = desk reject.**
- [ ] **Page limit.** Confirm ≤ 4 pages excluding references *under the COLM template* (margins
      differ from the stand-in, so length will shift).
- [ ] **Anonymization (double-blind).** No author names/affiliations; no repo names (e.g. the
      `EchoJEPA-VE` repo), PhysioNet usernames, lab/cluster paths (e.g. `/orcd/...`). Cite our own
      arXiv papers in the third person. Run a grep over the `.tex` for identifying strings.
- [ ] **References.** Complete author lists / verify MIMIC citations (`references.bib` TODOs).
- [ ] **W01 positioning.** Tighten the EchoingECG differentiation paragraph (§2) after reading
      arXiv:2509.25791.
- [ ] **LLM-usage statement** per the COLM 2026 policy.
- [ ] **Submit** on OpenReview (DAIH @ COLM 2026); declare the **Case Report** track; archive the
      confirmation + a PDF snapshot.

## Optional upgrades if access clears (keeps the same draft)
- **Tier 1 (BigQuery, no GPU):** run the cohort builder → real cohort N, EF≤40% prevalence,
  demographic coverage, and Figure 1 counts. Insert at the marked spots in §3 and §5.
- **Tier 2 (one GPU):** run the ECG-only probe → one real MAE/AUROC result. Insert in §5.
