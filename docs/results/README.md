# Sanitized result artifacts

Aggregate-only copies of the local result JSONs backing the numbers in the top-level
README — per-example predictions and checkpoints stay local (`results/` and `probes/`
are gitignored), so this is what reviewers can diff without data access.

Written by `scripts/export_result_bundle.py`, which strips the per-example
`predictions` / `predictions_val` blocks from the missing-modality JSON and copies the
rest verbatim. `SHA256SUMS` covers both these files and the local raw artifacts
(manifest, the four canonical checkpoints, the full missing-modality JSON); verify a
reproduction from the repo root with:

```bash
shasum -c docs/results/SHA256SUMS
```

Lines for local raw artifacts fail with "No such file" until you have rebuilt them —
that is the checklist of what your reproduction still has to produce.

## Which model produced which file

Every file here scores the cross-attention fused checkpoint
(`probes/fused/cross_attn_fused.pt`, SHA-256 `bac18bb8…`) **except**
`failure_report.ridge.json`, which comes from the harness's own Ridge on concatenated
embeddings with absent modalities zeroed at inference.

That is a different model, and it disagrees sharply on the dropped conditions — the ridge
report puts drop-echo MAE at 117.5 where the fused checkpoint gives 20.55 — so the two
must never share a table.

`failure_report.fused.json` — **not yet in this bundle** — is the file the paper figures
should be built from: the same three views computed off the fused checkpoint's own
per-example predictions, with no model fitted in between. Produce it with

```bash
python scripts/run_failure_analysis.py --predictions results/missing_modality.json
python scripts/export_result_bundle.py
```

on a machine that has the manifest and the raw `results/`. Every regenerated failure
artifact carries a `provenance` block naming its producing model; the ridge copy committed
here predates both that block and the fused route, so it is still the pre-#79 file.
