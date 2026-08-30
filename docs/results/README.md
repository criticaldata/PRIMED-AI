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

One caveat on what a pass certifies: probe training is not bit-reproducible across
machines (the same manifest and seed give different checkpoint bytes), so the four
checkpoint lines and the raw `results/missing_modality.json` line only pass on the
machine that last regenerated this bundle. On an independent rerun expect those five to
fail even when every reported number matches — the manifest line and the `docs/results/`
lines are the ones that certify a reproduction.

## Which model produced which file

Every file here scores the cross-attention fused checkpoint
(`probes/fused/cross_attn_fused.pt`, SHA-256 `7cae4f92…`) **except**
`failure_report.ridge.json`, which comes from the harness's own Ridge on concatenated
embeddings with absent modalities zeroed at inference.

That is a different model, and it disagrees sharply on the dropped conditions — the ridge
report puts drop-echo MAE at 117.5 where the fused checkpoint gives 20.54 — so the two
must never share a table. The committed copy carries a `provenance` block naming the
producing model. #79 adds the fused-checkpoint failure report that the paper figures
should be built from.
