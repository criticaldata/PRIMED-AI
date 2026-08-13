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
