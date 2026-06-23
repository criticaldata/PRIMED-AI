#!/usr/bin/env python3
"""Train the ECG-only LVEF probe (M06).

Reads a subject-level split cohort (with ``lvef``, ``ef_le_40``, ``split``) and a
per-record ECG embedding table, trains Ridge (regression) + LogisticRegression
(EF<=40 gate) heads, and writes a checkpoint + results JSON to ``--out-dir``.

Example:
  python scripts/train_ecg_probe.py \
      --cohort cohort/paired_with_splits.parquet \
      --embeddings ecg_hubert_cohort.csv \
      --out-dir probes/ecg_only
"""
from __future__ import annotations

import argparse

from primed_ai.probes.ecg_only import run


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True,
                    help="Split-labeled cohort (parquet/csv) with lvef, ef_le_40, split.")
    ap.add_argument("--embeddings", required=True,
                    help="Per-record ECG embedding table (parquet/csv).")
    ap.add_argument("--out-dir", default="probes/ecg_only")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()

    res = run(args.cohort, args.embeddings, out_dir=args.out_dir,
              seed=args.seed, n_bootstrap=args.n_bootstrap)

    n = res["n"]
    print(f"n train/val/test = {n['train']}/{n['val']}/{n['test']} | "
          f"dropped {res['n_dropped_nonfinite']} non-finite | "
          f"EF<=40 test positives = {res['ef40_test_positives']}")
    print(f"ridge alpha={res['ridge_alpha']:.3g} | logreg C={res['logreg_C']:.3g}")
    for nm in ("val", "test"):
        m = res["splits"][nm]
        print(f"{nm:5} baseMAE {m['baseline_mae']:.2f} | ridgeMAE {m['ridge_mae']:.2f} | "
              f"EF40 AUROC reg {m['ef40_auroc_from_regression']:.3f} "
              f"logreg {m['ef40_auroc_logreg']:.3f}")
    t = res["splits"]["test"]
    print(f"TEST MAE {t['ridge_mae']} CI {t['ridge_mae_ci95']} (baseline {t['baseline_mae']}) | "
          f"EF<=40 AUROC {t['ef40_auroc_logreg']} CI {t['ef40_auroc_logreg_ci95']}")
    print(f"Saved -> {args.out_dir}/results.json , {args.out_dir}/ecg_only.joblib")


if __name__ == "__main__":
    main()
