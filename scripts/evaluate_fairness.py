"""Post-hoc fairness audit: stratify test predictions by sex, age_band, race (E03).

This script loads test-set predictions from E01 (missing-modality eval) and
stratifies metrics (MAE, AUROC) by demographic attributes to assess equity.

Usage:
  python scripts/evaluate_fairness.py \
    --cohort cohort/paired.parquet \
    --echo-embeddings embeddings/echo/pairs.parquet \
    --ecg-embeddings embeddings/ecg/pairs.parquet \
    --checkpoint probes/cross_attn_fused/cross_attn_fused.pt \
    --out results/fairness
"""
from __future__ import annotations

import argparse

from primed_ai.evaluation.fairness import run_fairness


def main():
    p = argparse.ArgumentParser(description="E03: Post-hoc fairness stratification (wrapper).")
    p.add_argument("--cohort", required=True, help="Cohort path (with demographics)")
    p.add_argument("--echo-embeddings", required=True, help="Echo embedding path")
    p.add_argument("--ecg-embeddings", required=True, help="ECG embedding path")
    p.add_argument("--checkpoint", required=True, help="M09 checkpoint path")
    p.add_argument("--out", default="results/fairness", help="Output directory")
    p.add_argument("--embed-dim", type=int, default=16, help="Fusion dimension (matches the checkpoint).")
    p.add_argument("--echo-dim", type=int, default=None, help="Echo embedding dim, e.g. 1024.")
    p.add_argument("--ecg-dim", type=int, default=None, help="ECG embedding dim, e.g. 768.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--device", default=None, help="Device (cuda or cpu)")
    args = p.parse_args()

    run_fairness(
        args.cohort,
        args.echo_embeddings,
        args.ecg_embeddings,
        args.checkpoint,
        out_dir=args.out,
        embed_dim=args.embed_dim,
        echo_dim=args.echo_dim,
        ecg_dim=args.ecg_dim,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Wrote fairness outputs to: {args.out}")


if __name__ == "__main__":
    main()
