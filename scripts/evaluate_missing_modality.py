"""Evaluate a trained M09 fused probe under inference-time missing modalities.

Example:
  python scripts/evaluate_missing_modality.py \
    --checkpoint probes/cross_attn_fused/cross_attn_fused.pt \
    --cohort data/raw/cohort/paired_with_splits.parquet \
    --echo-embeddings data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet \
    --ecg-embeddings data/interim/hubert_ecg_embeddings.parquet \
    --embed-dim 256 \
    --echo-dim 1024 \
    --ecg-dim 768 \
    --output results/missing_modality.json
"""

from __future__ import annotations

import argparse

from primed_ai.evaluation.missing_modality import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="data/raw/cohort/paired_with_splits.parquet")
    parser.add_argument(
        "--echo-embeddings",
        default="data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet",
    )
    parser.add_argument("--ecg-embeddings", default="data/interim/hubert_ecg_embeddings.parquet")
    parser.add_argument("--checkpoint", default="probes/cross_attn_fused/cross_attn_fused.pt")
    parser.add_argument("--output", "--out", dest="output", default="results/missing_modality.json")
    parser.add_argument("--embed-dim", type=int, default=16, help="Shared fusion dimension.")
    parser.add_argument("--echo-dim", type=int, default=None)
    parser.add_argument("--ecg-dim", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    results = run(
        args.cohort,
        args.echo_embeddings,
        args.ecg_embeddings,
        args.checkpoint,
        args.output,
        embed_dim=args.embed_dim,
        echo_dim=args.echo_dim,
        ecg_dim=args.ecg_dim,
        hidden=args.hidden,
        batch_size=args.batch_size,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        device=args.device,
    )
    for row in results["metrics_table"]:
        print(f"{row['condition']:12} MAE {row['mae']:.4f} | EF<=40 AUROC {row['ef40_auroc']:.4f}")
    print(f"\nWrote missing-modality metrics to: {args.output}")


if __name__ == "__main__":
    main()
