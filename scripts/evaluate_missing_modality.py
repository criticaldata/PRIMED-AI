"""Evaluate a saved M09 cross-attention fused probe under missing-modality conditions.

Usage example:
python scripts/evaluate_missing_modality.py \
  --cohort cohort/paired.parquet \
  --echo-embeddings embeddings/echo/pairs.parquet \
  --ecg-embeddings embeddings/ecg/pairs.parquet \
  --checkpoint probes/cross_attn_fused/cross_attn_fused.pt \
  --out results/missing_modality.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from primed_ai.probes import cross_attn
from primed_ai.probes.common import read_table, save_results
from primed_ai.probes.echo_only import _ensure_tokens
from primed_ai.probes.concat_mlp import _ensure_ecg_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--echo-embeddings", required=True)
    p.add_argument("--ecg-embeddings", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="results/missing_modality.json")
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    coh = read_table(args.cohort)
    echo = read_table(args.echo_embeddings)
    ecg = read_table(args.ecg_embeddings)

    echo_key = "echo_study_id" if "echo_study_id" in coh.columns else "subject_id"
    ecg_key = "ecg_record_id"
    if echo_key not in echo.columns:
        echo = echo.rename(columns={echo.columns[0]: echo_key})
    if ecg_key not in ecg.columns:
        ecg = ecg.rename(columns={ecg.columns[0]: ecg_key})

    df = coh.merge(echo, on=echo_key, how="inner").merge(ecg, on=ecg_key, how="inner")
    df = _ensure_tokens(df)
    df = _ensure_ecg_tokens(df)

    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if len(parts["test"]) == 0:
        raise SystemExit("No test partition found in cohort; aborting")

    # build minimal dataloader using the same collate fn as training
    from torch.utils.data import DataLoader
    from primed_ai.probes.common import TokenEmbeddingDataset, collate_tokens

    ds = TokenEmbeddingDataset(parts["test"], ecg_col="ecg_tokens")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_tokens(b, pad_dim=args.embed_dim))

    model = cross_attn.CrossAttnFusedProbe(args.embed_dim).to(device)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)

    results = {
        "task": "E01_missing_modality_eval",
        "checkpoint": str(ckpt),
        "git_sha": cross_attn.git_sha(),
        "n": {"test": len(parts["test"])},
        "test": {},
    }

    for cond in ("full", "echo_dropped", "ecg_dropped"):
        metrics = cross_attn._eval_condition(model, loader, device, cond)
        results["test"][cond] = metrics

    save_results(Path(args.out), results)
    print(f"Saved missing-modality results to {args.out}")


if __name__ == "__main__":
    main()
