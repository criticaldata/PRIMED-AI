"""Train a tiny synthetic cross-attn probe and run missing-modality + fairness evaluation.

This script is for local testing to produce a checkpoint and result files
without access to actual MIMIC data. It writes outputs under `artifacts/` and `results/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from primed_ai.probes.cross_attn import run as train_cross


def _synthetic(root: Path, n=200, dim=8, seed=2):
    rng = np.random.default_rng(seed)
    subs = rng.integers(10_000_000, 99_999_999, size=n)
    signal = rng.standard_normal(dim)
    lvef = np.clip(50 + rng.standard_normal(n) * 8, 15, 85)
    echo_emb = rng.standard_normal((n, dim)) + signal * ((lvef - 50) / 20)[:, None]
    ecg_emb = rng.standard_normal((n, dim)) + signal * ((lvef - 50) / 25)[:, None]

    coh = pd.DataFrame({
        "subject_id": subs,
        "echo_study_id": np.arange(2000, 2000 + n),
        "ecg_record_id": np.arange(3000, 3000 + n),
        "lvef": lvef,
        "ef_le_40": lvef <= 40,
    })
    uniq = pd.unique(coh["subject_id"])
    rng.shuffle(uniq)
    ntr, nva = int(len(uniq) * 0.7), int(len(uniq) * 0.1)
    smap = {s: ("train" if i < ntr else "val" if i < ntr + nva else "test") for i, s in enumerate(uniq)}
    coh["split"] = coh["subject_id"].map(smap)

    echo = pd.DataFrame(echo_emb, columns=[f"echo_ve{i:04d}" for i in range(dim)])
    echo.insert(0, "echo_study_id", coh["echo_study_id"].values)
    ecg = pd.DataFrame(ecg_emb, columns=[f"ve{i:04d}" for i in range(dim)])
    ecg.insert(0, "ecg_record_id", coh["ecg_record_id"].values)

    root.mkdir(parents=True, exist_ok=True)
    cpath = root / "cohort.parquet"
    echo_path = root / "echo.parquet"
    ecg_path = root / "ecg.parquet"
    coh.to_parquet(cpath)
    echo.to_parquet(echo_path)
    ecg.to_parquet(ecg_path)
    return cpath, echo_path, ecg_path


def main():
    art = Path("artifacts")
    cpath, echo_path, ecg_path = _synthetic(art, n=250, dim=16)
    out = art / "probes" / "cross_attn_fused"
    # small epochs to keep runtime short
    res = train_cross(cpath, echo_path, ecg_path, out_dir=out, embed_dim=16, epochs=6, batch_size=64)
    print("Training done. Results summary:", res["test"]["full"]) 
    
    # Add demographic data to cohort for fairness eval
    coh = pd.read_parquet(cpath)
    rng = np.random.default_rng(42)
    coh["sex"] = rng.choice(["M", "F"], size=len(coh))
    coh["age_band"] = rng.choice(["18-40", "40-60", "60-80", "80+"], size=len(coh))
    coh["race"] = rng.choice(["White", "Black", "Asian", "Other"], size=len(coh))
    coh.to_parquet(cpath)
    print("Added demographics to cohort")
    
    # Run E01 evaluation using the new script
    ckpt = out / "cross_attn_fused.pt"
    from subprocess import run
    run([
        "python", "scripts/evaluate_missing_modality.py",
        "--cohort", str(cpath),
        "--echo-embeddings", str(echo_path),
        "--ecg-embeddings", str(ecg_path),
        "--checkpoint", str(ckpt),
        "--out", "results/missing_modality.json"
    ], check=True)
    
    # Run E03 fairness eval
    run([
        "python", "scripts/evaluate_fairness.py",
        "--cohort", str(cpath),
        "--echo-embeddings", str(echo_path),
        "--ecg-embeddings", str(ecg_path),
        "--checkpoint", str(ckpt),
        "--out", "results/fairness"
    ], check=True)


if __name__ == "__main__":
    main()
