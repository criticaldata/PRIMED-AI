"""Unit tests for fused probes (M08, M09)."""

import numpy as np
import pandas as pd

from primed_ai.probes.concat_mlp import run as run_concat
from primed_ai.probes.cross_attn import run as run_cross


def _synthetic(tmp_path, n=200, dim=8, seed=2):
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
    smap = {s: ("train" if i < ntr else "val" if i < ntr + nva else "test")
            for i, s in enumerate(uniq)}
    coh["split"] = coh["subject_id"].map(smap)

    echo = pd.DataFrame(echo_emb, columns=[f"echo_ve{i:04d}" for i in range(dim)])
    echo.insert(0, "echo_study_id", coh["echo_study_id"].values)
    ecg = pd.DataFrame(ecg_emb, columns=[f"ve{i:04d}" for i in range(dim)])
    ecg.insert(0, "ecg_record_id", coh["ecg_record_id"].values)

    cpath = tmp_path / "cohort.parquet"
    echo_path = tmp_path / "echo.parquet"
    ecg_path = tmp_path / "ecg.parquet"
    coh.to_parquet(cpath)
    echo.to_parquet(echo_path)
    ecg.to_parquet(ecg_path)
    return cpath, echo_path, ecg_path


def test_concat_mlp_probe(tmp_path):
    cpath, echo_path, ecg_path = _synthetic(tmp_path, n=400, dim=16)
    out = tmp_path / "probes" / "concat_mlp"
    res = run_concat(cpath, echo_path, ecg_path, out_dir=out,
                      echo_dim=16, ecg_dim=16, epochs=40)
    assert (out / "concat_mlp.pt").is_file()
    assert np.isfinite(res["val"]["mae"])


def test_cross_attn_missing_modality(tmp_path):
    cpath, echo_path, ecg_path = _synthetic(tmp_path)
    out = tmp_path / "probes" / "cross_attn"
    res = run_cross(cpath, echo_path, ecg_path, out_dir=out, embed_dim=8, epochs=20)
    assert res["supports_missing_modality_inference"] is True
    assert "echo_dropped" in res["test"]
    assert "ecg_dropped" in res["test"]
    assert (out / "cross_attn_fused.pt").is_file()
