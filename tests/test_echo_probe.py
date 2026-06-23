"""Unit tests for echo-only attentive probe (M07)."""

import numpy as np
import pandas as pd

from primed_ai.probes.echo_only import run


def _synthetic(tmp_path, n=240, dim=8, seed=1):
    rng = np.random.default_rng(seed)
    subs = rng.integers(10_000_000, 99_999_999, size=n)
    lvef = np.clip(50 + rng.standard_normal(n) * 10, 10, 90)
    weight = rng.standard_normal(dim)
    emb = rng.standard_normal((n, dim)) + weight * ((lvef - 50) / 15)[:, None]

    coh = pd.DataFrame({
        "subject_id": subs,
        "echo_study_id": np.arange(1000, 1000 + n),
        "lvef": lvef,
        "ef_le_40": lvef <= 40,
    })
    uniq = pd.unique(coh["subject_id"])
    rng.shuffle(uniq)
    ntr, nva = int(len(uniq) * 0.7), int(len(uniq) * 0.1)
    smap = {s: ("train" if i < ntr else "val" if i < ntr + nva else "test")
            for i, s in enumerate(uniq)}
    coh["split"] = coh["subject_id"].map(smap)

    echo = pd.DataFrame(emb, columns=[f"echo_ve{i:04d}" for i in range(dim)])
    echo.insert(0, "echo_study_id", coh["echo_study_id"].values)
    cpath, epath = tmp_path / "cohort.parquet", tmp_path / "echo_emb.parquet"
    coh.to_parquet(cpath)
    echo.to_parquet(epath)
    return cpath, epath


def test_echo_probe_trains_and_saves_checkpoint(tmp_path):
    cpath, epath = _synthetic(tmp_path, n=400, dim=16, seed=1)
    out = tmp_path / "probes" / "echo_only"
    res = run(cpath, epath, out_dir=out, embed_dim=16, epochs=40, batch_size=32)
    assert (out / "echo_only.pt").is_file()
    assert (out / "results.json").is_file()
    assert res["n"]["train"] > 0
    assert np.isfinite(res["val"]["attentive"]["mae"])
