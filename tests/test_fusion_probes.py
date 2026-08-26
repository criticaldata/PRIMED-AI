"""Unit tests for fused probes (M08, M09)."""

import numpy as np
import pandas as pd
import torch

from primed_ai.evaluation.missing_modality import _bootstrap_ci
from primed_ai.evaluation.missing_modality import run as run_missing_modality_eval
from primed_ai.probes.concat_mlp import run as run_concat
from primed_ai.probes.cross_attn import CrossAttnFusedProbe
from primed_ai.probes.cross_attn import run as run_cross


def _synthetic(tmp_path, n=200, dim=8, seed=2):
    rng = np.random.default_rng(seed)
    subs = rng.integers(10_000_000, 99_999_999, size=n)
    signal = rng.standard_normal(dim)
    lvef = np.clip(50 + rng.standard_normal(n) * 8, 15, 85)
    echo_emb = rng.standard_normal((n, dim)) + signal * ((lvef - 50) / 20)[:, None]
    ecg_emb = rng.standard_normal((n, dim)) + signal * ((lvef - 50) / 25)[:, None]

    coh = pd.DataFrame(
        {
            "subject_id": subs,
            "echo_study_id": np.arange(2000, 2000 + n),
            "ecg_record_id": np.arange(3000, 3000 + n),
            "lvef": lvef,
            "ef_le_40": lvef <= 40,
        }
    )
    uniq = pd.unique(coh["subject_id"])
    rng.shuffle(uniq)
    ntr, nva = int(len(uniq) * 0.7), int(len(uniq) * 0.1)
    smap = {
        s: ("train" if i < ntr else "val" if i < ntr + nva else "test") for i, s in enumerate(uniq)
    }
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
    res = run_concat(cpath, echo_path, ecg_path, out_dir=out, echo_dim=16, ecg_dim=16, epochs=40)
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


def test_cross_attn_projects_mismatched_modalities():
    model = CrossAttnFusedProbe(embed_dim=8, echo_dim=10, ecg_dim=6)
    pred = model(torch.randn(4, 3, 10), torch.randn(4, 2, 6))
    assert pred.shape == (4,)


def test_e01_missing_modality_eval_loads_checkpoint(tmp_path):
    cpath, echo_path, ecg_path = _synthetic(tmp_path)
    echo_df = pd.read_parquet(echo_path)
    echo_cols = [c for c in echo_df.columns if c.startswith("echo_ve")]
    echo_df = pd.DataFrame(
        {
            "echo_study_id": echo_df["echo_study_id"],
            "echo_embedding": list(echo_df[echo_cols].to_numpy()),
        }
    )
    echo_df.to_parquet(echo_path)

    ecg_df = pd.read_parquet(ecg_path)
    ecg_cols = [c for c in ecg_df.columns if c.startswith("ve")]
    ecg_df = pd.DataFrame(
        {
            "ecg_study_id": ecg_df["ecg_record_id"],
            "ecg_embedding": list(ecg_df[ecg_cols].to_numpy()),
        }
    )
    ecg_df.to_parquet(ecg_path)

    probe_dir = tmp_path / "probes" / "cross_attn"
    run_cross(cpath, echo_path, ecg_path, out_dir=probe_dir, embed_dim=8, epochs=5)

    output = tmp_path / "results" / "missing_modality.json"
    res = run_missing_modality_eval(
        cpath,
        echo_path,
        ecg_path,
        probe_dir / "cross_attn_fused.pt",
        output,
        embed_dim=8,
        batch_size=32,
        n_bootstrap=20,
        device="cpu",
    )

    assert output.is_file()
    assert res["task"] == "E01_missing_modality_evaluation"
    assert [row["condition"] for row in res["metrics_table"]] == [
        "full",
        "echo_dropped",
        "ecg_dropped",
    ]
    assert set(res["test"]) == {"full", "echo_dropped", "ecg_dropped"}
    assert set(res["predictions"]) == {"full", "echo_dropped", "ecg_dropped"}
    assert "mae_ci_low" in res["metrics_table"][0]
    assert "ef40_auroc_ci_high" in res["metrics_table"][0]
    row = res["metrics_table"][0]
    assert row["n_auroc_replicates"] + row["n_auroc_discarded"] == row["n_bootstrap"] == 20


def test_bootstrap_ci_counts_discarded_auroc_replicates():
    # Every replicate of a single-class EF vector has an undefined AUROC, so the interval
    # has nothing to stand on -- it must be absent and the discards counted, not silently
    # reported as if 50 replicates backed it.
    rng = np.random.default_rng(0)
    arrays = {
        "lvef": rng.uniform(50, 70, size=40),
        "prediction": rng.uniform(50, 70, size=40),
        "ef_le_40": np.zeros(40, dtype=bool),
    }
    ci = _bootstrap_ci(arrays, n_bootstrap=50, seed=1)

    assert ci["n_bootstrap"] == 50
    assert ci["n_auroc_replicates"] == 0
    assert ci["n_auroc_discarded"] == 50
    assert "ef40_auroc_ci_low" not in ci
    assert np.isfinite(ci["mae_ci_low"]) and np.isfinite(ci["mae_ci_high"])


def test_bootstrap_ci_counts_add_up_when_auroc_is_defined():
    rng = np.random.default_rng(3)
    lvef = rng.uniform(20, 70, size=60)
    arrays = {
        "lvef": lvef,
        "prediction": lvef + rng.standard_normal(60),
        "ef_le_40": lvef <= 40,
    }
    ci = _bootstrap_ci(arrays, n_bootstrap=100, seed=7)

    assert ci["n_auroc_replicates"] + ci["n_auroc_discarded"] == 100
    assert ci["n_auroc_replicates"] > 0
    assert ci["ef40_auroc_ci_low"] <= ci["ef40_auroc_ci_high"]
