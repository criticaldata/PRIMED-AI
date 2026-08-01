# Unit tests for the ECG-only probe (M06) on synthetic data.

import numpy as np
import pandas as pd

from primed_ai.probes.ecg_only import _canon, load_dataset, run


def _make_data(tmp_path, n=300, dim=16, seed=0):
    """Synthetic cohort + per-record embeddings with a planted LVEF signal."""
    rng = np.random.default_rng(seed)
    n_subj = int(n / 1.2)  # ~1.2 records/subject so some subjects repeat
    subj = rng.integers(10_000_000, 99_999_999, size=n_subj)
    paths, recs, subs = [], [], []
    for i in range(n):
        s = int(subj[i % n_subj])
        rec = 40_000_000 + i
        subs.append(s)
        recs.append(rec)
        paths.append(f"files/p{str(s)[:4]}/p{s}/s{rec}/{rec}")
    coh = pd.DataFrame({"subject_id": subs, "ecg_record_id": recs, "ecg_path": paths})

    weight = rng.standard_normal(dim)
    emb_mat = rng.standard_normal((n, dim))
    with np.errstate(all="ignore"):  # spurious numpy/Accelerate matmul warnings on darwin
        z = (emb_mat @ weight) / np.sqrt(dim)
    coh["lvef"] = np.clip(50 + 15 * z + rng.standard_normal(n) * 2, 5, 95)  # strong signal
    coh["ef_le_40"] = coh["lvef"] <= 40

    uniq = pd.unique(coh["subject_id"])
    rng.shuffle(uniq)
    ntr, nva = int(len(uniq) * 0.7), int(len(uniq) * 0.1)
    smap = {
        s: ("train" if i < ntr else "val" if i < ntr + nva else "test") for i, s in enumerate(uniq)
    }
    coh["split"] = coh["subject_id"].map(smap)

    emb = pd.DataFrame(emb_mat, columns=[f"ve{i:04d}" for i in range(dim)])
    emb.insert(0, "filename", coh["ecg_path"].values)
    cpath, epath = tmp_path / "cohort.parquet", tmp_path / "emb.csv"
    coh.to_parquet(cpath)
    emb.to_csv(epath, index=False)
    return cpath, epath


def test_canon_handles_path_s_and_bare():
    assert _canon("files/p1002/p10024736/s42282815/42282815") == "42282815"
    assert _canon("s42282815") == "42282815"
    assert _canon(42282815) == "42282815"


def test_load_drops_nonfinite(tmp_path):
    cpath, epath = _make_data(tmp_path)
    emb = pd.read_csv(epath)
    emb.loc[0, "ve0000"] = np.nan
    emb.to_csv(epath, index=False)
    df, ve, n_dropped = load_dataset(cpath, epath)
    assert n_dropped == 1
    assert len(ve) == 16
    assert np.isfinite(df[ve].to_numpy()).all()


def test_no_subject_leakage_in_splits(tmp_path):
    cpath, _ = _make_data(tmp_path)
    coh = pd.read_parquet(cpath)
    assert (coh.groupby("subject_id")["split"].nunique() == 1).all()


def test_run_beats_baseline_and_saves(tmp_path):
    cpath, epath = _make_data(tmp_path)
    out = tmp_path / "probes" / "ecg_only"
    res = run(cpath, epath, out_dir=out, n_bootstrap=200)
    assert (out / "results.json").is_file()
    assert (out / "ecg_only.joblib").is_file()
    t = res["splits"]["test"]
    assert t["ridge_mae"] < t["baseline_mae"]  # planted signal -> beats mean baseline
    assert t["ef40_auroc_from_regression"] > 0.6  # and discriminates EF<=40
