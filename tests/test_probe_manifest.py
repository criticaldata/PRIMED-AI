"""All four probes must train straight off the joined manifest (M10, #63).

Fixture data here is synthetic and exists only to exercise the wiring — it says
nothing about model quality. LVEF is a linear function of the embeddings so the
probes have something learnable to fit; the assertions check plumbing (shapes,
splits, checkpoints, metric keys), not accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("torch")

from primed_ai.probes import manifest
from primed_ai.probes.concat_mlp import run as run_concat
from primed_ai.probes.cross_attn import prepare_fused_probe_data
from primed_ai.probes.cross_attn import run as run_fused
from primed_ai.probes.ecg_only import run as run_ecg
from primed_ai.probes.echo_only import run as run_echo

ECHO_DIM, ECG_DIM = 8, 6


def _manifest(path, n=60):
    rng = np.random.default_rng(0)
    echo = rng.standard_normal((n, ECHO_DIM)).astype(np.float32)
    ecg = rng.standard_normal((n, ECG_DIM)).astype(np.float32)
    lvef = 45.0 + 8.0 * echo[:, 0] + 3.0 * ecg[:, 0]
    # deterministic split so every partition is non-empty regardless of n
    split = np.array(["train", "val", "test"])[np.arange(n) % 3]
    pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "echo_study_id": np.arange(n) + 1000,
            "ecg_study_id": np.arange(n) + 2000,
            "lvef": lvef,
            "ef_le_40": lvef <= 40.0,
            "split": split,
            "echo_embedding": list(echo),
            "ecg_embedding": list(ecg),
        }
    ).to_parquet(path, index=False)
    return path


def test_load_parses_embeddings_and_reports_dims(tmp_path):
    df = manifest.load(_manifest(tmp_path / "m.parquet"))
    assert manifest.dims(df) == (ECHO_DIM, ECG_DIM)
    assert np.asarray(df["echo_embedding"].iloc[0]).shape == (ECHO_DIM,)


def test_load_rejects_a_table_that_is_not_a_manifest(tmp_path):
    path = tmp_path / "plain.parquet"
    pd.DataFrame({"subject_id": [1], "lvef": [50.0]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="not a joined manifest"):
        manifest.load(path)


def test_load_drops_rows_missing_an_embedding(tmp_path):
    path = _manifest(tmp_path / "m.parquet", n=9)
    df = pd.read_parquet(path)
    df.loc[0, "ecg_embedding"] = None
    df.to_parquet(path, index=False)
    assert len(manifest.load(path)) == 8


def test_expand_flattens_ecg_into_prefixed_columns(tmp_path):
    df = manifest.load(_manifest(tmp_path / "m.parquet"))
    flat, cols = manifest.expand(df, manifest.ECG_COLUMN, "ve")
    assert cols == [f"ve{i}" for i in range(ECG_DIM)]
    assert manifest.ECG_COLUMN not in flat.columns
    np.testing.assert_allclose(
        flat[cols].iloc[0].to_numpy(), df["ecg_embedding"].iloc[0], rtol=1e-6
    )


def test_fused_data_prep_splits_without_an_explicit_join(tmp_path):
    parts = prepare_fused_probe_data(_manifest(tmp_path / "m.parquet"))
    assert set(parts) == {"train", "val", "test"}
    assert all(len(p) > 0 for p in parts.values())
    assert np.asarray(parts["train"]["echo_tokens"].iloc[0]).shape[-1] == ECHO_DIM


def test_half_specified_embedding_paths_are_rejected(tmp_path):
    path = _manifest(tmp_path / "m.parquet")
    with pytest.raises(ValueError, match="both embedding paths"):
        prepare_fused_probe_data(path, tmp_path / "echo.parquet", None)


def test_attentive_pooling_is_inert_on_study_level_embeddings():
    """A pooled study vector is tiled into identical tokens, so attention cannot discriminate.

    Softmax over identical scores is uniform, which makes ``AttentivePool`` an exact
    identity here — the attentive echo probe degenerates to an MLP on the mean-pooled
    vector, which TECHNICAL.md 4.1 says is the insufficient case. Pinned so the day
    clip-level tokens land, this test fails and flags that the collapse is gone.
    """
    import torch

    from primed_ai.probes.echo_only import _embedding_to_tokens
    from primed_ai.probes.layers import AttentivePool

    torch.manual_seed(0)
    vec = np.random.default_rng(0).standard_normal(32).astype(np.float32)
    tokens = torch.as_tensor(_embedding_to_tokens(vec, n_tokens=8))[None]

    pooled = AttentivePool(32)(tokens).detach().numpy()[0]
    np.testing.assert_allclose(pooled, vec, atol=1e-6)
    np.testing.assert_allclose(pooled, tokens.mean(1).numpy()[0], atol=1e-6)


@pytest.mark.parametrize("name", ["ecg", "echo", "concat", "fused"])
def test_every_probe_trains_from_the_manifest(tmp_path, name):
    path = _manifest(tmp_path / "m.parquet")
    out = tmp_path / name

    if name == "ecg":
        res = run_ecg(path, out_dir=str(out), n_bootstrap=20)
        assert res["embedding_dim"] == ECG_DIM
        assert (out / "ecg_only.joblib").exists()
    elif name == "echo":
        res = run_echo(path, out_dir=str(out), embed_dim=ECHO_DIM, epochs=2)
        assert (out / "echo_only.pt").exists()
    elif name == "concat":
        res = run_concat(path, out_dir=str(out), echo_dim=ECHO_DIM, ecg_dim=ECG_DIM, epochs=2)
        assert (out / "concat_mlp.pt").exists()
    else:
        res = run_fused(
            path,
            out_dir=str(out),
            embed_dim=ECHO_DIM,
            echo_dim=ECHO_DIM,
            ecg_dim=ECG_DIM,
            epochs=2,
        )
        assert (out / "cross_attn_fused.pt").exists()

    assert (out / "results.json").exists()
    assert sum(res["n"].values()) > 0
