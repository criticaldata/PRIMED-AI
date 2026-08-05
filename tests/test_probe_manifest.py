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
from primed_ai.probes.common import drop_non_finite
from primed_ai.probes.concat_mlp import run as run_concat
from primed_ai.probes.cross_attn import prepare_fused_probe_data
from primed_ai.probes.cross_attn import run as run_fused
from primed_ai.probes.ecg_only import run as run_ecg
from primed_ai.probes.echo_only import run as run_echo

ECHO_DIM, ECG_DIM = 8, 6


def _clip_manifest(path, n=60, max_clips=4):
    """Synthetic clip-level manifest fixture: ``echo_embedding`` is ``(n_clips, ECHO_DIM)``.

    Mirrors what ``build_echo_study_embeddings(..., max_clips=N)`` writes — ragged clip
    counts, nested lists in parquet. Fixture only; the numbers mean nothing.
    """
    rng = np.random.default_rng(1)
    ecg = rng.standard_normal((n, ECG_DIM)).astype(np.float32)
    clips = [
        rng.standard_normal((1 + i % max_clips, ECHO_DIM)).astype(np.float32) for i in range(n)
    ]
    lvef = 45.0 + 8.0 * np.array([c[:, 0].mean() for c in clips]) + 3.0 * ecg[:, 0]
    pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "echo_study_id": np.arange(n) + 1000,
            "ecg_study_id": np.arange(n) + 2000,
            "lvef": lvef,
            "ef_le_40": lvef <= 40.0,
            "split": np.array(["train", "val", "test"])[np.arange(n) % 3],
            "echo_embedding": [c.tolist() for c in clips],
            "ecg_embedding": list(ecg),
        }
    ).to_parquet(path, index=False)
    return path


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
    parts, n_dropped = prepare_fused_probe_data(_manifest(tmp_path / "m.parquet"))
    assert set(parts) == {"train", "val", "test"}
    assert all(len(p) > 0 for p in parts.values())
    assert np.asarray(parts["train"]["echo_tokens"].iloc[0]).shape[-1] == ECHO_DIM
    assert n_dropped == 0


def _manifest_with_non_finite_ecg(path, n=60, bad=(0, 1, 2)):
    """Manifest whose ECG vectors carry NaN/inf inside otherwise present cells.

    This is the real-data failure @kevzho hit on the 1,208-row cohort: twelve rows held
    non-finite HuBERT-ECG vectors. ``manifest.load``'s ``require_both`` misses them
    because ``.notna()`` asks whether the cell is null, not what the vector contains.
    """
    _manifest(path, n=n)
    df = pd.read_parquet(path)
    ecg = np.vstack([np.asarray(v, dtype=np.float32) for v in df["ecg_embedding"]])
    for row, value in zip(bad, (np.nan, np.inf, -np.inf)):
        ecg[row, 0] = value
    df["ecg_embedding"] = list(ecg)
    df.to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("name", ["ecg", "echo", "concat", "fused"])
def test_non_finite_embeddings_are_dropped_not_trained_on(tmp_path, name):
    """Every probe must survive non-finite vectors, and say how many rows it lost."""
    path = _manifest_with_non_finite_ecg(tmp_path / "m.parquet")
    out = tmp_path / name
    kwargs = {"out_dir": str(out)}
    if name == "ecg":
        res = run_ecg(path, n_bootstrap=20, **kwargs)
    elif name == "echo":
        res = run_echo(path, embed_dim=ECHO_DIM, epochs=2, **kwargs)
    elif name == "concat":
        res = run_concat(path, echo_dim=ECHO_DIM, ecg_dim=ECG_DIM, epochs=2, **kwargs)
    else:
        res = run_fused(
            path, embed_dim=ECHO_DIM, echo_dim=ECHO_DIM, ecg_dim=ECG_DIM, epochs=2, **kwargs
        )

    # echo vectors are untouched, so the echo-only probe legitimately keeps every row
    expected = 0 if name == "echo" else 3
    assert res["n_dropped_nonfinite"] == expected
    assert sum(res["n"].values()) == 60 - expected


def test_drop_non_finite_also_catches_a_bad_label(tmp_path):
    df = pd.read_parquet(_manifest(tmp_path / "m.parquet"))
    df.loc[0, "lvef"] = np.nan
    kept, dropped = drop_non_finite(df, ("ecg_embedding",))
    assert dropped == 1 and len(kept) == len(df) - 1


def test_half_specified_embedding_paths_are_rejected(tmp_path):
    path = _manifest(tmp_path / "m.parquet")
    with pytest.raises(ValueError, match="both embedding paths"):
        prepare_fused_probe_data(path, tmp_path / "echo.parquet", None)


def test_attentive_pooling_is_inert_on_study_level_embeddings():
    """Documents the default (mean-pooled) manifest: tiled tokens collapse attention.

    A 1-D study vector is tiled into identical tokens, softmax over identical scores is
    uniform, and ``AttentivePool`` becomes an exact identity — the attentive echo probe
    degenerates to an MLP on the mean-pooled vector, which TECHNICAL.md 4.1 calls the
    insufficient case. The query gets exactly zero gradient, so training cannot escape it.
    Build with ``--max-clips`` to get out of this regime.
    """
    import torch

    from primed_ai.probes.echo_only import _embedding_to_tokens
    from primed_ai.probes.layers import AttentivePool

    torch.manual_seed(0)
    vec = np.random.default_rng(0).standard_normal(32).astype(np.float32)
    tokens = torch.as_tensor(_embedding_to_tokens(vec, n_tokens=8))[None]

    pool = AttentivePool(32)
    pooled = pool(tokens)
    np.testing.assert_allclose(pooled.detach().numpy()[0], vec, atol=1e-6)
    np.testing.assert_allclose(pooled.detach().numpy()[0], tokens.mean(1).numpy()[0], atol=1e-6)

    pooled.sum().backward()
    assert float(pool.query.grad.abs().max()) == 0.0


def test_attentive_pooling_discriminates_between_clip_tokens():
    """The point of clip retention: with distinct clips, attention stops being an identity.

    Synthetic fixture (six random clip vectors). The query is pinned to coordinate 0, so a
    working softmax must pull the pooled vector above the plain mean towards the clip with
    the largest coordinate 0, and must pick up gradient — both impossible with tiled tokens.
    """
    import math

    import torch

    from primed_ai.probes.layers import AttentivePool

    clips = torch.as_tensor(
        np.random.default_rng(1).standard_normal((1, 6, ECHO_DIM)), dtype=torch.float32
    )
    pool = AttentivePool(ECHO_DIM)
    with torch.no_grad():
        pool.query.zero_()
        pool.query[0] = 4.0 * math.sqrt(ECHO_DIM)

    pooled = pool(clips)
    mean = clips.mean(1)
    assert not torch.allclose(pooled, mean, atol=1e-4)
    assert float(pooled[0, 0].detach()) > float(mean[0, 0])

    pooled.sum().backward()
    assert float(pool.query.grad.abs().max()) > 0.0


def _padded_pair(seed=2, lengths=(2, 5)):
    """Two ragged studies collated into one padded batch. Synthetic fixture."""
    import torch

    from primed_ai.probes.common import collate_tokens

    rng = np.random.default_rng(seed)
    clips = [
        torch.as_tensor(rng.standard_normal((n, ECHO_DIM)), dtype=torch.float32) for n in lengths
    ]
    batch = collate_tokens(
        [{"echo": c, "lvef": torch.tensor(50.0), "ef_le_40": torch.tensor(0.0)} for c in clips],
        pad_dim=ECHO_DIM,
    )
    return clips, batch


def test_collate_rejects_an_embed_dim_that_does_not_match_the_tokens():
    """``pad_dim`` comes from a config flag; a stale one has to fail here, not inside a matmul."""
    import torch

    from primed_ai.probes.common import collate_tokens

    item = {
        "echo": torch.zeros(2, ECHO_DIM),
        "lvef": torch.tensor(50.0),
        "ef_le_40": torch.tensor(0.0),
    }
    with pytest.raises(ValueError, match=f"expected embed dim {ECHO_DIM + 1}"):
        collate_tokens([item], pad_dim=ECHO_DIM + 1)


def test_collate_rejects_ragged_ecg_token_counts():
    """Unreachable today (ECG vectors are tiled to a fixed count) and guarded so it stays that way.

    Nothing downstream masks ECG: ``ConcatMLPProbe`` takes a plain ``mean(dim=1)`` and
    ``CrossAttentionFusion`` attends to ECG with no ``key_padding_mask``. Padding a ragged ECG
    batch would therefore pool zero rows as real tokens, silently, so it is refused instead.
    """
    import torch

    from primed_ai.probes.common import collate_tokens

    def item(n_ecg):
        return {
            "echo": torch.zeros(2, ECHO_DIM),
            "ecg": torch.zeros(n_ecg, ECG_DIM),
            "lvef": torch.tensor(50.0),
            "ef_le_40": torch.tensor(0.0),
        }

    assert collate_tokens([item(4), item(4)], pad_dim=ECHO_DIM)["ecg"].shape == (2, 4, ECG_DIM)
    with pytest.raises(ValueError, match=r"ragged ecg token counts \[3, 4\]"):
        collate_tokens([item(4), item(3)], pad_dim=ECHO_DIM)


def test_padding_never_leaks_into_attentive_pooling():
    """Ragged clip counts pad to the batch max; ``echo_mask`` has to make that invisible.

    Synthetic fixture (a 2-clip and a 5-clip study). Pooling the short study inside the
    padded batch must match pooling it on its own, and must differ from the unmasked result.
    """
    import torch

    from primed_ai.probes.layers import AttentivePool

    clips, batch = _padded_pair()
    assert batch["echo_mask"].tolist() == [[False, False, True, True, True], [False] * 5]

    pool = AttentivePool(ECHO_DIM)
    alone = pool(clips[0][None])[0]
    torch.testing.assert_close(pool(batch["echo"], batch["echo_mask"])[0], alone)
    assert not torch.allclose(pool(batch["echo"])[0], alone, atol=1e-4)


def test_linear_baseline_means_over_real_clips_not_the_padded_width():
    """The ablation baseline has to be length-correct or the comparison is rigged.

    Dividing by the padded width shrinks every short study toward zero, which degrades the
    linear probe for a reason that has nothing to do with mean vs attentive pooling.
    """
    import torch

    from primed_ai.probes.echo_only import LinearEchoProbe

    clips, batch = _padded_pair()
    lin = LinearEchoProbe(ECHO_DIM)
    with torch.no_grad():
        # identity-ish head so the assertion reads the pooled vector, not a projection of it
        lin.head.weight.copy_(torch.eye(1, ECHO_DIM))
        lin.head.bias.zero_()

    pooled = lin(batch["echo"], batch["echo_mask"])[0]
    torch.testing.assert_close(pooled, clips[0][:, 0].mean())
    assert not torch.allclose(lin(batch["echo"])[0], pooled, atol=1e-4)


def test_cross_attention_fusion_ignores_padded_clips():
    """The fusion block reads echo clips twice, and both reads need the mask.

    Synthetic fixture (a 2-clip and a 5-clip study). Without ``key_padding_mask`` the
    ecg->echo attention treats pad_sequence's zero rows as real clips; without the mask on
    ``echo_pool`` the padded context rows take softmax weight. Either way the short study's
    fused vector moves depending on who shares its batch.
    """
    import torch

    from primed_ai.probes.layers import CrossAttentionFusion

    clips, batch = _padded_pair(seed=3)
    ecg = torch.as_tensor(
        np.random.default_rng(4).standard_normal((2, 4, ECHO_DIM)), dtype=torch.float32
    )

    fusion = CrossAttentionFusion(ECHO_DIM).eval()
    alone = fusion(clips[0][None], ecg[:1])[0]
    torch.testing.assert_close(fusion(batch["echo"], ecg, echo_mask=batch["echo_mask"])[0], alone)
    assert not torch.allclose(fusion(batch["echo"], ecg)[0], alone, atol=1e-4)


def test_clip_manifest_reaches_attention_without_collapsing(tmp_path):
    """The whole manifest -> token boundary: parquet, ``manifest.load``, ``_embedding_to_tokens``,
    ``collate_tokens``. Mean-pooling anywhere along it makes every token identical and the
    softmax uniform again, which is the degeneracy clip retention exists to remove.
    """
    import math

    import torch

    from primed_ai.probes.common import TokenEmbeddingDataset, collate_tokens
    from primed_ai.probes.echo_only import _ensure_tokens
    from primed_ai.probes.layers import AttentivePool

    df = _ensure_tokens(manifest.load(_clip_manifest(tmp_path / "m.parquet", n=12)))
    multi = next(t for t in df["echo_tokens"] if len(t) > 1)
    assert not np.allclose(multi[0], multi[1])

    ds = TokenEmbeddingDataset(df)
    batch = collate_tokens([ds[i] for i in range(len(ds))], pad_dim=ECHO_DIM)
    rows = [i for i in range(len(ds)) if int((~batch["echo_mask"][i]).sum()) > 1]
    assert rows

    pool = AttentivePool(ECHO_DIM)
    with torch.no_grad():
        pool.query.zero_()
        pool.query[0] = 4.0 * math.sqrt(ECHO_DIM)
    pooled = pool(batch["echo"], batch["echo_mask"])

    for i in rows:
        clips = batch["echo"][i][~batch["echo_mask"][i]]
        # uniform weights would land pooled exactly on the mean of that study's clips
        assert not torch.allclose(pooled[i], clips.mean(0), atol=1e-4)
        assert float(pooled[i, 0].detach()) > float(clips[:, 0].mean())


@pytest.mark.parametrize("probe", ["attentive", "linear", "concat", "fused"])
def test_probe_predictions_ignore_the_padding_ragged_clip_counts_add(tmp_path, probe):
    """A study's prediction must not depend on what it was batched with.

    Batching to 12 pads up to the widest study, batching to 1 never pads, so the two eval
    passes only agree while ``echo_mask`` is still reaching every layer that pools or
    attends over echo clips — in the probe *and* in the eval loop that feeds it. Compared on
    raw predictions: ``regression_metrics`` rounds MAE to 4 dp, which hides small leaks.
    """
    import torch
    from torch.utils.data import DataLoader

    from primed_ai.probes.common import TokenEmbeddingDataset, collate_tokens
    from primed_ai.probes.concat_mlp import ConcatMLPProbe, _ensure_ecg_tokens
    from primed_ai.probes.concat_mlp import _predict as predict_concat
    from primed_ai.probes.cross_attn import CrossAttnFusedProbe, _condition_arrays
    from primed_ai.probes.echo_only import EchoOnlyProbe, LinearEchoProbe, _ensure_tokens
    from primed_ai.probes.echo_only import _predict as predict_echo

    torch.manual_seed(0)
    df = _ensure_tokens(manifest.load(_clip_manifest(tmp_path / "m.parquet", n=12)))
    assert df["echo_tokens"].map(len).nunique() > 1

    fused = probe in ("concat", "fused")
    if fused:
        df = _ensure_ecg_tokens(df)
    ds = TokenEmbeddingDataset(df, ecg_col="ecg_tokens" if fused else None)

    def loader(batch_size: int) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=lambda b: collate_tokens(b, pad_dim=ECHO_DIM),
        )

    if probe == "attentive":
        model, predict = EchoOnlyProbe(ECHO_DIM), predict_echo
    elif probe == "linear":
        model, predict = LinearEchoProbe(ECHO_DIM), predict_echo
    elif probe == "concat":
        model, predict = ConcatMLPProbe(ECHO_DIM, ECG_DIM), predict_concat
    else:
        model = CrossAttnFusedProbe(ECHO_DIM, echo_dim=ECHO_DIM, ecg_dim=ECG_DIM)
        predict = lambda m, ldr, dev: _condition_arrays(m, ldr, dev, "full")  # noqa: E731

    np.testing.assert_allclose(
        predict(model, loader(12), "cpu")["prediction"],
        predict(model, loader(1), "cpu")["prediction"],
        rtol=1e-6,
        atol=1e-6,
    )


def _widen_padding(collate, extra: int):
    """Add ``extra`` pad columns to every batch — as if each study met a longer batch peer."""
    import torch

    def wrapped(items):
        batch = collate(items)
        if extra:
            echo, mask = batch["echo"], batch["echo_mask"]
            n, _, dim = echo.shape
            batch["echo"] = torch.cat([echo, echo.new_zeros(n, extra, dim)], dim=1)
            batch["echo_mask"] = torch.cat([mask, mask.new_ones(n, extra)], dim=1)
        return batch

    return wrapped


@pytest.mark.parametrize("probe", ["echo", "concat", "fused"])
def test_training_is_invariant_to_the_padding_width(tmp_path, probe):
    """Training, not just eval, has to be blind to the pad rows batching adds.

    Every eval-time mask assertion still passes with the mask dropped from ``_train_epoch``,
    which is the worst case: the probe learns on pooled vectors that average in pad rows and
    is then scored under masked pooling it never saw. Widening the padding holds the data,
    the order, the batch size and the optimizer trajectory fixed and changes only how much
    padding each batch carries — so identical parameters after several epochs is exactly the
    claim that the mask reached the training forward pass.

    Dropout is switched off because the attention dropout draw is sized by the padded width;
    that is an artefact of the fixture, not of the property under test. Plain SGD for the same
    reason: summing a wider row reassociates the float32 adds, and Adam's ``m / sqrt(v)`` turns
    that 1-ULP gradient difference into an O(lr) parameter difference. Under SGD the clean run
    lands within 2.4e-7 while a dropped mask moves parameters by 5.6e-2 or more.
    """
    import torch
    from torch.utils.data import DataLoader

    from primed_ai.probes.common import TokenEmbeddingDataset, collate_tokens
    from primed_ai.probes.concat_mlp import ConcatMLPProbe, _ensure_ecg_tokens
    from primed_ai.probes.concat_mlp import _train_epoch as train_concat
    from primed_ai.probes.cross_attn import CrossAttnFusedProbe
    from primed_ai.probes.cross_attn import _train_epoch as train_fused
    from primed_ai.probes.echo_only import EchoOnlyProbe, _ensure_tokens
    from primed_ai.probes.echo_only import _train_epoch as train_echo

    df = _ensure_ecg_tokens(_ensure_tokens(manifest.load(_clip_manifest(tmp_path / "m.parquet"))))
    assert df["echo_tokens"].map(len).nunique() > 1
    ds = TokenEmbeddingDataset(df, ecg_col=None if probe == "echo" else "ecg_tokens")

    def train(extra: int) -> dict:
        torch.manual_seed(0)
        if probe == "echo":
            model, step = EchoOnlyProbe(ECHO_DIM), train_echo
        elif probe == "concat":
            model, step = ConcatMLPProbe(ECHO_DIM, ECG_DIM), train_concat
        else:
            model = CrossAttnFusedProbe(ECHO_DIM, echo_dim=ECHO_DIM, ecg_dim=ECG_DIM)
            step = train_fused
        for module in model.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                module.dropout = 0.0
            elif isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        loader = DataLoader(
            ds,
            batch_size=8,
            shuffle=False,
            collate_fn=_widen_padding(lambda b: collate_tokens(b, pad_dim=ECHO_DIM), extra),
        )
        optim = torch.optim.SGD(model.parameters(), lr=1e-3)
        for _ in range(4):
            step(model, loader, optim, "cpu")
        return model.state_dict()

    tight, wide = train(0), train(3)
    for key, value in tight.items():
        torch.testing.assert_close(value, wide[key], msg=lambda m, k=key: f"{k}:\n{m}")


def test_load_keeps_clip_matrices_and_still_reports_the_feature_dim(tmp_path):
    df = manifest.load(_clip_manifest(tmp_path / "m.parquet"))
    assert manifest.dims(df) == (ECHO_DIM, ECG_DIM)
    assert {np.asarray(v).shape[0] for v in df["echo_embedding"]} == {1, 2, 3, 4}
    with pytest.raises(ValueError, match="token matrices"):
        manifest.expand(df, manifest.ECHO_COLUMN, "ee")


def test_clip_level_build_trains_a_probe_end_to_end(tmp_path):
    """The one test that drives the real builder instead of imitating its output.

    Everywhere else the clip-level fixture hand-writes a parquet shaped like what
    ``build_echo_study_embeddings(max_clips=N)`` emits, so builder and consumer could drift
    apart without anything failing. This runs the actual chain — clip shards ->
    ``build_echo_study_embeddings`` -> ``convert_hubert_csv_to_parquet`` ->
    ``build_joined_manifest`` -> ``manifest.load`` -> ``run_echo``. Synthetic shards; the
    assertions are on shapes and plumbing, not on anything the probe learned.
    """
    from primed_ai.data.echo_hubert_manifest import (
        build_echo_study_embeddings,
        build_joined_manifest,
        convert_hubert_csv_to_parquet,
    )

    n, max_clips = 18, 3
    rng = np.random.default_rng(7)
    clip_counts = [1 + i % 5 for i in range(n)]

    echo_dir = tmp_path / "echo_clips"
    echo_dir.mkdir()
    pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(n), clip_counts),
            "study_id": np.repeat(np.arange(n) + 1000, clip_counts),
            "embedding": [
                row.tolist()
                for row in rng.standard_normal((sum(clip_counts), ECHO_DIM)).astype(np.float32)
            ],
        }
    ).to_parquet(echo_dir / "shard-00000.parquet", index=False)

    ecg = rng.standard_normal((n, ECG_DIM)).astype(np.float32)
    ecg_csv = tmp_path / "hubert.csv"
    pd.DataFrame(
        {"subject_id": np.arange(n), "ecg_study_id": np.arange(n) + 2000}
        | {f"ve{i:04d}": ecg[:, i] for i in range(ECG_DIM)}
    ).to_csv(ecg_csv, index=False)

    cohort_csv = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "echo_study_id": np.arange(n) + 1000,
            "ecg_study_id": np.arange(n) + 2000,
            "lvef": 45.0 + 5.0 * ecg[:, 0],
            "split": np.array(["train", "val", "test"])[np.arange(n) % 3],
        }
    ).to_csv(cohort_csv, index=False)

    echo_built = build_echo_study_embeddings(
        echo_dir, tmp_path / "echo_study.parquet", max_clips=max_clips
    )
    convert_hubert_csv_to_parquet(ecg_csv, tmp_path / "ecg.parquet", chunksize=5)
    built, summary = build_joined_manifest(
        cohort_csv,
        tmp_path / "echo_study.parquet",
        tmp_path / "ecg.parquet",
        tmp_path / "manifest.parquet",
        tmp_path / "metadata.csv",
        tmp_path / "summary.json",
    )
    assert summary["n_with_both_embeddings"] == n
    assert echo_built["n_echo_clips_retained"].tolist() == [min(c, max_clips) for c in clip_counts]

    df = manifest.load(tmp_path / "manifest.parquet")
    shapes = {np.asarray(v).shape for v in df["echo_embedding"]}
    assert shapes == {(k, ECHO_DIM) for k in (1, 2, max_clips)}
    assert built["n_echo_clips_retained"].tolist() == [
        len(np.asarray(v)) for v in df["echo_embedding"]
    ]

    res = run_echo(
        tmp_path / "manifest.parquet",
        out_dir=str(tmp_path / "probe"),
        embed_dim=ECHO_DIM,
        epochs=2,
        batch_size=4,
    )
    assert sum(res["n"].values()) == n
    assert (tmp_path / "probe" / "echo_only.pt").exists()


@pytest.mark.parametrize("build", [_manifest, _clip_manifest], ids=["pooled", "clip_level"])
@pytest.mark.parametrize("name", ["ecg", "echo", "concat", "fused"])
def test_every_probe_trains_from_the_manifest(tmp_path, name, build):
    path = build(tmp_path / "m.parquet")
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


def test_missing_modality_scores_a_manifest_trained_checkpoint(tmp_path):
    """E07 (#64) must score the checkpoint M10 produced, off the same manifest.

    ``missing_modality.run`` required both embedding paths, so the canonical rerun could
    not read a manifest even after the probes could. Fixture data; the metrics are
    meaningless, only the plumbing is asserted.
    """
    from primed_ai.evaluation.missing_modality import run as run_missing

    path = _manifest(tmp_path / "m.parquet")
    probe_dir = tmp_path / "fused"
    run_fused(
        path,
        out_dir=str(probe_dir),
        embed_dim=ECHO_DIM,
        echo_dim=ECHO_DIM,
        ecg_dim=ECG_DIM,
        epochs=2,
    )

    res = run_missing(
        path,
        checkpoint_path=probe_dir / "cross_attn_fused.pt",
        output_path=str(tmp_path / "missing.json"),
        embed_dim=ECHO_DIM,
        echo_dim=ECHO_DIM,
        ecg_dim=ECG_DIM,
        hidden=256,
        n_bootstrap=10,
    )
    assert set(res["test"]) == {"full", "echo_dropped", "ecg_dropped"}
    assert (tmp_path / "missing.json").exists()


def test_missing_modality_requires_a_checkpoint(tmp_path):
    from primed_ai.evaluation.missing_modality import run as run_missing

    with pytest.raises(ValueError, match="checkpoint_path is required"):
        run_missing(_manifest(tmp_path / "m.parquet"))
