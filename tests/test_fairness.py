def test_fairness_runs_on_real_dim_checkpoint(tmp_path):
    import numpy as np
    import pandas as pd
    import torch

    from primed_ai.evaluation.fairness import load_model, predict_on_df
    from primed_ai.probes import cross_attn

    EMBED, ECHO, ECG = 32, 64, 48  # unequal dims, like 256/1024/768 but small
    model = cross_attn.CrossAttnFusedProbe(EMBED, echo_dim=ECHO, ecg_dim=ECG)
    ckpt = tmp_path / "fused.pt"
    torch.save(model.state_dict(), ckpt)

    # B1: loader must accept the real checkpoint
    m, _ = load_model(ckpt, EMBED, device="cpu", echo_dim=ECHO, ecg_dim=ECG)

    # B2: prediction must collate at the echo token dim
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        [
            {
                "echo_tokens": rng.standard_normal((4, ECHO)).astype("float32"),
                "ecg_tokens": rng.standard_normal((4, ECG)).astype("float32"),
                "lvef": float(v),
                "ef_le_40": bool(v <= 40),
            }
            for v in (30, 35, 60, 65, 28, 70, 41, 55)
        ]
    )
    y_pred, y_true, ef = predict_on_df(m, df, EMBED, device="cpu", echo_dim=ECHO)
    assert y_pred.shape == (len(df),)


def test_fairness_predictions_ignore_clip_padding():
    """Clip-level manifests give studies ragged clip counts, so predict_on_df pads.

    Synthetic fixture. A subject's stratum metrics must not depend on batch composition,
    which only holds while the collated echo_mask is passed through to the model.
    """
    import numpy as np
    import pandas as pd
    import torch

    from primed_ai.evaluation.fairness import predict_on_df
    from primed_ai.probes import cross_attn

    EMBED, ECHO, ECG = 16, 16, 8
    torch.manual_seed(0)
    model = cross_attn.CrossAttnFusedProbe(EMBED, echo_dim=ECHO, ecg_dim=ECG).eval()

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        [
            {
                "echo_tokens": rng.standard_normal((1 + i % 4, ECHO)).astype("float32"),
                "ecg_tokens": rng.standard_normal((4, ECG)).astype("float32"),
                "lvef": 30.0 + 5.0 * i,
                "ef_le_40": i < 3,
            }
            for i in range(8)
        ]
    )
    batched, _, _ = predict_on_df(model, df, EMBED, batch_size=8, device="cpu", echo_dim=ECHO)
    alone, _, _ = predict_on_df(model, df, EMBED, batch_size=1, device="cpu", echo_dim=ECHO)
    np.testing.assert_allclose(batched, alone, atol=1e-5)


def test_run_fairness_reads_manifest_and_stratifies_per_condition(tmp_path):
    """E08 (#65): fairness must score the joined manifest under dropped conditions too.

    Synthetic fixture; only plumbing is asserted. One test row carries a non-finite ECG
    vector so the stratified n must line up with the canonical rerun's drop behavior.
    """
    import numpy as np
    import pandas as pd
    import torch

    from primed_ai.evaluation.fairness import MIMIC_BIAS_NOTE, run_fairness
    from primed_ai.probes import cross_attn

    EMBED, ECHO, ECG = 16, 24, 12
    n = 30
    rng = np.random.default_rng(0)
    echo = rng.standard_normal((n, ECHO)).astype("float32")
    ecg = rng.standard_normal((n, ECG)).astype("float32")
    lvef = np.clip(55 + 15 * echo[:, 0], 10, 80)
    ecg[5, 0] = np.nan  # index 5 % 3 == 2 lands in the test split below
    df = pd.DataFrame(
        {
            "subject_id": np.arange(n),
            "lvef": lvef,
            "ef_le_40": lvef <= 40.0,
            "split": np.array(["train", "val", "test"])[np.arange(n) % 3],
            "sex": np.where(np.arange(n) % 2 == 0, "F", "M"),
            "age_band": np.where(np.arange(n) < 15, "40-54", "65-74"),
            "race": np.where(np.arange(n) % 3 == 0, "WHITE", "BLACK/AFRICAN AMERICAN"),
            "echo_embedding": list(echo),
            "ecg_embedding": list(ecg),
        }
    )
    path = tmp_path / "manifest.parquet"
    df.to_parquet(path, index=False)

    torch.manual_seed(0)
    model = cross_attn.CrossAttnFusedProbe(EMBED, echo_dim=ECHO, ecg_dim=ECG)
    ckpt = tmp_path / "fused.pt"
    torch.save(model.state_dict(), ckpt)

    res = run_fairness(
        path,
        checkpoint=ckpt,
        out_dir=tmp_path / "fairness",
        embed_dim=EMBED,
        echo_dim=ECHO,
        ecg_dim=ECG,
        device="cpu",
        conditions=("full", "echo_dropped"),
    )

    assert set(res["conditions"]) == {"full", "echo_dropped"}
    assert res["n_dropped_nonfinite_all_splits"] == 1
    assert res["n_test"] == 9  # 10 test rows minus the non-finite one
    assert res["bias_note"] == MIMIC_BIAS_NOTE
    full = res["conditions"]["full"]
    assert set(full["by"]) == {"sex", "age_band", "race"}
    assert all(m["small_n"] for m in full["by"]["sex"].values())
    # aggregate.py reads overall/by at the top level of the combined payload
    assert res["overall"] == full["overall"] and res["by"] == full["by"]
    # masking the echo branch must actually change the predictions
    assert full["overall"]["mae"] != res["conditions"]["echo_dropped"]["overall"]["mae"]
    for condition in ("full", "echo_dropped"):
        assert (tmp_path / "fairness" / condition / "fairness_metrics.json").exists()
        assert (tmp_path / "fairness" / condition / "fairness_summary.csv").exists()
    assert (tmp_path / "fairness" / "fairness_metrics.json").exists()


def test_run_fairness_rejects_unknown_conditions(tmp_path):
    import pytest

    from primed_ai.evaluation.fairness import run_fairness

    with pytest.raises(ValueError, match="unknown conditions"):
        run_fairness("whatever.parquet", checkpoint="x.pt", conditions=("upside_down",))
