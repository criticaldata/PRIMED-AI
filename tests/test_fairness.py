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
