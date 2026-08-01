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
