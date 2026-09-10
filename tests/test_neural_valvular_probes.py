"""Unit tests for Neural Multimodal Valvular Probes (Task B)."""

import torch

from primed_ai.probes.neural_valvular import (
    ConcatMLPValvularProbe,
    CrossAttentionValvularProbe,
    MultiTaskValvularHead,
)


def test_multitask_valvular_head_forward():
    batch_size = 8
    in_dim = 512
    head = MultiTaskValvularHead(in_dim=in_dim, hidden_dim=128)

    x = torch.randn(batch_size, in_dim)
    out = head(x)

    assert "as_logits" in out
    assert "mr_logits" in out
    assert "tr_logits" in out
    assert "as_grade_logits" in out
    assert "mr_grade_logits" in out
    assert "tr_grade_logits" in out

    assert out["as_logits"].shape == (batch_size,)
    assert out["mr_logits"].shape == (batch_size,)
    assert out["tr_logits"].shape == (batch_size,)
    assert out["as_grade_logits"].shape == (batch_size, 4)
    assert out["mr_grade_logits"].shape == (batch_size, 4)
    assert out["tr_grade_logits"].shape == (batch_size, 4)


def test_concat_mlp_valvular_probe_forward_and_mask():
    batch_size = 4
    echo_dim = 1024
    ecg_dim = 768

    model = ConcatMLPValvularProbe(echo_dim=echo_dim, ecg_dim=ecg_dim, hidden_dim=128)

    echo = torch.randn(batch_size, echo_dim)
    ecg = torch.randn(batch_size, ecg_dim)

    # Full forward
    out_full = model(echo, ecg)
    assert out_full["as_logits"].shape == (batch_size,)

    # Mask echo
    out_drop_echo = model(echo, ecg, mask_echo=True)
    assert not torch.allclose(out_full["as_logits"], out_drop_echo["as_logits"])

    # Mask ecg
    out_drop_ecg = model(echo, ecg, mask_ecg=True)
    assert not torch.allclose(out_full["as_logits"], out_drop_ecg["as_logits"])


def test_cross_attention_valvular_probe_forward_and_tokens():
    batch_size = 4
    echo_dim = 1024
    ecg_dim = 768
    embed_dim = 128

    model = CrossAttentionValvularProbe(
        embed_dim=embed_dim,
        echo_dim=echo_dim,
        ecg_dim=ecg_dim,
        n_heads=2,
        hidden_dim=128,
    )

    # 3D token inputs: Echo (B, 8 clips, 1024), ECG (B, 1, 768)
    echo_tokens = torch.randn(batch_size, 8, echo_dim)
    ecg_tokens = torch.randn(batch_size, 1, ecg_dim)

    out = model(echo_tokens, ecg_tokens)
    assert out["as_logits"].shape == (batch_size,)
    assert out["mr_logits"].shape == (batch_size,)
    assert out["tr_logits"].shape == (batch_size,)
