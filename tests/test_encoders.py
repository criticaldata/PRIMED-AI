"""Unit tests for frozen encoder wrappers (M01, M03 / I05)."""

import torch
import torch.nn as nn

from primed_ai.encoders.base import freeze_module
from primed_ai.encoders.ecg_fm import ECGFMEncoder, synthetic_ecg_batch
from primed_ai.encoders.echojepa import EchoJEPALEncoder


class _StubEchoBackbone(nn.Module):
    def forward(self, x):
        b = x.size(0)
        return torch.randn(b, 5, 32)


def test_ecg_fm_stub_encode_shape():
    enc = ECGFMEncoder(None, use_stub=True, embed_dim=32, seq_len=8, frozen=True)
    batch = synthetic_ecg_batch(batch_size=3, length=1000)
    out = enc.encode(batch, return_tokens=True, pool="mean")
    assert out["tokens"].shape == (3, 8, 32)
    assert out["pooled"].shape == (3, 32)
    enc.verify_frozen()


def test_ecg_fm_no_grad():
    enc = ECGFMEncoder(None, use_stub=True, embed_dim=16, frozen=True)
    batch = synthetic_ecg_batch(batch_size=1, length=500)
    out = enc.encode(batch, pool="mean")
    assert out["pooled"].requires_grad is False


def test_echo_encoder_forward_tokens():
    enc = EchoJEPALEncoder.__new__(EchoJEPALEncoder)
    nn.Module.__init__(enc)
    enc.embed_dim = 32
    enc._backbone = _StubEchoBackbone()
    freeze_module(enc._backbone)
    video = torch.randn(2, 3, 4, 64, 64)
    out = enc.encode(video, return_tokens=True, pool="mean")
    assert out["tokens"].shape == (2, 5, 32)
    assert out["pooled"].shape == (2, 32)
