"""Frozen ECG-FM encoder wrapper (M03 / I05).

ECG-FM weights are hosted on Hugging Face (``wanglab/ecg-fm``) and load via the
``fairseq_signals`` stack. Unit tests and offline dev can opt into a shape-stable
stub backbone with ``use_stub=True``; asking for a checkpoint without the stack
installed raises rather than quietly substituting random weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from primed_ai.encoders.base import assert_frozen, freeze_module

DEFAULT_HF_REPO = "wanglab/ecg-fm"
DEFAULT_CHECKPOINT = "mimic_iv_ecg_physionet_pretrained.pt"
# MIMIC-IV-ECG: 12 leads, 500 Hz, 10 s -> 5000 samples per lead after resampling.
DEFAULT_SAMPLE_RATE = 500
DEFAULT_SIGNAL_LENGTH = 5000
DEFAULT_N_LEADS = 12


class _StubECGFMBackbone(nn.Module):
    """Minimal conv backbone for offline tests (no fairseq_signals required)."""

    def __init__(self, embed_dim: int = 768, seq_len: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.proj = nn.Conv1d(DEFAULT_N_LEADS, embed_dim, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 12, T] -> [B, seq_len, embed_dim]
        h = self.proj(x)
        h = nn.functional.adaptive_avg_pool1d(h, self.seq_len)
        return h.transpose(1, 2)


def download_checkpoint(
    repo_id: str = DEFAULT_HF_REPO,
    filename: str = DEFAULT_CHECKPOINT,
    cache_dir: str | Path | None = None,
) -> Path:
    """Download (or resolve cached) ECG-FM weights from Hugging Face."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("install huggingface_hub: pip install 'primed-ai[ecg]'") from exc

    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    return Path(path)


def _load_fairseq_model(checkpoint_path: str | Path) -> nn.Module:
    from fairseq_signals.models import build_model_from_checkpoint

    model = build_model_from_checkpoint(checkpoint_path=str(checkpoint_path))
    model.eval()
    return model


class ECGFMEncoder(nn.Module):
    """Frozen ECG-FM encoder: ``[B, 12, T]`` waveform -> sequence / pooled embeddings."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        embed_dim: int = 768,
        seq_len: int = 64,
        frozen: bool = True,
        use_stub: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.use_stub = use_stub
        self._backbone: nn.Module
        self._fairseq = False

        if use_stub or checkpoint_path is None:
            self._backbone = _StubECGFMBackbone(embed_dim=embed_dim, seq_len=seq_len)
        else:
            # Never fall back to the random-weight stub implicitly: a caller that asked for
            # a checkpoint and silently got noise would poison every downstream embedding.
            self._backbone = _load_fairseq_model(checkpoint_path)
            self._fairseq = True

        if frozen:
            freeze_module(self._backbone)
        self.to(device)

    @classmethod
    def from_config(cls, cfg, *, device: str | torch.device = "cpu") -> ECGFMEncoder:
        ckpt = cfg.get("checkpoint_path")
        use_stub = bool(cfg.get("use_stub", False))
        return cls(
            None if use_stub else ckpt,
            embed_dim=int(cfg.get("embed_dim", 768)),
            seq_len=int(cfg.get("seq_len", 64)),
            frozen=bool(cfg.get("frozen", True)),
            use_stub=use_stub,
            device=device,
        )

    def _forward_fairseq(self, waveform: torch.Tensor) -> torch.Tensor:
        out = self._backbone(source=waveform)
        tokens = out["encoder_out"]  # [B, T, D]
        return tokens

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        return_tokens: bool = True,
        pool: Literal["mean", "none"] = "none",
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of 12-lead waveforms.

        ``waveform`` shape: ``[B, 12, T]`` (leads x time samples).
        """
        with torch.set_grad_enabled(any(p.requires_grad for p in self._backbone.parameters())):
            if self._fairseq:
                tokens = self._forward_fairseq(waveform)
            else:
                tokens = self._backbone(waveform)

        result: dict[str, torch.Tensor] = {}
        if return_tokens:
            result["tokens"] = tokens
        if pool == "mean":
            if self._fairseq:
                denom = (tokens != 0).sum(dim=1).clamp(min=1)
                result["pooled"] = tokens.sum(dim=1) / denom
            else:
                result["pooled"] = tokens.mean(dim=1)
        return result

    def encode(self, waveform: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        return self.forward(waveform, **kwargs)

    def verify_frozen(self) -> None:
        assert_frozen(self._backbone)


def synthetic_ecg_batch(
    batch_size: int = 2,
    n_leads: int = DEFAULT_N_LEADS,
    length: int = DEFAULT_SIGNAL_LENGTH,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Random 12-lead waveform batch for smoke tests."""
    return torch.randn(batch_size, n_leads, length, device=device)
