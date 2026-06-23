"""Reusable probe layers and training helpers (M07–M09)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class AttentivePool(nn.Module):
    """Single-query attentive pooling over token sequences ``[B, T, D]`` -> ``[B, D]``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) / math.sqrt(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = (tokens * self.query).sum(dim=-1) / math.sqrt(tokens.size(-1))
        weights = torch.softmax(scores, dim=-1)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)


class MLPHead(nn.Module):
    """Small MLP regression head."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention fusion with optional branch masking (M09 / E01)."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.echo_to_ecg = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ecg_to_echo = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.echo_pool = AttentivePool(dim)
        self.ecg_pool = AttentivePool(dim)
        self.norm = nn.LayerNorm(dim * 2)

    def forward(
        self,
        echo_tokens: torch.Tensor,
        ecg_tokens: torch.Tensor,
        *,
        mask_echo: bool = False,
        mask_ecg: bool = False,
    ) -> torch.Tensor:
        if mask_echo:
            echo_tokens = torch.zeros_like(echo_tokens)
        if mask_ecg:
            ecg_tokens = torch.zeros_like(ecg_tokens)

        echo_ctx, _ = self.echo_to_ecg(echo_tokens, ecg_tokens, ecg_tokens)
        ecg_ctx, _ = self.ecg_to_echo(ecg_tokens, echo_tokens, echo_tokens)
        fused = torch.cat([self.echo_pool(echo_ctx), self.ecg_pool(ecg_ctx)], dim=-1)
        return self.norm(fused)
