"""Neural Multimodal Probes for Task B Valvular Hemodynamic Disease.

Implements multi-task neural architectures:
  1. ConcatMLPValvularProbe: Concatenation baseline with multi-task gating & severity heads.
  2. CrossAttentionValvularProbe: Bidirectional cross-attention between Echo & ECG tokens.
  3. MultiTaskValvularHead: Shared trunk with condition-specific classification heads for AS, MR, TR.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from primed_ai.probes.layers import CrossAttentionFusion

ModalityMask = Literal["full", "echo_dropped", "ecg_dropped"]


class MultiTaskValvularHead(nn.Module):
    """Shared hidden trunk with multi-task binary and ordinal classification heads for AS, MR, and TR."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        head_in = hidden_dim // 2

        # Binary clinical gate heads (AS, MR, TR >= Moderate) -> Logits
        self.as_gate = nn.Linear(head_in, 1)
        self.mr_gate = nn.Linear(head_in, 1)
        self.tr_gate = nn.Linear(head_in, 1)

        # Ordinal severity grade heads (0=None, 1=Mild, 2=Mod, 3=Sev) -> 4 logits
        self.as_grade = nn.Linear(head_in, 4)
        self.mr_grade = nn.Linear(head_in, 4)
        self.tr_grade = nn.Linear(head_in, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.trunk(x)
        return {
            "as_logits": self.as_gate(feat).squeeze(-1),
            "mr_logits": self.mr_gate(feat).squeeze(-1),
            "tr_logits": self.tr_gate(feat).squeeze(-1),
            "as_grade_logits": self.as_grade(feat),
            "mr_grade_logits": self.mr_grade(feat),
            "tr_grade_logits": self.tr_grade(feat),
        }


class ConcatMLPValvularProbe(nn.Module):
    """Concatenation baseline for Task B multi-task valvular disease prediction."""

    def __init__(
        self,
        echo_dim: int = 1024,
        ecg_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.echo_dim = echo_dim
        self.ecg_dim = ecg_dim
        self.norm = nn.LayerNorm(echo_dim + ecg_dim)
        self.head = MultiTaskValvularHead(echo_dim + ecg_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        echo_emb: torch.Tensor,
        ecg_emb: torch.Tensor,
        *,
        mask_echo: bool = False,
        mask_ecg: bool = False,
    ) -> dict[str, torch.Tensor]:
        # Handle 3D token inputs by pooling to 2D
        if echo_emb.ndim == 3:
            echo_emb = echo_emb.mean(dim=1)
        if ecg_emb.ndim == 3:
            ecg_emb = ecg_emb.mean(dim=1)

        if mask_echo:
            echo_emb = torch.zeros_like(echo_emb)
        if mask_ecg:
            ecg_emb = torch.zeros_like(ecg_emb)

        fused = torch.cat([echo_emb, ecg_emb], dim=-1)
        fused = self.norm(fused)
        return self.head(fused)


class CrossAttentionValvularProbe(nn.Module):
    """Bidirectional Cross-Attention Multimodal Probe for Task B Valvular Disease (Headline Model)."""

    def __init__(
        self,
        embed_dim: int = 256,
        echo_dim: int = 1024,
        ecg_dim: int = 768,
        n_heads: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.echo_proj = nn.Linear(echo_dim, embed_dim)
        self.ecg_proj = nn.Linear(ecg_dim, embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim, n_heads=n_heads, dropout=dropout)
        self.head = MultiTaskValvularHead(embed_dim * 2, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        echo_tokens: torch.Tensor,
        ecg_tokens: torch.Tensor,
        *,
        mask_echo: bool = False,
        mask_ecg: bool = False,
        echo_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Ensure 3D token representations [B, T, D]
        if echo_tokens.ndim == 2:
            echo_tokens = echo_tokens.unsqueeze(1)
        if ecg_tokens.ndim == 2:
            ecg_tokens = ecg_tokens.unsqueeze(1)

        echo_proj = self.echo_proj(echo_tokens)
        ecg_proj = self.ecg_proj(ecg_tokens)

        fused = self.fusion(
            echo_proj,
            ecg_proj,
            mask_echo=mask_echo,
            mask_ecg=mask_ecg,
            echo_mask=echo_mask,
        )
        return self.head(fused)
