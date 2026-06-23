"""Cross-attention fused LVEF probe with missing-modality masking (M09)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from primed_ai.probes.common import (
    TokenEmbeddingDataset,
    auroc,
    collate_tokens,
    git_sha,
    read_table,
    regression_metrics,
    save_results,
)
from primed_ai.probes.concat_mlp import _ensure_ecg_tokens
from primed_ai.probes.echo_only import _ensure_tokens
from primed_ai.probes.layers import CrossAttentionFusion, MLPHead

ModalityMask = Literal["full", "echo_dropped", "ecg_dropped"]


class CrossAttnFusedProbe(nn.Module):
    """Cross-attention fusion + regression head."""

    def __init__(self, embed_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.fusion = CrossAttentionFusion(embed_dim)
        self.head = MLPHead(embed_dim * 2, hidden=hidden)

    def forward(
        self,
        echo_tokens: torch.Tensor,
        ecg_tokens: torch.Tensor,
        *,
        mask_echo: bool = False,
        mask_ecg: bool = False,
    ) -> torch.Tensor:
        fused = self.fusion(echo_tokens, ecg_tokens, mask_echo=mask_echo, mask_ecg=mask_ecg)
        return self.head(fused)


def _train_epoch(model, loader, optim, device) -> None:
    model.train()
    loss_fn = nn.MSELoss()
    for batch in loader:
        echo = batch["echo"].to(device)
        ecg = batch["ecg"].to(device)
        y = batch["lvef"].to(device)
        optim.zero_grad()
        pred = model(echo, ecg)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()


@torch.no_grad()
def _eval_condition(
    model: CrossAttnFusedProbe,
    loader: DataLoader,
    device: str,
    condition: ModalityMask,
) -> dict:
    model.eval()
    mask_echo = condition == "echo_dropped"
    mask_ecg = condition == "ecg_dropped"
    ys, preds, ef = [], [], []
    for batch in loader:
        pred = model(
            batch["echo"].to(device),
            batch["ecg"].to(device),
            mask_echo=mask_echo,
            mask_ecg=mask_ecg,
        ).cpu().numpy()
        ys.append(batch["lvef"].numpy())
        preds.append(pred)
        ef.append(batch["ef_le_40"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(preds)
    ef = np.concatenate(ef).astype(bool)
    metrics = regression_metrics(y, p)
    metrics["ef40_auroc"] = round(auroc(ef, -p), 4)
    return metrics


def run(
    cohort_path,
    echo_embedding_path,
    ecg_embedding_path,
    out_dir="probes/cross_attn_fused",
    *,
    embed_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    coh = read_table(cohort_path)
    echo_emb = read_table(echo_embedding_path)
    ecg_emb = read_table(ecg_embedding_path)
    echo_key = "echo_study_id" if "echo_study_id" in coh.columns else "subject_id"
    ecg_key = "ecg_record_id"
    if echo_key not in echo_emb.columns:
        echo_emb = echo_emb.rename(columns={echo_emb.columns[0]: echo_key})
    if ecg_key not in ecg_emb.columns:
        ecg_emb = ecg_emb.rename(columns={ecg_emb.columns[0]: ecg_key})

    df = coh.merge(echo_emb, on=echo_key, how="inner").merge(ecg_emb, on=ecg_key, how="inner")
    df = _ensure_tokens(df)
    df = _ensure_ecg_tokens(df)
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if min(len(parts[s]) for s in parts) == 0:
        raise ValueError("train/val/test must all be non-empty")

    def loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = TokenEmbeddingDataset(frame, ecg_col="ecg_tokens")
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_tokens(b, pad_dim=embed_dim),
        )

    model = CrossAttnFusedProbe(embed_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    for _ in range(epochs):
        _train_epoch(model, loader(parts["train"], True), optim, device)
        val_mae = _eval_condition(model, loader(parts["val"], False), device, "full")["mae"]
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out / "cross_attn_fused.pt")

    val_loader = loader(parts["val"], False)
    test_loader = loader(parts["test"], False)
    results = {
        "task": "M09_cross_attn_fused_probe",
        "issue": 27,
        "seed": seed,
        "git_sha": git_sha(),
        "embed_dim": embed_dim,
        "n": {k: len(v) for k, v in parts.items()},
        "val": {
            "full": _eval_condition(model, val_loader, device, "full"),
            "echo_dropped": _eval_condition(model, val_loader, device, "echo_dropped"),
            "ecg_dropped": _eval_condition(model, val_loader, device, "ecg_dropped"),
        },
        "test": {
            "full": _eval_condition(model, test_loader, device, "full"),
            "echo_dropped": _eval_condition(model, test_loader, device, "echo_dropped"),
            "ecg_dropped": _eval_condition(model, test_loader, device, "ecg_dropped"),
        },
        "supports_missing_modality_inference": True,
    }
    save_results(out / "results.json", results)
    return results
