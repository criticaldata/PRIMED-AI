"""Echo-only attentive LVEF probe (M07)."""

from __future__ import annotations

from pathlib import Path

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
from primed_ai.probes.layers import AttentivePool, MLPHead


class EchoOnlyProbe(nn.Module):
    """Attentive pooling over echo token embeddings -> LVEF regression."""

    def __init__(self, embed_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.pool = AttentivePool(embed_dim)
        self.head = MLPHead(embed_dim, hidden=hidden)

    def forward(self, echo_tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(echo_tokens))


class LinearEchoProbe(nn.Module):
    """Mean-pool + linear baseline for the attentive vs linear ablation."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, echo_tokens: torch.Tensor) -> torch.Tensor:
        return self.head(echo_tokens.mean(dim=1)).squeeze(-1)


def _embedding_to_tokens(value, *, n_tokens: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return np.tile(arr, (n_tokens, 1)).reshape(n_tokens, arr.shape[0])
    if arr.ndim == 2:
        return arr
    raise ValueError(f"embedding must be 1D or 2D, got shape {arr.shape}")


def _ensure_tokens(df: pd.DataFrame, prefix: str = "echo_ve", n_tokens: int = 8) -> pd.DataFrame:
    """Build synthetic token arrays from flat embedding columns if needed."""
    if "echo_tokens" in df.columns:
        return df
    if "echo_embedding" in df.columns:
        out = df.copy()
        out["echo_tokens"] = [
            _embedding_to_tokens(value, n_tokens=n_tokens) for value in out["echo_embedding"]
        ]
        return out
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"need echo_tokens, echo_embedding, or columns prefixed {prefix!r}")
    mat = df[cols].to_numpy(np.float64)
    dim = len(cols)
    tokens = []
    for row in mat:
        # Repeat the pooled vector into a short token sequence for attentive pooling.
        tokens.append(np.tile(row, (n_tokens, 1)).reshape(n_tokens, dim))
    out = df.copy()
    out["echo_tokens"] = tokens
    return out


def _train_epoch(model, loader, optim, device) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    for batch in loader:
        echo = batch["echo"].to(device)
        y = batch["lvef"].to(device)
        optim.zero_grad()
        pred = model(echo)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()
        total += float(loss.item()) * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def _eval_model(model, loader, device) -> dict:
    model.eval()
    ys, preds, ef = [], [], []
    for batch in loader:
        echo = batch["echo"].to(device)
        pred = model(echo).cpu().numpy()
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
    embedding_path,
    out_dir="probes/echo_only",
    *,
    embed_dim: int = 16,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """Train attentive + linear echo probes; save best attentive checkpoint."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    coh = read_table(cohort_path)
    emb = read_table(embedding_path)
    key = "echo_study_id" if "echo_study_id" in coh.columns else coh.columns[0]
    if key not in emb.columns:
        emb = emb.rename(columns={emb.columns[0]: key})
    df = coh.merge(emb, on=key, how="inner")
    df = _ensure_tokens(df)

    def split(name: str) -> pd.DataFrame:
        return df[df["split"] == name].reset_index(drop=True)

    train_df, val_df = split("train"), split("val")
    test_df = split("test")
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("train/val/test splits must all be non-empty")

    def loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = TokenEmbeddingDataset(frame)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_tokens(b, pad_dim=embed_dim),
        )

    attn = EchoOnlyProbe(embed_dim).to(device)
    linear = LinearEchoProbe(embed_dim).to(device)
    attn_opt = torch.optim.Adam(attn.parameters(), lr=lr)
    lin_opt = torch.optim.Adam(linear.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None
    for _ in range(epochs):
        _train_epoch(attn, loader(train_df, True), attn_opt, device)
        _train_epoch(linear, loader(train_df, True), lin_opt, device)
        val_mae = _eval_model(attn, loader(val_df, False), device)["mae"]
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in attn.state_dict().items()}

    if best_state:
        attn.load_state_dict(best_state)
    torch.save(attn.state_dict(), out / "echo_only.pt")

    val_attn = _eval_model(attn, loader(val_df, False), device)
    val_lin = _eval_model(linear, loader(val_df, False), device)
    test_attn = _eval_model(attn, loader(test_df, False), device)

    results = {
        "task": "M07_echo_only_probe",
        "issue": 25,
        "seed": seed,
        "git_sha": git_sha(),
        "embed_dim": embed_dim,
        "n": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "val": {"attentive": val_attn, "linear": val_lin},
        "test": {"attentive": test_attn},
        "attentive_beats_linear_val_mae": val_attn["mae"] < val_lin["mae"],
    }
    save_results(out / "results.json", results)
    return results
