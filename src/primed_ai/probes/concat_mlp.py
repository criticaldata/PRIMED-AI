"""Concat-MLP fused LVEF probe (M08)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from primed_ai.probes import manifest
from primed_ai.probes.common import (
    TokenEmbeddingDataset,
    auroc,
    collate_tokens,
    drop_non_finite,
    git_sha,
    read_table,
    regression_metrics,
    save_results,
)
from primed_ai.probes.echo_only import _embedding_to_tokens, _ensure_tokens
from primed_ai.probes.layers import AttentivePool, MLPHead


class ConcatMLPProbe(nn.Module):
    """Attentive echo pool + mean ECG pool -> concat -> MLP."""

    def __init__(self, echo_dim: int, ecg_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.echo_pool = AttentivePool(echo_dim)
        self.head = MLPHead(echo_dim + ecg_dim, hidden=hidden)

    def forward(
        self,
        echo_tokens: torch.Tensor,
        ecg_tokens: torch.Tensor,
        echo_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        echo_vec = self.echo_pool(echo_tokens, echo_mask)
        ecg_vec = ecg_tokens.mean(dim=1)
        return self.head(torch.cat([echo_vec, ecg_vec], dim=-1))


def _ensure_ecg_tokens(df: pd.DataFrame, prefix: str = "ve", n_tokens: int = 4) -> pd.DataFrame:
    if "ecg_tokens" in df.columns:
        return df
    if "ecg_embedding" in df.columns:
        out = df.copy()
        out["ecg_tokens"] = [
            _embedding_to_tokens(value, n_tokens=n_tokens) for value in out["ecg_embedding"]
        ]
        return out
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"need ecg_tokens, ecg_embedding, or columns prefixed {prefix!r}")
    dim = len(cols)
    tokens = [np.tile(row, (n_tokens, 1)).reshape(n_tokens, dim) for row in df[cols].to_numpy()]
    out = df.copy()
    out["ecg_tokens"] = tokens
    return out


def _train_epoch(model, loader, optim, device) -> None:
    model.train()
    loss_fn = nn.MSELoss()
    for batch in loader:
        echo = batch["echo"].to(device)
        ecg = batch["ecg"].to(device)
        y = batch["lvef"].to(device)
        optim.zero_grad()
        pred = model(echo, ecg, batch["echo_mask"].to(device))
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()


@torch.no_grad()
def _predict(model, loader, device) -> dict:
    model.eval()
    ys, preds, ef = [], [], []
    for batch in loader:
        pred = (
            model(
                batch["echo"].to(device),
                batch["ecg"].to(device),
                batch["echo_mask"].to(device),
            )
            .cpu()
            .numpy()
        )
        ys.append(batch["lvef"].numpy())
        preds.append(pred)
        ef.append(batch["ef_le_40"].numpy())
    return {
        "lvef": np.concatenate(ys),
        "prediction": np.concatenate(preds),
        "ef_le_40": np.concatenate(ef).astype(bool),
    }


def _eval_model(model, loader, device) -> dict:
    arrays = _predict(model, loader, device)
    metrics = regression_metrics(arrays["lvef"], arrays["prediction"])
    metrics["ef40_auroc"] = round(auroc(arrays["ef_le_40"], -arrays["prediction"]), 4)
    return metrics


def run(
    cohort_path,
    echo_embedding_path=None,
    ecg_embedding_path=None,
    out_dir="probes/concat_mlp",
    *,
    echo_dim: int = 16,
    ecg_dim: int = 16,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if echo_embedding_path is None and ecg_embedding_path is None:
        df = manifest.load(cohort_path)  # embeddings already inline, nothing to join
    elif echo_embedding_path is None or ecg_embedding_path is None:
        raise ValueError(
            "pass both embedding paths for the two-table layout, or neither to read "
            "cohort_path as a joined manifest"
        )
    else:
        coh = read_table(cohort_path)
        echo_emb = read_table(echo_embedding_path)
        ecg_emb = read_table(ecg_embedding_path)

        echo_key = "echo_study_id" if "echo_study_id" in coh.columns else "subject_id"
        ecg_key = "ecg_record_id"
        if echo_key not in echo_emb.columns:
            echo_emb = echo_emb.rename(columns={echo_emb.columns[0]: echo_key})
        if ecg_key not in ecg_emb.columns:
            ecg_emb = ecg_emb.rename(columns={ecg_emb.columns[0]: ecg_key})

        df = coh.merge(echo_emb, on=echo_key, how="inner").merge(
            ecg_emb, on=ecg_key, how="inner", suffixes=("", "_ecg")
        )
    df = _ensure_tokens(df)
    df = _ensure_ecg_tokens(df)
    df, n_dropped = drop_non_finite(df, ("echo_tokens", "ecg_tokens"))

    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    if min(len(parts[s]) for s in parts) == 0:
        raise ValueError("train/val/test must all be non-empty")

    def loader(frame: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = TokenEmbeddingDataset(frame, ecg_col="ecg_tokens")
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_tokens(b, pad_dim=echo_dim),
        )

    model = ConcatMLPProbe(echo_dim, ecg_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    for _ in range(epochs):
        _train_epoch(model, loader(parts["train"], True), optim, device)
        val_mae = _eval_model(model, loader(parts["val"], False), device)["mae"]
        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out / "concat_mlp.pt")

    results = {
        "task": "M08_concat_mlp_probe",
        "issue": 26,
        "seed": seed,
        "git_sha": git_sha(),
        "echo_dim": echo_dim,
        "ecg_dim": ecg_dim,
        "n_dropped_nonfinite": n_dropped,
        "n": {k: len(v) for k, v in parts.items()},
        "val": _eval_model(model, loader(parts["val"], False), device),
        "test": _eval_model(model, loader(parts["test"], False), device),
    }
    save_results(out / "results.json", results)
    return results
