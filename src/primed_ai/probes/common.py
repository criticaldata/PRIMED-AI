"""Shared dataset / metrics helpers for probe training."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch.utils.data import Dataset


def read_table(path: str | Path) -> pd.DataFrame:
    path = str(path)
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def git_sha() -> str:
    try:
        repo = Path(__file__).resolve().parents[3]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    if y_true.any() and (~y_true).any():
        return float(roc_auc_score(y_true, score))
    return float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    baseline = np.full_like(y_true, y_true.mean())
    return {
        "baseline_mae": round(float(mean_absolute_error(y_true, baseline)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


class TokenEmbeddingDataset(Dataset):
    """Cohort rows with variable-length token embeddings stored as arrays."""

    def __init__(self, df: pd.DataFrame, echo_col: str = "echo_tokens", ecg_col: str | None = None):
        self.df = df.reset_index(drop=True)
        self.echo_col = echo_col
        self.ecg_col = ecg_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        item = {
            "echo": torch.as_tensor(row[self.echo_col], dtype=torch.float32),
            "lvef": torch.tensor(row["lvef"], dtype=torch.float32),
            "ef_le_40": torch.tensor(bool(row["ef_le_40"]), dtype=torch.float32),
        }
        if self.ecg_col:
            item["ecg"] = torch.as_tensor(row[self.ecg_col], dtype=torch.float32)
        return item


def collate_tokens(batch: list[dict], *, pad_dim: int) -> dict[str, torch.Tensor]:
    """Pad echo token sequences to the max length in the batch.

    Studies carry different clip counts, so ``echo_mask`` (True where padded) rides along
    and keeps attentive pooling from weighting the zero rows pad_sequence appends.

    ECG is stacked, not padded: every producer tiles a study vector to a fixed token count,
    and no ECG consumer takes a mask, so a ragged batch is rejected rather than silently
    pooled with pad rows.
    """
    dims = {b["echo"].size(-1) for b in batch}
    if dims != {pad_dim}:
        raise ValueError(f"expected embed dim {pad_dim}, got {sorted(dims)}")
    lengths = torch.tensor([b["echo"].size(0) for b in batch])
    echo = torch.nn.utils.rnn.pad_sequence([b["echo"] for b in batch], batch_first=True)
    out = {
        "echo": echo,
        "echo_mask": torch.arange(echo.size(1))[None, :] >= lengths[:, None],
        "lvef": torch.stack([b["lvef"] for b in batch]),
        "ef_le_40": torch.stack([b["ef_le_40"] for b in batch]),
    }
    if "ecg" in batch[0]:
        ecg_lengths = {b["ecg"].size(0) for b in batch}
        if len(ecg_lengths) > 1:
            raise ValueError(
                f"ragged ecg token counts {sorted(ecg_lengths)} are unsupported — "
                "ConcatMLPProbe and CrossAttentionFusion pool ECG without a mask"
            )
        out["ecg"] = torch.stack([b["ecg"] for b in batch])
    return out
