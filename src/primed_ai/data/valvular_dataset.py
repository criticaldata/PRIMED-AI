"""PyTorch Dataset & DataLoader for Task B Valvular Hemodynamic Disease.

Loads synchronized EchoJEPA and HuBERT-ECG embeddings alongside multi-task
valvular disease labels (Aortic Stenosis, Mitral Regurgitation, Tricuspid Regurgitation).

Supports:
  - Subject-level split isolation (train/val/test)
  - Missing-modality simulation (mask_echo, mask_ecg)
  - Multi-task binary gates and ordinal severity grades
  - Demographic metadata preservation for fairness audits
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

VALVULAR_TARGETS = [
    "as_moderate_or_severe",
    "mr_moderate_or_severe",
    "tr_moderate_or_severe",
]

VALVULAR_GRADES = [
    "as_grade",
    "mr_grade",
    "tr_grade",
]


@dataclass
class ValvularBatch:
    subject_id: torch.Tensor
    echo_emb: torch.Tensor  # (B, 1024) or (B, N_clips, 1024)
    ecg_emb: torch.Tensor   # (B, 768)
    labels: torch.Tensor    # (B, 3) binary targets [AS, MR, TR]
    grades: torch.Tensor    # (B, 3) ordinal severity grades [0..3]
    has_echo: torch.Tensor  # (B,) boolean mask
    has_ecg: torch.Tensor   # (B,) boolean mask
    demographics: dict[str, list]


class ValvularMultimodalDataset(Dataset):
    """Dataset for multimodal Task B valvular hemodynamic disease evaluation."""

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        split: str | None = None,
        mask_modality: str | None = None,  # 'echo', 'ecg', or None
        echo_dim: int = 1024,
        ecg_dim: int = 768,
    ):
        if split is not None:
            self.df = manifest_df[manifest_df["split"] == split].copy().reset_index(drop=True)
        else:
            self.df = manifest_df.copy().reset_index(drop=True)

        self.mask_modality = mask_modality
        self.echo_dim = echo_dim
        self.ecg_dim = ecg_dim

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Extract embeddings
        if "echo_embedding" in row and row["echo_embedding"] is not None and not (isinstance(row["echo_embedding"], float) and np.isnan(row["echo_embedding"])):
            echo_vec = np.asarray(row["echo_embedding"], dtype=np.float32)
            has_echo = True
        else:
            echo_vec = np.zeros(self.echo_dim, dtype=np.float32)
            has_echo = False

        if "ecg_embedding" in row and row["ecg_embedding"] is not None and not (isinstance(row["ecg_embedding"], float) and np.isnan(row["ecg_embedding"])):
            ecg_vec = np.asarray(row["ecg_embedding"], dtype=np.float32)
            has_ecg = True
        else:
            ecg_vec = np.zeros(self.ecg_dim, dtype=np.float32)
            has_ecg = False

        # Apply runtime missing modality dropout
        if self.mask_modality == "echo":
            echo_vec = np.zeros_like(echo_vec)
            has_echo = False
        elif self.mask_modality == "ecg":
            ecg_vec = np.zeros_like(ecg_vec)
            has_ecg = False

        # Multi-task binary labels
        labels = np.array([
            float(bool(row.get("as_moderate_or_severe", False))),
            float(bool(row.get("mr_moderate_or_severe", False))),
            float(bool(row.get("tr_moderate_or_severe", False))),
        ], dtype=np.float32)

        # Multi-task ordinal grades (-1 if unassigned)
        grades = np.array([
            float(row.get("as_grade", -1) if pd.notna(row.get("as_grade")) else -1),
            float(row.get("mr_grade", -1) if pd.notna(row.get("mr_grade")) else -1),
            float(row.get("tr_grade", -1) if pd.notna(row.get("tr_grade")) else -1),
        ], dtype=np.float32)

        return {
            "subject_id": int(row.get("subject_id", 0)),
            "echo_emb": torch.from_numpy(echo_vec),
            "ecg_emb": torch.from_numpy(ecg_vec),
            "labels": torch.from_numpy(labels),
            "grades": torch.from_numpy(grades),
            "has_echo": has_echo,
            "has_ecg": has_ecg,
            "sex": str(row.get("sex", "UNKNOWN")),
            "age": float(row.get("age", 0)) if pd.notna(row.get("age")) else 0.0,
            "race": str(row.get("race", "UNKNOWN")),
        }


def collate_valvular_batch(items: Sequence[dict]) -> ValvularBatch:
    subject_ids = torch.tensor([item["subject_id"] for item in items], dtype=torch.int64)
    echo_embs = torch.stack([item["echo_emb"] for item in items], dim=0)
    ecg_embs = torch.stack([item["ecg_emb"] for item in items], dim=0)
    labels = torch.stack([item["labels"] for item in items], dim=0)
    grades = torch.stack([item["grades"] for item in items], dim=0)
    has_echo = torch.tensor([item["has_echo"] for item in items], dtype=torch.bool)
    has_ecg = torch.tensor([item["has_ecg"] for item in items], dtype=torch.bool)

    demographics = {
        "sex": [item["sex"] for item in items],
        "age": [item["age"] for item in items],
        "race": [item["race"] for item in items],
    }

    return ValvularBatch(
        subject_id=subject_ids,
        echo_emb=echo_embs,
        ecg_emb=ecg_embs,
        labels=labels,
        grades=grades,
        has_echo=has_echo,
        has_ecg=has_ecg,
        demographics=demographics,
    )


def create_valvular_dataloaders(
    manifest_path: str | Path,
    batch_size: int = 64,
    num_workers: int = 0,
    mask_modality_eval: str | None = None,
) -> dict[str, DataLoader]:
    df = pd.read_parquet(manifest_path) if str(manifest_path).endswith(".parquet") else pd.read_csv(manifest_path)

    train_ds = ValvularMultimodalDataset(df, split="train")
    val_ds = ValvularMultimodalDataset(df, split="val", mask_modality=mask_modality_eval)
    test_ds = ValvularMultimodalDataset(df, split="test", mask_modality=mask_modality_eval)

    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_valvular_batch, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_valvular_batch, num_workers=num_workers),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_valvular_batch, num_workers=num_workers),
    }
