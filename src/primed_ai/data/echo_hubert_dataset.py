"""Dataset for synchronized EchoJEPA and HuBERT embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from primed_ai.data.echo_hubert_manifest import parse_embedding


class EchoHubertDataset:
    """Load rows from an EchoJEPA + HuBERT joined manifest parquet."""

    def __init__(
        self,
        manifest_path: str | Path = "data/processed/echo_hubert_manifest.parquet",
        *,
        split: str | None = None,
        require_both_embeddings: bool = True,
        return_numpy: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.require_both_embeddings = require_both_embeddings
        self.return_numpy = return_numpy

        frame = pd.read_parquet(self.manifest_path)
        if split is not None:
            frame = frame[frame["split"] == split]
        if require_both_embeddings:
            frame = frame[frame["echo_embedding"].notna() & frame["ecg_embedding"].notna()]
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        echo_embedding = parse_embedding(row["echo_embedding"])
        ecg_embedding = parse_embedding(row["ecg_embedding"])
        if self.return_numpy:
            echo_embedding = np.asarray(echo_embedding, dtype=np.float32)
            ecg_embedding = np.asarray(ecg_embedding, dtype=np.float32)

        metadata_columns = [
            col for col in self.frame.columns if col not in {"echo_embedding", "ecg_embedding"}
        ]
        return {
            "subject_id": row["subject_id"],
            "echo_study_id": row["echo_study_id"],
            "ecg_study_id": row["ecg_study_id"],
            "echo_embedding": echo_embedding,
            "ecg_embedding": ecg_embedding,
            "lvef": row["lvef"],
            "ef_le_40": row["ef_le_40"],
            "split": row["split"],
            "metadata": row[metadata_columns].to_dict(),
        }
