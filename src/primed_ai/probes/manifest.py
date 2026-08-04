"""Feed the joined EchoJEPA + HuBERT manifest to the existing probe training paths.

The probes predate the manifest: each expects a cohort table plus one or two separate
embedding tables to join on study/record id. The manifest written by
``scripts/build_echo_hubert_manifest.py`` already carries both embeddings inline, so that
join is a no-op. These helpers reshape it into what the probes already read, which keeps
the training/eval/checkpoint code in each probe untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from primed_ai.data.echo_hubert_manifest import parse_embedding

ECHO_COLUMN = "echo_embedding"
ECG_COLUMN = "ecg_embedding"


def has_embeddings(df: pd.DataFrame, *columns: str) -> bool:
    """True when the frame already carries the named embedding columns (default: both)."""
    return all(col in df.columns for col in (columns or (ECHO_COLUMN, ECG_COLUMN)))


def load(path: str | Path, *, require_both: bool = True) -> pd.DataFrame:
    """Read a manifest parquet, parsing both embedding columns into float32 arrays.

    Rows whose embeddings failed to parse are dropped when ``require_both`` is set;
    the probes cannot do anything with a half-present pair.
    """
    df = pd.read_parquet(path)
    missing = [c for c in (ECHO_COLUMN, ECG_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is not a joined manifest — missing {missing}")

    for col in (ECHO_COLUMN, ECG_COLUMN):
        df[col] = [
            None if (vec := parse_embedding(v)) is None else np.asarray(vec, dtype=np.float32)
            for v in df[col]
        ]

    if require_both:
        keep = df[ECHO_COLUMN].notna() & df[ECG_COLUMN].notna()
        df = df[keep]
    return df.reset_index(drop=True)


def dims(df: pd.DataFrame) -> tuple[int, int]:
    """(echo_dim, ecg_dim) taken from the first row carrying both embeddings."""
    for _, row in df.iterrows():
        echo, ecg = row[ECHO_COLUMN], row[ECG_COLUMN]
        if echo is not None and ecg is not None:
            return int(np.asarray(echo).shape[-1]), int(np.asarray(ecg).shape[-1])
    raise ValueError("no row carries both embeddings")


def expand(df: pd.DataFrame, column: str, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    """Explode a list-valued embedding column into flat numeric columns.

    ``ecg_only`` selects its features by column prefix rather than reading an array
    column, so the manifest has to be flattened before it can train.

    Clip-level (2-D) columns are rejected: ravelling them would silently emit
    ``n_clips * dim`` feature columns whose meaning changes row to row.
    """
    vectors = [np.asarray(v, dtype=np.float64) for v in df[column]]
    ragged = {v.ndim for v in vectors} - {1}
    if ragged:
        raise ValueError(
            f"{column} holds {sorted(ragged)}-D token matrices; expand() needs vectors"
        )
    mat = np.vstack(vectors)
    cols = [f"{prefix}{i}" for i in range(mat.shape[1])]
    flat = pd.DataFrame(mat, columns=cols, index=df.index)
    return df.drop(columns=[column]).join(flat), cols
