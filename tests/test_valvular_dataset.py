"""Unit tests for Task B Valvular Dataset, DataLoader, and Manifest Builder."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from primed_ai.data.valvular_dataset import (
    ValvularMultimodalDataset,
    collate_valvular_batch,
    create_valvular_dataloaders,
)
from scripts.build_valvular_manifest import build_valvular_manifest


@pytest.fixture
def dummy_valvular_df():
    rng = np.random.default_rng(42)
    n = 20
    return pd.DataFrame({
        "subject_id": np.arange(100, 100 + n),
        "split": ["train"] * 14 + ["val"] * 2 + ["test"] * 4,
        "as_moderate_or_severe": rng.choice([False, True], n),
        "mr_moderate_or_severe": rng.choice([False, True], n),
        "tr_moderate_or_severe": rng.choice([False, True], n),
        "as_grade": rng.choice([0, 1, 2, 3], n),
        "mr_grade": rng.choice([0, 1, 2, 3], n),
        "tr_grade": rng.choice([0, 1, 2, 3], n),
        "echo_embedding": [rng.normal(0, 1, 1024).astype(np.float32).tolist() for _ in range(n)],
        "ecg_embedding": [rng.normal(0, 1, 768).astype(np.float32).tolist() for _ in range(n)],
        "sex": ["M", "F"] * (n // 2),
        "age": np.linspace(30, 80, n),
        "race": ["WHITE"] * n,
    })


def test_valvular_dataset_shapes_and_splits(dummy_valvular_df):
    train_ds = ValvularMultimodalDataset(dummy_valvular_df, split="train")
    assert len(train_ds) == 14

    item = train_ds[0]
    assert item["echo_emb"].shape == (1024,)
    assert item["ecg_emb"].shape == (768,)
    assert item["labels"].shape == (3,)
    assert item["grades"].shape == (3,)
    assert item["has_echo"] is True
    assert item["has_ecg"] is True


def test_valvular_missing_modality_dropout(dummy_valvular_df):
    # Mask echo
    ds_drop_echo = ValvularMultimodalDataset(dummy_valvular_df, mask_modality="echo")
    item = ds_drop_echo[0]
    assert torch.all(item["echo_emb"] == 0)
    assert not torch.all(item["ecg_emb"] == 0)
    assert item["has_echo"] is False
    assert item["has_ecg"] is True

    # Mask ecg
    ds_drop_ecg = ValvularMultimodalDataset(dummy_valvular_df, mask_modality="ecg")
    item2 = ds_drop_ecg[0]
    assert not torch.all(item2["echo_emb"] == 0)
    assert torch.all(item2["ecg_emb"] == 0)
    assert item2["has_echo"] is True
    assert item2["has_ecg"] is False


def test_collate_and_dataloader(dummy_valvular_df, tmp_path):
    parquet_path = tmp_path / "test_manifest.parquet"
    dummy_valvular_df.to_parquet(parquet_path, index=False)

    loaders = create_valvular_dataloaders(parquet_path, batch_size=4)
    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders

    for batch in loaders["train"]:
        assert batch.echo_emb.shape == (4, 1024)
        assert batch.ecg_emb.shape == (4, 768)
        assert batch.labels.shape == (4, 3)
        assert batch.grades.shape == (4, 3)
        break


def test_build_valvular_manifest_runner(dummy_valvular_df, tmp_path):
    cohort_path = tmp_path / "cohort.parquet"
    dummy_valvular_df.drop(columns=["echo_embedding", "ecg_embedding"]).to_parquet(cohort_path, index=False)

    out_manifest = tmp_path / "out_manifest.parquet"
    out_meta = tmp_path / "out_meta.csv"
    out_json = tmp_path / "out_summary.json"

    summary = build_valvular_manifest(
        cohort_path=cohort_path,
        echo_path=None,
        ecg_path=None,
        out_manifest=out_manifest,
        out_meta_csv=out_meta,
        out_summary_json=out_json,
    )

    assert summary["n_rows"] == 20
    assert out_manifest.exists()
    assert out_meta.exists()
    assert out_json.exists()
