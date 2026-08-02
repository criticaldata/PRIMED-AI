from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from primed_ai.data.echo_hubert_dataset import EchoHubertDataset
from primed_ai.data.echo_hubert_manifest import (
    build_echo_study_embeddings,
    build_joined_manifest,
    convert_hubert_csv_to_parquet,
)


def test_echo_hubert_manifest_and_dataset(tmp_path):
    echo_dir = tmp_path / "echo"
    echo_dir.mkdir()
    pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "study_id": [10, 10, 20],
            "embedding": [[1.0, 3.0], [3.0, 5.0], [10.0, 20.0]],
        }
    ).to_parquet(echo_dir / "train-00000-of-00001.parquet", index=False)

    hubert_csv = tmp_path / "hubert.csv"
    pd.DataFrame(
        {
            "filename": [
                "files/p0000/p00000001/s00000100/00000100",
                "files/p0000/p00000002/s00000200/00000200",
            ],
            "ve0001": [0.5, 1.5],
            "ve0002": [1.5, 2.5],
        }
    ).to_csv(hubert_csv, index=False)

    cohort_path = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "echo_study_id": [10, 20],
            "ecg_study_id": [100, 200],
            "lvef": [35.0, 55.0],
            "split": ["train", "test"],
            "sex": ["F", "M"],
            "age": [72, 65],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN"],
        }
    ).to_csv(cohort_path, index=False)

    echo_out = tmp_path / "echo_study.parquet"
    ecg_out = tmp_path / "ecg.parquet"
    manifest_out = tmp_path / "manifest.parquet"
    metadata_out = tmp_path / "manifest_metadata.csv"
    summary_out = tmp_path / "summary.json"

    echo_frame = build_echo_study_embeddings(echo_dir, echo_out)
    convert_hubert_csv_to_parquet(hubert_csv, ecg_out, chunksize=1)
    manifest, summary = build_joined_manifest(
        cohort_path,
        echo_out,
        ecg_out,
        manifest_out,
        metadata_out,
        summary_out,
    )

    pooled = echo_frame.loc[echo_frame["echo_study_id"] == 10, "echo_embedding"].iloc[0]
    assert pooled == [2.0, 4.0]
    assert summary["n_cohort_rows"] == 2
    assert summary["n_with_both_embeddings"] == 2
    assert summary["echo_embedding_dim"] == 2
    assert summary["ecg_embedding_dim"] == 2
    assert manifest["ef_le_40"].tolist() == [True, False]
    assert "echo_embedding" not in pd.read_csv(metadata_out).columns
    assert json.loads(summary_out.read_text())["test_count"] == 1

    dataset = EchoHubertDataset(manifest_out, split="train")
    assert len(dataset) == 1
    item = dataset[0]
    assert item["subject_id"] == 1
    assert item["echo_embedding"].tolist() == [2.0, 4.0]
    assert item["ecg_embedding"].tolist() == [0.5, 1.5]
    assert item["lvef"] == 35.0


def test_subject_split_leakage_raises(tmp_path):
    echo_path = tmp_path / "echo.parquet"
    ecg_path = tmp_path / "ecg.parquet"
    cohort_path = tmp_path / "cohort.parquet"
    pd.DataFrame(
        {
            "subject_id": [1, 1],
            "echo_study_id": [10, 11],
            "n_echo_clips": [1, 1],
            "echo_embedding": [[1.0], [2.0]],
            "echo_model": ["echo", "echo"],
        }
    ).to_parquet(echo_path, index=False)
    pd.DataFrame(
        {
            "subject_id": [1, 1],
            "ecg_study_id": [100, 101],
            "ecg_embedding": [[1.0], [2.0]],
            "ecg_model": ["ecg", "ecg"],
        }
    ).to_parquet(ecg_path, index=False)
    pd.DataFrame(
        {
            "subject_id": [1, 1],
            "echo_study_id": [10, 11],
            "ecg_study_id": [100, 101],
            "lvef": [35.0, 50.0],
            "split": ["train", "test"],
        }
    ).to_parquet(cohort_path, index=False)

    with pytest.raises(ValueError, match="Subject leakage"):
        build_joined_manifest(
            cohort_path,
            echo_path,
            ecg_path,
            tmp_path / "manifest.parquet",
            tmp_path / "metadata.csv",
            tmp_path / "summary.json",
        )
