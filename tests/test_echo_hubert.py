from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from primed_ai.data import echo_hubert_manifest
from primed_ai.data.echo_hubert_dataset import EchoHubertDataset
from primed_ai.data.echo_hubert_manifest import (
    build_echo_study_embeddings,
    build_joined_manifest,
    convert_hubert_csv_to_parquet,
    replace_manifest_echo_embeddings,
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
    assert pooled.tolist() == [2.0, 4.0]
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


def test_max_clips_retains_an_even_stride_subsample(tmp_path):
    """Synthetic clip shard fixture: five clips tagged 0..4 in the first coordinate."""
    echo_dir = tmp_path / "echo"
    echo_dir.mkdir()
    pd.DataFrame(
        {
            "subject_id": [1] * 5 + [2],
            "study_id": [10] * 5 + [20],
            "embedding": [[float(i), 0.0] for i in range(5)] + [[9.0, 9.0]],
        }
    ).to_parquet(echo_dir / "shard.parquet", index=False)

    clip_path = tmp_path / "clips.parquet"
    frame = build_echo_study_embeddings(echo_dir, clip_path, max_clips=3)
    kept = frame.loc[frame["echo_study_id"] == 10, "echo_embedding"].iloc[0]
    assert [row[0] for row in kept] == [0.0, 2.0, 4.0]
    assert frame.loc[frame["echo_study_id"] == 10, "n_echo_clips"].iloc[0] == 5
    assert frame.loc[frame["echo_study_id"] == 10, "n_echo_clips_retained"].iloc[0] == 3

    single = frame.loc[frame["echo_study_id"] == 20, "echo_embedding"].iloc[0]
    assert single.tolist() == [[9.0, 9.0]]

    # float32 on disk: a .tolist() round-trip would silently double the file
    assert pq.read_schema(clip_path).field("echo_embedding").type == pa.list_(
        pa.list_(pa.float32())
    )

    # default stays mean-pooled and 1-D
    pooled_path = tmp_path / "pooled.parquet"
    pooled = build_echo_study_embeddings(echo_dir, pooled_path)
    mean_vector = pooled.loc[pooled["echo_study_id"] == 10, "echo_embedding"].iloc[0]
    assert mean_vector.tolist() == [2.0, 0.0]
    assert "n_echo_clips_retained" not in pooled.columns
    assert pq.read_schema(pooled_path).field("echo_embedding").type == pa.list_(pa.float32())


def test_null_clips_neither_burn_retention_slots_nor_drop_a_study(tmp_path):
    """Synthetic clip shard fixture: study 10 has nulls on exactly the old stride picks.

    Under a stride taken before parsing, 0/2/4/6/8 are all null and study 10 disappears
    from the clip build while the pooled build keeps it.
    """
    echo_dir = tmp_path / "echo"
    echo_dir.mkdir()
    embeddings = [None if i % 2 == 0 else [float(i), 0.0] for i in range(10)]
    pd.DataFrame(
        {"subject_id": [1] * 10, "study_id": [10] * 10, "embedding": embeddings}
    ).to_parquet(echo_dir / "shard.parquet", index=False)

    clipped = build_echo_study_embeddings(echo_dir, tmp_path / "clips.parquet", max_clips=5)
    pooled = build_echo_study_embeddings(echo_dir, tmp_path / "pooled.parquet")

    kept = clipped["echo_embedding"].iloc[0]
    assert [row[0] for row in kept] == [1.0, 5.0, 9.0]
    assert clipped["n_echo_clips_retained"].iloc[0] == 3

    # n_echo_clips counts parsed clips in both modes, so QC can reconcile the two builds
    assert clipped["echo_study_id"].tolist() == pooled["echo_study_id"].tolist()
    assert clipped["n_echo_clips"].tolist() == pooled["n_echo_clips"].tolist() == [5]


def test_clip_payload_past_the_list_offset_ceiling_is_refused(tmp_path, monkeypatch):
    """Parquet list offsets are int32, so the clip column has a hard 2^31-value ceiling.

    ``pa.ListArray.from_arrays`` narrows int64 offsets to int32 whatever it is handed, and
    ``build_joined_manifest`` re-writes the column through pandas as a plain ``list<>`` as
    well, so the ceiling survives any large_list change made in the builder alone. It is
    unreachable at the sizes this pipeline writes, so the real ceiling is monkeypatched down
    rather than allocating 8 GB to reach it.
    """
    echo_dir = tmp_path / "echo"
    echo_dir.mkdir()
    pd.DataFrame(
        {
            "subject_id": [1, 1],
            "study_id": [10, 10],
            "embedding": [[1.0, 2.0], [3.0, 4.0]],
        }
    ).to_parquet(echo_dir / "shard.parquet", index=False)

    monkeypatch.setattr(echo_hubert_manifest, "MAX_LIST_VALUES", 3)
    with pytest.raises(ValueError, match="parquet list-offset ceiling"):
        build_echo_study_embeddings(echo_dir, tmp_path / "clips.parquet", max_clips=2)


def test_build_echo_raises_when_nothing_parses(tmp_path):
    echo_dir = tmp_path / "echo"
    echo_dir.mkdir()
    pd.DataFrame(
        {"subject_id": [1], "study_id": [10], "embedding": pd.Series([None], dtype=object)}
    ).to_parquet(echo_dir / "shard.parquet", index=False)

    with pytest.raises(ValueError, match="No parseable echo embeddings"):
        build_echo_study_embeddings(echo_dir, tmp_path / "clips.parquet", max_clips=2)


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


def test_replace_manifest_echo_embeddings_preserves_base_cohort_and_ecg(tmp_path):
    base_path = tmp_path / "base.parquet"
    echo_path = tmp_path / "clip_echo.parquet"
    manifest_path = tmp_path / "clip_manifest.parquet"
    metadata_path = tmp_path / "clip_metadata.csv"
    summary_path = tmp_path / "clip_summary.json"

    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "echo_study_id": [10, 20],
            "ecg_study_id": [100, 200],
            "lvef": [35.0, 60.0],
            "ef_le_40": [True, False],
            "split": ["train", "val"],
            "n_echo_clips": [5, 4],
            "echo_embedding": [[1.0, 2.0], [3.0, 4.0]],
            "echo_model": ["pooled", "pooled"],
            "ecg_embedding": [[0.1, 0.2], [0.3, 0.4]],
            "ecg_model": ["hubert", "hubert"],
        }
    ).to_parquet(base_path, index=False)
    pd.DataFrame(
        {
            "subject_id": [1, 2],
            "echo_study_id": [10, 20],
            "n_echo_clips": [5, 4],
            "n_echo_clips_retained": [2, 2],
            "echo_embedding": [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            "echo_model": ["clip", "clip"],
        }
    ).to_parquet(echo_path, index=False)

    manifest, summary = replace_manifest_echo_embeddings(
        base_manifest_path=base_path,
        echo_embeddings_path=echo_path,
        manifest_path=manifest_path,
        metadata_csv_path=metadata_path,
        summary_json_path=summary_path,
    )

    assert np.asarray(manifest.loc[0, "ecg_embedding"]).tolist() == [0.1, 0.2]
    assert np.asarray(manifest.loc[1, "ecg_embedding"]).tolist() == [0.3, 0.4]
    assert np.allclose(
        np.stack(manifest.loc[0, "echo_embedding"]),
        [[1.0, 2.0], [3.0, 4.0]],
    )
    assert manifest["n_echo_clips_retained"].tolist() == [2, 2]
    assert manifest["has_echo_embedding"].tolist() == [True, True]
    assert summary["n_with_both_embeddings"] == 2
    assert summary["base_manifest_path"] == str(base_path)
    assert "echo_embedding" not in pd.read_csv(metadata_path).columns
    assert json.loads(summary_path.read_text())["echo_embeddings_path"] == str(echo_path)
