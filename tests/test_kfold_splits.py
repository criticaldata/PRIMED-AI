import json
import sys

import pandas as pd
import pytest
from make_kfold_splits import (
    main,
    make_subject_folds,
    verify_no_subject_overlap,
    verify_test_fold_coverage,
)


def test_five_fold_sizes_match_70_10_20():
    subjects = list(range(100))

    folds = make_subject_folds(
        subjects,
        n_folds=5,
        val_frac=0.10,
        seed=42,
    )

    assert len(folds) == 5

    for fold in folds:
        assert len(fold["train"]) == 70
        assert len(fold["val"]) == 10
        assert len(fold["test"]) == 20


def test_no_subject_leakage_within_folds():
    subjects = list(range(100))

    folds = make_subject_folds(
        subjects,
        n_folds=5,
        val_frac=0.10,
        seed=42,
    )

    for fold in folds:
        verify_no_subject_overlap(fold)


def test_every_subject_is_tested_once():
    subjects = list(range(100))

    folds = make_subject_folds(
        subjects,
        n_folds=5,
        val_frac=0.10,
        seed=42,
    )

    verify_test_fold_coverage(subjects, folds)

    test_subjects = [subject for fold in folds for subject in fold["test"]]

    assert len(test_subjects) == 100
    assert len(set(test_subjects)) == 100


def test_same_seed_is_reproducible():
    subjects = list(range(100))

    folds_a = make_subject_folds(
        subjects,
        n_folds=5,
        val_frac=0.10,
        seed=42,
    )

    folds_b = make_subject_folds(
        subjects,
        n_folds=5,
        val_frac=0.10,
        seed=42,
    )

    assert folds_a == folds_b


def test_different_seed_changes_assignment():
    subjects = list(range(100))

    folds_a = make_subject_folds(
        subjects,
        n_folds=5,
        val_frac=0.10,
        seed=42,
    )

    folds_b = make_subject_folds(
        subjects,
        n_folds=5,
        val_frac=0.10,
        seed=43,
    )

    assert folds_a != folds_b


def test_val_frac_cannot_empty_training_split():
    subjects = list(range(100))

    with pytest.raises(ValueError, match="training split would be empty"):
        make_subject_folds(
            subjects,
            n_folds=2,
            val_frac=0.5,
            seed=42,
        )


def test_main_writes_ef40_counts_and_preserves_canonical_split(
    tmp_path,
    monkeypatch,
):
    input_path = tmp_path / "cohort.parquet"
    out_dir = tmp_path / "kfold"

    cohort = pd.DataFrame(
        {
            "subject_id": list(range(100)),
            "split": ["train"] * 70 + ["val"] * 10 + ["test"] * 20,
            "ef_le_40": [i % 4 == 0 for i in range(100)],
        }
    )
    cohort.to_parquet(input_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_kfold_splits.py",
            "--input",
            str(input_path),
            "--out-dir",
            str(out_dir),
            "--n-folds",
            "5",
            "--val-frac",
            "0.10",
            "--seed",
            "42",
        ],
    )

    main()

    metadata = json.loads((out_dir / "kfold_manifest.json").read_text(encoding="utf-8"))

    fold_df = pd.read_parquet(out_dir / "fold_0.parquet")
    fold_metadata = metadata["folds"][0]

    assert "split_canonical" in fold_df.columns

    expected_canonical = cohort.sort_values("subject_id")["split"].tolist()
    actual_canonical = fold_df.sort_values("subject_id")["split_canonical"].tolist()
    assert actual_canonical == expected_canonical

    for split in ("train", "val", "test"):
        expected_count = int(
            fold_df.loc[
                fold_df["split"] == split,
                "ef_le_40",
            ].sum()
        )
        assert fold_metadata["ef_le_40_counts"][split] == expected_count
