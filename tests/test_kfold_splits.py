import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from make_kfold_splits import (  # noqa: E402
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
