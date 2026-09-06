"""Create reproducible patient-level k-fold splits for the paired LVEF cohort.

E13 requirements:
- Split by unique subject_id, never by study/record.
- Each subject belongs to exactly one outer test fold.
- Within each fold, assign every row to train/val/test.
- Verify zero subject overlap within every fold.
- Preserve approximately the canonical 70/10/20 train/val/test proportions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from check_ef40_prevalence import normalize_ef_le_40
from make_splits import (
    file_sha256,
    hash_values,
    read_cohort,
    verify_no_subject_overlap,
    write_cohort,
)


def make_subject_folds(
    subjects: list[Any],
    *,
    n_folds: int = 5,
    val_frac: float = 0.10,
    seed: int = 42,
) -> list[dict[str, list[Any]]]:
    """Create patient-level outer folds with an inner validation holdout."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")

    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")

    if val_frac >= 1.0 - (1.0 / n_folds):
        raise ValueError(
            "val_frac is too large for the requested number of folds; "
            "the training split would be empty."
        )
    subjects = sorted(subjects, key=lambda x: str(x))

    if len(subjects) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} subjects for {n_folds}-fold CV; got {len(subjects)}."
        )

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(subjects))

    test_folds = [list(x) for x in np.array_split(shuffled, n_folds)]
    folds: list[dict[str, list[Any]]] = []

    for fold_idx, test_subjects in enumerate(test_folds):
        test_set = set(test_subjects)
        remaining = [s for s in shuffled if s not in test_set]

        # val_frac is expressed relative to the full cohort.
        # With 5 folds, test is ~20%; selecting 10% of the full cohort
        # for validation leaves approximately 70% for training.
        n_val = max(1, int(round(len(subjects) * val_frac)))

        fold_rng = np.random.default_rng(seed + fold_idx + 1)
        remaining_shuffled = list(fold_rng.permutation(remaining))

        val_subjects = remaining_shuffled[:n_val]
        train_subjects = remaining_shuffled[n_val:]

        split = {
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects,
        }
        verify_no_subject_overlap(split)
        folds.append(split)

    return folds


def verify_test_fold_coverage(
    subjects: list[Any],
    folds: list[dict[str, list[Any]]],
) -> None:
    """Verify that every subject appears in exactly one outer test fold."""
    test_subjects = [subject for fold in folds for subject in fold["test"]]

    if len(test_subjects) != len(set(test_subjects)):
        raise AssertionError("A subject appears in more than one outer test fold.")

    if set(test_subjects) != set(subjects):
        raise AssertionError("Outer test folds do not cover every subject exactly once.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/echo_hubert_manifest.parquet"),
        help="Joined EchoJEPA + HuBERT manifest.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/kfold"),
        help="Directory where fold-specific manifests and metadata are written.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cohort = read_cohort(args.input)

    if "subject_id" not in cohort.columns:
        raise KeyError("Input cohort must contain a subject_id column.")

    if cohort["subject_id"].isna().any():
        raise ValueError("Input cohort contains missing subject_id values.")

    subjects = cohort["subject_id"].drop_duplicates().tolist()

    folds = make_subject_folds(
        subjects,
        n_folds=args.n_folds,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    verify_test_fold_coverage(subjects, folds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_metadata = []

    for fold_idx, splits in enumerate(folds):
        verify_no_subject_overlap(splits)

        split_lookup = {
            subject_id: split for split, subject_ids in splits.items() for subject_id in subject_ids
        }

        fold_df = cohort.copy()

        if "split" in fold_df.columns:
            fold_df["split_canonical"] = fold_df["split"]

        fold_df["split"] = fold_df["subject_id"].map(split_lookup)

        if fold_df["split"].isna().any():
            raise AssertionError(f"Fold {fold_idx}: some cohort rows were not assigned a split.")

        output_splits = {
            split: (fold_df.loc[fold_df["split"] == split, "subject_id"].drop_duplicates().tolist())
            for split in ("train", "val", "test")
        }
        verify_no_subject_overlap(output_splits)

        fold_path = args.out_dir / f"fold_{fold_idx}.parquet"
        write_cohort(fold_df, fold_path)

        subject_split_df = pd.DataFrame(
            [
                {
                    "subject_id": subject_id,
                    "split": split,
                }
                for split, subject_ids in splits.items()
                for subject_id in subject_ids
            ]
        ).sort_values(["split", "subject_id"])

        subject_split_path = args.out_dir / f"fold_{fold_idx}_subjects.csv"
        subject_split_df.to_csv(subject_split_path, index=False)

        row_counts = {
            split: int((fold_df["split"] == split).sum()) for split in ("train", "val", "test")
        }

        subject_counts = {
            split: int(fold_df.loc[fold_df["split"] == split, "subject_id"].nunique())
            for split in ("train", "val", "test")
        }

        ef40_bool = normalize_ef_le_40(fold_df["ef_le_40"])

        ef40_counts = {
            split: int(ef40_bool.loc[fold_df["split"] == split].sum())
            for split in ("train", "val", "test")
        }

        fold_metadata.append(
            {
                "fold": fold_idx,
                "manifest_path": str(fold_path),
                "subject_splits_path": str(subject_split_path),
                "row_counts": row_counts,
                "subject_counts": subject_counts,
                "ef_le_40_counts": ef40_counts,
                "split_subject_id_hashes": {
                    split: hash_values(subject_ids) for split, subject_ids in splits.items()
                },
                "leakage_check": {
                    "train_val_overlap": 0,
                    "train_test_overlap": 0,
                    "val_test_overlap": 0,
                },
            }
        )

        print(f"Fold {fold_idx}: rows={row_counts}, subjects={subject_counts}")

    metadata = {
        "task": "E13_patient_level_kfold",
        "input_path": str(args.input),
        "input_sha256": file_sha256(args.input),
        "seed": args.seed,
        "n_folds": args.n_folds,
        "val_frac": args.val_frac,
        "n_rows": int(len(cohort)),
        "n_subjects": int(len(subjects)),
        "subject_id_hash": hash_values(subjects),
        "outer_test_coverage": {
            "n_unique_test_subjects": int(
                len({subject for fold in folds for subject in fold["test"]})
            ),
            "expected_subjects": int(len(subjects)),
            "each_subject_tested_once": True,
        },
        "folds": fold_metadata,
    }

    metadata_path = args.out_dir / "kfold_manifest.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {args.n_folds} fold manifests to: {args.out_dir}")
    print(f"Wrote k-fold metadata to: {metadata_path}")
    print("Leakage check passed for every fold.")
    print("Every subject appears in exactly one outer test fold.")


if __name__ == "__main__":
    main()
