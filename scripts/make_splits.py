"""Create subject-level train/val/test splits for the paired LVEF cohort.

D04 requirements:
- Split by unique subject_id, not by study/record.
- Assign a split label to every cohort row.
- Verify zero subject_id overlap across train/val/test.
- Save a reproducible split manifest with seed and hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def file_sha256(path: Path) -> str:
    """Compute SHA256 hash for an input file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_values(values: list[Any]) -> str:
    """Hash sorted values for reproducibility without relying on file order."""
    text = "\n".join(str(x) for x in sorted(values, key=lambda y: str(y)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_cohort(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet cohort file."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def write_cohort(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV or Parquet cohort file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
        return

    raise ValueError(f"Unsupported output format: {path.suffix}")


def make_subject_splits(
    subjects: list[Any],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[Any]]:
    """Split unique subject IDs into train/val/test sets."""
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0; got {total}")

    subjects = sorted(subjects, key=lambda x: str(x))
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(subjects))

    n_subjects = len(shuffled)
    n_train = int(round(n_subjects * train_frac))
    n_val = int(round(n_subjects * val_frac))

    train_subjects = shuffled[:n_train]
    val_subjects = shuffled[n_train : n_train + n_val]
    test_subjects = shuffled[n_train + n_val :]

    return {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }


def verify_no_subject_overlap(splits: dict[str, list[Any]]) -> None:
    """Assert that no subject appears in more than one split."""
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    if train & val:
        raise AssertionError("Subject leakage detected: train and val overlap.")
    if train & test:
        raise AssertionError("Subject leakage detected: train and test overlap.")
    if val & test:
        raise AssertionError("Subject leakage detected: val and test overlap.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cohort/paired.csv"),
        help="Input paired cohort file: CSV or Parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cohort/paired_with_splits.csv"),
        help="Output cohort file with split column: CSV or Parquet.",
    )
    parser.add_argument(
        "--subject-splits",
        type=Path,
        default=Path("cohort/subject_splits.csv"),
        help="Output subject_id-to-split mapping.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("cohort/splits.json"),
        help="Output JSON split manifest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.20)
    args = parser.parse_args()

    cohort = read_cohort(args.input)

    if "subject_id" not in cohort.columns:
        raise KeyError("Input cohort must contain a subject_id column.")

    if cohort["subject_id"].isna().any():
        raise ValueError("Input cohort contains missing subject_id values.")

    subjects = cohort["subject_id"].drop_duplicates().tolist()

    splits = make_subject_splits(
        subjects=subjects,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    verify_no_subject_overlap(splits)

    split_lookup = {
        subject_id: split
        for split, subject_ids in splits.items()
        for subject_id in subject_ids
    }

    cohort = cohort.copy()
    cohort["split"] = cohort["subject_id"].map(split_lookup)

    if cohort["split"].isna().any():
        raise AssertionError("Some cohort rows were not assigned a split.")

    # Verify no leakage in the actual output table.
    output_splits = {
        split: cohort.loc[cohort["split"] == split, "subject_id"]
        .drop_duplicates()
        .tolist()
        for split in ["train", "val", "test"]
    }
    verify_no_subject_overlap(output_splits)

    # Write row-level cohort with split labels.
    write_cohort(cohort, args.output)

    # Write subject-level mapping for later joins onto D03/final cohort outputs.
    subject_split_df = pd.DataFrame(
        [
            {"subject_id": subject_id, "split": split}
            for split, subject_ids in splits.items()
            for subject_id in subject_ids
        ]
    ).sort_values(["split", "subject_id"])

    args.subject_splits.parent.mkdir(parents=True, exist_ok=True)
    subject_split_df.to_csv(args.subject_splits, index=False)

    row_counts = {
        split: int((cohort["split"] == split).sum())
        for split in ["train", "val", "test"]
    }
    subject_counts = {
        split: int(cohort.loc[cohort["split"] == split, "subject_id"].nunique())
        for split in ["train", "val", "test"]
    }

    manifest = {
        "task": "D04_subject_level_train_val_test_split",
        "input_path": str(args.input),
        "input_sha256": file_sha256(args.input),
        "output_path": str(args.output),
        "subject_splits_path": str(args.subject_splits),
        "seed": args.seed,
        "ratios": {
            "train": args.train_frac,
            "val": args.val_frac,
            "test": args.test_frac,
        },
        "n_rows": int(len(cohort)),
        "n_subjects": int(len(subjects)),
        "row_counts": row_counts,
        "subject_counts": subject_counts,
        "subject_id_hash": hash_values(subjects),
        "split_subject_id_hashes": {
            split: hash_values(subject_ids)
            for split, subject_ids in splits.items()
        },
        "leakage_check": {
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
        },
    }

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote split cohort to: {args.output}")
    print(f"Wrote subject split mapping to: {args.subject_splits}")
    print(f"Wrote split manifest to: {args.manifest}")
    print(f"Rows by split: {row_counts}")
    print(f"Subjects by split: {subject_counts}")
    print("Leakage check passed: no subject_id overlap across splits.")


if __name__ == "__main__":
    main()