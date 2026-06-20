"""Check EF<=40% prevalence overall and by train/val/test split.

D06 requirements:
- Report EF<=40% prevalence overall and per split.
- Flag whether AUROC is likely reportable based on per-class counts.
- Save prevalence table to results/ef40_prevalence.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def read_cohort(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet cohort file."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported input format: {path.suffix}")


def normalize_ef_le_40(series: pd.Series) -> pd.Series:
    """Convert common ef_le_40 encodings to booleans."""
    if series.dtype == bool:
        return series

    if str(series.dtype) == "boolean":
        return series.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int).astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }

    normalized = series.astype(str).str.strip().str.lower().map(mapping)

    if normalized.isna().any():
        bad_values = sorted(series[normalized.isna()].dropna().unique().tolist())
        raise ValueError(f"Could not parse ef_le_40 values: {bad_values}")

    return normalized.astype(bool)


def summarize_group(
    df: pd.DataFrame,
    split_label: str,
    min_per_class: int,
) -> dict:
    """Summarize EF<=40 prevalence for one split or the full cohort."""
    n_rows = int(len(df))
    n_subjects = int(df["subject_id"].nunique()) if "subject_id" in df.columns else None

    n_positive = int(df["ef_le_40_bool"].sum())
    n_negative = int(n_rows - n_positive)

    prevalence = n_positive / n_rows if n_rows else None

    has_both_classes = n_positive > 0 and n_negative > 0
    meets_min_per_class = n_positive >= min_per_class and n_negative >= min_per_class
    auroc_reportable = bool(has_both_classes and meets_min_per_class)

    if not has_both_classes:
        notes = "AUROC not reportable: split does not contain both classes."
    elif not meets_min_per_class:
        notes = (
            f"AUROC may be unstable: fewer than {min_per_class} examples "
            "in at least one class. Consider bootstrap CIs or limitation note."
        )
    else:
        notes = "AUROC prevalence check passed."

    return {
        "split": split_label,
        "n_rows": n_rows,
        "n_subjects": n_subjects,
        "n_ef_le_40": n_positive,
        "n_ef_gt_40": n_negative,
        "ef_le_40_prevalence": round(prevalence, 6) if prevalence is not None else None,
        "min_per_class_threshold": min_per_class,
        "has_both_classes": has_both_classes,
        "auroc_reportable": auroc_reportable,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("cohort/paired_with_splits.parquet"),
        help="Input split-labeled cohort file, CSV or Parquet.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/ef40_prevalence.csv"),
        help="Output prevalence table CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/ef40_prevalence.json"),
        help="Output prevalence summary JSON.",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=30,
        help=(
            "Minimum positives and negatives required in a split before "
            "marking AUROC as reportable. This is a configurable review flag, "
            "not a formal statistical guarantee."
        ),
    )
    args = parser.parse_args()

    cohort = read_cohort(args.input)

    required_columns = {"split", "ef_le_40"}
    missing = required_columns - set(cohort.columns)
    if missing:
        raise KeyError(f"Input cohort is missing required columns: {sorted(missing)}")

    cohort = cohort.copy()
    n_null_splits = cohort["split"].isna().sum()
    if n_null_splits:
        raise ValueError(f"Input cohort has {n_null_splits} rows with missing split labels.")
    cohort["split"] = cohort["split"].astype(str).str.lower().str.strip()
    cohort["ef_le_40_bool"] = normalize_ef_le_40(cohort["ef_le_40"])

    expected_splits = {"train", "val", "test"}
    observed_splits = set(cohort["split"].unique())

    missing_splits = expected_splits - observed_splits
    if missing_splits:
        raise ValueError(f"Missing expected splits: {sorted(missing_splits)}")

    extra_splits = observed_splits - expected_splits
    if extra_splits:
        raise ValueError(f"Unexpected split labels: {sorted(extra_splits)}")

    rows = [summarize_group(cohort, "overall", args.min_per_class)]

    for split in ["train", "val", "test"]:
        split_df = cohort[cohort["split"] == split]
        rows.append(summarize_group(split_df, split, args.min_per_class))

    prevalence = pd.DataFrame(rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    prevalence.to_csv(args.output_csv, index=False)

    summary = {
        "task": "D06_EF40_prevalence_check",
        "input_path": str(args.input),
        "output_csv": str(args.output_csv),
        "min_per_class_threshold": args.min_per_class,
        "all_splits_auroc_reportable": bool(
            prevalence.loc[
                prevalence["split"].isin(["train", "val", "test"]),
                "auroc_reportable",
            ].all()
        ),
        "rows": rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(prevalence.to_string(index=False))
    print(f"\nWrote prevalence table to: {args.output_csv}")
    print(f"Wrote JSON summary to: {args.output_json}")

    test_row = prevalence[prevalence["split"] == "test"].iloc[0]
    if not bool(test_row["auroc_reportable"]):
        print("\nWARNING: Test-set AUROC may not be reportable without caveats.")
        print(test_row["notes"])


if __name__ == "__main__":
    main()