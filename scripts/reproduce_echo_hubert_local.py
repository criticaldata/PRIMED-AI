"""Reproduce the local EchoJEPA + HuBERT synchronized manifest.

This wrapper assumes the large embedding artifacts already exist locally:

- data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet
- data/interim/hubert_ecg_embeddings.parquet

It regenerates the cohort splits, EF<=40 prevalence logs, joined manifest,
metadata CSV, join summary JSON, and runs a dataloader smoke test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("../outputs_build_cohort/paired.csv"),
        help="Paired cohort CSV/Parquet with LVEF labels and ECG record IDs.",
    )
    parser.add_argument(
        "--echo",
        type=Path,
        default=Path("data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet"),
        help="Mean-pooled EchoJEPA study embedding parquet.",
    )
    parser.add_argument(
        "--ecg",
        type=Path,
        default=Path("data/interim/hubert_ecg_embeddings.parquet"),
        help="HuBERT ECG embedding parquet.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    require_file(args.cohort, "paired cohort")
    require_file(args.echo, "EchoJEPA study embeddings")
    require_file(args.ecg, "HuBERT ECG embeddings")

    split_cohort = Path("data/raw/cohort/paired_with_splits.parquet")
    subject_splits = Path("data/raw/cohort/subject_splits.csv")
    split_manifest = Path("logs/splits.json")
    ef40_csv = Path("logs/ef40_prevalence.csv")
    ef40_json = Path("logs/ef40_prevalence.json")
    manifest = Path("data/processed/echo_hubert_manifest.parquet")
    metadata_csv = Path("data/processed/echo_hubert_manifest_metadata.csv")
    summary_json = Path("logs/echo_hubert_join_summary.json")

    run(
        [
            sys.executable,
            "scripts/make_splits.py",
            "--input",
            str(args.cohort),
            "--output",
            str(split_cohort),
            "--subject-splits",
            str(subject_splits),
            "--manifest",
            str(split_manifest),
            "--seed",
            str(args.seed),
            "--train-frac",
            str(args.train_frac),
            "--val-frac",
            str(args.val_frac),
            "--test-frac",
            str(args.test_frac),
        ]
    )

    run(
        [
            sys.executable,
            "scripts/check_ef40_prevalence.py",
            "--input",
            str(split_cohort),
            "--output-csv",
            str(ef40_csv),
            "--output-json",
            str(ef40_json),
        ]
    )

    run(
        [
            sys.executable,
            "scripts/build_echo_hubert_manifest.py",
            "join",
            "--cohort",
            str(split_cohort),
            "--echo",
            str(args.echo),
            "--ecg",
            str(args.ecg),
            "--manifest",
            str(manifest),
            "--metadata-csv",
            str(metadata_csv),
            "--summary-json",
            str(summary_json),
            "--ecg-study-col",
            "ecg_record_id",
        ]
    )

    from primed_ai.data.echo_hubert_dataset import EchoHubertDataset

    smoke = {}
    for split in ["train", "val", "test"]:
        dataset = EchoHubertDataset(manifest, split=split)
        item = dataset[0]
        smoke[split] = {
            "rows": len(dataset),
            "echo_shape": list(item["echo_embedding"].shape),
            "ecg_shape": list(item["ecg_embedding"].shape),
        }

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    print(json.dumps({"join_summary": summary, "loader_smoke": smoke}, indent=2))


if __name__ == "__main__":
    main()
