"""Build EchoJEPA study embeddings, HuBERT parquet, and the joined manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_ECHO_MODEL = "vjepa2.1-vitl-mimic-pt-100"
DEFAULT_ECG_MODEL = "hubert"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    echo = subparsers.add_parser("build-echo", help="Mean-pool EchoJEPA shards.")
    echo.add_argument("--input", type=Path, required=True)
    echo.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet"),
    )
    echo.add_argument("--subject-col")
    echo.add_argument("--study-col")
    echo.add_argument("--embedding-col")
    echo.add_argument("--echo-model", default=DEFAULT_ECHO_MODEL)
    echo.add_argument(
        "--max-clips",
        type=int,
        help=(
            "keep an even-stride subsample of up to N clip vectors per study instead of "
            "mean-pooling them; echo_embedding becomes (n_clips, dim). 16 is a sane start. "
            "Holds n_studies * N * dim * 4 bytes in memory until the write (7,251 studies "
            "x 16 x 1024 is ~475 MB, ~4x that in peak RSS), so raise it against the machine."
        ),
    )

    ecg = subparsers.add_parser("convert-ecg", help="Convert HuBERT CSV to parquet.")
    ecg.add_argument("--csv", type=Path, required=True)
    ecg.add_argument(
        "--output",
        type=Path,
        default=Path("data/interim/hubert_ecg_embeddings.parquet"),
    )
    ecg.add_argument("--subject-col")
    ecg.add_argument("--ecg-study-col")
    ecg.add_argument("--ecg-embedding-col")
    ecg.add_argument("--filename-col")
    ecg.add_argument("--ecg-feature-prefix", action="append", default=None)
    ecg.add_argument("--ecg-model", default=DEFAULT_ECG_MODEL)
    ecg.add_argument("--chunksize", type=int, default=50_000)

    join = subparsers.add_parser("join", help="Join cohort, echo, and ECG parquet.")
    join.add_argument("--cohort", type=Path, required=True)
    join.add_argument(
        "--echo",
        type=Path,
        default=Path("data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet"),
    )
    join.add_argument(
        "--ecg", type=Path, default=Path("data/interim/hubert_ecg_embeddings.parquet")
    )
    join.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/echo_hubert_manifest.parquet"),
    )
    join.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/processed/echo_hubert_manifest_metadata.csv"),
    )
    join.add_argument(
        "--summary-json",
        type=Path,
        default=Path("logs/echo_hubert_join_summary.json"),
    )
    for name in [
        "subject-col",
        "echo-study-col",
        "ecg-study-col",
        "lvef-col",
        "ef-le-40-col",
        "split-col",
        "sex-col",
        "age-col",
        "race-col",
    ]:
        join.add_argument(f"--{name}")

    all_steps = subparsers.add_parser("all", help="Run all build steps.")
    all_steps.add_argument("--echo-input", type=Path, required=True)
    all_steps.add_argument("--hubert-csv", type=Path, required=True)
    all_steps.add_argument("--cohort", type=Path, required=True)
    all_steps.add_argument(
        "--echo-output",
        type=Path,
        default=Path("data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet"),
    )
    all_steps.add_argument(
        "--ecg-output",
        type=Path,
        default=Path("data/interim/hubert_ecg_embeddings.parquet"),
    )
    all_steps.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/echo_hubert_manifest.parquet"),
    )
    all_steps.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/processed/echo_hubert_manifest_metadata.csv"),
    )
    all_steps.add_argument(
        "--summary-json",
        type=Path,
        default=Path("logs/echo_hubert_join_summary.json"),
    )
    all_steps.add_argument("--chunksize", type=int, default=50_000)
    all_steps.add_argument("--max-clips", type=int)
    return parser.parse_args()


def cohort_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "subject_col": getattr(args, "subject_col", None),
        "echo_study_col": getattr(args, "echo_study_col", None),
        "ecg_study_col": getattr(args, "ecg_study_col", None),
        "lvef_col": getattr(args, "lvef_col", None),
        "ef_le_40_col": getattr(args, "ef_le_40_col", None),
        "split_col": getattr(args, "split_col", None),
        "sex_col": getattr(args, "sex_col", None),
        "age_col": getattr(args, "age_col", None),
        "race_col": getattr(args, "race_col", None),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, indent=2, default=str))


def main() -> None:
    args = parse_args()
    from primed_ai.data.echo_hubert_manifest import (
        build_echo_study_embeddings,
        build_joined_manifest,
        convert_hubert_csv_to_parquet,
    )

    if args.command == "build-echo":
        frame = build_echo_study_embeddings(
            args.input,
            args.output,
            subject_col=args.subject_col,
            study_col=args.study_col,
            embedding_col=args.embedding_col,
            echo_model=args.echo_model,
            max_clips=args.max_clips,
        )
        print(f"Wrote {len(frame)} echo study embeddings to {args.output}")
    elif args.command == "convert-ecg":
        convert_hubert_csv_to_parquet(
            args.csv,
            args.output,
            subject_col=args.subject_col,
            ecg_study_col=args.ecg_study_col,
            embedding_col=args.ecg_embedding_col,
            filename_col=args.filename_col,
            feature_prefixes=args.ecg_feature_prefix or ("ve", "emb_", "embedding_", "feature_"),
            ecg_model=args.ecg_model,
            chunksize=args.chunksize,
        )
        print(f"Wrote HuBERT ECG embeddings to {args.output}")
    elif args.command == "join":
        _, summary = build_joined_manifest(
            args.cohort,
            args.echo,
            args.ecg,
            args.manifest,
            args.metadata_csv,
            args.summary_json,
            **cohort_kwargs(args),
        )
        print_summary(summary)
    elif args.command == "all":
        build_echo_study_embeddings(args.echo_input, args.echo_output, max_clips=args.max_clips)
        convert_hubert_csv_to_parquet(args.hubert_csv, args.ecg_output, chunksize=args.chunksize)
        _, summary = build_joined_manifest(
            args.cohort,
            args.echo_output,
            args.ecg_output,
            args.manifest,
            args.metadata_csv,
            args.summary_json,
        )
        print_summary(summary)


if __name__ == "__main__":
    main()
