"""Check whether EchoJEPA + HuBERT inputs are ready for the joined manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from primed_ai.data.echo_hubert_manifest import embedding_dim, read_table  # noqa: E402

REQUIRED_COHORT_COLUMNS = {
    "subject_id",
    "echo_study_id",
    "ecg_study_id",
    "lvef",
}
RECOMMENDED_COHORT_COLUMNS = {
    "ef_le_40",
    "split",
    "sex",
    "age",
    "race",
}


def parquet_summary(path: Path, embedding_column: str) -> dict[str, object]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    parquet_file = pq.ParquetFile(path)
    sample = pd.read_parquet(path, columns=[embedding_column]).head(20)
    return {
        "exists": True,
        "path": str(path),
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "n_rows": int(parquet_file.metadata.num_rows),
        "n_row_groups": int(parquet_file.metadata.num_row_groups),
        "embedding_dim": embedding_dim(sample[embedding_column]),
    }


def cohort_summary(path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "exists": False,
            "required_columns": sorted(REQUIRED_COHORT_COLUMNS),
            "recommended_columns": sorted(RECOMMENDED_COHORT_COLUMNS),
        }
    if not path.exists():
        return {"exists": False, "path": str(path)}
    frame = read_table(path)
    columns = set(frame.columns)
    return {
        "exists": True,
        "path": str(path),
        "n_rows": int(len(frame)),
        "columns": sorted(columns),
        "missing_required_columns": sorted(REQUIRED_COHORT_COLUMNS - columns),
        "missing_recommended_columns": sorted(RECOMMENDED_COHORT_COLUMNS - columns),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--echo",
        type=Path,
        default=Path("data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet"),
    )
    parser.add_argument(
        "--ecg",
        type=Path,
        default=Path("data/interim/hubert_ecg_embeddings.parquet"),
    )
    parser.add_argument("--cohort", type=Path, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/echo_hubert_readiness.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = {
        "echo": parquet_summary(args.echo, "echo_embedding"),
        "ecg": parquet_summary(args.ecg, "ecg_embedding"),
        "cohort": cohort_summary(args.cohort),
    }
    summary["ready_for_join"] = bool(
        summary["echo"].get("exists")
        and summary["ecg"].get("exists")
        and summary["cohort"].get("exists")
        and not summary["cohort"].get("missing_required_columns", [])
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
