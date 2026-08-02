"""Inspect MIMIC-IV-Echo JEPA embedding parquet shards.

This script is intentionally lightweight for laptop use. By default it
downloads/inspects one shard rather than the full gated Hugging Face dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

DEFAULT_REPO_ID = "MITCriticalData/mimic-iv-echo-jepa-embeddings"
DEFAULT_SHARD = "vjepa2.1-vitl-mimic-pt-100/train-00000-of-00010.parquet"
METADATA_COLUMNS = [
    "subject_id",
    "study_id",
    "dicom_id",
    "file_path",
    "acquisition_datetime",
    "study_datetime",
    "note_id",
    "note_seq",
    "note_charttime",
]


def download_shard(repo_id: str, filename: str, cache_dir: Path | None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for downloads. Install with: pip install huggingface_hub"
        ) from exc

    token = os.environ.get("HF_TOKEN")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache_dir) if cache_dir else None,
            token=token,
        )
    )


def first_embedding_info(path: Path) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=16, columns=["embedding"]):
        for value in batch.column("embedding").to_pylist():
            if value is None:
                continue
            return {
                "embedding_dim": len(value),
                "embedding_value_type": type(value[0]).__name__ if value else "unknown",
                "first_embedding_preview": value[:5],
            }
    return {
        "embedding_dim": None,
        "embedding_value_type": None,
        "first_embedding_preview": [],
    }


def inspect_parquet(path: Path, sample_rows: int) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    available_columns = set(schema.names)
    metadata_columns = [col for col in METADATA_COLUMNS if col in available_columns]

    metadata_table = pq.read_table(path, columns=metadata_columns)
    sample_table = metadata_table.slice(0, sample_rows)

    summary: dict[str, Any] = {
        "path": str(path),
        "file_size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "num_rows": parquet_file.metadata.num_rows,
        "num_row_groups": parquet_file.metadata.num_row_groups,
        "schema": str(schema),
        "columns": schema.names,
        "sample_rows": sample_table.to_pylist(),
        "null_counts": {},
        "unique_counts": {},
        "duplicate_counts": {},
    }

    for col in metadata_columns:
        column = metadata_table[col]
        null_count = pc.sum(pc.is_null(column)).as_py()
        unique_count = len(pc.unique(column))
        summary["null_counts"][col] = null_count
        summary["unique_counts"][col] = unique_count
        summary["duplicate_counts"][col] = metadata_table.num_rows - unique_count

    if "embedding" in available_columns:
        summary.update(first_embedding_info(path))

    return summary


def write_outputs(summary: dict[str, Any], output_json: Path | None) -> None:
    text = json.dumps(summary, indent=2, default=str)
    print(text)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--local-parquet",
        type=Path,
        help="Path to a local parquet shard to inspect.",
    )
    source.add_argument(
        "--download-shard",
        action="store_true",
        help="Download one shard from Hugging Face before inspection.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_SHARD)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory. Useful if your home disk is tight.",
    )
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/jepa_embedding_inspection.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = (
        download_shard(args.repo_id, args.filename, args.cache_dir)
        if args.download_shard
        else args.local_parquet
    )
    summary = inspect_parquet(path, args.sample_rows)
    write_outputs(summary, args.output_json)


if __name__ == "__main__":
    main()
