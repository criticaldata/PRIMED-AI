"""Build EchoJEPA + HuBERT synchronized embedding manifests."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ECHO_MODEL = "vjepa2.1-vitl-mimic-pt-100"
DEFAULT_ECG_MODEL = "hubert"
ECG_FILENAME_RE = re.compile(r"(?:^|/)p\d+/p(?P<subject_id>\d+)/s(?P<study_id>\d+)/")


@dataclass(frozen=True)
class ColumnMap:
    """Resolved source columns for cohort, echo, or ECG inputs."""

    subject_id: str
    study_id: str | None = None
    embedding: str | None = None
    lvef: str | None = None
    ef_le_40: str | None = None
    split: str | None = None
    sex: str | None = None
    age: str | None = None
    race: str | None = None


def first_existing(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    lower_to_original = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def require_column(
    columns: Iterable[str],
    explicit: str | None,
    candidates: Sequence[str],
    label: str,
) -> str:
    if explicit:
        return explicit
    resolved = first_existing(columns, candidates)
    if resolved:
        return resolved
    raise ValueError(f"Could not infer {label} column from: {sorted(columns)}")


def parse_embedding(value: Any) -> list[float] | None:
    """Return a float vector from common CSV/Parquet embedding representations."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    if hasattr(value, "as_py"):
        return parse_embedding(value.as_py())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = text.strip("[]").replace(",", " ").split()
        return [float(item) for item in parsed]
    return [float(value)]


def embedding_dim(series: pd.Series) -> int | None:
    for value in series:
        parsed = parse_embedding(value)
        if parsed is not None:
            return len(parsed)
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for {path}")


def parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.glob("*.parquet"))
    if not files:
        files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return files


def build_echo_study_embeddings(
    input_path: Path,
    output_path: Path,
    *,
    subject_col: str | None = None,
    study_col: str | None = None,
    embedding_col: str | None = None,
    echo_model: str = DEFAULT_ECHO_MODEL,
    batch_size: int = 1024,
) -> pd.DataFrame:
    """Mean-pool clip-level EchoJEPA embeddings to one vector per echo study."""

    sums: dict[tuple[Any, Any], np.ndarray] = {}
    counts: defaultdict[tuple[Any, Any], int] = defaultdict(int)

    for file_path in parquet_files(input_path):
        parquet_file = pq.ParquetFile(file_path)
        columns = parquet_file.schema_arrow.names
        resolved_subject = require_column(columns, subject_col, ["subject_id"], "echo subject_id")
        resolved_study = require_column(
            columns, study_col, ["study_id", "echo_study_id"], "echo study_id"
        )
        resolved_embedding = require_column(
            columns, embedding_col, ["embedding", "echo_embedding"], "echo embedding"
        )
        read_columns = [resolved_subject, resolved_study, resolved_embedding]
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_columns):
            frame = batch.to_pandas()
            for row in frame.itertuples(index=False):
                subject_id, study_id, raw_embedding = row
                vector = parse_embedding(raw_embedding)
                if vector is None:
                    continue
                key = (subject_id, study_id)
                arr = np.asarray(vector, dtype=np.float64)
                if key not in sums:
                    sums[key] = np.zeros_like(arr)
                if sums[key].shape != arr.shape:
                    raise ValueError(
                        f"Inconsistent echo embedding dimension for {key}: "
                        f"{sums[key].shape[0]} vs {arr.shape[0]}"
                    )
                sums[key] += arr
                counts[key] += 1

    rows = [
        {
            "subject_id": subject_id,
            "echo_study_id": echo_study_id,
            "n_echo_clips": counts[(subject_id, echo_study_id)],
            "echo_embedding": (
                sums[(subject_id, echo_study_id)] / counts[(subject_id, echo_study_id)]
            )
            .astype(np.float32)
            .tolist(),
            "echo_model": echo_model,
        }
        for subject_id, echo_study_id in sorted(sums)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(output_path, index=False)
    return frame


def infer_embedding_feature_columns(columns: Iterable[str], prefixes: Sequence[str]) -> list[str]:
    candidates = [
        col for col in columns if any(col.lower().startswith(prefix.lower()) for prefix in prefixes)
    ]
    return sorted(candidates, key=lambda value: (len(value), value))


def parse_ecg_filename_ids(value: Any) -> tuple[int | None, int | None]:
    """Parse MIMIC-IV-ECG subject/study IDs from paths like p1000/p10000032/s40689238."""

    if value is None:
        return None, None
    match = ECG_FILENAME_RE.search(str(value))
    if not match:
        return None, None
    return int(match.group("subject_id")), int(match.group("study_id"))


def convert_hubert_csv_to_parquet(
    csv_path: Path,
    output_path: Path,
    *,
    subject_col: str | None = None,
    ecg_study_col: str | None = None,
    embedding_col: str | None = None,
    filename_col: str | None = None,
    feature_prefixes: Sequence[str] = ("ve", "emb_", "embedding_", "feature_"),
    ecg_model: str = DEFAULT_ECG_MODEL,
    chunksize: int = 50_000,
) -> pd.DataFrame | None:
    """Convert a large HuBERT ECG CSV into vector-column Parquet."""

    sample = pd.read_csv(csv_path, nrows=5)
    columns = sample.columns.tolist()
    resolved_filename = filename_col or first_existing(columns, ["filename", "file_path", "path"])
    resolved_subject = first_existing(columns, [subject_col] if subject_col else ["subject_id"])
    resolved_study = first_existing(
        columns,
        [ecg_study_col]
        if ecg_study_col
        else ["ecg_study_id", "study_id", "record_id", "ecg_record_id"],
    )
    if not (resolved_subject and resolved_study) and not resolved_filename:
        raise ValueError(
            "Could not infer ECG IDs. Pass subject/study columns or a filename "
            "column containing MIMIC-IV-ECG p*/p<subject_id>/s<study_id>/ paths."
        )
    resolved_embedding = embedding_col or first_existing(
        columns, ["ecg_embedding", "embedding", "embeddings"]
    )
    feature_columns = (
        [] if resolved_embedding else infer_embedding_feature_columns(columns, feature_prefixes)
    )
    if not resolved_embedding and not feature_columns:
        raise ValueError(
            "Could not infer ECG embeddings. Pass --ecg-embedding-col or --ecg-feature-prefix."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    preview: pd.DataFrame | None = None
    read_columns = []
    if resolved_subject:
        read_columns.append(resolved_subject)
    if resolved_study:
        read_columns.append(resolved_study)
    if resolved_filename:
        read_columns.append(resolved_filename)
    read_columns += [resolved_embedding] if resolved_embedding else feature_columns
    read_columns = list(dict.fromkeys(read_columns))

    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=read_columns):
            if resolved_subject and resolved_study:
                subject_ids = chunk[resolved_subject]
                study_ids = chunk[resolved_study]
            else:
                parsed_ids = chunk[resolved_filename].map(parse_ecg_filename_ids)
                subject_ids = parsed_ids.map(lambda item: item[0])
                study_ids = parsed_ids.map(lambda item: item[1])
            if resolved_embedding:
                vectors = chunk[resolved_embedding].map(parse_embedding)
            else:
                vectors = chunk[feature_columns].astype(float).values.tolist()
            out = pd.DataFrame(
                {
                    "subject_id": subject_ids,
                    "ecg_study_id": study_ids,
                    "ecg_embedding": vectors,
                    "ecg_model": ecg_model,
                }
            )
            out = out.dropna(subset=["subject_id", "ecg_study_id"])
            out["subject_id"] = out["subject_id"].astype("int64")
            out["ecg_study_id"] = out["ecg_study_id"].astype("int64")
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            if preview is None:
                preview = out.head(20).copy()
    finally:
        if writer is not None:
            writer.close()

    return preview


def normalize_cohort_columns(
    cohort: pd.DataFrame,
    *,
    subject_col: str | None = None,
    echo_study_col: str | None = None,
    ecg_study_col: str | None = None,
    lvef_col: str | None = None,
    ef_le_40_col: str | None = None,
    split_col: str | None = None,
    sex_col: str | None = None,
    age_col: str | None = None,
    race_col: str | None = None,
) -> pd.DataFrame:
    columns = cohort.columns.tolist()
    mapping = {
        require_column(columns, subject_col, ["subject_id"], "cohort subject_id"): "subject_id",
        require_column(
            columns,
            echo_study_col,
            ["echo_study_id", "study_id", "echo_id"],
            "cohort echo_study_id",
        ): "echo_study_id",
        require_column(
            columns,
            ecg_study_col,
            ["ecg_study_id", "ecg_record_id", "record_id"],
            "cohort ecg_study_id",
        ): "ecg_study_id",
        require_column(columns, lvef_col, ["lvef", "lvef_value"], "cohort lvef"): "lvef",
    }
    optional = {
        first_existing(columns, [ef_le_40_col] if ef_le_40_col else ["ef_le_40"]): "ef_le_40",
        first_existing(columns, [split_col] if split_col else ["split", "fold"]): "split",
        first_existing(columns, [sex_col] if sex_col else ["sex", "gender"]): "sex",
        first_existing(columns, [age_col] if age_col else ["age", "anchor_age"]): "age",
        first_existing(columns, [race_col] if race_col else ["race", "ethnicity"]): "race",
    }
    mapping.update({key: value for key, value in optional.items() if key})
    out = cohort.rename(columns=mapping).copy()
    for col in ["ef_le_40", "split", "sex", "age", "race"]:
        if col not in out:
            out[col] = pd.NA
    missing_label = out["ef_le_40"].isna()
    out.loc[missing_label, "ef_le_40"] = out.loc[missing_label, "lvef"].astype(float) <= 40
    return out


def build_joined_manifest(
    cohort_path: Path,
    echo_embeddings_path: Path,
    ecg_embeddings_path: Path,
    manifest_path: Path,
    metadata_csv_path: Path,
    summary_json_path: Path,
    **cohort_column_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join cohort labels/splits to EchoJEPA and HuBERT study embeddings."""

    cohort = normalize_cohort_columns(read_table(cohort_path), **cohort_column_kwargs)
    echo = pd.read_parquet(echo_embeddings_path)
    ecg = pd.read_parquet(ecg_embeddings_path)

    manifest = cohort.merge(
        echo,
        on=["subject_id", "echo_study_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        ecg,
        on=["subject_id", "ecg_study_id"],
        how="left",
        validate="many_to_one",
    )

    manifest["has_echo_embedding"] = manifest["echo_embedding"].notna()
    manifest["has_ecg_embedding"] = manifest["ecg_embedding"].notna()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)

    metadata_columns = [
        "subject_id",
        "echo_study_id",
        "ecg_study_id",
        "lvef",
        "ef_le_40",
        "split",
        "sex",
        "age",
        "race",
        "n_echo_clips",
        "has_echo_embedding",
        "has_ecg_embedding",
        "echo_model",
        "ecg_model",
    ]
    existing_metadata_columns = [col for col in metadata_columns if col in manifest]
    manifest[existing_metadata_columns].to_csv(metadata_csv_path, index=False)

    summary = summarize_manifest(manifest)
    summary["outputs"] = {
        "manifest_path": str(manifest_path),
        "metadata_csv_path": str(metadata_csv_path),
        "summary_json_path": str(summary_json_path),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return manifest, summary


def summarize_manifest(manifest: pd.DataFrame) -> dict[str, Any]:
    split_counts = manifest["split"].value_counts(dropna=False).to_dict()
    leakage = (
        manifest.dropna(subset=["split"])
        .groupby("subject_id")["split"]
        .nunique()
        .loc[lambda values: values > 1]
    )
    duplicates = manifest.duplicated(
        subset=["subject_id", "echo_study_id", "ecg_study_id"], keep=False
    )
    summary: dict[str, Any] = {
        "n_cohort_rows": int(len(manifest)),
        "n_with_echo_embedding": int(manifest["has_echo_embedding"].sum()),
        "n_with_ecg_embedding": int(manifest["has_ecg_embedding"].sum()),
        "n_with_both_embeddings": int(
            (manifest["has_echo_embedding"] & manifest["has_ecg_embedding"]).sum()
        ),
        "n_missing_lvef": int(manifest["lvef"].isna().sum()),
        "n_missing_split": int(manifest["split"].isna().sum()),
        "n_duplicate_subject_echo_ecg": int(duplicates.sum()),
        "train_count": int(split_counts.get("train", 0)),
        "val_count": int(split_counts.get("val", 0)),
        "test_count": int(split_counts.get("test", 0)),
        "echo_embedding_dim": embedding_dim(manifest["echo_embedding"]),
        "ecg_embedding_dim": embedding_dim(manifest["ecg_embedding"]),
        "join_keys_used": {
            "echo": ["subject_id", "echo_study_id"],
            "ecg": ["subject_id", "ecg_study_id"],
        },
        "subject_split_leakage_count": int(len(leakage)),
        "subject_split_leakage_subject_ids": [str(value) for value in leakage.index],
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
    }
    if summary["subject_split_leakage_count"]:
        raise ValueError(
            "Subject leakage across splits detected for "
            f"{summary['subject_split_leakage_subject_ids'][:10]}"
        )
    return summary
