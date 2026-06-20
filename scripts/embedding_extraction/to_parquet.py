#!/usr/bin/env python3
"""Convert per-folder .pt embedding dicts to sharded Parquet with full MIMIC-IV-Echo metadata."""

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

EMBEDDINGS_DIR = "/orcd/pool/006/lceli_shared/jepa-embeddings-mimiciv-echo"
METADATA_DIR = "/orcd/pool/006/lceli_shared/mimic-iv-echo"
FOLDERS = [f"p{i}" for i in range(10, 20)]


def build_record_lookup(csv_path: str) -> dict:
    """Load echo-record-list.csv, key by mp4 path (strip 'files/', .dcm -> .mp4)."""
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        # files/p10/p10002221/s94106955/94106955_0001.dcm -> p10002221/s94106955/94106955_0001.mp4
        # PT keys don't include the pXX folder prefix, so strip "files/pXX/"
        dicom = row["dicom_filepath"]
        # Remove "files/pXX/" prefix (first 2 path components)
        stripped = "/".join(dicom.split("/")[2:])
        mp4_path = stripped.replace(".dcm", ".mp4")
        lookup[mp4_path] = {
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "acquisition_datetime": str(row["acquisition_datetime"]) if pd.notna(row["acquisition_datetime"]) else None,
        }
    return lookup


def build_study_lookup(csv_path: str) -> dict:
    """Load echo-study-list.csv, key by (subject_id, study_id)."""
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        key = (int(row["subject_id"]), int(row["study_id"]))
        lookup[key] = {
            "study_datetime": str(row["study_datetime"]) if pd.notna(row["study_datetime"]) else None,
            "note_id": str(row["note_id"]) if pd.notna(row["note_id"]) else None,
            "note_seq": str(row["note_seq"]) if pd.notna(row["note_seq"]) else None,
            "note_charttime": str(row["note_charttime"]) if pd.notna(row["note_charttime"]) else None,
        }
    return lookup


def parse_path_metadata(file_path: str) -> dict:
    """Fallback: parse subject_id, study_id, dicom_id from path.

    Expected path: p10/p10790436/s99115579/99115579_0019.mp4
                   [0]  [1]       [2]       [3]
    """
    parts = file_path.split("/")
    dicom_id = Path(parts[-1]).stem if parts else None  # 99115579_0019

    subject_id = None
    study_id = None
    for part in parts:
        if part.startswith("p") and len(part) > 3 and part[1:].isdigit():
            subject_id = int(part[1:])
        elif part.startswith("s") and part[1:].isdigit():
            study_id = int(part[1:])

    return {"subject_id": subject_id, "study_id": study_id, "dicom_id": dicom_id}


def main():
    parser = argparse.ArgumentParser(description="Convert .pt embeddings to Parquet")
    parser.add_argument("--model", default="vitl", help="Model name (default: vitl)")
    parser.add_argument("--embeddings_dir", default=EMBEDDINGS_DIR)
    parser.add_argument("--metadata_dir", default=METADATA_DIR)
    args = parser.parse_args()

    out_dir = os.path.join(args.embeddings_dir, "mimic-iv-echo-jepa-embeddings", f"jepa-{args.model[3:]}-embeddings")
    os.makedirs(out_dir, exist_ok=True)

    # Load metadata
    print("Loading echo-record-list.csv...")
    record_lookup = build_record_lookup(os.path.join(args.metadata_dir, "echo-record-list.csv"))
    print(f"  {len(record_lookup)} records")

    print("Loading echo-study-list.csv...")
    study_lookup = build_study_lookup(os.path.join(args.metadata_dir, "echo-study-list.csv"))
    print(f"  {len(study_lookup)} studies")

    schema = pa.schema([
        ("subject_id", pa.int64()),
        ("study_id", pa.int64()),
        ("dicom_id", pa.string()),
        ("file_path", pa.string()),
        ("acquisition_datetime", pa.string()),
        ("study_datetime", pa.string()),
        ("note_id", pa.string()),
        ("note_seq", pa.string()),
        ("note_charttime", pa.string()),
        ("embedding", pa.list_(pa.float32(), 1024)),
    ])

    total_rows = 0
    total_matched_record = 0
    total_matched_study = 0
    num_shards = len(FOLDERS)

    for i, folder in enumerate(FOLDERS):
        pt_path = os.path.join(args.embeddings_dir, f"{args.model}_embeddings_{folder}.pt")
        if not os.path.exists(pt_path):
            print(f"  Skipping {folder}: {pt_path} not found")
            continue

        print(f"\n[{i+1}/{num_shards}] Loading {folder}...")
        data = torch.load(pt_path, map_location="cpu", weights_only=False)

        rows = {
            "subject_id": [],
            "study_id": [],
            "dicom_id": [],
            "file_path": [],
            "acquisition_datetime": [],
            "study_datetime": [],
            "note_id": [],
            "note_seq": [],
            "note_charttime": [],
            "embedding": [],
        }

        matched_record = 0
        matched_study = 0

        for file_path, tensor in data.items():
            path_meta = parse_path_metadata(file_path)

            # Try record lookup first
            rec = record_lookup.get(file_path)
            if rec:
                subject_id = rec["subject_id"]
                study_id = rec["study_id"]
                acq_dt = rec["acquisition_datetime"]
                matched_record += 1
            else:
                subject_id = path_meta["subject_id"]
                study_id = path_meta["study_id"]
                acq_dt = None

            # Study lookup
            study = study_lookup.get((subject_id, study_id))
            if study:
                matched_study += 1
            else:
                study = {"study_datetime": None, "note_id": None, "note_seq": None, "note_charttime": None}

            rows["subject_id"].append(subject_id)
            rows["study_id"].append(study_id)
            rows["dicom_id"].append(path_meta["dicom_id"])
            rows["file_path"].append(file_path)
            rows["acquisition_datetime"].append(acq_dt)
            rows["study_datetime"].append(study["study_datetime"])
            rows["note_id"].append(study["note_id"])
            rows["note_seq"].append(study["note_seq"])
            rows["note_charttime"].append(study["note_charttime"])
            rows["embedding"].append(tensor.tolist())

        table = pa.table(rows, schema=schema)
        shard_path = os.path.join(out_dir, f"train-{i:05d}-of-{num_shards:05d}.parquet")
        pq.write_table(table, shard_path, compression="snappy")

        size_mb = os.path.getsize(shard_path) / 1e6
        print(f"  {folder}: {len(data)} rows → {shard_path} ({size_mb:.0f} MB)")
        print(f"  Record match: {matched_record}/{len(data)} ({100*matched_record/len(data):.1f}%)")
        print(f"  Study match:  {matched_study}/{len(data)} ({100*matched_study/len(data):.1f}%)")

        total_rows += len(data)
        total_matched_record += matched_record
        total_matched_study += matched_study

    print(f"\nDone! {total_rows} rows across {num_shards} shards")
    print(f"Record coverage: {total_matched_record}/{total_rows} ({100*total_matched_record/total_rows:.1f}%)")
    print(f"Study coverage:  {total_matched_study}/{total_rows} ({100*total_matched_study/total_rows:.1f}%)")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    main()
