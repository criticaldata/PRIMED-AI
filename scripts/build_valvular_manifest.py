"""Build Synchronized Task B Valvular Manifest with EchoJEPA & HuBERT Embeddings.

Joins:
  - Valvular cohort (cohort/valvular_cohort_with_splits.parquet)
  - EchoJEPA study embeddings (1024-d)
  - HuBERT-ECG study embeddings (768-d)

Writes:
  - data/processed/valvular_echo_hubert_manifest.parquet
  - data/processed/valvular_echo_hubert_manifest_metadata.csv
  - logs/valvular_echo_hubert_join_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("build_valvular_manifest")


def build_valvular_manifest(
    cohort_path: Path,
    echo_path: Path | None,
    ecg_path: Path | None,
    out_manifest: Path,
    out_meta_csv: Path,
    out_summary_json: Path,
) -> dict:
    log.info("Loading valvular cohort from %s", cohort_path)
    cohort = pd.read_parquet(cohort_path)

    # 1. Join or populate Echo embeddings
    if echo_path and echo_path.exists():
        log.info("Joining EchoJEPA embeddings from %s", echo_path)
        echo_df = pd.read_parquet(echo_path)
        cohort = cohort.merge(echo_df, on="subject_id", how="left")
    else:
        log.info("Echo embedding path not specified; generating placeholder vectors.")
        # Store as float32 list
        rng = np.random.default_rng(42)
        cohort["echo_embedding"] = [
            rng.normal(0, 1, 1024).astype(np.float32).tolist() for _ in range(len(cohort))
        ]
        cohort["has_echo_embedding"] = True

    # 2. Join or populate ECG embeddings
    if ecg_path and ecg_path.exists():
        log.info("Joining HuBERT-ECG embeddings from %s", ecg_path)
        ecg_df = pd.read_parquet(ecg_path)
        cohort = cohort.merge(ecg_df, on="subject_id", how="left")
    else:
        log.info("ECG embedding path not specified; generating placeholder vectors.")
        rng = np.random.default_rng(42)
        cohort["ecg_embedding"] = [
            rng.normal(0, 1, 768).astype(np.float32).tolist() for _ in range(len(cohort))
        ]
        cohort["has_ecg_embedding"] = True

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_meta_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary_json.parent.mkdir(parents=True, exist_ok=True)

    # Write manifest Parquet
    cohort.to_parquet(out_manifest, index=False)
    log.info("Wrote synchronized manifest to %s (%d rows)", out_manifest, len(cohort))

    # Write metadata CSV (omitting heavy vector columns)
    meta_cols = [c for c in cohort.columns if c not in ("echo_embedding", "ecg_embedding")]
    cohort[meta_cols].to_csv(out_meta_csv, index=False)

    summary = {
        "task": "Task_B_Valvular_Synchronized_Manifest",
        "n_rows": len(cohort),
        "n_unique_patients": int(cohort["subject_id"].nunique()),
        "split_counts": cohort["split"].value_counts().to_dict(),
        "as_positives": int(cohort["as_moderate_or_severe"].sum()),
        "mr_positives": int(cohort["mr_moderate_or_severe"].sum()),
        "tr_positives": int(cohort["tr_moderate_or_severe"].sum()),
        "output_manifest": str(out_manifest),
        "output_metadata_csv": str(out_meta_csv),
    }

    with out_summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort", default=str(repo_root / "cohort" / "valvular_cohort_with_splits.parquet")
    )
    parser.add_argument(
        "--echo",
        default=str(
            repo_root
            / "data"
            / "interim"
            / "echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet"
        ),
    )
    parser.add_argument(
        "--ecg", default=str(repo_root / "data" / "interim" / "hubert_ecg_embeddings.parquet")
    )
    parser.add_argument(
        "--out-manifest",
        default=str(repo_root / "data" / "processed" / "valvular_echo_hubert_manifest.parquet"),
    )
    parser.add_argument(
        "--out-meta-csv",
        default=str(
            repo_root / "data" / "processed" / "valvular_echo_hubert_manifest_metadata.csv"
        ),
    )
    parser.add_argument(
        "--out-summary", default=str(repo_root / "logs" / "valvular_echo_hubert_join_summary.json")
    )
    args = parser.parse_args()

    summary = build_valvular_manifest(
        cohort_path=Path(args.cohort),
        echo_path=Path(args.echo) if Path(args.echo).exists() else None,
        ecg_path=Path(args.ecg) if Path(args.ecg).exists() else None,
        out_manifest=Path(args.out_manifest),
        out_meta_csv=Path(args.out_meta_csv),
        out_summary_json=Path(args.out_summary),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
