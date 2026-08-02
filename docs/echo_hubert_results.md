# EchoJEPA + HuBERT Sync Results

## Status

The EchoJEPA + HuBERT synchronized labeled manifest is complete for the local
workflow using `../outputs_build_cohort/paired.csv` as the cohort backbone.

The cohort includes timestamp-derived Echo-ECG pairs, LVEF labels, EF<=40
labels, and demographics. Subject-level train/val/test splits were generated
after cohort construction.

## Generated Artifacts

Generated data and logs are intentionally ignored by git.

```text
data/raw/cohort/paired_with_splits.parquet
data/raw/cohort/subject_splits.csv
logs/splits.json
data/processed/echo_hubert_manifest.parquet
data/processed/echo_hubert_manifest_metadata.csv
logs/echo_hubert_join_summary.json
```

## Reproduce Locally

The preferred one-command reproduction path is:

```bash
make reproduce-echo-hubert-local
```

Required local inputs:

```text
../outputs_build_cohort/paired.csv
data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet
data/interim/hubert_ecg_embeddings.parquet
```

The command regenerates the split cohort, subject split file, split manifest,
EF<=40 prevalence logs, joined manifest, metadata CSV, join summary JSON, and a
train/val/test dataloader smoke test.

Previously generated embedding artifacts used by the join:

```text
data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet
data/interim/hubert_ecg_embeddings.parquet
data/processed/echo_hubert_subject_overlap_metadata.csv
logs/jepa_embedding_inspection.json
logs/echo_study_embedding_summary.json
logs/hubert_ecg_conversion_summary.json
logs/echo_hubert_subject_overlap_summary.json
logs/echo_hubert_readiness.json
```

## Manifest Columns

The metadata CSV contains:

```text
subject_id
echo_study_id
ecg_study_id
lvef
ef_le_40
split
sex
age
race
n_echo_clips
has_echo_embedding
has_ecg_embedding
echo_model
ecg_model
```

The full Parquet manifest also contains:

```text
echo_embedding
ecg_embedding
```

## Validation Summary

```text
n_cohort_rows: 1208
n_with_echo_embedding: 1208
n_with_ecg_embedding: 1208
n_with_both_embeddings: 1208
n_missing_lvef: 0
n_missing_split: 0
n_duplicate_subject_echo_ecg: 0
train_count: 829
val_count: 131
test_count: 248
echo_embedding_dim: 1024
ecg_embedding_dim: 768
subject_split_leakage_count: 0
```

Subject counts by split:

```text
train: 702
val: 100
test: 201
```

Loader smoke test:

```text
train: 829 rows, echo (1024,), ECG (768,)
val: 131 rows, echo (1024,), ECG (768,)
test: 248 rows, echo (1024,), ECG (768,)
```

Unit test result:

```text
pytest tests/test_echo_hubert.py -q
2 passed
```

## Join Details

Echo join keys:

```text
subject_id
echo_study_id
```

ECG join keys:

```text
subject_id
ecg_study_id
```

The source cohort column `ecg_record_id` is normalized to `ecg_study_id` during
manifest construction.

Direct `echo_study_id == ecg_study_id` matching is not used. The final cohort
uses timestamp-derived Echo-ECG pairs from `../outputs_build_cohort/paired.csv`.
