# EchoJEPA + HuBERT Loader

This workflow synchronizes EchoJEPA echo embeddings, HuBERT ECG embeddings, and
the paired LVEF cohort file without creating a giant mixed CSV.

## Current Status

Completed locally from the files currently available:

- EchoJEPA `vjepa2.1-vitl-mimic-pt-100` shards were downloaded from Hugging Face.
- EchoJEPA clip-level embeddings were mean-pooled to one vector per echo study.
- HuBERT ECG embeddings were downloaded as an 11.25 GB CSV from Drive.
- HuBERT ECG embeddings were converted to Parquet.
- HuBERT `subject_id` and `ecg_study_id` were parsed from `filename` paths.
- `outputs_build_cohort/paired.csv` supplied the final paired LVEF cohort.
- Subject-level train/val/test splits were generated from that cohort.
- The synchronized EchoJEPA + HuBERT manifest and metadata CSV were built.
- Dataloader, manifest-building, validation, and readiness-check code is ready.
- Subject-level Echo + ECG overlap was quantified from the embedding files.

Completed artifacts:

```text
data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet
data/interim/hubert_ecg_embeddings.parquet
data/processed/echo_hubert_subject_overlap_metadata.csv
data/raw/cohort/paired_with_splits.parquet
data/raw/cohort/subject_splits.csv
data/processed/echo_hubert_manifest.parquet
data/processed/echo_hubert_manifest_metadata.csv
logs/splits.json
logs/jepa_embedding_inspection.json
logs/echo_study_embedding_summary.json
logs/hubert_ecg_conversion_summary.json
logs/echo_hubert_subject_overlap_summary.json
logs/echo_hubert_readiness.json
logs/echo_hubert_join_summary.json
```

Current validation numbers:

```text
Echo studies: 7,251
Echo subjects: 4,585
Echo embedding dim: 1,024

ECG studies: 800,035
ECG subjects: 161,352
ECG embedding dim: 768

Shared Echo + ECG subjects: 4,103
Echo studies among shared subjects: 6,617
ECG studies among shared subjects: 57,459
Subject-only candidate Echo+ECG pairs: 116,754
Exact echo_study_id == ecg_study_id matches: 0

Final joined manifest rows: 1,208
Rows with both EchoJEPA and HuBERT embeddings: 1,208
Train/val/test rows: 829/131/248
Train/val/test subjects: 702/100/201
Joined echo embedding dim: 1,024
Joined ECG embedding dim: 768
Missing LVEF: 0
Missing split: 0
Duplicate subject/echo/ECG rows: 0
Subject split leakage count: 0
```

Important interpretation:

- The Hugging Face dataset viewer split named `train` is a storage split, not a
  modeling train split.
- The 3.15M rows shown by the dataset viewer correspond to all available
  embedding variants together. The selected EchoJEPA variant contains about
  525K clip rows, which were pooled into 7,251 echo-study rows.
- Direct `echo_study_id == ecg_study_id` pairing is not the right join. The
  final cohort uses timestamp-derived Echo↔ECG pairs from
  `outputs_build_cohort/paired.csv`.
- `ecg_record_id` from the cohort is normalized to `ecg_study_id` in the joined
  manifest so the loader has one consistent ECG key.
- The train/val/test split was generated at subject level, so no patient appears
  in more than one split.

See [`echo_hubert_results.md`](echo_hubert_results.md) for the concise results
summary and final artifact list.

The current branch includes the merged PRIMED-AI helper scripts needed for that
last step:

```text
scripts/build_cohort.py
scripts/make_splits.py
scripts/check_ef40_prevalence.py
```

These scripts do not include private data, but they define the expected cohort
construction path.

## Local Inputs

Keep raw and generated data local unless the team explicitly approves sharing or
committing it.

Expected local paths:

```bash
data/raw/echo_jepa/vjepa2.1-vitl-mimic-pt-100/*.parquet
data/raw/hubert_ecg/<hubert_embeddings>.csv
data/raw/cohort/<paired_lvef_split>.csv            # optional if already built
data/raw/mimic_ecg/record_list.csv                 # needed to derive ECG timestamps locally
data/raw/mimic_echo/structured_measurement.csv     # needed to derive LVEF locally
```

Install the project and Parquet/Hugging Face dependencies:

```bash
pip install -e .
pip install huggingface_hub datasets pyarrow pandas polars scikit-learn
```

Download only the first EchoJEPA variant initially:

```bash
hf download MITCriticalData/mimic-iv-echo-jepa-embeddings \
  --repo-type dataset \
  --include "vjepa2.1-vitl-mimic-pt-100/*.parquet" \
  --local-dir data/raw/echo_jepa
```

If the default transfer stalls, disable the Xet backend and fetch the shards
sequentially:

```bash
for i in $(seq -f "%05g" 0 9); do
  HF_HUB_DISABLE_XET=1 hf download MITCriticalData/mimic-iv-echo-jepa-embeddings \
    "vjepa2.1-vitl-mimic-pt-100/train-${i}-of-00010.parquet" \
    --repo-type dataset \
    --local-dir data/raw/echo_jepa \
    --max-workers 1
done
```

## Build Workflow

### Build The Labeled Cohort

If BigQuery/PhysioNet access is available, build the paired Echo + ECG + LVEF
cohort directly from the official MIMIC tables:

```bash
gcloud auth application-default login
export GCP_PROJECT_ID=<your-billing-gcp-project>

python scripts/build_cohort.py \
  --project "$GCP_PROJECT_ID" \
  --pair-by window \
  --window-hours 48 \
  --output-dir data/raw/cohort \
  --logs-dir logs \
  --format parquet
```

`scripts/build_cohort.py` handles:

```text
Echo↔ECG temporal pairing by subject_id and nearest ECG timestamp
LVEF extraction from MIMIC-IV-Echo structured_measurement
ef_le_40 = lvef <= 40
demographics: sex, age, age_band, race
cohort funnel/exclusion logs
LVEF source summary
demographics coverage report
```

Then add a subject-level train/val/test split:

```bash
python scripts/make_splits.py \
  --input data/raw/cohort/paired.parquet \
  --output data/raw/cohort/paired_with_splits.parquet \
  --subject-splits data/raw/cohort/subject_splits.csv \
  --manifest data/raw/cohort/splits.json \
  --seed 42 \
  --train-frac 0.70 \
  --val-frac 0.10 \
  --test-frac 0.20
```

Check EF<=40 prevalence before training/reporting AUROC:

```bash
python scripts/check_ef40_prevalence.py \
  --input data/raw/cohort/paired_with_splits.parquet \
  --output-csv logs/ef40_prevalence.csv \
  --output-json logs/ef40_prevalence.json
```

For the local completed run, `../outputs_build_cohort/paired.csv` was used as
the cohort source:

```bash
python scripts/make_splits.py \
  --input ../outputs_build_cohort/paired.csv \
  --output data/raw/cohort/paired_with_splits.parquet \
  --subject-splits data/raw/cohort/subject_splits.csv \
  --manifest logs/splits.json \
  --seed 42 \
  --train-frac 0.70 \
  --val-frac 0.10 \
  --test-frac 0.20
```

After that, `data/raw/cohort/paired_with_splits.parquet` is the cohort input for
the EchoJEPA + HuBERT manifest join.

Build EchoJEPA study-level embeddings:

```bash
python scripts/build_echo_hubert_manifest.py build-echo \
  --input data/raw/echo_jepa/vjepa2.1-vitl-mimic-pt-100 \
  --output data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet
```

For the current HuBERT Drive CSV, IDs are stored in `filename` paths like
`files/p1000/p10000032/s40689238/40689238`, and embeddings are spread across
`ve0001` to `ve0768`. The converter parses those paths automatically:

```bash
python scripts/build_echo_hubert_manifest.py convert-ecg \
  --csv data/raw/hubert_ecg/ecg_hubert_embedding.csv \
  --output data/interim/hubert_ecg_embeddings.parquet \
  --filename-col filename \
  --ecg-feature-prefix ve
```

After a true paired cohort file exists, build the final joined manifest:

```bash
python scripts/build_echo_hubert_manifest.py join \
  --cohort data/raw/cohort/paired_with_splits.parquet \
  --echo data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet \
  --ecg data/interim/hubert_ecg_embeddings.parquet \
  --manifest data/processed/echo_hubert_manifest.parquet \
  --metadata-csv data/processed/echo_hubert_manifest_metadata.csv \
  --summary-json logs/echo_hubert_join_summary.json \
  --ecg-study-col ecg_record_id
```

Outputs:

```text
data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet
data/interim/hubert_ecg_embeddings.parquet
data/processed/echo_hubert_manifest.parquet
data/processed/echo_hubert_manifest_metadata.csv
logs/echo_hubert_join_summary.json
```

Check whether the local embedding artifacts and cohort file are ready:

```bash
python scripts/check_echo_hubert_readiness.py \
  --cohort data/raw/cohort/<paired_lvef_split>.csv
```

Current completed embedding and partial-overlap artifacts:

```text
data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet
data/interim/hubert_ecg_embeddings.parquet
data/processed/echo_hubert_subject_overlap_metadata.csv
logs/echo_study_embedding_summary.json
logs/hubert_ecg_conversion_summary.json
logs/echo_hubert_subject_overlap_summary.json
```

If schemas use different names, pass explicit columns:

```bash
python scripts/build_echo_hubert_manifest.py join \
  --cohort data/raw/cohort/<paired_lvef_split>.csv \
  --subject-col subject_id \
  --echo-study-col echo_study_id \
  --ecg-study-col ecg_study_id \
  --lvef-col lvef \
  --split-col split
```

## Load The Dataset

```python
from primed_ai.data import EchoHubertDataset

dataset = EchoHubertDataset(
    manifest_path="data/processed/echo_hubert_manifest.parquet",
    split="train",
)

item = dataset[0]
echo = item["echo_embedding"]
ecg = item["ecg_embedding"]
lvef = item["lvef"]
```

## Validation Summary

`logs/echo_hubert_join_summary.json` includes matched counts, missing labels,
missing embeddings, duplicate `(subject_id, echo_study_id, ecg_study_id)` rows,
train/val/test counts, embedding dimensions, join keys, and subject leakage
across splits. The join step raises an error if any `subject_id` appears in
more than one split.

`logs/echo_hubert_subject_overlap_summary.json` is not the final cohort summary.
It documents the maximum overlap derivable from embedding IDs alone and explains
why timestamp/label metadata is still needed.

## Suggested Team Update

```text
I used the columns available in the files and got as far as the files allow.
EchoJEPA gives subject_id, echo study_id, echo timestamps, and embeddings;
HuBERT gives filename plus embeddings, from which I parse subject_id and
ecg_study_id.

I prepared both embedding tables and found 4,103 shared subjects, 6,617 echo
studies, and 57,459 ECG studies. There are 0 exact echo_study_id/ecg_study_id
matches, so study-level pairing requires ECG timestamps from MIMIC-IV-ECG
record_list.csv or BigQuery.

The HF/Drive files do not contain LVEF labels, ECG timestamps, demographics, or
modeling splits. I can generate the final synchronized manifest as soon as I
have the source metadata tables or PhysioNet/BigQuery access.
```
