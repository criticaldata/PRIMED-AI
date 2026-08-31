"""Extract Valvular Disease Labels (AS, MR, TR) from MIMIC-IV BigQuery.

This script parses structured diagnoses (ICD-9 / ICD-10) and detailed echocardiographic
findings embedded within MIMIC-IV Clinical Notes (physionet-data.mimiciv_note.discharge)
to construct gold-standard labels for Task B:
  1. Aortic Stenosis (AS): None (0), Mild (1), Moderate (2), Severe (3)
  2. Mitral Regurgitation (MR): None (0), Mild (1), Moderate (2), Severe (3)
  3. Tricuspid Regurgitation (TR): None (0), Mild (1), Moderate (2), Severe (3)

Also computes binary clinical gates:
  - as_moderate_or_severe: AS grade >= 2 or ICD diagnosis positive
  - mr_moderate_or_severe: MR grade >= 2 or ICD diagnosis positive
  - tr_moderate_or_severe: TR grade >= 2 or ICD diagnosis positive

Pairs each finding with the nearest 12-lead ECG in MIMIC-IV-ECG.
Outputs:
  - cohort/valvular_cohort.parquet
  - cohort/valvular_cohort.csv
  - logs/valvular_extraction_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import google.auth
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("extract_valvular_labels")

import medspacy
from medspacy.context import ConTextRule
from medspacy.ner import TargetRule
from loguru import logger as loguru_logger

# Silence internal verbose logs
loguru_logger.disable("PyRuSH")
logging.getLogger("PyRuSH").setLevel(logging.WARNING)

# Build and cache MedSpacy Clinical NER + ConText NLP pipeline (fast & silent)
def build_clinical_nlp_pipeline():
    nlp = medspacy.load(medspacy_enable=["medspacy_target_matcher", "medspacy_context"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", before="medspacy_target_matcher")
    
    # 1. Target Rules for Valvular Pathology
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_rules = [
        # Aortic Stenosis & Sclerosis
        TargetRule("aortic stenosis", "AORTIC_STENOSIS"),
        TargetRule("aortic valve stenosis", "AORTIC_STENOSIS"),
        TargetRule("valvular aortic stenosis", "AORTIC_STENOSIS"),
        TargetRule("critical aortic stenosis", "AORTIC_STENOSIS"),
        TargetRule("calcific aortic stenosis", "AORTIC_STENOSIS"),
        TargetRule("aortic sclerosis", "AORTIC_SCLEROSIS"),
        TargetRule("aortic valve sclerosis", "AORTIC_SCLEROSIS"),
        
        # Mitral Regurgitation
        TargetRule("mitral regurgitation", "MITRAL_REGURGITATION"),
        TargetRule("mitral valve regurgitation", "MITRAL_REGURGITATION"),
        TargetRule("mitral insufficiency", "MITRAL_REGURGITATION"),
        TargetRule("mitral valve insufficiency", "MITRAL_REGURGITATION"),
        
        # Tricuspid Regurgitation
        TargetRule("tricuspid regurgitation", "TRICUSPID_REGURGITATION"),
        TargetRule("tricuspid valve regurgitation", "TRICUSPID_REGURGITATION"),
        TargetRule("tricuspid insufficiency", "TRICUSPID_REGURGITATION"),
        TargetRule("tricuspid valve insufficiency", "TRICUSPID_REGURGITATION"),
    ]
    target_matcher.add(target_rules)
    
    # 2. ConText Severity Modifiers (Bidirectional matching)
    context = nlp.get_pipe("medspacy_context")
    context_rules = [
        # Severe
        ConTextRule("severe", "SEVERITY_SEVERE", direction="BIDIRECTIONAL"),
        ConTextRule("critical", "SEVERITY_SEVERE", direction="BIDIRECTIONAL"),
        ConTextRule("moderate to severe", "SEVERITY_SEVERE", direction="BIDIRECTIONAL"),
        ConTextRule("severe to critical", "SEVERITY_SEVERE", direction="BIDIRECTIONAL"),
        
        # Moderate
        ConTextRule("moderate", "SEVERITY_MODERATE", direction="BIDIRECTIONAL"),
        ConTextRule("mild to moderate", "SEVERITY_MODERATE", direction="BIDIRECTIONAL"),
        ConTextRule("moderate degree of", "SEVERITY_MODERATE", direction="BIDIRECTIONAL"),
        
        # Mild
        ConTextRule("mild", "SEVERITY_MILD", direction="BIDIRECTIONAL"),
        ConTextRule("trace to mild", "SEVERITY_MILD", direction="BIDIRECTIONAL"),
        ConTextRule("mild degree of", "SEVERITY_MILD", direction="BIDIRECTIONAL"),
        ConTextRule("minimal", "SEVERITY_MILD", direction="BIDIRECTIONAL"),
        
        # Trace / Trivial / Physiologic -> Grade 0
        ConTextRule("trivial", "SEVERITY_NONE", direction="BIDIRECTIONAL"),
        ConTextRule("trace", "SEVERITY_NONE", direction="BIDIRECTIONAL"),
        ConTextRule("physiologic", "SEVERITY_NONE", direction="BIDIRECTIONAL"),
    ]
    context.add(context_rules)
    return nlp

CLINICAL_NLP = build_clinical_nlp_pipeline()

# Quantitative Doppler and LVEF patterns
RE_AVA_SEV = re.compile(r"aortic\s+valve\s+area\s*(?:<|<=|is|of|\:)?\s*0?\.[0-9]\s*cm\^?2", re.IGNORECASE)
RE_GRAD_SEV = re.compile(r"(?:mean\s+(?:aortic\s+)?gradient|gradient)\s*(?:>|>=|is|of|\:)?\s*(?:4[0-9]|[5-9][0-9]|1[0-9]{2})\s*mm\s*hg", re.IGNORECASE)
RE_LVEF = re.compile(
    r"(?:lvef|ejection fraction|ef)\s*(?:is|of|was|\:|\=)?\s*(?:approximately\s*|approx\s*)?([1-9][0-9])\s*\%|"
    r"ejection fraction\s*(?:is|estimated at|of)?\s*(?:>|>=)\s*55\%|"
    r"lvef\s*(?:>|>=)\s*55\%",
    re.IGNORECASE,
)

VALVULAR_KEYWORDS = ("valve", "stenosis", "regurgitation", "echo", "aortic", "mitral", "tricuspid", "lvef")


def parse_echo_text_with_ner(raw_text: str) -> dict:
    """Extract valvular severity using MedSpacy Clinical NER and ConText assertion on relevant sections."""
    # Filter only lines/paragraphs containing relevant keywords for 50x faster speed
    lines = [line.strip() for line in raw_text.split("\n") if any(k in line.lower() for k in VALVULAR_KEYWORDS)]
    if not lines:
        return {
            "as_grade": None, "as_evidence": None,
            "mr_grade": None, "mr_evidence": None,
            "tr_grade": None, "tr_evidence": None,
            "lvef_extracted": None,
        }
    
    text = " ".join(lines)
    doc = CLINICAL_NLP(text)

    res = {
        "as_grade": None,
        "as_evidence": None,
        "mr_grade": None,
        "mr_evidence": None,
        "tr_grade": None,
        "tr_evidence": None,
        "lvef_extracted": None,
    }

    for ent in doc.ents:
        mod_cats = [m.category for m in ent._.modifiers]
        is_negated = ent._.is_negated
        
        # 1. Skip non-patient and non-certain assertions
        if any(cat in ("FAMILY", "POSSIBLE_EXISTENCE", "HYPOTHETICAL") for cat in mod_cats):
            continue
        
        # 2. Determine grade from NER ConText modifiers & Negation
        if is_negated:
            grade = 0
        elif "SEVERITY_SEVERE" in mod_cats:
            grade = 3
        elif "SEVERITY_MODERATE" in mod_cats:
            grade = 2
        elif "SEVERITY_MILD" in mod_cats:
            grade = 1
        elif "SEVERITY_NONE" in mod_cats:
            grade = 0
        else:
            # Positive mention without explicit modifier in echo section -> Grade 1 (mild)
            grade = 1

        evidence = f"{ent.text} (mods: {mod_cats}, neg: {is_negated})"

        if ent.label_ in ("AORTIC_STENOSIS", "AORTIC_SCLEROSIS"):
            if res["as_grade"] is None or grade > res["as_grade"]:
                res["as_grade"] = grade
                res["as_evidence"] = evidence
        elif ent.label_ == "MITRAL_REGURGITATION":
            if res["mr_grade"] is None or grade > res["mr_grade"]:
                res["mr_grade"] = grade
                res["mr_evidence"] = evidence
        elif ent.label_ == "TRICUSPID_REGURGITATION":
            if res["tr_grade"] is None or grade > res["tr_grade"]:
                res["tr_grade"] = grade
                res["tr_evidence"] = evidence

    # Check quantitative Doppler overrides for AS
    if RE_AVA_SEV.search(text) or RE_GRAD_SEV.search(text):
        res["as_grade"] = 3
        res["as_evidence"] = "Quantitative Doppler (AVA <= 1.0 cm2 or Mean Grad >= 40 mmHg)"

    # LVEF extraction
    m_lvef = RE_LVEF.search(text)
    if m_lvef:
        val = m_lvef.group(1)
        if val:
            res["lvef_extracted"] = float(val)
        elif ">" in m_lvef.group(0):
            res["lvef_extracted"] = 60.0

    return res


def build_bigquery_client(project: str) -> bigquery.Client:
    credentials, _ = google.auth.default()
    if hasattr(credentials, "with_quota_project"):
        credentials = credentials.with_quota_project(project)
    return bigquery.Client(project=project, credentials=credentials)


def fetch_discharge_notes_with_ecg(client: bigquery.Client, limit: int | None = None) -> pd.DataFrame:
    """Fetch discharge summaries matched with nearest ECG record within admission."""
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    WITH ecg_per_adm AS (
      -- Nearest ECG to admission charttime
      SELECT 
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        pt.gender AS sex,
        pt.anchor_age AS age,
        a.race,
        r.study_id AS ecg_record_id,
        r.ecg_time,
        r.file_name AS ecg_file_name,
        r.path AS ecg_path,
        DATETIME_DIFF(r.ecg_time, a.admittime, SECOND) / 3600.0 AS ecg_hours_from_admit
      FROM `physionet-data.mimiciv_3_1_hosp.admissions` a
      JOIN `physionet-data.mimiciv_3_1_hosp.patients` pt ON a.subject_id = pt.subject_id
      JOIN `physionet-data.mimiciv_ecg.record_list` r 
        ON a.subject_id = r.subject_id
       AND r.ecg_time BETWEEN a.admittime AND a.dischtime
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY a.hadm_id 
        ORDER BY ABS(DATETIME_DIFF(r.ecg_time, a.admittime, SECOND)), r.ecg_time
      ) = 1
    ),
    valvular_icd AS (
      SELECT 
        hadm_id,
        MAX(CASE WHEN icd_code IN ('I350', 'I352', '4241', '3950', '3952') THEN 1 ELSE 0 END) AS icd_as,
        MAX(CASE WHEN icd_code IN ('I340', '4240', '3941', 'I051') THEN 1 ELSE 0 END) AS icd_mr,
        MAX(CASE WHEN icd_code IN ('I361', '4242', '3970', 'I071') THEN 1 ELSE 0 END) AS icd_tr
      FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
      GROUP BY hadm_id
    )
    SELECT 
      n.note_id,
      n.subject_id,
      n.hadm_id,
      e.sex,
      e.age,
      e.race,
      e.ecg_record_id,
      e.ecg_time,
      e.ecg_file_name,
      e.ecg_path,
      COALESCE(v.icd_as, 0) AS icd_as,
      COALESCE(v.icd_mr, 0) AS icd_mr,
      COALESCE(v.icd_tr, 0) AS icd_tr,
      n.text
    FROM `physionet-data.mimiciv_note.discharge` n
    JOIN ecg_per_adm e ON n.hadm_id = e.hadm_id
    LEFT JOIN valvular_icd v ON n.hadm_id = v.hadm_id
    WHERE (LOWER(n.text) LIKE '%aortic valve%' OR LOWER(n.text) LIKE '%mitral valve%' OR LOWER(n.text) LIKE '%tricuspid valve%' OR LOWER(n.text) LIKE '%echocardiogram%')
    {limit_clause}
    """
    log.info("Running BigQuery query for discharge notes with matched ECGs...")
    return client.query(query).to_dataframe()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID", "datascience-373702"))
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for test runs")
    parser.add_argument("--output-dir", default=str(repo_root / "cohort"))
    parser.add_argument("--logs-dir", default=str(repo_root / "logs"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    client = build_bigquery_client(args.project)
    df = fetch_discharge_notes_with_ecg(client, limit=args.limit)
    log.info("Fetched %d candidate discharge notes with matched ECGs.", len(df))

    log.info("Parsing valvular text findings with MedSpacy Clinical NER...")
    extracted_records = []
    for _, row in df.iterrows():
        parsed = parse_echo_text_with_ner(row["text"])
        
        # Consolidate clinical gates (combining NLP grade >= 2 or ICD diagnosis)
        as_grade = parsed["as_grade"]
        mr_grade = parsed["mr_grade"]
        tr_grade = parsed["tr_grade"]
        
        as_mod_sev = bool((as_grade is not None and as_grade >= 2) or row["icd_as"] == 1)
        mr_mod_sev = bool((mr_grade is not None and mr_grade >= 2) or row["icd_mr"] == 1)
        tr_mod_sev = bool((tr_grade is not None and tr_grade >= 2) or row["icd_tr"] == 1)
        
        # Only keep if at least one valve was parsed or ICD diagnosed
        has_any_label = bool((as_grade is not None) or (mr_grade is not None) or (tr_grade is not None) or bool(row["icd_as"]) or bool(row["icd_mr"]) or bool(row["icd_tr"]))
        
        extracted_records.append({
            "subject_id": int(row["subject_id"]),
            "hadm_id": int(row["hadm_id"]),
            "note_id": row["note_id"],
            "ecg_record_id": row["ecg_record_id"],
            "ecg_time": str(row["ecg_time"]),
            "ecg_file_name": row["ecg_file_name"],
            "ecg_path": row["ecg_path"],
            "sex": row["sex"],
            "age": int(row["age"]) if pd.notna(row["age"]) else None,
            "race": row["race"],
            "as_grade": as_grade,
            "as_evidence": parsed["as_evidence"],
            "as_moderate_or_severe": as_mod_sev,
            "icd_as": int(row["icd_as"]),
            "mr_grade": mr_grade,
            "mr_evidence": parsed["mr_evidence"],
            "mr_moderate_or_severe": mr_mod_sev,
            "icd_mr": int(row["icd_mr"]),
            "tr_grade": tr_grade,
            "tr_evidence": parsed["tr_evidence"],
            "tr_moderate_or_severe": tr_mod_sev,
            "icd_tr": int(row["icd_tr"]),
            "lvef_extracted": parsed["lvef_extracted"],
            "has_valvular_label": has_any_label,
        })

    cohort_df = pd.DataFrame(extracted_records)
    labeled_cohort = cohort_df[cohort_df["has_valvular_label"].astype(bool)].copy().reset_index(drop=True)
    
    log.info("Total cohort rows: %d | Labeled valvular rows: %d | Unique subjects: %d",
             len(cohort_df), len(labeled_cohort), labeled_cohort["subject_id"].nunique())

    # Write files
    parquet_path = out_dir / "valvular_cohort.parquet"
    csv_path = out_dir / "valvular_cohort.csv"
    labeled_cohort.to_parquet(parquet_path, index=False)
    labeled_cohort.to_csv(csv_path, index=False)
    log.info("Saved valvular cohort to %s and %s", parquet_path, csv_path)

    # Summary stats
    summary = {
        "n_total_notes_with_ecg": len(cohort_df),
        "n_labeled_valvular_rows": len(labeled_cohort),
        "n_unique_patients": int(labeled_cohort["subject_id"].nunique()),
        "prevalence": {
            "aortic_stenosis_mod_or_sev": {
                "n_positive": int(labeled_cohort["as_moderate_or_severe"].sum()),
                "prevalence": round(float(labeled_cohort["as_moderate_or_severe"].mean()), 4),
            },
            "mitral_regurgitation_mod_or_sev": {
                "n_positive": int(labeled_cohort["mr_moderate_or_severe"].sum()),
                "prevalence": round(float(labeled_cohort["mr_moderate_or_severe"].mean()), 4),
            },
            "tricuspid_regurgitation_mod_or_sev": {
                "n_positive": int(labeled_cohort["tr_moderate_or_severe"].sum()),
                "prevalence": round(float(labeled_cohort["tr_moderate_or_severe"].mean()), 4),
            },
        },
        "grade_distributions": {
            "as_grades": {str(k): int(v) for k, v in labeled_cohort["as_grade"].value_counts(dropna=False).items()},
            "mr_grades": {str(k): int(v) for k, v in labeled_cohort["mr_grade"].value_counts(dropna=False).items()},
            "tr_grades": {str(k): int(v) for k, v in labeled_cohort["tr_grade"].value_counts(dropna=False).items()},
        },
    }

    summary_path = logs_dir / "valvular_extraction_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary JSON to %s", summary_path)

    print("\n" + "=" * 50)
    print("TASK B: VALVULAR EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total labeled cohort: {len(labeled_cohort):,} studies ({labeled_cohort['subject_id'].nunique():,} unique patients)")
    print(f"  - AS (Moderate/Severe): {summary['prevalence']['aortic_stenosis_mod_or_sev']['n_positive']:,} ({summary['prevalence']['aortic_stenosis_mod_or_sev']['prevalence']*100:.1f}%)")
    print(f"  - MR (Moderate/Severe): {summary['prevalence']['mitral_regurgitation_mod_or_sev']['n_positive']:,} ({summary['prevalence']['mitral_regurgitation_mod_or_sev']['prevalence']*100:.1f}%)")
    print(f"  - TR (Moderate/Severe): {summary['prevalence']['tricuspid_regurgitation_mod_or_sev']['n_positive']:,} ({summary['prevalence']['tricuspid_regurgitation_mod_or_sev']['prevalence']*100:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    main()
