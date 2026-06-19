"""Build the paired Echo+ECG LVEF cohort from MIMIC-IV on BigQuery.

All filtering happens server-side in BigQuery so we never pull the full ~525K
echo corpus to the client. The script downloads only the final paired cohort
plus a small inclusion/exclusion funnel.

Inclusion criteria (applied in order; counts logged at each step):
  1. Patient has an echocardiogram study            (mimiciv_echo.echo_study_list)
  2. A structured LVEF measurement was recorded      (mimiciv_echo.structured_measurement)
  3. A matched ECG study exists within +/- WINDOW_HOURS of the echo
                                                     (mimiciv_ecg.record_list)
  4. The echo falls inside a hospital admission so   (mimiciv_3_1_hosp.admissions)
     demographics (race etc.) are available

Pairing rule (--pair-by): for each echo study, take the ECG with the smallest
absolute time delta, restricted either to a time window ('window', default) or
to the same hospital admission ('admission'). The window can be asymmetric via
--window-before-hours / --window-after-hours (e.g. allow an ECG up to 30 days
before but only 24h after the echo, matching the screen-first deployment story);
--window-hours sets a symmetric default for both sides. Admission pairing is
encounter-aligned and robust to the documented ECG machine-clock skew. Ties are
broken deterministically so reruns from the same inputs produce the same cohort.

Sources:
- MIMIC-IV          physionet-data.mimiciv_3_1_hosp.{admissions,patients}
- MIMIC-IV-ECG      physionet-data.mimiciv_ecg.record_list
- MIMIC-IV-Echo     physionet-data.mimiciv_echo.{echo_study_list,structured_measurement}

Authentication (per team member, no shared secrets):
  gcloud auth application-default login
  export GCP_PROJECT_ID=<your-billing-gcp-project>
Each member must be PhysioNet-credentialed, have signed the DUAs, and have
requested BigQuery access on the MIMIC-IV, MIMIC-IV-ECG and MIMIC-IV-Echo
PhysioNet project pages.

Outputs (written to cohort/):
- paired.parquet (or paired.csv)  - one row per paired echo study
- cohort_funnel.json / .csv       - inclusion/exclusion counts per step
- cohort_flowchart.md             - Mermaid CONSORT-style flow diagram

Note: the lead's brief allows a 24-48h window; the cohort definition here uses a
symmetric +/-24h window by default. Override with --window-hours.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import google.auth
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("build_cohort")

# --- Fully-qualified BigQuery tables --------------------------------------
ECHO_STUDY_LIST = "physionet-data.mimiciv_echo.echo_study_list"
ECHO_RECORD_LIST = "physionet-data.mimiciv_echo.echo_record_list"
ECHO_STRUCTURED = "physionet-data.mimiciv_echo.structured_measurement"
ECG_RECORD_LIST = "physionet-data.mimiciv_ecg.record_list"
HOSP_ADMISSIONS = "physionet-data.mimiciv_3_1_hosp.admissions"
HOSP_PATIENTS = "physionet-data.mimiciv_3_1_hosp.patients"

# LVEF is not stored in a single column: an echo-lab system transition (and the
# stress-echo protocol's separate rest/stress phases) means it appears under
# several measurement identifiers. We take the first available in priority order
# and record which one was used (`lvef_measurement`) so fallbacks are auditable.
#
# Ordering rationale:
#   - Resting point estimates first (lvef -> biplane -> rest_lvef -> rest_biplane
#     -> lvef_3d). These are mostly disjoint across studies (two lab systems), so
#     ordering among them rarely changes the chosen value, only inclusion.
#   - Range upper-bounds (`*_upper`) are last: they are the high end of a reported
#     range, not a point EF, so only used when nothing better exists.
#   - `stress_lvef` is intentionally EXCLUDED: it is EF measured during stress, a
#     different physiological state than the resting EF we are labelling (and it
#     recovers ~1 study in practice). Add it here only if you explicitly want it.
#
# Empirically (this dataset): for the DICOM-linked imaging cohort the rest_*/
# stress_* variants never occur (it is 100% resting TTE), so the extra fallbacks
# change nothing there; they matter only for a measurement-only / ECG-only arm.
LVEF_PRIORITY = [
    "lvef",
    "biplane_lvef",
    "rest_lvef",
    "rest_biplane_lvef",
    "lvef_3d",
    "lvef_upper",
    "rest_lvef_upper",
]
LVEF_MEASUREMENTS = tuple(LVEF_PRIORITY)
LVEF_IN_LIST = ", ".join(f"'{m}'" for m in LVEF_PRIORITY)
LVEF_PRIORITY_SQL = (
    "CASE measurement "
    + " ".join(f"WHEN '{m}' THEN {i}" for i, m in enumerate(LVEF_PRIORITY))
    + f" ELSE {len(LVEF_PRIORITY)} END"
)

def funnel_stages(pair_by: str, require_admission: bool = True
                  ) -> list[tuple[str, str]]:
    """(cte_name, label) per inclusion step, in the order they are applied.

    The last two steps swap order between modes: window pairing matches an ECG by
    time then attaches an admission for demographics; admission pairing requires
    the echo to be inside an admission first, then matches an ECG in that same
    admission.
    """
    common = [
        ("echo_base", "1. Echo studies (all)"),
        ("echo_with_meas", "2. Linked to a structured measurement"),
        ("echo_with_lvef", "3. Patient has an LVEF reading"),
        ("echo_with_ecg_subject", "4. Patient also has an ECG (subject overlap)"),
    ]
    if pair_by == "admission":
        return common + [
            ("echo_adm", "5. Echo within a hospital admission"),
            ("ecg_pairs", "6. ECG matched in same admission"),
        ]
    adm_label = ("6. Within a hospital admission (demographics)"
                 if require_admission
                 else "6. Has admission demographics (subset, not a filter)")
    return common + [
        ("ecg_pairs", "5. ECG within window of the echo"),
        ("adm_match", adm_label),
    ]


def get_client(project: str) -> bigquery.Client:
    """BigQuery client billed to the member's own project (ADC, no secrets)."""
    if not project:
        raise RuntimeError(
            "Set GCP_PROJECT_ID to your billing GCP project "
            "(export GCP_PROJECT_ID=<project> and run "
            "`gcloud auth application-default login`)."
        )
    credentials, _ = google.auth.default()
    # Attach a quota project so ADC user-credentials don't warn / throttle.
    if hasattr(credentials, "with_quota_project"):
        credentials = credentials.with_quota_project(project)
    return bigquery.Client(project=project, credentials=credentials)


def build_cte_sql(window_before_hours: float, window_after_hours: float,
                  lvef_min: float, lvef_max: float,
                  require_admission: bool, pair_by: str = "window") -> str:
    """Return the shared WITH ... clause used by both the funnel and the fetch.

    Every stage CTE is reduced to exactly one row per echo study so that
    COUNT(*) at any stage is an unambiguous study count and the final cohort
    cannot contain duplicate (subject_id, echo_study_id) rows.

    pair_by:
      "window"     - match the nearest ECG whose time relative to the echo is in
                     [-window_before_hours, +window_after_hours], then attach an
                     admission for demographics. `require_admission` controls
                     whether the admission is mandatory (set it False to keep
                     outpatient pairs). The window may be asymmetric.
      "admission"  - require the echo to fall inside a hospital admission first,
                     then match the nearest ECG occurring in that *same* admission.
                     This is encounter-aligned and robust to ECG clock skew, but
                     ignores the window (delta_hours can span the whole stay).
                     An admission is always mandatory in this mode.
    """
    head = f"""
WITH echo_base AS (
  -- One row per echo study (defensive dedup on study_id).
  SELECT subject_id,
         study_id        AS echo_study_id,
         study_datetime  AS echo_datetime,
         measurement_id
  FROM `{ECHO_STUDY_LIST}`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY study_id ORDER BY measurement_id) = 1
),

echo_with_meas AS (
  SELECT * FROM echo_base WHERE measurement_id IS NOT NULL
),

lvef AS (
  -- One LVEF value per measurement_id, chosen by the priority above.
  SELECT measurement_id,
         SAFE_CAST(TRIM(REPLACE(result, '%', '')) AS FLOAT64) AS lvef_value,
         measurement AS lvef_measurement
  FROM `{ECHO_STRUCTURED}`
  WHERE measurement IN ({LVEF_IN_LIST})
    AND SAFE_CAST(TRIM(REPLACE(result, '%', '')) AS FLOAT64)
        BETWEEN {lvef_min} AND {lvef_max}
  QUALIFY ROW_NUMBER() OVER (
            PARTITION BY measurement_id
            ORDER BY {LVEF_PRIORITY_SQL}, measurement_datetime, result
          ) = 1
),

echo_with_lvef AS (
  SELECT m.subject_id, m.echo_study_id, m.echo_datetime, m.measurement_id,
         l.lvef_value, l.lvef_measurement
  FROM echo_with_meas m
  JOIN lvef l USING (measurement_id)
),

dicom_counts AS (
  -- Number of echo DICOM files per study (used to tally imaging volume at each
  -- funnel step and to annotate the final cohort).
  SELECT study_id AS echo_study_id, COUNT(*) AS n_dicom_files
  FROM `{ECHO_RECORD_LIST}`
  GROUP BY study_id
),

ecg_subjects AS (
  SELECT DISTINCT subject_id FROM `{ECG_RECORD_LIST}`
),

echo_with_ecg_subject AS (
  -- LVEF echo studies whose patient has >=1 ECG anywhere (subject overlap).
  -- This is the hard ceiling on the paired cohort: an echo can only ever be
  -- paired if its patient appears in MIMIC-IV-ECG at all (~4,002 LVEF patients).
  SELECT el.*
  FROM echo_with_lvef el
  WHERE el.subject_id IN (SELECT subject_id FROM ecg_subjects)
),
"""

    if pair_by == "admission":
        tail = f"""
echo_adm AS (
  -- Echo study that falls inside an admission (earliest if several overlap).
  SELECT el.subject_id, el.echo_study_id, el.echo_datetime,
         el.lvef_value, el.lvef_measurement,
         a.hadm_id, a.admittime, a.dischtime,
         a.race, a.insurance, a.marital_status, a.language
  FROM echo_with_ecg_subject el
  JOIN `{HOSP_ADMISSIONS}` a
    ON a.subject_id = el.subject_id
   AND el.echo_datetime BETWEEN a.admittime AND a.dischtime
  QUALIFY ROW_NUMBER() OVER (
            PARTITION BY el.echo_study_id ORDER BY a.admittime, a.hadm_id
          ) = 1
),

ecg_pairs AS (
  -- Nearest ECG occurring within the SAME admission window. Deterministic ties.
  SELECT ea.subject_id, ea.echo_study_id, ea.echo_datetime,
         ea.lvef_value, ea.lvef_measurement,
         ea.hadm_id, ea.admittime, ea.dischtime,
         ea.race, ea.insurance, ea.marital_status, ea.language,
         r.study_id  AS ecg_record_id,
         r.ecg_time,
         r.file_name AS ecg_file_name,
         r.path      AS ecg_path,
         DATETIME_DIFF(r.ecg_time, ea.echo_datetime, SECOND) / 3600.0 AS delta_hours
  FROM echo_adm ea
  JOIN `{ECG_RECORD_LIST}` r
    ON r.subject_id = ea.subject_id
   AND r.ecg_time BETWEEN ea.admittime AND ea.dischtime
  QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ea.echo_study_id
            ORDER BY ABS(DATETIME_DIFF(r.ecg_time, ea.echo_datetime, SECOND)),
                     r.ecg_time, r.study_id
          ) = 1
),

final AS (
  SELECT
    p.subject_id,
    p.echo_study_id,
    p.ecg_record_id,
    p.echo_datetime,
    p.ecg_time AS ecg_datetime,
    ROUND(p.delta_hours, 4) AS delta_hours,
    p.lvef_value,
    p.lvef_measurement,
    (p.lvef_value <= 40) AS ef_le_40,
    p.ecg_file_name,
    p.ecg_path,
    COALESCE(dc.n_dicom_files, 0) AS n_dicom_files,
    p.hadm_id, p.admittime, p.dischtime,
    pt.gender, pt.anchor_age,
    p.race, p.insurance, p.marital_status, p.language
  FROM ecg_pairs p
  LEFT JOIN `{HOSP_PATIENTS}` pt ON pt.subject_id = p.subject_id
  LEFT JOIN dicom_counts dc ON dc.echo_study_id = p.echo_study_id
)
"""
        return head + tail

    # pair_by == "window"
    before_seconds = int(round(window_before_hours * 3600))
    after_seconds = int(round(window_after_hours * 3600))
    adm_join = "JOIN" if require_admission else "LEFT JOIN"
    tail = f"""
ecg_pairs AS (
  -- Nearest ECG in [-before, +after] of the echo. Negative delta = ECG precedes
  -- the echo. Nearest by absolute gap; deterministic ties.
  SELECT el.subject_id, el.echo_study_id, el.echo_datetime, el.measurement_id,
         el.lvef_value, el.lvef_measurement,
         r.study_id  AS ecg_record_id,
         r.ecg_time,
         r.file_name AS ecg_file_name,
         r.path      AS ecg_path,
         DATETIME_DIFF(r.ecg_time, el.echo_datetime, SECOND) / 3600.0 AS delta_hours
  FROM echo_with_ecg_subject el
  JOIN `{ECG_RECORD_LIST}` r
    ON r.subject_id = el.subject_id
   AND DATETIME_DIFF(r.ecg_time, el.echo_datetime, SECOND)
       BETWEEN -{before_seconds} AND {after_seconds}
  QUALIFY ROW_NUMBER() OVER (
            PARTITION BY el.echo_study_id
            ORDER BY ABS(DATETIME_DIFF(r.ecg_time, el.echo_datetime, SECOND)),
                     r.ecg_time, r.study_id
          ) = 1
),

adm_match AS (
  -- Admission whose stay window contains the echo (earliest if several).
  SELECT p.subject_id, p.echo_study_id,
         a.hadm_id, a.admittime, a.dischtime,
         a.race, a.insurance, a.marital_status, a.language
  FROM ecg_pairs p
  JOIN `{HOSP_ADMISSIONS}` a
    ON a.subject_id = p.subject_id
   AND p.echo_datetime BETWEEN a.admittime AND a.dischtime
  QUALIFY ROW_NUMBER() OVER (
            PARTITION BY p.echo_study_id ORDER BY a.admittime, a.hadm_id
          ) = 1
),

final AS (
  SELECT
    p.subject_id,
    p.echo_study_id,
    p.ecg_record_id,
    p.echo_datetime,
    p.ecg_time AS ecg_datetime,
    ROUND(p.delta_hours, 4) AS delta_hours,
    p.lvef_value,
    p.lvef_measurement,
    (p.lvef_value <= 40) AS ef_le_40,
    p.ecg_file_name,
    p.ecg_path,
    COALESCE(dc.n_dicom_files, 0) AS n_dicom_files,
    am.hadm_id, am.admittime, am.dischtime,
    pt.gender, pt.anchor_age,
    am.race, am.insurance, am.marital_status, am.language
  FROM ecg_pairs p
  {adm_join} adm_match am USING (echo_study_id)
  LEFT JOIN `{HOSP_PATIENTS}` pt ON pt.subject_id = p.subject_id
  LEFT JOIN dicom_counts dc ON dc.echo_study_id = p.echo_study_id
)
"""
    return head + tail


def run_funnel(client: bigquery.Client, cte_sql: str, pair_by: str,
               require_admission: bool) -> pd.DataFrame:
    """Count distinct studies and subjects remaining at each inclusion step."""
    parts = [
        f"SELECT '{label}' AS stage, {i} AS step, "
        f"COUNT(*) AS n_studies, "
        f"COUNT(DISTINCT s.subject_id) AS n_subjects, "
        f"COALESCE(SUM(d.n_dicom_files), 0) AS n_dicom_files "
        f"FROM {cte} s LEFT JOIN dicom_counts d USING (echo_study_id)"
        for i, (cte, label) in enumerate(funnel_stages(pair_by, require_admission))
    ]
    sql = cte_sql + "\n UNION ALL \n".join(parts) + "\nORDER BY step"
    df = client.query(sql).to_dataframe()

    df["excluded_studies"] = df["n_studies"].shift(1).fillna(0).astype(int) - df["n_studies"]
    df["excluded_dicom_files"] = (
        df["n_dicom_files"].shift(1).fillna(0).astype(int) - df["n_dicom_files"]
    )
    df.loc[0, ["excluded_studies", "excluded_dicom_files"]] = 0
    return df


def run_fetch(client: bigquery.Client, cte_sql: str) -> pd.DataFrame:
    sql = cte_sql + "SELECT * FROM final ORDER BY subject_id, echo_study_id"
    return client.query(sql).to_dataframe()


def run_lvef_breakdown(client: bigquery.Client, cte_sql: str) -> pd.DataFrame:
    """Count how often each LVEF variant was used at the LVEF-selection stage.

    Reported over `echo_with_lvef` (i.e. all LVEF-valid echo studies, before the
    ECG / admission filters) so fallback frequency is visible independent of the
    downstream pairing. Ordered by the configured priority.
    """
    order = " ".join(
        f"WHEN '{m}' THEN {i}" for i, m in enumerate(LVEF_PRIORITY)
    )
    sql = cte_sql + f"""
SELECT lvef_measurement,
       COUNT(*) AS n_studies,
       ROUND(AVG(lvef_value), 1) AS mean_lvef
FROM echo_with_lvef
GROUP BY lvef_measurement
ORDER BY CASE lvef_measurement {order} ELSE {len(LVEF_PRIORITY)} END
"""
    df = client.query(sql).to_dataframe()
    total = int(df["n_studies"].sum()) or 1
    df["is_fallback"] = df["lvef_measurement"] != LVEF_PRIORITY[0]
    df["pct"] = (df["n_studies"] / total * 100).round(2)
    return df


def write_flowchart(funnel: pd.DataFrame, path: Path, pairing_desc: str) -> None:
    """Write a Mermaid CONSORT-style inclusion/exclusion flow diagram."""
    lines = [
        "# Cohort construction flow",
        "",
        f"Paired Echo+ECG LVEF cohort ({pairing_desc}). "
        "Generated by `scripts/build_cohort.py`.",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    rows = list(funnel.itertuples(index=False))
    for i, r in enumerate(rows):
        lines.append(
            f'    S{i}["{r.stage}<br/>studies = {r.n_studies:,}<br/>'
            f'patients = {r.n_subjects:,}<br/>DICOM files = {r.n_dicom_files:,}"]'
        )
    for i in range(len(rows) - 1):
        excluded = rows[i].n_studies - rows[i + 1].n_studies
        excluded_dcm = rows[i].n_dicom_files - rows[i + 1].n_dicom_files
        lines.append(
            f'    E{i}["excluded: {excluded:,} studies, {excluded_dcm:,} DICOM files"]'
        )
    for i in range(len(rows) - 1):
        lines.append(f"    S{i} --> S{i + 1}")
        lines.append(f"    S{i} -.-> E{i}")
    lines.append("```")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=os.getenv("GCP_PROJECT_ID"),
                    help="Billing GCP project (default: $GCP_PROJECT_ID).")
    ap.add_argument("--pair-by", choices=["window", "admission"], default="window",
                    help="ECG pairing strategy: nearest within a time window "
                         "('window', default) or nearest ECG in the same hospital "
                         "admission ('admission', ignores the window).")
    ap.add_argument("--window-hours", type=float, default=24.0,
                    help="Symmetric echo<->ECG gap in hours; sets the default for "
                         "both sides (window mode only; default: 24).")
    ap.add_argument("--window-before-hours", type=float, default=None,
                    help="Max hours an ECG may precede the echo (overrides "
                         "--window-hours for the 'before' side, e.g. 720 = 30 days).")
    ap.add_argument("--window-after-hours", type=float, default=None,
                    help="Max hours an ECG may follow the echo (overrides "
                         "--window-hours for the 'after' side).")
    ap.add_argument("--lvef-min", type=float, default=0.0,
                    help="Drop LVEF results below this value (default: 0).")
    ap.add_argument("--lvef-max", type=float, default=100.0,
                    help="Drop LVEF results above this value (default: 100).")
    ap.add_argument("--no-require-admission", dest="require_admission",
                    action="store_false",
                    help="Keep pairs without a matching admission (race may be null).")
    ap.add_argument("--output-dir", default=str(repo_root / "cohort"),
                    help="Directory for cohort outputs (default: <repo>/cohort).")
    ap.add_argument("--format", choices=["parquet", "csv"], default="csv",
                    help="Paired cohort output format (default: csv).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the funnel only; do not download/write the cohort.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    before_h = (args.window_before_hours if args.window_before_hours is not None
                else args.window_hours)
    after_h = (args.window_after_hours if args.window_after_hours is not None
               else args.window_hours)
    if before_h < 0 or after_h < 0:
        ap.error("window-before-hours / window-after-hours must be non-negative.")

    client = get_client(args.project)
    if args.pair_by == "admission":
        pairing_desc = "admission (same-stay)"
    elif before_h == after_h:
        pairing_desc = f"window +/-{before_h:.3g}h"
    else:
        pairing_desc = f"window -{before_h:.3g}h..+{after_h:.3g}h (before..after)"
    log.info("BigQuery project: %s | pair_by: %s | LVEF in [%g, %g] | "
             "require_admission=%s",
             args.project, pairing_desc, args.lvef_min, args.lvef_max,
             args.require_admission or args.pair_by == "admission")

    cte_sql = build_cte_sql(before_h, after_h, args.lvef_min, args.lvef_max,
                            args.require_admission, args.pair_by)

    log.info("Computing inclusion/exclusion funnel ...")
    funnel = run_funnel(client, cte_sql, args.pair_by, args.require_admission)
    for r in funnel.itertuples(index=False):
        log.info("  %-46s studies=%-7d patients=%-7d dicoms=%-9d (excluded %d studies)",
                 r.stage, r.n_studies, r.n_subjects, r.n_dicom_files,
                 r.excluded_studies)

    funnel.to_json(out_dir / "cohort_funnel.json", orient="records", indent=2)
    funnel.to_csv(out_dir / "cohort_funnel.csv", index=False)
    write_flowchart(funnel, out_dir / "cohort_flowchart.md", pairing_desc)
    log.info("Wrote funnel + flowchart to %s", out_dir)

    log.info("LVEF source breakdown (at LVEF-selection stage):")
    lvef_src = run_lvef_breakdown(client, cte_sql)
    for r in lvef_src.itertuples(index=False):
        log.info("  %-18s studies=%-7d (%.2f%%) mean=%-5s %s",
                 r.lvef_measurement, r.n_studies, r.pct, r.mean_lvef,
                 "[fallback]" if r.is_fallback else "[primary]")
    lvef_src.to_json(out_dir / "cohort_lvef_sources.json", orient="records", indent=2)
    lvef_src.to_csv(out_dir / "cohort_lvef_sources.csv", index=False)

    if args.dry_run:
        log.info("--dry-run set; skipping cohort download.")
        return

    log.info("Fetching paired cohort ...")
    cohort = run_fetch(client, cte_sql)

    dupes = cohort.duplicated(subset=["subject_id", "echo_study_id"]).sum()
    if dupes:
        raise RuntimeError(f"Found {dupes} duplicate (subject_id, echo_study_id) rows.")
    log.info("Paired cohort: %d rows | %d unique patients | dupes=0",
             len(cohort), cohort["subject_id"].nunique())

    if args.format == "parquet":
        out_path = out_dir / "paired.parquet"
        cohort.to_parquet(out_path, index=False)
    else:
        out_path = out_dir / "paired.csv"
        cohort.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
