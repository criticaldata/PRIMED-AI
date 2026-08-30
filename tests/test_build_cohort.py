"""Unit tests for cohort builder helpers (D01, D02, D05) — no BigQuery."""

import json

import pandas as pd
from build_cohort import (
    AGE_BAND_LABELS,
    LVEF_EXCLUDED_BOUNDS,
    LVEF_PRIORITY,
    add_age_bands,
    build_cte_sql,
    funnel_stages,
    write_cohort_summary,
    write_demographics_coverage,
    write_flowchart,
)
from run_cohort_sensitivity import summarize_sensitivity


def test_funnel_stages_window_mode():
    stages = funnel_stages("window", require_admission=True)
    labels = [s[1] for s in stages]
    assert labels[0].startswith("1.")
    assert any("ECG within window" in x for x in labels)
    assert any("admission" in x.lower() for x in labels)
    # the #75 exclusion is a visible funnel step, not a silent drop
    assert any("range bounds excluded" in x for x in labels)


def test_funnel_stages_admission_mode():
    stages = funnel_stages("admission")
    labels = [s[1] for s in stages]
    assert any("same admission" in x for x in labels)


def test_lvef_priority_excludes_range_upper_bounds():
    assert LVEF_PRIORITY[0] == "lvef"
    assert not any(m.endswith("_upper") for m in LVEF_PRIORITY)
    assert LVEF_EXCLUDED_BOUNDS == ("lvef_upper", "rest_lvef_upper")


def test_cte_sql_keeps_upper_bounds_out_of_the_label_chain():
    kept = ", ".join(f"'{m}'" for m in LVEF_PRIORITY)
    for pair_by in ("window", "admission"):
        sql = build_cte_sql(24, 24, 0, 100, require_admission=True, pair_by=pair_by)
        # chain membership is exactly the kept list, and the cohort path
        # filters on it; the bounds enter only as counted candidates
        assert f"measurement IN ({kept}) AS accepted" in sql
        assert "WHERE accepted" in sql
        for m in LVEF_EXCLUDED_BOUNDS:
            assert f"'{m}'" in sql


def test_add_age_bands():
    df = pd.DataFrame({"age": [25, 45, 91, None]})
    add_age_bands(df)
    assert str(df.loc[0, "age_band"]) == "18-39"
    assert str(df.loc[2, "age_band"]) == "90+"
    assert pd.isna(df.loc[3, "age_band"])
    assert len(AGE_BAND_LABELS) == 6


def test_write_cohort_summary(tmp_path):
    cohort = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "lvef": [55.0, 35.0, 60.0],
            "ef_le_40": [False, True, False],
        }
    )
    path = tmp_path / "summary.json"
    summary = write_cohort_summary(cohort, path, lvef_min=0, lvef_max=100)
    assert summary["n_rows"] == 3
    assert summary["lvef"]["n_missing"] == 0
    assert summary["ef_le_40"]["n_positive"] == 1
    assert summary["drop_rules"]["range_upper_bounds_excluded"] == list(LVEF_EXCLUDED_BOUNDS)
    assert path.is_file()


def test_write_demographics_coverage(tmp_path):
    df = pd.DataFrame(
        {
            "sex": ["M", "F"],
            "age": [50, None],
            "race": ["WHITE", "UNKNOWN"],
        }
    )
    add_age_bands(df)
    path = tmp_path / "coverage.json"
    report = write_demographics_coverage(df, path)
    assert report["fields"]["sex"]["coverage"] == 1.0
    assert report["fields"]["race"]["n_present"] == 1
    assert "gender_curation" in report["limitations"]
    assert json.loads(path.read_text())["n_rows"] == 2


def test_write_flowchart(tmp_path):
    funnel = pd.DataFrame(
        {
            "stage": ["1. Echo", "2. LVEF", "3. Paired"],
            "step": [0, 1, 2],
            "n_studies": [1000, 500, 200],
            "n_subjects": [800, 400, 150],
            "n_dicom_files": [5000, 2500, 1000],
            "excluded_studies": [0, 500, 300],
            "excluded_dicom_files": [0, 2500, 1500],
        }
    )
    path = tmp_path / "flow.md"
    write_flowchart(funnel, path, "window +/-24h")
    text = path.read_text()
    assert "flowchart TD" in text
    assert "studies = 200" in text


def test_summarize_sensitivity_compares_final_rows():
    funnels = {
        "window_24h": pd.DataFrame(
            {
                "stage": ["1. Echo", "2. Paired"],
                "n_studies": [1000, 200],
                "n_subjects": [900, 180],
                "excluded_studies": [0, 800],
            }
        ),
        "window_48h": pd.DataFrame(
            {
                "stage": ["1. Echo", "2. Paired"],
                "n_studies": [1000, 260],
                "n_subjects": [900, 230],
                "excluded_studies": [0, 740],
            }
        ),
    }

    summary = summarize_sensitivity(funnels)

    assert summary.loc[0, "scenario"] == "window_24h"
    assert summary.loc[0, "final_rows"] == 200
    assert summary.loc[1, "gain_vs_first"] == 60
    assert summary.loc[1, "gain_vs_first_pct"] == 30.0
