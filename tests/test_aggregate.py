"""Tests for E05 results aggregation."""
from __future__ import annotations

import json

import pytest

from primed_ai.evaluation.aggregate import (
    aggregate_results,
    to_latex,
    to_markdown,
    write_all,
)


def _seed_results(root):
    (root / "missing_modality.json").write_text(json.dumps({
        "n": {"test": 245},
        "metrics_table": [
            {"condition": "full", "mae": 10.28, "ef40_auroc": 0.766},
            {"condition": "echo_dropped", "mae": 18.57, "ef40_auroc": 0.693},
            {"condition": "ecg_dropped", "mae": 15.13, "ef40_auroc": 0.750},
        ],
    }))
    fdir = root / "fairness"
    fdir.mkdir()
    (fdir / "fairness_metrics.json").write_text(json.dumps({
        "overall": {"n": 245, "mae": 10.28, "ef40_auroc": 0.766, "small_n": False},
        "by": {
            "sex": {
                "F": {"n": 130, "mae": 10.1, "ef40_auroc": 0.77, "small_n": False},
                "M": {"n": 115, "mae": 10.5, "ef40_auroc": 0.76, "small_n": False},
            },
            "race": {"Other": {"n": 6, "mae": 12.0, "ef40_auroc": None, "small_n": True}},
        },
    }))


def test_aggregate_collects_sources(tmp_path):
    _seed_results(tmp_path)
    agg = aggregate_results(tmp_path)
    assert set(agg["sources"]) == {"missing_modality", "fairness"}
    assert agg["missing_modality"]["n_test"] == 245
    assert len(agg["missing_modality"]["conditions"]) == 3
    assert "git_sha" in agg


def test_renderers_and_write(tmp_path):
    _seed_results(tmp_path)
    agg = aggregate_results(tmp_path)

    md = to_markdown(agg)
    assert "Missing-modality degradation" in md
    assert "Echo dropped (ECG-only)" in md
    assert "18.57" in md

    tex = to_latex(agg)
    assert "\\begin{table}" in tex and "\\bottomrule" in tex
    assert "tab:degradation-agg" in tex

    paths = write_all(agg, tmp_path)
    for path in paths.values():
        assert (tmp_path / "").exists()  # dir present
    assert json.loads((tmp_path / "results.json").read_text())["task"] == "E05_results_aggregation"
    assert (tmp_path / "tables" / "results.tex").exists()


def test_missing_inputs_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate_results(tmp_path)


def test_null_auroc_renders_dash(tmp_path):
    _seed_results(tmp_path)
    agg = aggregate_results(tmp_path)
    md = to_markdown(agg)
    # the "Other" race stratum has null AUROC -> rendered as a dash
    assert "| Race | Other" in md
    assert "--" in md
