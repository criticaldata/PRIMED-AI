"""Tests for E04 EF<=40% calibration."""
from __future__ import annotations

import json

import numpy as np

from primed_ai.evaluation.calibration import (
    expected_calibration_error,
    platt_probabilities,
    reliability_bins,
    run_calibration,
)


def test_ece_zero_when_probs_match_labels():
    y = np.array([0, 1, 0, 1, 1, 0])
    # probabilities exactly equal to the binary labels => perfectly calibrated
    assert expected_calibration_error(y, y.astype(float), n_bins=10) == 0.0


def test_ece_detects_miscalibration():
    y = np.ones(50)
    confident_wrong = np.full(50, 0.05)  # high confidence the event is absent, but it's present
    assert expected_calibration_error(y, confident_wrong, n_bins=10) > 0.8


def test_reliability_bins_conserve_count():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    bins = reliability_bins(y, p, n_bins=10)
    assert sum(b["count"] for b in bins) == 200
    assert len(bins) == 10


def test_platt_bounded_and_separating():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=200)
    scores = labels + rng.normal(0, 0.5, size=200)  # scores correlated with labels
    prob = platt_probabilities(scores, labels)
    assert prob.min() > 0.0 and prob.max() < 1.0
    assert prob[labels == 1].mean() > prob[labels == 0].mean()


def test_run_calibration_end_to_end(tmp_path):
    rng = np.random.default_rng(2)
    n = 80
    lvef = rng.uniform(15, 70, size=n)
    ef_le_40 = (lvef <= 40).astype(int).tolist()
    # predicted LVEF tracks truth with noise
    pred = (lvef + rng.normal(0, 5, size=n)).tolist()
    pj = tmp_path / "missing_modality.json"
    pj.write_text(json.dumps({"predictions": {"full": {"ef_le_40": ef_le_40, "prediction": pred}}}))

    res = run_calibration(pj, tmp_path / "calibration", condition="full", n_bins=8)
    assert res["task"] == "E04_ef40_calibration"
    assert 0.0 <= res["ece"] <= 1.0
    assert res["n"] == n
    assert (tmp_path / "calibration" / "calibration.json").exists()
    assert (tmp_path / "calibration" / "reliability.pdf").exists()
