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


def test_run_calibration_fits_on_val_when_present(tmp_path):
    """E09 (#66): the scaler must be fit on val predictions, never on test.

    Val and test carry opposite score-label relationships, so a val-fit scaler is
    confidently wrong on test (huge ECE) while an in-sample fit would look calibrated.
    """
    rng = np.random.default_rng(3)
    n = 100
    test_pred = rng.uniform(20, 70, size=n)
    test_labels = (test_pred <= 40).astype(int).tolist()  # low prediction => positive
    val_pred = rng.uniform(20, 70, size=n)
    val_labels = (val_pred > 40).astype(int).tolist()  # inverted relationship
    pj = tmp_path / "missing_modality.json"
    pj.write_text(
        json.dumps(
            {
                "checkpoint": "probes/fused/cross_attn_fused.pt",
                "checkpoint_sha256": "abc123",
                "seed": 42,
                "predictions": {
                    "full": {"ef_le_40": test_labels, "prediction": test_pred.tolist()}
                },
                "predictions_val": {
                    "full": {"ef_le_40": val_labels, "prediction": val_pred.tolist()}
                },
            }
        )
    )

    res = run_calibration(pj, tmp_path / "calibration", condition="full", n_bins=10)
    assert res["scaler_fit_on"] == "val"
    assert res["ece"] > 0.5
    # the artifact must be traceable on its own: source checkpoint/seed ride along
    assert res["predictions_path"] == str(pj)
    assert res["prediction_source"] == {
        "checkpoint": "probes/fused/cross_attn_fused.pt",
        "checkpoint_sha256": "abc123",
        "seed": 42,
    }
