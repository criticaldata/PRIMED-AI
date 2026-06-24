"""Validate the modality-failure harness recovers planted ground-truth structure."""
from __future__ import annotations

import json

import numpy as np

from primed_ai.failure import (
    analyze_modality_failure,
    classify_taxonomy,
    make_synthetic_multimodal,
    masked_ridge_predict_fn,
    stratified_train_mask,
)


def _run(seed: int = 0):
    emb, lvef, ef, planted = make_synthetic_multimodal(n=600, seed=seed)
    train = stratified_train_mask(ef, 0.7, seed=seed + 12345)
    test = ~train
    predict_full = masked_ridge_predict_fn(emb, lvef, train)
    emb_te = {m: emb[m][test] for m in emb}
    report = analyze_modality_failure(
        emb_te, lvef[test], ef[test], lambda present: predict_full(present)[test]
    )
    return report.to_dict()


def test_recovers_echo_dominance():
    d = _run()
    mv = d["complementarity"]["marginal_value"]
    assert mv["echo"] > mv["ecg"]                                 # robust aggregate: echo worth more
    winners = d["complementarity"]["per_example_winners"]
    assert winners["echo"] > winners["ecg"]                       # planted: echo dominant globally
    # fusion is at least as good as the best single modality
    assert d["complementarity"]["full_mae"] <= min(d["complementarity"]["solo_mae"].values()) + 1e-6


def test_ecg_still_wins_some_examples():
    # the planted ECG-only subset means ECG must "win" for a nonzero count
    d = _run()
    assert d["complementarity"]["per_example_winners"]["ecg"] > 0


def test_dropping_dominant_modality_induces_more_and_silent_failures():
    d = _run()
    assert (d["dropout"]["drop_echo"]["induced_critical"]
            >= d["dropout"]["drop_ecg"]["induced_critical"])
    assert d["dropout"]["drop_echo"]["induced_critical"] > 0      # dropping echo induces gate failures


def test_taxonomy_degrades_under_dropout():
    d = _run()
    assert (d["conditions"]["full"]["taxonomy"]["correct"]
            > d["conditions"]["drop_echo"]["taxonomy"]["correct"])


def test_report_is_json_serializable():
    d = _run()
    assert "complementarity" in json.dumps(d)
    assert d["complementarity"]["matrix"]["modalities"] == ["echo", "ecg"]


def test_classify_taxonomy_categories():
    y = np.array([35.0, 55.0, 50.0])
    pred = np.array([36.0, 54.0, 20.0])   # 3rd: confidently predicts HFrEF but truth is 50 -> critical
    gate_true = y <= 40
    cats = classify_taxonomy(y, pred, gate_true, threshold=40, tolerance=5)
    assert list(cats) == ["correct", "correct", "critical"]


def test_requires_two_modalities():
    import pytest

    with pytest.raises(ValueError):
        analyze_modality_failure({"echo": np.zeros((3, 2))}, [1, 2, 3], [True, False, True],
                                 lambda present: np.zeros(3))


def test_three_modality_generality():
    from primed_ai.failure import make_synthetic_modalities

    emb, lvef, ef = make_synthetic_modalities({"echo": 2.0, "ecg": 0.8, "labs": 0.8}, n=450, seed=1)
    train = stratified_train_mask(ef, 0.7, seed=7)
    test = ~train
    pf = masked_ridge_predict_fn(emb, lvef, train)
    d = analyze_modality_failure(
        {m: emb[m][test] for m in emb}, lvef[test], ef[test], lambda present: pf(present)[test]
    ).to_dict()
    matrix = d["complementarity"]["matrix"]
    assert matrix["modalities"] == ["echo", "ecg", "labs"]          # genuine 3x3 matrix
    assert len(matrix["values"]) == 3 and all(len(r) == 3 for r in matrix["values"])
    assert d["complementarity"]["marginal_value"]["echo"] == max(
        d["complementarity"]["marginal_value"].values())            # echo planted strongest
    assert set(d["dropout"]) == {"drop_echo", "drop_ecg", "drop_labs"}


def test_split_has_healthy_prevalence():
    # guards against the generation/split RNG correlation that can zero out test-set prevalence
    emb, lvef, ef, _ = make_synthetic_multimodal(n=600, seed=0)
    test = ~stratified_train_mask(ef, 0.7, seed=12345)
    prevalence = float(ef[test].mean())
    assert 0.15 < prevalence < 0.6   # both gate classes well represented in the held-out split
