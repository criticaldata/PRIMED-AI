import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from primed_ai.probes.valvular import (
    VALVULAR_TARGETS,
    MultiTargetValvularClassifier,
    auprc_safe,
    auroc_safe,
    bootstrap_ci,
)
from scripts.extract_valvular_labels import parse_echo_text_with_ner


def test_valvular_targets_defined():
    assert len(VALVULAR_TARGETS) == 3
    targets = [t[0] for t in VALVULAR_TARGETS]
    assert "as_moderate_or_severe" in targets
    assert "mr_moderate_or_severe" in targets
    assert "tr_moderate_or_severe" in targets


def test_metrics_safety():
    # All true / all false edge case
    all_true = np.ones(10, dtype=bool)
    scores = np.linspace(0, 1, 10)
    assert np.isnan(auroc_safe(all_true, scores))
    assert np.isnan(auprc_safe(all_true, scores))

    # Standard valid case
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc_safe(y, s) == 1.0
    assert auprc_safe(y, s) == 1.0


def test_multitarget_valvular_classifier_fit_and_predict():
    rng = np.random.default_rng(42)
    n = 100
    dim = 32
    X = rng.normal(0, 1, (n, dim))

    y_as = rng.choice([False, True], size=n)
    y_mr = rng.choice([False, True], size=n)
    y_tr = rng.choice([False, True], size=n)

    y_dict = {
        "as_moderate_or_severe": y_as,
        "mr_moderate_or_severe": y_mr,
        "tr_moderate_or_severe": y_tr,
    }

    clf = MultiTargetValvularClassifier(cv=2, seed=42)
    clf.fit(X, y_dict)

    probs = clf.predict_proba(X)
    assert "as_moderate_or_severe" in probs
    assert len(probs["as_moderate_or_severe"]) == n
    assert (probs["as_moderate_or_severe"] >= 0).all() and (probs["as_moderate_or_severe"] <= 1).all()

    eval_results = clf.evaluate(X, y_dict, n_bootstrap=50)
    assert "as_moderate_or_severe" in eval_results
    assert eval_results["as_moderate_or_severe"].auroc >= 0.0


def test_clinical_ner_extraction_negation_and_severity():
    # Severe AS
    text1 = "Echocardiogram shows calcified aortic leaflets with severe aortic stenosis."
    res1 = parse_echo_text_with_ner(text1)
    assert res1["as_grade"] == 3

    # Negated MR
    text2 = "Mitral valve: no evidence of mitral regurgitation."
    res2 = parse_echo_text_with_ner(text2)
    assert res2["mr_grade"] == 0

    # Family history exclusion
    text3 = "Patient states father had aortic stenosis. Current echo is unremarkable."
    res3 = parse_echo_text_with_ner(text3)
    assert res3["as_grade"] is None or res3["as_grade"] == 0
