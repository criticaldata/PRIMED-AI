"""Validate the modality-failure harness recovers planted ground-truth structure."""

from __future__ import annotations

import json

import numpy as np
import pytest

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
    assert mv["echo"] > mv["ecg"]  # robust aggregate: echo worth more
    winners = d["complementarity"]["per_example_winners"]
    assert winners["echo"] > winners["ecg"]  # planted: echo dominant globally
    # fusion is at least as good as the best single modality
    assert d["complementarity"]["full_mae"] <= min(d["complementarity"]["solo_mae"].values()) + 1e-6


def test_ecg_still_wins_some_examples():
    # the planted ECG-only subset means ECG must "win" for a nonzero count
    d = _run()
    assert d["complementarity"]["per_example_winners"]["ecg"] > 0


def test_dropping_dominant_modality_induces_more_and_silent_failures():
    d = _run()
    assert (
        d["dropout"]["drop_echo"]["induced_critical"]
        >= d["dropout"]["drop_ecg"]["induced_critical"]
    )
    assert d["dropout"]["drop_echo"]["induced_critical"] > 0  # dropping echo induces gate failures


def test_taxonomy_degrades_under_dropout():
    d = _run()
    assert (
        d["conditions"]["full"]["taxonomy"]["correct"]
        > d["conditions"]["drop_echo"]["taxonomy"]["correct"]
    )


def test_report_is_json_serializable():
    d = _run()
    assert "complementarity" in json.dumps(d)
    assert d["complementarity"]["matrix"]["modalities"] == ["echo", "ecg"]


def test_shapley_efficiency_and_closed_form():
    # exact Shapley over v(S) = -MAE(f(S)): the values must sum to the full-vs-empty MAE
    # gap, and at N = 2 each phi has a closed form over the four subset MAEs
    emb, lvef, ef, _ = make_synthetic_multimodal(n=600, seed=0)
    train = stratified_train_mask(ef, 0.7, seed=12345)
    test = ~train
    pf = masked_ridge_predict_fn(emb, lvef, train)
    report = analyze_modality_failure(
        {m: emb[m][test] for m in emb}, lvef[test], ef[test], lambda present: pf(present)[test]
    )
    shap = report.complementarity["shapley_value"]
    y = report.predictions["__labels__"]

    def mae(key):
        return float(np.abs(report.predictions[key] - y).mean())

    assert sum(shap.values()) == pytest.approx(mae("") - mae("ecg,echo"), abs=1e-3)
    closed_echo = 0.5 * ((mae("") - mae("echo")) + (mae("ecg") - mae("ecg,echo")))
    assert shap["echo"] == pytest.approx(closed_echo, abs=1e-3)
    assert shap["echo"] > shap["ecg"]  # planted dominance survives the attribution change


def test_shapley_credits_redundant_modalities_beyond_loo():
    # two equally strong modalities: leave-one-out under-credits each (the twin covers),
    # Shapley counts the coalitions where the twin is absent and pays both more
    from primed_ai.failure import make_synthetic_modalities

    emb, lvef, ef = make_synthetic_modalities({"m1": 2.0, "m2": 2.0, "weak": 0.5}, n=750, seed=0)
    train = stratified_train_mask(ef, 0.7, seed=1000)
    test = ~train
    pf = masked_ridge_predict_fn(emb, lvef, train)
    c = analyze_modality_failure(
        {m: emb[m][test] for m in emb}, lvef[test], ef[test], lambda present: pf(present)[test]
    ).complementarity
    loo, shap = c["marginal_value"], c["shapley_value"]
    assert shap["m1"] > loo["m1"] and shap["m2"] > loo["m2"]
    assert abs(shap["m1"] - shap["m2"]) < 1.0  # symmetric plants get symmetric credit
    assert shap["weak"] < min(shap["m1"], shap["m2"])


def test_shapley_none_when_empty_set_unsupported():
    rng = np.random.default_rng(0)
    y = rng.uniform(15, 75, 40)
    gate = y <= 40

    def predict_fn(present: frozenset) -> np.ndarray:
        if not present:
            raise ValueError("cannot mask everything")
        return y + rng.normal(0, len(present), 40)

    report = analyze_modality_failure(["echo", "ecg"], y, gate, predict_fn)
    assert report.complementarity["shapley_value"] is None
    assert report.complementarity["marginal_value"]  # the rest of the report still works


def test_classify_taxonomy_categories():
    y = np.array([35.0, 55.0, 50.0])
    pred = np.array(
        [36.0, 54.0, 20.0]
    )  # 3rd: confidently predicts HFrEF but truth is 50 -> critical
    gate_true = y <= 40
    cats = classify_taxonomy(y, pred, gate_true, threshold=40, tolerance=5)
    assert list(cats) == ["correct", "correct", "critical"]


def test_requires_two_modalities():
    import pytest

    with pytest.raises(ValueError):
        analyze_modality_failure(
            {"echo": np.zeros((3, 2))}, [1, 2, 3], [True, False, True], lambda present: np.zeros(3)
        )


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
    assert matrix["modalities"] == ["echo", "ecg", "labs"]  # genuine 3x3 matrix
    assert len(matrix["values"]) == 3 and all(len(r) == 3 for r in matrix["values"])
    assert d["complementarity"]["marginal_value"]["echo"] == max(
        d["complementarity"]["marginal_value"].values()
    )  # echo planted strongest
    assert set(d["dropout"]) == {"drop_echo", "drop_ecg", "drop_labs"}


def test_split_has_healthy_prevalence():
    # guards against the generation/split RNG correlation that can zero out test-set prevalence
    emb, lvef, ef, _ = make_synthetic_multimodal(n=600, seed=0)
    test = ~stratified_train_mask(ef, 0.7, seed=12345)
    prevalence = float(ef[test].mean())
    assert 0.15 < prevalence < 0.6  # both gate classes well represented in the held-out split


def test_emitted_report_names_the_producing_model(tmp_path):
    # A failure report and a fused-checkpoint report disagree badly on the dropped
    # conditions, so an artifact that doesn't name its model can be read as the wrong one.
    import run_failure_analysis

    out = tmp_path / "failure_demo"
    run_failure_analysis._run_demo(0, str(out))
    payload = json.loads((out / "failure_report.json").read_text())

    assert payload["provenance"]["model"] == run_failure_analysis.RIDGE_MODEL
    assert payload["provenance"]["data"] == "synthetic_planted"
    assert payload["conditions"]["full"]["mae"] > 0


def _prediction_dump(tmp_path):
    """A minimal `results/missing_modality.json`, shaped the way the fused eval writes it."""
    rng = np.random.default_rng(4)
    lvef = rng.uniform(15.0, 75.0, size=80)
    ef = lvef <= 40.0
    # echo carries the signal; the ECG-only arm is deliberately useless, so which condition
    # maps to which modality shows up in the numbers rather than only in the field names.
    full = lvef + rng.standard_normal(80)
    echo_only = lvef + rng.standard_normal(80) * 3.0
    ecg_only = np.full(80, 55.0)

    def block(pred):
        return {
            "lvef": lvef.tolist(),
            "prediction": np.asarray(pred, dtype=float).tolist(),
            "ef_le_40": ef.tolist(),
        }

    payload = {
        "seed": 42,
        "git_sha": "deadbee",
        "checkpoint": "probes/fused/cross_attn_fused.pt",
        "checkpoint_sha256": "bac18bb8",
        "manifest_sha256": "81694c9b",
        "predictions": {
            "full": block(full),
            "echo_dropped": block(ecg_only),
            "ecg_dropped": block(echo_only),
        },
    }
    path = tmp_path / "missing_modality.json"
    path.write_text(json.dumps(payload))
    return path, lvef


def test_fused_route_maps_each_condition_to_the_surviving_modality(tmp_path):
    import run_failure_analysis

    path, _ = _prediction_dump(tmp_path)
    out = tmp_path / "failure_fused"
    run_failure_analysis._run_predictions(str(path), str(out))
    d = json.loads((out / "failure_report.json").read_text())

    # `echo_dropped` leaves ECG alone, and ECG-only is the useless arm in the fixture --
    # a flipped mapping would blame echo instead.
    solo = d["complementarity"]["solo_mae"]
    assert solo["ecg"] > solo["echo"]
    assert d["complementarity"]["marginal_value"]["echo"] > 0
    assert d["conditions"]["drop_echo"]["mae"] > d["conditions"]["drop_ecg"]["mae"]
    assert d["n"] == 80


def test_fused_route_records_checkpoint_provenance(tmp_path):
    import run_failure_analysis

    path, _ = _prediction_dump(tmp_path)
    out = tmp_path / "failure_fused"
    run_failure_analysis._run_predictions(str(path), str(out))
    prov = json.loads((out / "failure_report.json").read_text())["provenance"]

    assert prov["model"] == run_failure_analysis.FUSED_MODEL
    assert prov["checkpoint"] == "probes/fused/cross_attn_fused.pt"
    assert prov["checkpoint_sha256"] == "bac18bb8"
    assert prov["manifest_sha256"] == "81694c9b"
    assert prov["seed"] == 42
    assert prov["source_git_sha"] == "deadbee"


def test_fused_route_rejects_the_sanitized_bundle_copy(tmp_path):
    import run_failure_analysis

    # docs/results/ ships the aggregate copy with the per-example blocks stripped; pointed
    # at that, the route has nothing to analyse and must say so rather than half-run.
    stripped = tmp_path / "missing_modality.metrics.json"
    stripped.write_text(json.dumps({"test": {"full": {"mae": 10.4}}}))
    with pytest.raises(SystemExit, match="no per-example"):
        run_failure_analysis._run_predictions(str(stripped), str(tmp_path / "out"))


def test_fused_route_rejects_misaligned_conditions(tmp_path):
    import run_failure_analysis

    path, _ = _prediction_dump(tmp_path)
    payload = json.loads(path.read_text())
    labels = payload["predictions"]["echo_dropped"]["lvef"]
    payload["predictions"]["echo_dropped"]["lvef"] = labels[1:] + labels[:1]
    path.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="different labels"):
        run_failure_analysis._run_predictions(str(path), str(tmp_path / "out"))


def test_fused_route_rejects_a_dump_missing_a_condition(tmp_path):
    import run_failure_analysis

    path, _ = _prediction_dump(tmp_path)
    payload = json.loads(path.read_text())
    del payload["predictions"]["ecg_dropped"]
    path.write_text(json.dumps(payload))

    with pytest.raises(SystemExit, match="missing condition"):
        run_failure_analysis._run_predictions(str(path), str(tmp_path / "out"))
