"""Unit tests for the E10 baseline-gap bootstrap helpers (#67)."""

import numpy as np
from diagnose_baseline_gap import _bootstrap, _ci95, _paired_delta


def _arrays(seed, *, single_class=False, noise=1.0):
    rng = np.random.default_rng(seed)
    lvef = rng.uniform(50, 70, size=50) if single_class else rng.uniform(20, 70, size=50)
    return {
        "lvef": lvef,
        "prediction": lvef + rng.standard_normal(50) * noise,
        "ef_le_40": lvef <= 40,
    }


def test_ci95_is_none_when_every_replicate_was_discarded():
    assert _ci95([]) is None
    assert _ci95([1.0, 2.0, 3.0]) == [1.05, 2.95]


def test_bootstrap_counts_add_up():
    res = _bootstrap(_arrays(11), n_bootstrap=100, seed=5)

    assert res["n_bootstrap"] == 100
    assert res["n_auroc_replicates"] + res["n_auroc_discarded"] == 100
    assert res["n_auroc_replicates"] > 0
    assert res["ef40_auroc_ci95"][0] <= res["ef40_auroc_ci95"][1]


def test_bootstrap_survives_a_single_class_frame():
    # Previously this raised inside np.quantile on the empty AUROC sample.
    res = _bootstrap(_arrays(12, single_class=True), n_bootstrap=30, seed=5)

    assert res["ef40_auroc_ci95"] is None
    assert res["n_auroc_replicates"] == 0
    assert res["n_auroc_discarded"] == 30
    assert res["mae_ci95"][0] <= res["mae_ci95"][1]


def test_paired_delta_counts_replicates_where_both_models_are_defined():
    fused = _arrays(13, noise=0.5)
    other = dict(fused, prediction=fused["lvef"] + np.full(50, 4.0))
    res = _paired_delta(fused, other, n_bootstrap=100, seed=5)

    assert res["n_bootstrap"] == 100
    assert res["n_auroc_replicates"] + res["n_auroc_discarded"] == 100
    # fused tracks the label, the comparator is biased high -> fusion wins on MAE
    assert res["delta_mae"] < 0


def test_paired_delta_survives_a_single_class_frame():
    # Previously this raised inside np.quantile on the empty delta sample.
    fused = _arrays(14, single_class=True)
    other = dict(fused, prediction=fused["prediction"] + 1.0)
    res = _paired_delta(fused, other, n_bootstrap=20, seed=5)

    assert res["delta_ef40_auroc_ci95"] is None
    assert res["n_auroc_replicates"] == 0
    assert res["n_auroc_discarded"] == 20
    assert res["delta_mae_ci95"][0] <= res["delta_mae_ci95"][1]
