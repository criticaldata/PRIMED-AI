"""Model-agnostic modality-failure analysis (MFA) harness.

Evaluation outlives models. Given N modality embeddings, a mask-aware prediction
function, and labels for a regression target with a binary clinical gate, the harness
reports three things that transfer across model generations:

1. **Failure taxonomy** -- per-example category (``correct`` / ``imprecise`` / ``critical``)
   under full-modality inference and under each missing-modality condition.
2. **Complementarity** -- each modality's leave-one-out marginal value, its exact Shapley
   value over modality coalitions, a complementarity matrix over modality subsets, and
   per-example attribution ("which modality wins").
3. **Dropout profile (loud vs. silent)** -- when a modality is missing at inference, does the
   model fail *loudly* (output shifts / sits near the decision boundary -> monitorable) or
   *silently* (confident, stable-looking output that is now clinically wrong)?

The harness is model-agnostic: callers pass ``predict_fn(present) -> np.ndarray`` returning a
continuous prediction per example given the *set of present modalities* (others masked at
inference). It works for any number of modalities and any probe/backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import factorial
from typing import Callable

import numpy as np

PredictFn = Callable[[frozenset], np.ndarray]
CATEGORIES = ("correct", "imprecise", "critical")


def _abs_err(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float))


def _gate(pred: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(pred, dtype=float) <= threshold


def classify_taxonomy(y, pred, gate_true, *, threshold: float, tolerance: float) -> np.ndarray:
    """Per-example error category.

    ``critical``  : the clinical gate decision is wrong (e.g. HFrEF missed / false-alarmed).
    ``imprecise`` : gate correct but regression error exceeds ``tolerance``.
    ``correct``   : gate correct and regression error within ``tolerance``.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    gate_true = np.asarray(gate_true, dtype=bool)
    err = _abs_err(pred, y)
    gate_wrong = _gate(pred, threshold) != gate_true
    return np.where(gate_wrong, "critical", np.where(err <= tolerance, "correct", "imprecise"))


def _counts(cats: np.ndarray) -> dict:
    return {c: int(np.sum(cats == c)) for c in CATEGORIES}


@dataclass
class FailureReport:
    """Serializable modality-failure report (raw predictions kept in ``predictions``)."""

    modalities: list
    n: int
    threshold: float
    tolerance: float
    conditions: dict
    complementarity: dict
    dropout: dict
    predictions: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        return {
            "task": "modality_failure_analysis",
            "modalities": self.modalities,
            "n": self.n,
            "threshold": self.threshold,
            "tolerance": self.tolerance,
            "conditions": self.conditions,
            "complementarity": self.complementarity,
            "dropout": self.dropout,
        }


def analyze_modality_failure(
    modalities,
    lvef,
    ef_le_40,
    predict_fn: PredictFn,
    *,
    threshold: float = 40.0,
    tolerance: float = 5.0,
    confidence_margin: float = 5.0,
    groups: dict | None = None,
) -> FailureReport:
    """Run the full modality-failure analysis.

    Parameters
    ----------
    modalities : the modality names, or any mapping keyed by them (``{name: array}``) -- the
        embedding *values* are never read here, so a caller that already holds predictions
        can pass the names alone. Masking is the caller's job, via ``predict_fn``.
    predict_fn : ``predict_fn(present: frozenset[str]) -> np.ndarray`` -- continuous prediction
        per example given the present modalities (others masked).
    threshold : clinical gate threshold on the continuous target (e.g. EF <= 40).
    tolerance : regression tolerance separating ``correct`` from ``imprecise``.
    confidence_margin : loud-vs-silent threshold = distance from the decision boundary (in
        target units). A newly-wrong case is *silent* if its deployment-observable prediction
        sits at least this far on the wrong side of the gate (confidently wrong), else *loud*.
    groups : optional ``{attribute: array}`` for per-stratum win attribution.
    """
    modalities = list(modalities)
    if len(modalities) < 2:
        raise ValueError("Need >= 2 modalities for failure attribution / dropout analysis.")
    y = np.asarray(lvef, dtype=float)
    gate_true = np.asarray(ef_le_40, dtype=bool)
    n = int(len(y))
    allm = frozenset(modalities)

    # Subsets we need: full, every singleton, every leave-one-out, every pair (for the matrix).
    needed = {allm}
    for m in modalities:
        needed.add(frozenset({m}))
        needed.add(allm - {m})
    for a, b in combinations(modalities, 2):
        needed.add(frozenset({a, b}))
    # Exact Shapley attribution needs every coalition, the empty set (all branches
    # masked) included. 2^N calls are cheap at the N this harness sees; past the cap
    # we keep leave-one-out attribution only rather than blow up predict_fn calls.
    shapley_ok = len(modalities) <= 8
    if shapley_ok:
        for k in range(len(modalities) + 1):
            for combo in combinations(modalities, k):
                needed.add(frozenset(combo))
    preds = {}
    for s in needed:
        try:
            preds[s] = np.asarray(predict_fn(s), dtype=float)
        except Exception:
            if s:
                raise
            # predict_fn that cannot mask everything: drop Shapley, keep the rest
            shapley_ok = False
    full = preds[allm]
    err_full = _abs_err(full, y)

    # --- 1. Per-condition taxonomy (full + each dropout) -------------------------------
    def _condition(pred) -> dict:
        cats = classify_taxonomy(y, pred, gate_true, threshold=threshold, tolerance=tolerance)
        return {"mae": round(float(_abs_err(pred, y).mean()), 4), "taxonomy": _counts(cats)}

    conditions = {"full": _condition(full)}
    for m in modalities:
        conditions[f"drop_{m}"] = _condition(preds[allm - {m}])

    # --- 2. Complementarity ------------------------------------------------------------
    solo_err = {m: _abs_err(preds[frozenset({m})], y) for m in modalities}
    drop_err = {m: _abs_err(preds[allm - {m}], y) for m in modalities}
    marginal_value = {m: round(float(drop_err[m].mean() - err_full.mean()), 4) for m in modalities}
    solo_mae = {m: round(float(solo_err[m].mean()), 4) for m in modalities}

    # per-example winner = modality whose removal hurts this example most
    drop_minus_full = np.stack([drop_err[m] - err_full for m in modalities], axis=1)  # N x M
    winner_idx = np.argmax(drop_minus_full, axis=1)
    winners = {m: int(np.sum(winner_idx == i)) for i, m in enumerate(modalities)}

    # complementarity matrix: C[i][j] = MAE with only {i, j} present (singleton on diagonal)
    matrix = []
    for a in modalities:
        row = []
        for b in modalities:
            subset = frozenset({a}) if a == b else frozenset({a, b})
            row.append(round(float(_abs_err(preds[subset], y).mean()), 4))
        matrix.append(row)

    # Exact Shapley value of each modality over the coalition game v(S) = -MAE(f(S)):
    # the weighted mean MAE drop from adding m across all coalitions, so positive phi
    # is error the modality removes. Unlike leave-one-out, two redundant modalities
    # split the credit instead of both reading as worthless.
    shapley = None
    if shapley_ok:
        mae_of = {s: float(_abs_err(p, y).mean()) for s, p in preds.items()}
        nm = len(modalities)
        shapley = {}
        for m in modalities:
            rest = [x for x in modalities if x != m]
            phi = 0.0
            for k in range(nm):
                w = factorial(k) * factorial(nm - k - 1) / factorial(nm)
                for combo in combinations(rest, k):
                    s = frozenset(combo)
                    phi += w * (mae_of[s] - mae_of[s | {m}])
            shapley[m] = round(phi, 4)

    best_solo = min(solo_mae.values())
    complementarity = {
        "full_mae": round(float(err_full.mean()), 4),
        "solo_mae": solo_mae,
        "marginal_value": marginal_value,  # how much each modality is worth (LOO)
        "shapley_value": shapley,  # coalition-fair attribution; None past the subset cap
        "fusion_gain_vs_best_solo": round(best_solo - float(err_full.mean()), 4),
        "per_example_winners": winners,  # which modality "wins" overall
        "matrix": {"modalities": modalities, "values": matrix},
    }
    if groups:
        complementarity["winners_by_group"] = {
            attr: {
                str(g): {
                    modalities[i]: int(np.sum(winner_idx[np.asarray(arr) == g] == i))
                    for i in range(len(modalities))
                }
                for g in np.unique(np.asarray(arr))
            }
            for attr, arr in groups.items()
        }

    # --- 3. Dropout profile: loud vs. silent ------------------------------------------
    dropout = {}
    for m in modalities:
        pred_drop = preds[allm - {m}]
        was_right = _gate(full, threshold) == gate_true
        now_wrong = _gate(pred_drop, threshold) != gate_true
        induced = was_right & now_wrong  # newly clinically wrong from dropping m
        confidence = np.abs(pred_drop - threshold)  # distance from boundary (deployment-observable)
        shift = np.abs(pred_drop - full)  # diagnostic only; needs the full counterfactual
        silent = induced & (confidence >= confidence_margin)  # confidently wrong -> undetectable
        loud = induced & (confidence < confidence_margin)  # near boundary -> output signals doubt
        dropout[f"drop_{m}"] = {
            "induced_critical": int(induced.sum()),
            "silent": int(silent.sum()),
            "loud": int(loud.sum()),
            "silent_rate": (
                round(float(silent.sum() / induced.sum()), 4) if induced.sum() else None
            ),
            "mean_output_shift": round(float(shift.mean()), 4),
            "mae_increase": round(float(drop_err[m].mean() - err_full.mean()), 4),
        }

    return FailureReport(
        modalities=modalities,
        n=n,
        threshold=threshold,
        tolerance=tolerance,
        conditions=conditions,
        complementarity=complementarity,
        dropout=dropout,
        predictions={
            "__labels__": y,
            "__gate__": gate_true,
            **{",".join(sorted(s)): p for s, p in preds.items()},
        },
    )
