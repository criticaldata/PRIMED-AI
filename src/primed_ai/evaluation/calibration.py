"""E04: calibration of the EF<=40% clinical gate (reliability diagram + ECE).

A deployable risk score must be calibrated, not merely discriminative. This module turns
continuous LVEF predictions into EF<=40% probabilities (Platt scaling), then reports the
expected calibration error (ECE) and a reliability diagram. Optional stretch (issue #10).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np


def platt_probabilities(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    fit_scores: np.ndarray | None = None,
    fit_labels: np.ndarray | None = None,
) -> np.ndarray:
    """Map arbitrary real-valued scores to calibrated probabilities via 1-D logistic fit.

    For the EF<=40% gate the natural score is the (negated) predicted LVEF; Platt scaling
    learns the sign and slope. Pass ``fit_scores``/``fit_labels`` from a held-out split to
    fit there and apply to ``scores``; fitting in-sample gives an optimistic ECE.
    """
    from sklearn.linear_model import LogisticRegression

    if (fit_scores is None) != (fit_labels is None):
        raise ValueError("pass fit_scores and fit_labels together, or neither")
    scores = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    if fit_scores is None:
        fit_scores, fit_labels = scores, labels
    fit_scores = np.asarray(fit_scores, dtype=np.float64).reshape(-1, 1)
    fit_labels = np.asarray(fit_labels).astype(int)
    lr = LogisticRegression()
    lr.fit(fit_scores, fit_labels)
    return lr.predict_proba(scores)[:, 1]


def reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-width reliability bins with per-bin confidence, accuracy, and count."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        bins.append(
            {
                "bin_low": round(float(edges[b]), 4),
                "bin_high": round(float(edges[b + 1]), 4),
                "count": count,
                "confidence": round(float(y_prob[mask].mean()), 4) if count else None,
                "accuracy": round(float(y_true[mask].mean()), 4) if count else None,
            }
        )
    return bins


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Standard ECE: count-weighted mean |accuracy - confidence| across bins."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return float("nan")
    ece = 0.0
    for b in reliability_bins(y_true, y_prob, n_bins):
        if b["count"]:
            ece += (b["count"] / n) * abs(b["accuracy"] - b["confidence"])
    return float(ece)


def plot_reliability(
    y_true: np.ndarray, y_prob: np.ndarray, output_pdf: str | Path, n_bins: int = 10
) -> Path:
    """Write a reliability diagram (PDF + PNG) with the diagonal and per-bin gaps."""
    cache_root = Path(tempfile.gettempdir()) / "primed-ai-plot-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [b for b in reliability_bins(y_true, y_prob, n_bins) if b["count"]]
    conf = [b["confidence"] for b in bins]
    acc = [b["accuracy"] for b in bins]
    ece = expected_calibration_error(y_true, y_prob, n_bins)

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "--", color="#9CA3AF", linewidth=1.0, label="Perfect calibration")
    ax.plot(conf, acc, "o-", color="#0072B2", linewidth=1.4, markersize=4.5, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(EF $\\leq$ 40%)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"EF $\\leq$ 40% reliability (ECE = {ece:.3f})")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_pdf


def run_calibration(
    predictions_path: str | Path,
    out_dir: str | Path = "results/calibration",
    condition: str = "full",
    n_bins: int = 10,
) -> dict:
    """Compute EF<=40% calibration from a missing-modality results JSON's predictions."""
    from primed_ai.probes.common import git_sha

    payload = json.loads(Path(predictions_path).read_text())
    preds = payload.get("predictions", {}).get(condition)
    if not preds:
        raise ValueError(
            f"No per-example predictions for condition '{condition}' in {predictions_path}. "
            "Re-run the missing-modality evaluator with prediction dumping enabled."
        )
    labels = np.asarray(preds["ef_le_40"]).astype(int)
    # Lower predicted LVEF => higher P(EF<=40); Platt learns the mapping. Fit on the val
    # split's predictions when the results JSON carries them, so test stays untouched.
    val_preds = payload.get("predictions_val", {}).get(condition)
    fit_kwargs = {}
    if val_preds:
        fit_kwargs = {
            "fit_scores": -np.asarray(val_preds["prediction"], dtype=np.float64),
            "fit_labels": np.asarray(val_preds["ef_le_40"]).astype(int),
        }
    prob = platt_probabilities(
        -np.asarray(preds["prediction"], dtype=np.float64), labels, **fit_kwargs
    )

    out_dir = Path(out_dir)
    figure = plot_reliability(labels, prob, out_dir / "reliability.pdf", n_bins=n_bins)
    result = {
        "task": "E04_ef40_calibration",
        "git_sha": git_sha(),
        "condition": condition,
        "n": int(len(labels)),
        "n_bins": n_bins,
        "scaler_fit_on": "val" if val_preds else "test (in-sample)",
        "ece": round(expected_calibration_error(labels, prob, n_bins), 4),
        "bins": reliability_bins(labels, prob, n_bins),
        "figure": str(figure),
        "note": (
            "Platt scaler fit on val predictions, applied to test."
            if val_preds
            else "In-sample Platt scaling; use a held-out calibration split for a deployment estimate."
        ),
    }
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2))
    return result
