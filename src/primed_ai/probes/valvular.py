"""Valvular Hemodynamic Disease Probes (Task B).

Implements multi-target classifiers for Aortic Stenosis (AS), Mitral Regurgitation (MR),
and Tricuspid Regurgitation (TR) on top of frozen foundation model embeddings (EchoJEPA + HuBERT-ECG).

Supports:
  1. ECG-only Probe (LogisticRegressionCV / MLP)
  2. Echo-only Probe (Attentive Pool / Ridge / LogReg)
  3. Multimodal Concat-MLP Probe
  4. Modality-Failure Analysis (MFA) on Valvular Gates
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("valvular_probes")

VALVULAR_TARGETS = [
    ("as_moderate_or_severe", "Aortic Stenosis (Mod/Sev)"),
    ("mr_moderate_or_severe", "Mitral Regurgitation (Mod/Sev)"),
    ("tr_moderate_or_severe", "Tricuspid Regurgitation (Mod/Sev)"),
]


def auroc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    if y_true.any() and (~y_true).any():
        return float(roc_auc_score(y_true, y_score))
    return float("nan")


def auprc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    if y_true.any() and (~y_true).any():
        return float(average_precision_score(y_true, y_score))
    return float("nan")


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[float]:
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        val = metric_fn(y_true[idx], y_score[idx])
        if not np.isnan(val):
            scores.append(val)
    if not scores:
        return [float("nan"), float("nan")]
    return [
        round(float(np.percentile(scores, 2.5)), 4),
        round(float(np.percentile(scores, 97.5)), 4),
    ]


@dataclass
class ValvularEvaluationResult:
    target: str
    target_name: str
    prevalence: float
    auroc: float
    auroc_ci95: list[float]
    auprc: float
    brier: float
    probabilities: np.ndarray

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "target_name": self.target_name,
            "prevalence": round(self.prevalence, 4),
            "auroc": round(self.auroc, 4),
            "auroc_ci95": self.auroc_ci95,
            "auprc": round(self.auprc, 4),
            "brier_score": round(self.brier, 4),
        }


class MultiTargetValvularClassifier:
    """Trains independent regularized classifiers for each valvular disease target."""

    def __init__(self, cs=None, cv: int = 5, seed: int = 42):
        self.cs = cs or np.logspace(-3, 2, 10)
        self.cv = cv
        self.seed = seed
        self.scaler = StandardScaler()
        self.models: dict[str, LogisticRegressionCV] = {}

    def fit(self, X: np.ndarray, y_dict: dict[str, np.ndarray]) -> MultiTargetValvularClassifier:
        Z = self.scaler.fit_transform(X)
        for target, name in VALVULAR_TARGETS:
            if target in y_dict:
                y = np.asarray(y_dict[target], dtype=bool)
                clf = LogisticRegressionCV(
                    Cs=self.cs,
                    cv=self.cv,
                    max_iter=1500,
                    scoring="roc_auc",
                    random_state=self.seed,
                )
                clf.fit(Z, y)
                self.models[target] = clf
        return self

    def predict_proba(self, X: np.ndarray) -> dict[str, np.ndarray]:
        Z = self.scaler.transform(X)
        out = {}
        for target, clf in self.models.items():
            out[target] = clf.predict_proba(Z)[:, 1]
        return out

    def evaluate(
        self, X: np.ndarray, y_dict: dict[str, np.ndarray], n_bootstrap: int = 1000
    ) -> dict[str, ValvularEvaluationResult]:
        probs = self.predict_proba(X)
        results = {}
        for target, name in VALVULAR_TARGETS:
            if target in y_dict and target in probs:
                y_true = np.asarray(y_dict[target], dtype=bool)
                score = probs[target]
                auc = auroc_safe(y_true, score)
                ci = bootstrap_ci(
                    y_true, score, auroc_safe, n_bootstrap=n_bootstrap, seed=self.seed
                )
                prc = auprc_safe(y_true, score)
                brier = float(brier_score_loss(y_true, score))
                results[target] = ValvularEvaluationResult(
                    target=target,
                    target_name=name,
                    prevalence=float(y_true.mean()),
                    auroc=auc,
                    auroc_ci95=ci,
                    auprc=prc,
                    brier=brier,
                    probabilities=score,
                )
        return results
