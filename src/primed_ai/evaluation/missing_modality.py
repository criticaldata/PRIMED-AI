"""Checkpoint-only missing-modality evaluation for the M09 fused probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from primed_ai.probes.common import (
    auroc,
    git_sha,
    regression_metrics,
    save_results,
    sha256_file,
)
from primed_ai.probes.cross_attn import (
    MISSING_MODALITY_CONDITIONS,
    CrossAttnFusedProbe,
    fused_probe_loader,
    predict_missing_modality,
    prepare_fused_probe_data,
)


def _bootstrap_ci(
    arrays: dict,
    *,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap MAE and EF<=40 AUROC intervals from per-example predictions.

    A resample that happens to draw a single EF class has no defined AUROC. Those
    replicates are dropped, so the AUROC interval rests on ``n_auroc_replicates``
    rather than ``n_bootstrap``; both are recorded so the gap is visible.
    """
    y = np.asarray(arrays["lvef"], dtype=np.float64)
    pred = np.asarray(arrays["prediction"], dtype=np.float64)
    ef = np.asarray(arrays["ef_le_40"], dtype=bool)
    if n_bootstrap <= 0:
        return {}

    rng = np.random.default_rng(seed)
    n = len(y)
    mae_samples = []
    auroc_samples = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        mae_samples.append(regression_metrics(y[idx], pred[idx])["mae"])
        auc = auroc(ef[idx], -pred[idx])
        if np.isfinite(auc):
            auroc_samples.append(auc)

    lo, hi = alpha / 2, 1 - alpha / 2
    ci = {
        "n_bootstrap": n_bootstrap,
        "mae_ci_low": round(float(np.quantile(mae_samples, lo)), 4),
        "mae_ci_high": round(float(np.quantile(mae_samples, hi)), 4),
        "n_auroc_replicates": len(auroc_samples),
        "n_auroc_discarded": n_bootstrap - len(auroc_samples),
    }
    if auroc_samples:
        ci["ef40_auroc_ci_low"] = round(float(np.quantile(auroc_samples, lo)), 4)
        ci["ef40_auroc_ci_high"] = round(float(np.quantile(auroc_samples, hi)), 4)
    return ci


def _serializable_predictions(arrays: dict) -> dict:
    return {
        "lvef": np.asarray(arrays["lvef"], dtype=float).round(4).tolist(),
        "prediction": np.asarray(arrays["prediction"], dtype=float).round(4).tolist(),
        "ef_le_40": np.asarray(arrays["ef_le_40"], dtype=bool).tolist(),
    }


def _metrics_from_predictions(arrays: dict) -> dict:
    metrics = regression_metrics(arrays["lvef"], arrays["prediction"])
    metrics["ef40_auroc"] = round(auroc(arrays["ef_le_40"], -arrays["prediction"]), 4)
    return metrics


def run(
    cohort_path,
    echo_embedding_path=None,
    ecg_embedding_path=None,
    checkpoint_path=None,
    output_path="results/missing_modality.json",
    *,
    embed_dim: int = 16,
    echo_dim: int | None = None,
    ecg_dim: int | None = None,
    hidden: int = 256,
    batch_size: int = 64,
    seed: int = 42,
    n_bootstrap: int = 1000,
    device: str | None = None,
) -> dict:
    """Load one full-modality M09 checkpoint and evaluate held-out test conditions.

    Omit both embedding paths to read ``cohort_path`` as a joined manifest, matching how
    the probes are trained -- E07 scores the checkpoint M10 produced, off the same table.
    """
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    echo_dim = echo_dim or embed_dim
    ecg_dim = ecg_dim or embed_dim
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"M09 checkpoint not found: {checkpoint}")

    parts, n_dropped = prepare_fused_probe_data(
        cohort_path, echo_embedding_path, ecg_embedding_path
    )
    val_loader = fused_probe_loader(parts["val"], embed_dim=echo_dim, batch_size=batch_size)
    test_loader = fused_probe_loader(parts["test"], embed_dim=echo_dim, batch_size=batch_size)

    model = CrossAttnFusedProbe(
        embed_dim=embed_dim,
        hidden=hidden,
        echo_dim=echo_dim,
        ecg_dim=ecg_dim,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)

    # Val predictions ride along so E09 can fit its Platt scaler on val without rescoring.
    val_predictions = predict_missing_modality(model, val_loader, device)
    test_predictions = predict_missing_modality(model, test_loader, device)
    test_metrics = {
        condition: _metrics_from_predictions(arrays)
        for condition, arrays in test_predictions.items()
    }
    bootstrap = {
        condition: _bootstrap_ci(
            arrays,
            n_bootstrap=n_bootstrap,
            seed=seed + idx,
        )
        for idx, (condition, arrays) in enumerate(test_predictions.items())
    }
    payload = {
        "task": "E01_missing_modality_evaluation",
        "issue": 7,
        "source_task": "M09_cross_attn_fused_probe",
        "seed": seed,
        "git_sha": git_sha(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(cohort_path),
        "config": {
            "cohort_path": str(cohort_path),
            "echo_embedding_path": str(echo_embedding_path),
            "ecg_embedding_path": str(ecg_embedding_path),
            "fusion_dim": embed_dim,
            "echo_dim": echo_dim,
            "ecg_dim": ecg_dim,
            "hidden": hidden,
            "batch_size": batch_size,
            "n_bootstrap": n_bootstrap,
            "device": device,
        },
        "n": {"val": len(parts["val"]), "test": len(parts["test"])},
        "test": test_metrics,
        "bootstrap": bootstrap,
        "predictions": {
            condition: _serializable_predictions(arrays)
            for condition, arrays in test_predictions.items()
        },
        "predictions_val": {
            condition: _serializable_predictions(arrays)
            for condition, arrays in val_predictions.items()
        },
        "metrics_table": [
            {
                "condition": condition,
                "mae": metrics["mae"],
                "ef40_auroc": metrics["ef40_auroc"],
                "baseline_mae": metrics["baseline_mae"],
                **bootstrap.get(condition, {}),
            }
            for condition, metrics in test_metrics.items()
            if condition in MISSING_MODALITY_CONDITIONS
        ],
        "notes": "Dropped conditions mask one branch at inference only; no unimodal retraining.",
    }
    save_results(Path(output_path), payload)
    return payload
