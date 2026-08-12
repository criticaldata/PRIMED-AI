from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from primed_ai.probes import cross_attn
from primed_ai.probes.common import (
    TokenEmbeddingDataset,
    auroc,
    collate_tokens,
    git_sha,
    regression_metrics,
)
from primed_ai.probes.cross_attn import (
    MISSING_MODALITY_CONDITIONS,
    _condition_arrays,
    fused_probe_loader,
    prepare_fused_probe_data,
)

logger = logging.getLogger(__name__)

MIMIC_BIAS_NOTE = (
    "MIMIC-IV records administrative gender as 'sex' and admission-reported race; both "
    "carry known curation bias (see MIMIC-IV documentation), so strata inherit it."
)


def _safe_json(obj):
    # Convert NaN/inf to None recursively for JSON compatibility
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj


def load_model(
    checkpoint: str | Path,
    embed_dim: int,
    device: str | None = None,
    echo_dim: int | None = None,
    ecg_dim: int | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = cross_attn.CrossAttnFusedProbe(embed_dim, echo_dim=echo_dim, ecg_dim=ecg_dim).to(device)
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    try:
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint {ckpt}: {e}") from e
    model.eval()
    return model, device


def predict_on_df(
    model: torch.nn.Module,
    df: pd.DataFrame,
    embed_dim: int,
    batch_size: int = 64,
    device: str | None = None,
    echo_dim: int | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = TokenEmbeddingDataset(df, ecg_col="ecg_tokens")
    pad_dim = echo_dim or embed_dim
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_tokens(b, pad_dim=pad_dim),
    )
    preds, ys, ef = [], [], []
    with torch.no_grad():
        for batch in loader:
            try:
                p = (
                    model(
                        batch["echo"].to(device),
                        batch["ecg"].to(device),
                        echo_mask=batch["echo_mask"].to(device),
                    )
                    .cpu()
                    .numpy()
                )
            except ValueError as e:
                raise ValueError(
                    f"Model forward failed: {e}. Check embed_dim={embed_dim} vs token feature dim (first batch: {batch['echo'].shape})"
                ) from e
            preds.append(p)
            ys.append(batch["lvef"].numpy())
            ef.append(batch["ef_le_40"].numpy())
    y_pred = np.concatenate(preds).flatten()
    y_true = np.concatenate(ys)
    ef_flags = np.concatenate(ef).astype(bool)
    return y_pred, y_true, ef_flags


def _compute_stratum_metrics(y_true: np.ndarray, y_pred: np.ndarray, ef_flags: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"n": 0, "mae": None, "ef40_auroc": None}
    metrics = regression_metrics(y_true, y_pred)
    au = auroc(ef_flags, -y_pred) if ef_flags.any() and (~ef_flags).any() else float("nan")
    return {
        "n": int(len(y_true)),
        "mae": float(metrics["mae"]),
        "baseline_mae": float(metrics.get("baseline_mae", float("nan"))),
        "ef40_auroc": None if np.isnan(au) else float(au),
        "small_n": len(y_true) < 20,
    }


def compute_stratified_results(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ef_flags: np.ndarray,
    strata: Sequence[str] = ("sex", "age_band", "race"),
) -> dict:
    def _coarsen_race(value) -> str:
        v = str(value).upper()
        for label in ("WHITE", "BLACK", "ASIAN"):
            if v.startswith(label):
                return label
        return "HISPANIC/LATINO" if v.startswith("HISPANIC") else "OTHER/UNKNOWN"

    df = df.copy()
    if "race" in df.columns:
        df["race"] = df["race"].map(_coarsen_race)
    results = {"overall": _compute_stratum_metrics(y_true, y_pred, ef_flags), "by": {}}
    for s in strata:
        if s not in df.columns:
            logger.warning("Stratum column %s not in DataFrame; filling 'unknown'", s)
            df[s] = "unknown"
        # cohorts with missing demographics mix NaN into string columns; sorted() would choke
        df[s] = df[s].where(df[s].notna(), "unknown")
        groups = {}
        for val in sorted(df[s].unique()):
            mask = (df[s] == val).to_numpy()
            groups[str(val)] = _compute_stratum_metrics(y_true[mask], y_pred[mask], ef_flags[mask])
        results["by"][s] = groups
    return results


def save_outputs(results: dict, out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe = _safe_json(results)
    (out / "fairness_metrics.json").write_text(json.dumps(safe, indent=2), encoding="utf-8")

    # produce CSV summary
    rows = []
    ov = results["overall"]
    rows.append(["Overall", "", ov.get("n"), ov.get("mae"), ov.get("ef40_auroc")])
    for s, mapping in results["by"].items():
        for val, metrics in mapping.items():
            rows.append([s, val, metrics.get("n"), metrics.get("mae"), metrics.get("ef40_auroc")])
    pd.DataFrame(
        rows, columns=["Stratum Type", "Stratum Value", "N", "MAE", "EF<=40 AUROC"]
    ).to_csv(out / "fairness_summary.csv", index=False)

    # plotting (best-effort)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        sex_keys = sorted(results["by"].get("sex", {}).keys())
        if sex_keys:
            sex_data = [(s, results["by"]["sex"][s]["mae"]) for s in sex_keys]
            sexes, maes = zip(*sex_data)
            axes[0].bar(sexes, maes, alpha=0.7, color="steelblue")
            axes[0].axhline(results["overall"]["mae"], color="red", linestyle="--", label="Overall")
            axes[0].set_ylabel("MAE (LVEF)")
            axes[0].set_xlabel("Sex")
            axes[0].set_title("MAE by Sex")
            axes[0].legend()

        age_keys = sorted(results["by"].get("age_band", {}).keys())
        if age_keys:
            age_data = [(a, results["by"]["age_band"][a]["mae"]) for a in age_keys]
            ages, maes = zip(*age_data)
            axes[1].bar(ages, maes, alpha=0.7, color="seagreen")
            axes[1].axhline(results["overall"]["mae"], color="red", linestyle="--", label="Overall")
            axes[1].set_ylabel("MAE (LVEF)")
            axes[1].set_xlabel("Age Band")
            axes[1].set_title("MAE by Age Band")
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].legend()

        race_keys = sorted(results["by"].get("race", {}).keys())
        race_data = [
            (r, results["by"]["race"][r]["ef40_auroc"])
            for r in race_keys
            if results["by"]["race"][r]["ef40_auroc"] is not None
        ]
        if race_data:
            races, aurocs = zip(*race_data)
            axes[2].bar(races, aurocs, alpha=0.7, color="coral")
            axes[2].axhline(
                results["overall"]["ef40_auroc"], color="red", linestyle="--", label="Overall"
            )
            axes[2].set_ylabel("AUROC")
            axes[2].set_xlabel("Race")
            axes[2].set_title("EF≤40% AUROC by Race")
            axes[2].set_ylim([0, 1])
            axes[2].tick_params(axis="x", rotation=45)
            axes[2].legend()

        plt.tight_layout()
        plot_path = out / "fairness_stratification.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning("Plotting failed: %s", e)


def run_fairness(
    cohort_path,
    echo_path=None,
    ecg_path=None,
    checkpoint=None,
    out_dir="results/fairness",
    embed_dim=16,
    echo_dim=None,
    ecg_dim=None,
    batch_size=64,
    device=None,
    conditions=("full",),
):
    """Stratify test predictions by demographics, per missing-modality condition.

    Omit both embedding paths to read ``cohort_path`` as a joined manifest (matching the
    probes and the missing-modality eval). Either way the data goes through
    ``prepare_fused_probe_data``, so non-finite rows are dropped the same way and the
    stratified n lines up with the canonical rerun.
    """
    if checkpoint is None:
        raise ValueError("checkpoint is required")
    unknown = set(conditions) - set(MISSING_MODALITY_CONDITIONS)
    if unknown:
        raise ValueError(
            f"unknown conditions {sorted(unknown)}; pick from {MISSING_MODALITY_CONDITIONS}"
        )

    parts, n_dropped = prepare_fused_probe_data(cohort_path, echo_path, ecg_path)
    test_df = parts["test"]

    model, device = load_model(checkpoint, embed_dim, device, echo_dim=echo_dim, ecg_dim=ecg_dim)
    loader = fused_probe_loader(test_df, embed_dim=echo_dim or embed_dim, batch_size=batch_size)
    results = {
        "task": "E03_fairness_stratification",
        "checkpoint": str(checkpoint),
        "git_sha": git_sha(),
        "n_test": int(len(test_df)),
        # aggregate count across train/val/test, not test-only — mirrors the run metadata
        "n_dropped_nonfinite_all_splits": n_dropped,
        "bias_note": MIMIC_BIAS_NOTE,
        "conditions": {},
    }
    out = Path(out_dir)
    provenance = dict(results)
    del provenance["conditions"]
    for condition in conditions:
        arrays = _condition_arrays(model, loader, device, condition)
        strat = compute_stratified_results(
            test_df,
            np.asarray(arrays["lvef"]),
            np.asarray(arrays["prediction"]).flatten(),
            np.asarray(arrays["ef_le_40"], dtype=bool),
        )
        results["conditions"][condition] = strat
        save_outputs({**provenance, "condition": condition, **strat}, out / condition)
    # keep the full-condition metrics at top level too: aggregate.py reads overall/by there
    results.update(results["conditions"].get("full", {}))
    out.mkdir(parents=True, exist_ok=True)
    (out / "fairness_metrics.json").write_text(
        json.dumps(_safe_json(results), indent=2), encoding="utf-8"
    )
    return results
