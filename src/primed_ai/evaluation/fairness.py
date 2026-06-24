from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from primed_ai.probes.common import (
    read_table,
    TokenEmbeddingDataset,
    collate_tokens,
    regression_metrics,
    auroc,
    git_sha,
)
from primed_ai.probes.echo_only import _ensure_tokens
from primed_ai.probes.concat_mlp import _ensure_ecg_tokens
from primed_ai.probes import cross_attn

logger = logging.getLogger(__name__)


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


def load_and_merge(cohort_path: str | Path, echo_path: str | Path, ecg_path: str | Path) -> pd.DataFrame:
    coh = read_table(cohort_path)
    echo = read_table(echo_path)
    ecg = read_table(ecg_path)

    echo_key = "echo_study_id" if "echo_study_id" in coh.columns else "subject_id"
    ecg_key = "ecg_record_id"
    if echo_key not in echo.columns:
        logger.debug("Renaming first echo embedding column -> %s", echo_key)
        echo = echo.rename(columns={echo.columns[0]: echo_key})
    if ecg_key not in ecg.columns:
        logger.debug("Renaming first ecg embedding column -> %s", ecg_key)
        ecg = ecg.rename(columns={ecg.columns[0]: ecg_key})

    df = coh.merge(echo, on=echo_key, how="inner").merge(ecg, on=ecg_key, how="inner")
    return df


def prepare_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df2 = _ensure_tokens(df)
    df2 = _ensure_ecg_tokens(df2)
    return df2


def load_model(checkpoint: str | Path, embed_dim: int, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = cross_attn.CrossAttnFusedProbe(embed_dim).to(device)
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


def predict_on_df(model: torch.nn.Module, df: pd.DataFrame, embed_dim: int, batch_size: int = 64, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = TokenEmbeddingDataset(df, ecg_col="ecg_tokens")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_tokens(b, pad_dim=embed_dim))
    preds, ys, ef = [], [], []
    with torch.no_grad():
        for batch in loader:
            try:
                p = model(batch["echo"].to(device), batch["ecg"].to(device)).cpu().numpy()
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


def compute_stratified_results(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, ef_flags: np.ndarray, strata: Sequence[str] = ("sex", "age_band", "race")) -> dict:
    results = {"overall": _compute_stratum_metrics(y_true, y_pred, ef_flags), "by": {}}
    for s in strata:
        if s not in df.columns:
            logger.warning("Stratum column %s not in DataFrame; filling 'unknown'", s)
            df[s] = "unknown"
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
    pd.DataFrame(rows, columns=["Stratum Type", "Stratum Value", "N", "MAE", "EF<=40 AUROC"]).to_csv(out / "fairness_summary.csv", index=False)

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
        race_data = [(r, results["by"]["race"][r]["ef40_auroc"]) for r in race_keys if results["by"]["race"][r]["ef40_auroc"] is not None]
        if race_data:
            races, aurocs = zip(*race_data)
            axes[2].bar(races, aurocs, alpha=0.7, color="coral")
            axes[2].axhline(results["overall"]["ef40_auroc"], color="red", linestyle="--", label="Overall")
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


def run_fairness(cohort_path, echo_path, ecg_path, checkpoint, out_dir="results/fairness", embed_dim=16, batch_size=64, device=None):
    logging = __import__("logging")
    logging.basicConfig(level=logging.INFO)
    df = load_and_merge(cohort_path, echo_path, ecg_path)
    df = prepare_tokens(df)
    test_df = df[df.get("split") == "test"].reset_index(drop=True)
    if test_df.empty:
        raise RuntimeError("test split is empty; ensure 'split' column contains 'test' partition")
    model, device = load_model(checkpoint, embed_dim, device)
    y_pred, y_true, ef_flags = predict_on_df(model, test_df, embed_dim, batch_size, device)
    results = {
        "task": "E03_fairness_stratification",
        "checkpoint": str(checkpoint),
        "git_sha": git_sha(),
        "n_test": int(len(test_df)),
    }
    strat = compute_stratified_results(test_df, y_true, y_pred, ef_flags)
    results.update(strat)
    save_outputs(results, out_dir)
    return results
