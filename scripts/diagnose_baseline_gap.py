#!/usr/bin/env python3
"""E10 (#67): in-cohort fused vs solo comparison on one shared test frame.

The published baselines (EchoJEPA 5.97 MAE, ECG-FM 0.929 AUROC) come from different
cohorts, so they cannot say whether fusion itself is broken. This scores the four
trained probes on the identical 245-row test frame (non-finite rows dropped the same
way the fused eval drops them), with bootstrap CIs and paired deltas, plus an LVEF
label-source error breakdown and a fused overfit check.

Example:
  python scripts/diagnose_baseline_gap.py \
    --manifest data/processed/echo_hubert_manifest.parquet \
    --probes-dir probes --out results/baseline_gap.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from primed_ai.probes import manifest as manifest_io
from primed_ai.probes.common import auroc, git_sha, regression_metrics, save_results
from primed_ai.probes.concat_mlp import ConcatMLPProbe
from primed_ai.probes.concat_mlp import _predict as predict_concat
from primed_ai.probes.cross_attn import (
    CrossAttnFusedProbe,
    _condition_arrays,
    fused_probe_loader,
    prepare_fused_probe_data,
)
from primed_ai.probes.echo_only import EchoOnlyProbe
from primed_ai.probes.echo_only import _predict as predict_echo


def _ci95(a) -> list:
    return [round(float(np.quantile(a, 0.025)), 4), round(float(np.quantile(a, 0.975)), 4)]


def _metrics(arrays: dict) -> dict:
    m = regression_metrics(arrays["lvef"], arrays["prediction"])
    m["ef40_auroc"] = round(auroc(arrays["ef_le_40"], -np.asarray(arrays["prediction"])), 4)
    return m


def _bootstrap(arrays: dict, *, n_bootstrap: int, seed: int) -> dict:
    y = np.asarray(arrays["lvef"], dtype=np.float64)
    p = np.asarray(arrays["prediction"], dtype=np.float64)
    ef = np.asarray(arrays["ef_le_40"], dtype=bool)
    rng = np.random.default_rng(seed)
    maes, aucs = [], []
    for _ in range(n_bootstrap):
        i = rng.integers(0, len(y), size=len(y))
        maes.append(np.abs(y[i] - p[i]).mean())
        a = auroc(ef[i], -p[i])
        if np.isfinite(a):
            aucs.append(a)
    return {"mae_ci95": _ci95(maes), "ef40_auroc_ci95": _ci95(aucs)}


def _paired_delta(fused: dict, other: dict, *, n_bootstrap: int, seed: int) -> dict:
    """Paired bootstrap of (fused - other) on the shared rows; negative MAE delta = fusion wins."""
    y = np.asarray(fused["lvef"], dtype=np.float64)
    pf = np.asarray(fused["prediction"], dtype=np.float64)
    po = np.asarray(other["prediction"], dtype=np.float64)
    ef = np.asarray(fused["ef_le_40"], dtype=bool)
    rng = np.random.default_rng(seed)
    d_mae, d_auc = [], []
    for _ in range(n_bootstrap):
        i = rng.integers(0, len(y), size=len(y))
        d_mae.append(np.abs(y[i] - pf[i]).mean() - np.abs(y[i] - po[i]).mean())
        af, ao = auroc(ef[i], -pf[i]), auroc(ef[i], -po[i])
        if np.isfinite(af) and np.isfinite(ao):
            d_auc.append(af - ao)
    return {
        "delta_mae": round(float(np.abs(y - pf).mean() - np.abs(y - po).mean()), 4),
        "delta_mae_ci95": _ci95(d_mae),
        "delta_ef40_auroc": round(auroc(ef, -pf) - auroc(ef, -po), 4),
        "delta_ef40_auroc_ci95": _ci95(d_auc),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default="data/processed/echo_hubert_manifest.parquet")
    ap.add_argument("--probes-dir", default="probes", help="Root holding ecg/echo/concat/fused")
    ap.add_argument("--fusion-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/baseline_gap.json")
    args = ap.parse_args()

    device = "cpu"
    probes = Path(args.probes_dir)
    parts, n_dropped = prepare_fused_probe_data(args.manifest)
    echo_dim, ecg_dim = manifest_io.dims(parts["test"])

    loaders = {
        s: fused_probe_loader(parts[s], embed_dim=echo_dim, batch_size=args.batch_size)
        for s in ("train", "val", "test")
    }

    fused = CrossAttnFusedProbe(args.fusion_dim, echo_dim=echo_dim, ecg_dim=ecg_dim).to(device)
    fused.load_state_dict(torch.load(probes / "fused" / "cross_attn_fused.pt", map_location=device))
    fused.eval()

    concat = ConcatMLPProbe(echo_dim, ecg_dim).to(device)
    concat.load_state_dict(torch.load(probes / "concat" / "concat_mlp.pt", map_location=device))
    concat.eval()

    echo = EchoOnlyProbe(echo_dim).to(device)
    echo.load_state_dict(torch.load(probes / "echo" / "echo_only.pt", map_location=device))
    echo.eval()

    ecg_bundle = joblib.load(probes / "ecg" / "ecg_only.joblib")
    X = np.vstack([np.asarray(v, dtype=np.float64) for v in parts["test"]["ecg_embedding"]])
    ecg_pred = ecg_bundle["ridge"].predict(ecg_bundle["scaler"].transform(X))

    test = parts["test"]
    arrays = {
        "fused": _condition_arrays(fused, loaders["test"], device, "full"),
        "concat": predict_concat(concat, loaders["test"], device),
        "echo_only": predict_echo(echo, loaders["test"], device),
        "ecg_only": {
            "lvef": test["lvef"].to_numpy(np.float64),
            "prediction": ecg_pred,
            "ef_le_40": test["ef_le_40"].astype(bool).to_numpy(),
        },
    }

    models = {}
    for i, (name, arr) in enumerate(arrays.items()):
        models[name] = {
            **_metrics(arr),
            **_bootstrap(arr, n_bootstrap=args.n_bootstrap, seed=args.seed + i),
        }

    deltas = {
        f"fused_vs_{name}": _paired_delta(
            arrays["fused"], arrays[name], n_bootstrap=args.n_bootstrap, seed=args.seed + 10 + i
        )
        for i, name in enumerate(("echo_only", "ecg_only", "concat"))
    }

    # Label-noise check: fused error stratified by which LVEF field supplied the label
    by_source = {}
    if "lvef_measurement" in test.columns:
        err = np.abs(
            np.asarray(arrays["fused"]["lvef"]) - np.asarray(arrays["fused"]["prediction"])
        )
        for src in sorted(test["lvef_measurement"].dropna().unique()):
            mask = (test["lvef_measurement"] == src).to_numpy()
            by_source[str(src)] = {"n": int(mask.sum()), "mae": round(float(err[mask].mean()), 4)}

    # Overfit check: the same fused checkpoint across splits
    fused_splits = {
        s: _metrics(_condition_arrays(fused, loaders[s], device, "full"))
        for s in ("train", "val", "test")
    }

    payload = {
        "task": "E10_baseline_gap_diagnosis",
        "issue": 67,
        "seed": args.seed,
        "git_sha": git_sha(),
        "manifest": str(args.manifest),
        "probes_dir": str(probes),
        "device": device,
        "n_test": len(test),
        "n_dropped_nonfinite": n_dropped,
        "n_bootstrap": args.n_bootstrap,
        "test_metrics": models,
        "paired_deltas_vs_fused": deltas,
        "fused_mae_by_lvef_source": by_source,
        "fused_metrics_by_split": fused_splits,
        "published_baselines_for_context": {
            "echojepa_solo_mae": 5.97,
            "ecg_fm_solo_auroc": 0.929,
            "note": "Different cohorts and setups; not like-for-like with this manifest.",
        },
    }
    save_results(Path(args.out), payload)

    print(f"n_test={len(test)} (dropped {n_dropped} non-finite)")
    for name, m in models.items():
        print(
            f"{name:10s} MAE {m['mae']:6.2f} {m['mae_ci95']}  AUROC {m['ef40_auroc']:.3f} {m['ef40_auroc_ci95']}"
        )
    for name, d in deltas.items():
        print(
            f"{name:20s} dMAE {d['delta_mae']:+.2f} {d['delta_mae_ci95']}  dAUROC {d['delta_ef40_auroc']:+.3f} {d['delta_ef40_auroc_ci95']}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
