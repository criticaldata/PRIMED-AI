"""ECG-only LVEF probe (M06).

Pooled ECG embedding (one vector per record) -> LVEF, two heads:
  * RidgeCV regression  -> continuous LVEF (MAE); EF<=40 AUROC derived from it.
  * LogisticRegressionCV -> EF<=40 gate (AUROC).

The cohort table must already carry a subject-level ``split`` column (D04) plus
``lvef`` and ``ef_le_40``. Metrics are reported on val and test against a
mean-LVEF baseline, with a seeded bootstrap CI on the test split. A checkpoint
(scaler + both heads) and a results JSON are written to ``out_dir``.

Run on **HuBERT-ECG** embeddings (``mimic-iv-ecg-ve``). The probe itself
is model-agnostic — any per-record pooled embedding table works.
"""

from __future__ import annotations

import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from primed_ai.probes import manifest

DEFAULT_ALPHAS = np.logspace(-1, 5, 25)
DEFAULT_CS = np.logspace(-3, 2, 12)


def _read(path: str | Path) -> pd.DataFrame:
    path = str(path)
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def _canon(v) -> str:
    """Record id = digits of the last '/'-segment ('files/.../s42282815/42282815' -> '42282815')."""
    return "".join(ch for ch in str(v).rsplit("/", 1)[-1] if ch.isdigit())


def load_dataset(cohort_path, embedding_path=None, *, record_col="ecg_record_id"):
    """Join cohort labels/splits to per-record embeddings; drop non-finite rows.

    Returns (df, embedding_columns, n_dropped). Matching is on the canonical
    record id, so the embedding key may be a bare id, an ``s``-prefixed id, or a
    full path.

    With ``embedding_path`` omitted, ``cohort_path`` is read as a joined manifest and
    its inline ECG vectors are flattened into ``ve*`` columns — this probe selects its
    features by prefix, so it cannot read the array column directly.
    """
    if embedding_path is None:
        flat, ve = manifest.expand(manifest.load(cohort_path), manifest.ECG_COLUMN, "ve")
        flat = flat.drop(columns=[manifest.ECHO_COLUMN], errors="ignore")
        finite = np.isfinite(flat[ve].to_numpy(np.float64)).all(axis=1)
        return flat[finite].reset_index(drop=True), ve, int((~finite).sum())

    coh = _read(cohort_path)
    emb = _read(embedding_path)

    ve = [c for c in emb.columns if c.startswith("ve")]
    if not ve:
        ve = [c for c in emb.columns if pd.api.types.is_numeric_dtype(emb[c])]
    non_numeric = [c for c in emb.columns if not pd.api.types.is_numeric_dtype(emb[c])]
    key = next(
        (c for c in (non_numeric or list(emb.columns)) if emb[c].map(_canon).str.len().gt(0).all()),
        None,
    )
    if key is None:
        raise ValueError("could not find an id/path column in the embedding table")

    emb = emb.assign(_rec=emb[key].map(_canon))
    coh = coh.assign(_rec=coh[record_col].astype("int64").astype(str))
    df = coh.merge(emb[["_rec", *ve]], on="_rec", how="left")

    finite = np.isfinite(df[ve].to_numpy(np.float64)).all(axis=1)
    n_dropped = int((~finite).sum())
    return df[finite].reset_index(drop=True), ve, n_dropped


def _git_sha() -> str:
    try:
        repo = Path(__file__).resolve().parents[3]
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _auroc(ef: np.ndarray, score: np.ndarray) -> float:
    return roc_auc_score(ef, score) if (ef.any() and (~ef).any()) else float("nan")


def _ci(a) -> list:
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def run(
    cohort_path,
    embedding_path=None,
    out_dir="probes/ecg_only",
    *,
    seed: int = 42,
    alphas=DEFAULT_ALPHAS,
    cs=DEFAULT_CS,
    n_bootstrap: int = 2000,
) -> dict:
    """Train + evaluate the ECG-only probe; write checkpoint + results JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    df, ve, n_dropped = load_dataset(cohort_path, embedding_path)

    def xy(name):
        s = df[df["split"] == name]
        return (
            s[ve].to_numpy(np.float64),
            s["lvef"].to_numpy(np.float64),
            s["ef_le_40"].astype(bool).to_numpy(),
        )

    Xtr, ytr, eftr = xy("train")
    Xva, yva, efva = xy("val")
    Xte, yte, efte = xy("test")
    if min(len(ytr), len(yva), len(yte)) == 0:
        raise ValueError("each of train/val/test must be non-empty (check the `split` column)")

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", ConvergenceWarning)
        sc = StandardScaler().fit(Xtr)
        ztr, zva, zte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
        ridge = RidgeCV(alphas=alphas).fit(ztr, ytr)
        clf = LogisticRegressionCV(
            Cs=cs, cv=5, max_iter=2000, scoring="roc_auc", random_state=seed
        ).fit(ztr, eftr)

        def metrics(z, y, ef):
            lp, pr = ridge.predict(z), clf.predict_proba(z)[:, 1]
            return {
                "baseline_mae": round(mean_absolute_error(y, np.full_like(y, ytr.mean())), 4),
                "ridge_mae": round(mean_absolute_error(y, lp), 4),
                "ef40_auroc_from_regression": round(_auroc(ef, -lp), 4),
                "ef40_auroc_logreg": round(_auroc(ef, pr), 4),
            }

        splits = {"val": metrics(zva, yva, efva), "test": metrics(zte, yte, efte)}

        # seeded bootstrap CI on test
        lp_te, pr_te = ridge.predict(zte), clf.predict_proba(zte)[:, 1]
        mae_b, areg_b, aclf_b = [], [], []
        idx = np.arange(len(yte))
        for _ in range(n_bootstrap):
            b = rng.choice(idx, len(idx), replace=True)
            mae_b.append(mean_absolute_error(yte[b], lp_te[b]))
            if efte[b].any() and (~efte[b]).any():
                areg_b.append(roc_auc_score(efte[b], -lp_te[b]))
                aclf_b.append(roc_auc_score(efte[b], pr_te[b]))

    splits["test"].update(
        {
            "ridge_mae_ci95": _ci(mae_b),
            "ef40_auroc_from_regression_ci95": _ci(areg_b),
            "ef40_auroc_logreg_ci95": _ci(aclf_b),
        }
    )

    results = {
        "task": "M06_ecg_only_probe",
        "issue": 24,
        "model_source": "HuBERT-ECG (mimic-iv-ecg-ve), pooled per-record",
        "embedding_dim": len(ve),
        "n_dropped_nonfinite": n_dropped,
        "seed": seed,
        "git_sha": _git_sha(),
        "ridge_alpha": float(ridge.alpha_),
        "logreg_C": float(clf.C_[0]),
        "lvef_train_mean": round(float(ytr.mean()), 3),
        "n": {"train": len(ytr), "val": len(yva), "test": len(yte)},
        "ef40_test_positives": int(efte.sum()),
        "splits": splits,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    dump(
        {"scaler": sc, "ridge": ridge, "logreg": clf, "embedding_dim": len(ve)},
        out / "ecg_only.joblib",
    )
    return results
