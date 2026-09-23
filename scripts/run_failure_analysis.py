"""Run the modality-failure harness: failure taxonomy, complementarity matrix, dropout profile.

Demo (synthetic data with planted echo-dominant + ECG-only structure):
  python scripts/run_failure_analysis.py --demo --out results/failure_demo

Real cached embeddings (Ridge probe; uses the cohort `split` column for train/test):
  python scripts/run_failure_analysis.py \
    --cohort data/raw/cohort/paired_with_splits.parquet \
    --echo-embeddings data/interim/echo_study_embeddings_vjepa2.1-vitl-mimic-pt-100.parquet \
    --ecg-embeddings  data/interim/hubert_ecg_embeddings.parquet \
    --out results/failure

Joined manifest (same table the probes train from; trains on `split == "train"`,
reports on `split == "test"`):
  python scripts/run_failure_analysis.py \
    --manifest data/processed/echo_hubert_manifest.parquet --out results/failure

Fused checkpoint (E14/#79 -- the three views for the model the paper actually reports,
read straight off the per-example dump `evaluate_missing_modality.py` already writes):
  python scripts/run_failure_analysis.py \
    --predictions results/missing_modality.json --out results/failure_fused
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from primed_ai.failure import (
    analyze_modality_failure,
    make_synthetic_multimodal,
    masked_ridge_predict_fn,
    stratified_train_mask,
)
from primed_ai.failure.plots import (
    plot_complementarity_matrix,
    plot_dropout_profile,
    plot_taxonomy,
)

RIDGE_MODEL = "masked_ridge_harness"
RIDGE_NOTE = (
    "Predictions come from a Ridge fitted on the concatenated embeddings with absent "
    "modalities zeroed at inference -- NOT from the cross-attention fused checkpoint the "
    "README results and the missing-modality eval report. The two disagree substantially "
    "on the dropped conditions; do not mix them in one table."
)

FUSED_MODEL = "cross_attn_fused_checkpoint"

# Which modalities are present under each masking condition the fused eval scores. With two
# modalities these three subsets are everything the harness asks predict_fn for (full, each
# singleton), so no refitting is needed -- the dump already holds every prediction.
CONDITION_PRESENT = {
    "full": frozenset({"echo", "ecg"}),
    "echo_dropped": frozenset({"ecg"}),
    "ecg_dropped": frozenset({"echo"}),
}


def _emit(report, out: str, provenance: dict) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = {**report.to_dict(), "provenance": provenance}
    (out_dir / "failure_report.json").write_text(json.dumps(d, indent=2))
    plot_complementarity_matrix(report, out_dir / "complementarity.pdf")
    plot_dropout_profile(report, out_dir / "dropout_profile.pdf")
    plot_taxonomy(report, out_dir / "taxonomy.pdf")
    print("modalities          :", d["modalities"], f"(n_test={d['n']})")
    print("per-example winners :", d["complementarity"]["per_example_winners"])
    print("marginal value (LOO):", d["complementarity"]["marginal_value"])
    print(
        "dropout profile     :",
        {
            k: {kk: v[kk] for kk in ("induced_critical", "silent", "silent_rate")}
            for k, v in d["dropout"].items()
        },
    )
    print("producing model     :", provenance["model"])
    print("wrote               :", out_dir)


def _run_demo(seed: int, out: str) -> None:
    emb, lvef, ef, _ = make_synthetic_multimodal(seed=seed)
    train = stratified_train_mask(ef, 0.7, seed=seed + 12345)
    test = ~train
    predict_full = masked_ridge_predict_fn(emb, lvef, train)
    emb_te = {m: emb[m][test] for m in emb}
    report = analyze_modality_failure(
        emb_te, lvef[test], ef[test], lambda present: predict_full(present)[test]
    )
    _emit(report, out, {"model": RIDGE_MODEL, "data": "synthetic_planted", "seed": seed})


def _run_predictions(predictions_path: str, out: str) -> None:
    """Failure views for the fused checkpoint, off the per-example predictions it already wrote.

    The ridge routes fit their own model, so their taxonomy/complementarity/dropout numbers
    describe something the paper never reports. Here nothing is fitted: the conditions in
    `missing_modality.json` are exactly the modality subsets the harness needs.
    """
    payload = json.loads(Path(predictions_path).read_text())
    dump = payload.get("predictions")
    if not dump:
        raise SystemExit(
            f"{predictions_path} has no per-example 'predictions' block -- point this at the "
            "raw results/missing_modality.json, not the sanitized docs/results copy"
        )
    missing = set(CONDITION_PRESENT) - set(dump)
    if missing:
        raise SystemExit(f"predictions are missing condition(s): {sorted(missing)}")

    preds = {
        CONDITION_PRESENT[cond]: np.asarray(dump[cond]["prediction"], dtype=float)
        for cond in CONDITION_PRESENT
    }
    lvef = np.asarray(dump["full"]["lvef"], dtype=float)
    ef = np.asarray(dump["full"]["ef_le_40"], dtype=bool)
    # All three conditions score the same rows in the same order; if they ever stop doing so
    # the per-example views would silently compare different patients.
    for cond, arrays in dump.items():
        if cond in CONDITION_PRESENT and not np.array_equal(
            np.asarray(arrays["lvef"], dtype=float), lvef
        ):
            raise SystemExit(f"condition '{cond}' has different labels than 'full'")

    def predict(present):
        if present not in preds:
            # KeyError, not SystemExit: the harness degrades gracefully when an optional
            # subset (the all-masked one Shapley probes) has no scored condition behind it.
            raise KeyError(f"no scored condition covers the modality subset {sorted(present)}")
        return preds[present]

    try:
        report = analyze_modality_failure(["echo", "ecg"], lvef, ef, predict)
    except KeyError as e:
        raise SystemExit(e.args[0]) from None
    _emit(
        report,
        out,
        {
            "model": FUSED_MODEL,
            "predictions_path": str(predictions_path),
            "checkpoint": payload.get("checkpoint"),
            "checkpoint_sha256": payload.get("checkpoint_sha256"),
            "manifest_sha256": payload.get("manifest_sha256"),
            "seed": payload.get("seed"),
            "source_git_sha": payload.get("git_sha"),
            "note": (
                "Same predictions, checkpoint and test rows as the reported missing-modality "
                "table -- no model is fitted here."
            ),
        },
    )


def _run_manifest(manifest_path: str, out: str) -> None:
    from primed_ai.probes import manifest as manifest_io
    from primed_ai.probes.common import drop_non_finite, git_sha, sha256_file

    df = manifest_io.load(manifest_path)
    if manifest_io.is_clip_level(df):
        raise SystemExit(
            "clip-level manifest: the ridge harness works on study vectors — "
            "rebuild the manifest pooled, or mean-pool the clips first"
        )
    df, n_dropped = drop_non_finite(df, ("echo_embedding", "ecg_embedding"))
    if n_dropped:
        print(f"dropped {n_dropped} rows with non-finite values")
    embeddings = {
        "echo": np.vstack(df["echo_embedding"].to_numpy()),
        "ecg": np.vstack(df["ecg_embedding"].to_numpy()),
    }
    lvef = df["lvef"].to_numpy(float)
    ef = df["ef_le_40"].to_numpy(bool)
    train = (df["split"] == "train").to_numpy()
    test = (df["split"] == "test").to_numpy()  # strictly test; val stays out
    if not train.any() or not test.any():
        raise SystemExit("manifest needs non-empty 'train' and 'test' splits")
    groups = {a: df[a].to_numpy() for a in ("sex", "age_band", "race") if a in df}
    predict_full = masked_ridge_predict_fn(embeddings, lvef, train)
    emb_te = {m: embeddings[m][test] for m in embeddings}
    g_te = {a: v[test] for a, v in groups.items()}
    report = analyze_modality_failure(
        emb_te,
        lvef[test],
        ef[test],
        lambda present: predict_full(present)[test],
        groups=g_te or None,
    )
    _emit(
        report,
        out,
        {
            "model": RIDGE_MODEL,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "git_sha": git_sha(),
            "n_dropped_nonfinite": n_dropped,
            "note": RIDGE_NOTE,
        },
    )


def _run_real(cohort: str, echo: str, ecg: str, out: str) -> None:
    import pandas as pd

    from primed_ai.probes.common import git_sha

    coh = pd.read_parquet(cohort)
    e = pd.read_parquet(echo)
    g = pd.read_parquet(ecg)
    # merge embeddings onto cohort by their id columns (first column of each embedding table)
    coh = coh.merge(e, left_on="echo_study_id", right_on=e.columns[0], how="inner")
    coh = coh.merge(g, left_on="ecg_record_id", right_on=g.columns[0], how="inner")
    echo_cols = [c for c in e.columns[1:]]
    ecg_cols = [c for c in g.columns[1:]]
    embeddings = {"echo": coh[echo_cols].to_numpy(float), "ecg": coh[ecg_cols].to_numpy(float)}
    lvef = coh["lvef"].to_numpy(float)
    ef = coh["ef_le_40"].to_numpy(bool) if "ef_le_40" in coh else (lvef <= 40.0)
    train = (coh["split"] == "train").to_numpy() if "split" in coh else np.ones(len(coh), bool)
    test = ~train if train.any() and (~train).any() else np.ones(len(coh), bool)
    groups = {a: coh[a].to_numpy() for a in ("sex", "age_band", "race") if a in coh}
    predict_full = masked_ridge_predict_fn(embeddings, lvef, train)
    emb_te = {m: embeddings[m][test] for m in embeddings}
    g_te = {a: v[test] for a, v in groups.items()}
    report = analyze_modality_failure(
        emb_te,
        lvef[test],
        ef[test],
        lambda present: predict_full(present)[test],
        groups=g_te or None,
    )
    _emit(
        report,
        out,
        {
            "model": RIDGE_MODEL,
            "cohort": str(cohort),
            "echo_embeddings": str(echo),
            "ecg_embeddings": str(ecg),
            "git_sha": git_sha(),
            "note": RIDGE_NOTE,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--demo", action="store_true", help="run on planted synthetic data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--predictions",
        help="per-example dump from evaluate_missing_modality.py; scores the fused checkpoint",
    )
    p.add_argument("--manifest", help="joined manifest; supersedes the three-table flags")
    p.add_argument("--cohort")
    p.add_argument("--echo-embeddings")
    p.add_argument("--ecg-embeddings")
    p.add_argument(
        "--out", help="default: results/failure, or results/failure_fused for --predictions"
    )
    a = p.parse_args()
    # Separate default per producer: a fused report landing on results/failure would
    # overwrite the ridge one, which is the confusion this route exists to end.
    out = a.out or ("results/failure_fused" if a.predictions and not a.demo else "results/failure")
    if a.predictions and not a.demo:
        _run_predictions(a.predictions, out)
    elif a.manifest and not a.demo:
        _run_manifest(a.manifest, out)
    elif a.demo or not (a.cohort and a.echo_embeddings and a.ecg_embeddings):
        _run_demo(a.seed, out if a.demo else "results/failure_demo")
    else:
        _run_real(a.cohort, a.echo_embeddings, a.ecg_embeddings, out)


if __name__ == "__main__":
    main()
