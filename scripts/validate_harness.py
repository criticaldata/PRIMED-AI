"""Validate the modality-failure harness on synthetic data with known ground truth.

  python scripts/validate_harness.py --seeds 12 --out results/validation --figdir results/figures

(1) Multi-seed recovery (echo/ECG): does the harness reliably recover planted echo-dominance and
    the modality-specific silent-failure asymmetry across seeds?
(2) N-modality generality (echo/ECG/labs): does it produce a genuine 3x3 complementarity matrix?
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from primed_ai.failure import (
    analyze_modality_failure,
    make_synthetic_modalities,
    make_synthetic_multimodal,
    masked_ridge_predict_fn,
    stratified_train_mask,
)
from primed_ai.failure.plots import plot_complementarity_matrix


def _analyze(emb, lvef, ef, seed):
    train = stratified_train_mask(ef, 0.7, seed=seed + 1000)
    test = ~train
    predict_full = masked_ridge_predict_fn(emb, lvef, train)
    return analyze_modality_failure(
        {m: emb[m][test] for m in emb},
        lvef[test],
        ef[test],
        lambda present: predict_full(present)[test],
    )


def _meanstd(vals):
    return [round(st.mean(vals), 3), round(st.pstdev(vals), 3)]


def multiseed(k: int, out: str) -> dict:
    rec = []
    for s in range(k):
        emb, lvef, ef, _ = make_synthetic_multimodal(seed=s)
        d = _analyze(emb, lvef, ef, s).to_dict()
        w = d["complementarity"]["per_example_winners"]
        tot = max(sum(w.values()), 1)
        mv = d["complementarity"]["marginal_value"]
        rec.append(
            dict(
                marg_echo=mv["echo"],
                marg_ecg=mv["ecg"],
                winfrac_echo=w["echo"] / tot,
                silent_echo=d["dropout"]["drop_echo"]["silent_rate"],
                silent_ecg=d["dropout"]["drop_ecg"]["silent_rate"],
                echo_val=mv["echo"] > mv["ecg"],
                ecg_silent=d["dropout"]["drop_ecg"]["silent_rate"]
                > d["dropout"]["drop_echo"]["silent_rate"],
            )
        )
    summary = dict(
        seeds=k,
        marginal_echo=_meanstd([r["marg_echo"] for r in rec]),
        marginal_ecg=_meanstd([r["marg_ecg"] for r in rec]),
        winfrac_echo=_meanstd([r["winfrac_echo"] for r in rec]),
        silent_rate_drop_echo=_meanstd([r["silent_echo"] for r in rec]),
        silent_rate_drop_ecg=_meanstd([r["silent_ecg"] for r in rec]),
        echo_more_valuable_frac=round(sum(r["echo_val"] for r in rec) / k, 3),
        ecg_more_silent_frac=round(sum(r["ecg_silent"] for r in rec) / k, 3),
    )
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "validation_multiseed.json").write_text(json.dumps(summary, indent=2))
    print("=== multi-seed (", k, "seeds) ===")
    print(json.dumps(summary, indent=2))
    return summary


def threemodal(out: str, figdir: str) -> dict:
    emb, lvef, ef = make_synthetic_modalities({"echo": 2.0, "ecg": 0.8, "labs": 0.8}, n=750, seed=0)
    report = _analyze(emb, lvef, ef, 0)
    d = report.to_dict()
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "validation_3modal.json").write_text(json.dumps(d, indent=2))
    Path(figdir).mkdir(parents=True, exist_ok=True)
    plot_complementarity_matrix(report, Path(figdir) / "complementarity3.pdf")
    print("=== 3-modality generality ===")
    print("winners       :", d["complementarity"]["per_example_winners"])
    print("marginal value:", d["complementarity"]["marginal_value"])
    print("matrix        :", d["complementarity"]["matrix"])
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default="results/validation")
    ap.add_argument("--figdir", default="results/figures")
    a = ap.parse_args()
    multiseed(a.seeds, a.out)
    threemodal(a.out, a.figdir)


if __name__ == "__main__":
    main()
