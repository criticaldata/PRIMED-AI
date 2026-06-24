"""Synthetic data with *planted* modality structure, for validating the harness.

A standard way to validate an evaluation method is to run it on data whose ground-truth
structure is known and check that it is recovered. ``make_synthetic_multimodal`` plants:
  - a globally **echo-dominant** signal (echo encodes LVEF strongly, ECG weakly), and
  - an **ECG-only subset** where echo is uninformative so ECG must "win",
so the harness should report echo dominance overall, ECG wins on that subset, and silent
failures when the dominant modality (echo) is dropped. No real PHI involved.
"""
from __future__ import annotations

import numpy as np


def make_synthetic_multimodal(
    n: int = 500,
    dim: int = 16,
    seed: int = 0,
    echo_strength: float = 2.0,
    ecg_strength: float = 0.5,
    ecg_only_frac: float = 0.18,
):
    """Return (embeddings, lvef, ef_le_40, planted) with known modality structure."""
    rng = np.random.default_rng(seed)
    lvef = rng.uniform(15.0, 75.0, size=n)
    signal = (lvef - 45.0) / 15.0  # standardized LVEF signal

    echo_dir = rng.standard_normal(dim)
    echo_dir /= np.linalg.norm(echo_dir)
    ecg_dir = rng.standard_normal(dim)
    ecg_dir /= np.linalg.norm(ecg_dir)

    echo = rng.standard_normal((n, dim))
    ecg = rng.standard_normal((n, dim))

    # echo carries the signal strongly for everyone...
    echo += echo_strength * np.outer(signal, echo_dir)
    # ...ECG carries it weakly for everyone...
    ecg += ecg_strength * np.outer(signal, ecg_dir)

    # ...except an ECG-only subset: echo signal removed, ECG signal boosted (ECG must win here)
    ecg_only = rng.random(n) < ecg_only_frac
    echo[ecg_only] = rng.standard_normal((int(ecg_only.sum()), dim))  # echo = pure noise here
    ecg[ecg_only] += 1.8 * np.outer(signal[ecg_only], ecg_dir)

    embeddings = {"echo": echo.astype(np.float32), "ecg": ecg.astype(np.float32)}
    ef_le_40 = lvef <= 40.0
    planted = {"ecg_only_idx": np.where(ecg_only)[0], "echo_dominant_globally": True}
    return embeddings, lvef, ef_le_40, planted


def masked_ridge_predict_fn(embeddings: dict, lvef, train_mask, alpha: float = 1.0):
    """Train one Ridge model on the full concatenated embeddings; mask absent modalities at
    inference by zeroing their columns. This mimics a *single deployed multimodal model*
    evaluated under missing inputs (no per-condition retraining), which is exactly what the
    dropout analysis assumes.
    """
    from sklearn.linear_model import Ridge

    mods = list(embeddings)
    cols, off = {}, 0
    for m in mods:
        d = embeddings[m].shape[1]
        cols[m] = slice(off, off + d)
        off += d
    X = np.concatenate([np.asarray(embeddings[m], dtype=float) for m in mods], axis=1)
    y = np.asarray(lvef, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)
    model = Ridge(alpha=alpha).fit(X[train_mask], y[train_mask])

    def predict_fn(present: frozenset) -> np.ndarray:
        Xm = X.copy()
        for m in mods:
            if m not in present:
                Xm[:, cols[m]] = 0.0
        return model.predict(Xm)

    return predict_fn


def make_synthetic_modalities(strengths: dict, n: int = 600, dim: int = 16, seed: int = 0,
                              exclusive_frac: float = 0.12):
    """N-modality synthetic data (generality demo). Each modality ``m`` encodes LVEF with its own
    ``strengths[m]`` along a random direction; an ``exclusive_frac`` slice is split across modalities
    so that, on each sub-slice, only one modality is informative (the others are noise)---planting
    both per-modality dominance and a complementary subset that the harness should recover.
    """
    rng = np.random.default_rng(seed)
    lvef = rng.uniform(15.0, 75.0, size=n)
    signal = (lvef - 45.0) / 15.0
    mods = list(strengths)
    dirs = {}
    for m in mods:
        v = rng.standard_normal(dim)
        dirs[m] = v / np.linalg.norm(v)
    emb = {m: rng.standard_normal((n, dim)) + strengths[m] * np.outer(signal, dirs[m]) for m in mods}
    order = rng.permutation(n)
    per = int(exclusive_frac * n)
    for k, m in enumerate(mods):
        sel = order[k * per:(k + 1) * per]
        if len(sel) == 0:
            continue
        for other in mods:
            if other != m:
                emb[other][sel] = rng.standard_normal((len(sel), dim))
        emb[m][sel] = rng.standard_normal((len(sel), dim)) + 1.6 * np.outer(signal[sel], dirs[m])
    embeddings = {m: emb[m].astype(np.float32) for m in mods}
    return embeddings, lvef, lvef <= 40.0


def stratified_train_mask(gate, train_frac: float = 0.7, seed: int = 0) -> np.ndarray:
    """Boolean train mask stratified by the binary gate, so both classes appear in train *and* test.

    Splitting must be independent of data generation: a seed shared between the two can correlate the
    split with the label and drive test-set prevalence to zero, silently degrading the gate.
    """
    gate = np.asarray(gate, dtype=bool)
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(gate), dtype=bool)
    for cls in (False, True):
        idx = np.where(gate == cls)[0]
        rng.shuffle(idx)
        mask[idx[: int(round(train_frac * len(idx)))]] = True
    return mask
