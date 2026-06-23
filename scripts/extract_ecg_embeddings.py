#!/usr/bin/env python3
"""Extract ECG-FM embeddings for the paired cohort into M05 cache (M04).

Maps ``ecg_record_id`` / ``ecg_path`` to waveform files, runs the frozen ECG-FM
encoder (or stub when weights unavailable), and writes ``embeddings/ecg/<id>.npy``.

Example:
  python scripts/extract_ecg_embeddings.py \\
      --cohort cohort/paired.parquet \\
      --ecg-root /path/to/mimic-iv-ecg/files \\
      --encoder-config configs/encoder/ecg_fm.yaml
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from primed_ai.embeddings.cache import EmbeddingCache
from primed_ai.encoders.ecg_fm import ECGFMEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("extract_ecg")


def _read_cohort(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _load_waveform(path: Path, sample_rate: int = 500, length: int = 5000) -> np.ndarray | None:
    """Load a 12-lead waveform via wfdb when available; else return None."""
    try:
        import wfdb
    except ImportError:
        return None
    try:
        record = wfdb.rdrecord(str(path.with_suffix("")))
        sig = record.p_signal.T  # [leads, samples]
        if sig.shape[0] != 12:
            return None
        # Simple crop/pad to fixed length.
        if sig.shape[1] >= length:
            sig = sig[:, :length]
        else:
            pad = np.zeros((12, length - sig.shape[1]), dtype=sig.dtype)
            sig = np.concatenate([sig, pad], axis=1)
        return sig.astype(np.float32)
    except Exception:
        return None


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default=str(repo_root / "cohort" / "paired.parquet"))
    ap.add_argument("--ecg-root", help="MIMIC-IV-ECG files root (optional if ecg_path is absolute).")
    ap.add_argument("--cache-root", default=str(repo_root / "embeddings"))
    ap.add_argument("--encoder-config", default=str(repo_root / "configs" / "encoder" / "ecg_fm.yaml"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use-stub", action="store_true", help="Force stub encoder (offline dev).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.encoder_config).read_text())
    ckpt = cfg.get("checkpoint_path")
    encoder_version = Path(ckpt).name if ckpt and not args.use_stub else "stub"
    cache = EmbeddingCache(args.cache_root, "ecg", encoder_version=encoder_version)
    cohort = _read_cohort(Path(args.cohort))
    ecg_root = Path(args.ecg_root) if args.ecg_root else None

    encoder = ECGFMEncoder(
        None if args.use_stub else ckpt,
        embed_dim=int(cfg.get("embed_dim", 768)),
        frozen=True,
        use_stub=args.use_stub or not ckpt,
        device=args.device,
    )

    pending = []
    for _, row in cohort.iterrows():
        rid = row["ecg_record_id"]
        if cache.has(rid):
            continue
        rel = row.get("ecg_path") or row.get("ecg_file_name")
        if rel is None:
            continue
        path = Path(rel) if Path(str(rel)).is_absolute() else (ecg_root / rel)
        pending.append((rid, path))
        if args.limit and len(pending) >= args.limit:
            break

    log.info("Encoding %d ECG records", len(pending))
    t0 = time.time()
    ok = err = 0
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start:start + args.batch_size]
        waves, ids = [], []
        for rid, path in chunk:
            wave = _load_waveform(path)
            if wave is None:
                err += 1
                log.warning("skip ecg_record_id=%s path=%s", rid, path)
                continue
            waves.append(wave)
            ids.append(rid)
        if not waves:
            continue
        batch = torch.as_tensor(np.stack(waves), device=args.device)
        with torch.inference_mode():
            pooled = encoder.encode(batch, return_tokens=False, pool="mean")["pooled"]
        for rid, vec in zip(ids, pooled.cpu().numpy()):
            cache.write(rid, vec)
            ok += 1

    log.info("Done: ok=%d err=%d elapsed=%.1fs", ok, err, time.time() - t0)


if __name__ == "__main__":
    main()
