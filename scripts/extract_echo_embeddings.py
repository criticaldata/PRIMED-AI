#!/usr/bin/env python3
"""Extract EchoJEPA embeddings for the paired cohort into M05 cache (M02).

Reads ``cohort/paired.parquet`` (or CSV), resolves echo MP4 paths, runs the frozen
EchoJEPA-L encoder, and writes one ``.npy`` per ``echo_study_id`` under
``embeddings/echo/``. Skips IDs already present (resume-safe).

Example (ORCD):
  python scripts/extract_echo_embeddings.py \\
      --cohort cohort/paired.parquet \\
      --mp4-root /orcd/pool/006/lceli_shared/mimic-iv-echo-mp4 \\
      --encoder-config configs/encoder/echojepa.yaml
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from primed_ai.embeddings.cache import EmbeddingCache
from primed_ai.encoders.echojepa import EchoJEPALEncoder, build_video_transform

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("extract_echo")


def _read_cohort(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _resolve_mp4(row, mp4_root: Path) -> Path | None:
    study = str(row.get("echo_study_id", ""))
    # Common ORCD layout: pXX/pSUBJECT/sSTUDY/STUDY.mp4 — search shallowly if needed.
    candidates = list(mp4_root.rglob(f"*{study}*.mp4"))
    if candidates:
        return candidates[0]
    n_dicom = int(row.get("n_dicom_files", 0) or 0)
    if n_dicom == 0:
        return None
    return None


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default=str(repo_root / "cohort" / "paired.parquet"))
    ap.add_argument("--mp4-root", required=True, help="Root directory of echo MP4 files.")
    ap.add_argument("--cache-root", default=str(repo_root / "embeddings"))
    ap.add_argument("--encoder-config", default=str(repo_root / "configs" / "encoder" / "echojepa.yaml"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Max studies (0 = all).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.encoder_config).read_text())
    encoder_version = Path(cfg["checkpoint_path"]).name
    cache = EmbeddingCache(args.cache_root, "echo", encoder_version=encoder_version)
    cohort = _read_cohort(Path(args.cohort))
    mp4_root = Path(args.mp4_root)

    device = args.device
    encoder = EchoJEPALEncoder(
        cfg["checkpoint_path"],
        embed_dim=int(cfg.get("embed_dim", 1024)),
        img_size=int(cfg.get("img_size", 256)),
        num_frames=int(cfg.get("num_frames", 16)),
        frozen=True,
        device=device,
    )
    transform = build_video_transform(int(cfg.get("img_size", 256)))

    pending = []
    for _, row in cohort.iterrows():
        sid = row["echo_study_id"]
        if cache.has(sid):
            continue
        mp4 = _resolve_mp4(row, mp4_root)
        if mp4 is None:
            log.warning("missing MP4 for echo_study_id=%s", sid)
            continue
        pending.append((sid, mp4))
        if args.limit and len(pending) >= args.limit:
            break

    log.info("Encoding %d echo studies (skipped %d already cached)",
             len(pending), len(cohort) - len(pending))
    t0 = time.time()
    ok = err = 0
    for start in range(0, len(pending), args.batch_size):
        chunk = pending[start:start + args.batch_size]
        tensors, ids = [], []
        for sid, mp4 in chunk:
            try:
                import numpy as np
                from decord import VideoReader, cpu

                vr = VideoReader(str(mp4), num_threads=1, ctx=cpu(0))
                idx = np.linspace(0, len(vr) - 1, num=int(cfg.get("num_frames", 16)), dtype=int)
                video = vr.get_batch(idx).asnumpy()
                tensor = torch.from_numpy(video).permute(0, 3, 1, 2)
                tensor = transform(tensor).unsqueeze(0)
                tensors.append(tensor)
                ids.append(sid)
            except Exception as exc:
                err += 1
                log.warning("load failed echo_study_id=%s: %s", sid, exc)
        if not tensors:
            continue
        batch = torch.stack(tensors).squeeze(1).to(device)
        with torch.inference_mode():
            pooled = encoder.encode(batch, return_tokens=False, pool="mean")["pooled"]
        for sid, vec in zip(ids, pooled.cpu().numpy()):
            cache.write(sid, vec)
            ok += 1
        if ok and ok % 50 == 0:
            rate = ok / max(time.time() - t0, 1e-6)
            log.info("progress ok=%d err=%d rate=%.2f studies/s", ok, err, rate)

    elapsed = time.time() - t0
    log.info("Done: ok=%d err=%d elapsed=%.1fs cache=%s", ok, err, elapsed, cache.dir)


if __name__ == "__main__":
    main()
