#!/usr/bin/env python3
"""ECG-FM weights smoke test (I05).

Downloads (or resolves cached) weights from Hugging Face when available, runs a
forward pass on a synthetic 12-lead batch, and writes shape / dtype / frozen-mode
checks to ``logs/ecg_fm_smoke.txt``.

Requires ``fairseq_signals`` + downloaded checkpoint for a real forward pass; falls
back to the stub encoder when deps are missing (still validates the wrapper API).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from primed_ai.encoders.ecg_fm import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HF_REPO,
    ECGFMEncoder,
    download_checkpoint,
    synthetic_ecg_batch,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=DEFAULT_HF_REPO)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--cache-dir", default=str(repo_root / "weights" / "ecg_fm"))
    ap.add_argument("--log-path", default=str(repo_root / "logs" / "ecg_fm_smoke.txt"))
    ap.add_argument("--skip-download", action="store_true",
                    help="Use stub encoder only (offline CI).")
    args = ap.parse_args()

    lines = [f"ECG-FM smoke test @ {datetime.now(timezone.utc).isoformat()}"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lines.append(f"device={device}")

    ckpt_path = None
    if not args.skip_download:
        try:
            ckpt_path = download_checkpoint(args.repo_id, args.checkpoint, args.cache_dir)
            lines.append(f"checkpoint={ckpt_path}")
        except Exception as exc:
            lines.append(f"download_failed={exc!r}")

    encoder = ECGFMEncoder(
        ckpt_path,
        use_stub=ckpt_path is None,
        frozen=True,
        device=device,
    )
    batch = synthetic_ecg_batch(device=device)
    out = encoder.encode(batch, return_tokens=True, pool="mean")

    tokens = out["tokens"]
    pooled = out.get("pooled")
    lines.extend([
        f"use_stub={encoder.use_stub}",
        f"tokens_shape={tuple(tokens.shape)}",
        f"tokens_dtype={tokens.dtype}",
        f"pooled_shape={tuple(pooled.shape) if pooled is not None else None}",
    ])
    encoder.verify_frozen()
    lines.append("frozen_check=pass")

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
