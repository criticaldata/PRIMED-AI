#!/usr/bin/env python3
"""Train the LVEF probes from the joined EchoJEPA + HuBERT manifest (M10).

Only the ECG-only probe had a CLI before this; the other three were library-only.
All four now read the single manifest written by ``build_echo_hubert_manifest.py``
instead of the older cohort + separate-embedding-table layout.

Encoder dims are read off the manifest rather than passed in — mismatched ``--embed-dim``
against a checkpoint is the failure mode called out in the README. The fused probe's
internal width is separate and stays a flag (``--fusion-dim``), since it is a capacity
choice rather than a property of the data.

Example:
  python scripts/train_probes.py --manifest data/processed/echo_hubert_manifest.parquet
  python scripts/train_probes.py --probe fused --epochs 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from primed_ai.probes import manifest as manifest_io
from primed_ai.probes import run_concat_mlp, run_cross_attn, run_ecg_only, run_echo_only
from primed_ai.probes.common import sha256_file
from primed_ai.utils.run_manifest import save_run_metadata

PROBES = ("ecg", "echo", "concat", "fused")


def _train(
    name: str, manifest_path: str, out_root: Path, *, epochs: int, seed: int, fusion_dim: int
) -> dict:
    out = str(out_root / name)
    if name == "ecg":
        return run_ecg_only(manifest_path, out_dir=out, seed=seed)

    echo_dim, ecg_dim = manifest_io.dims(manifest_io.load(manifest_path))
    if name == "echo":
        return run_echo_only(
            manifest_path, out_dir=out, embed_dim=echo_dim, epochs=epochs, seed=seed
        )
    if name == "concat":
        return run_concat_mlp(
            manifest_path,
            out_dir=out,
            echo_dim=echo_dim,
            ecg_dim=ecg_dim,
            epochs=epochs,
            seed=seed,
        )
    return run_cross_attn(
        manifest_path,
        out_dir=out,
        embed_dim=fusion_dim,
        echo_dim=echo_dim,
        ecg_dim=ecg_dim,
        epochs=epochs,
        seed=seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default="data/processed/echo_hubert_manifest.parquet")
    ap.add_argument("--probe", choices=(*PROBES, "all"), default="all")
    ap.add_argument("--out-dir", default="probes")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument(
        "--fusion-dim",
        type=int,
        default=256,
        help="Width the fused probe projects both modalities to. Defaults to the 256 the "
        "existing checkpoint used; passing the echo dim instead is an 8.7x parameter jump "
        "for no measured gain.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not Path(args.manifest).exists():
        raise SystemExit(
            f"manifest not found: {args.manifest}\n"
            "Build it first with `make reproduce-echo-hubert-local` (needs the embedding "
            "parquets and the paired cohort — see docs/echo_hubert_loader.md)."
        )

    out_root = Path(args.out_dir)
    selected = PROBES if args.probe == "all" else (args.probe,)
    summary = {}
    for name in selected:
        res = _train(
            name,
            args.manifest,
            out_root,
            epochs=args.epochs,
            seed=args.seed,
            fusion_dim=args.fusion_dim,
        )
        summary[name] = res
        # ecg_only nests its metrics under "splits"; the torch probes put them at top level.
        metrics = res.get("splits", res)
        print(f"{name:7s} val={metrics.get('val', {})} test={metrics.get('test', {})}")

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    checkpoint_files = {
        "ecg": "ecg_only.joblib",
        "echo": "echo_only.pt",
        "concat": "concat_mlp.pt",
        "fused": "cross_attn_fused.pt",
    }
    echo_dim, ecg_dim = manifest_io.dims(manifest_io.load(args.manifest))
    save_run_metadata(
        out_root,
        {
            "task": "M10_probe_training",
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "checkpoint_sha256": {
                name: sha256_file(out_root / name / checkpoint_files[name]) for name in selected
            },
            "probes": list(selected),
            "seed": args.seed,
            "epochs": args.epochs,
            "fusion_dim": args.fusion_dim,
            "echo_dim": echo_dim,
            "ecg_dim": ecg_dim,
            # a clip-level manifest carries (n_clips, dim) per study; pooled carries (dim,)
            "echo_tokens_are_clip_level": manifest_io.is_clip_level(args.manifest),
            "n_dropped_nonfinite": {k: v.get("n_dropped_nonfinite") for k, v in summary.items()},
        },
    )
    print(f"\nwrote {out_root / 'summary.json'} and {out_root / 'run_metadata.json'}")


if __name__ == "__main__":
    main()
