#!/usr/bin/env python3
"""Export a sanitized, checksummed copy of the aggregate result artifacts.

`results/` and `probes/` are gitignored (per-example predictions and checkpoints stay
local), so reviewers had nothing committed to diff reported numbers against. This copies
the aggregate-only artifacts into docs/results/ — stripping the per-example prediction
blocks from the missing-modality JSON — and writes a SHA256SUMS covering both the bundle
and the local raw artifacts (manifest, checkpoints, full results), so an independent
reproduction can be checked file-by-file.

Example:
  python scripts/export_result_bundle.py \
    --manifest data/processed/echo_hubert_manifest.parquet \
    --results results --probes probes --out docs/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from primed_ai.probes.common import sha256_file

PER_EXAMPLE_KEYS = ("predictions", "predictions_val")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default="data/processed/echo_hubert_manifest.parquet")
    ap.add_argument("--results", default="results")
    ap.add_argument("--probes", default="probes")
    ap.add_argument("--out", default="docs/results")
    args = ap.parse_args()

    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    exported: list[Path] = []

    def export_json(src: Path, dst_name: str, strip: tuple[str, ...] = ()) -> None:
        if not src.is_file():
            print(f"skip (missing): {src}")
            return
        payload = json.loads(src.read_text())
        for key in strip:
            payload.pop(key, None)
        dst = out / dst_name
        dst.write_text(json.dumps(payload, indent=2) + "\n")
        exported.append(dst)

    export_json(
        results / "missing_modality.json", "missing_modality.metrics.json", PER_EXAMPLE_KEYS
    )
    export_json(results / "fairness" / "fairness_metrics.json", "fairness_metrics.json")
    for cond in ("full", "echo_dropped", "ecg_dropped"):
        export_json(results / "calibration" / cond / "calibration.json", f"calibration_{cond}.json")
    export_json(results / "baseline_gap.json", "baseline_gap.json")
    # Named for the harness that produced it: the failure views come from a Ridge on
    # concatenated embeddings, not from the fused checkpoint the other artifacts score.
    export_json(results / "failure" / "failure_report.json", "failure_report.ridge.json")

    # checksums: the bundle itself, plus the local raw artifacts a reproduction must match.
    # Only the canonical M10 checkpoint locations — stale dirs from older layouts (e.g.
    # probes/cross_attn_fused/) must not end up looking canonical in the sums.
    probes = Path(args.probes)
    raw = [Path(args.manifest)]
    raw += [
        probes / "ecg" / "ecg_only.joblib",
        probes / "echo" / "echo_only.pt",
        probes / "concat" / "concat_mlp.pt",
        probes / "fused" / "cross_attn_fused.pt",
    ]
    raw.append(results / "missing_modality.json")
    # Anything already in the bundle whose source was missing this run is still committed,
    # so it must still get a line -- dropping it would let `shasum -c` pass over a file no
    # run has verified. Checksum it and say so.
    fresh = set(exported)
    stale = sorted(p for p in out.glob("*.json") if p not in fresh)
    if stale:
        print("stale (source missing, checksummed as-is): " + ", ".join(p.name for p in stale))

    lines = []
    for path in exported + stale + [p for p in raw if p.is_file()]:
        lines.append(f"{sha256_file(path)}  {path.as_posix()}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"exported {len(exported)} sanitized artifacts + SHA256SUMS to {out}")


if __name__ == "__main__":
    main()
