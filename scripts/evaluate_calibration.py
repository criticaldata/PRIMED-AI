"""E04: EF<=40% gate calibration (reliability diagram + ECE) from a results JSON.

Artifacts land in <out>/<condition>/, which is both what export_result_bundle.py reads and
what keeps three conditions from overwriting each other's calibration.json.

Example (all three conditions):
  for c in full echo_dropped ecg_dropped; do
    python scripts/evaluate_calibration.py \
      --predictions results/missing_modality.json --condition "$c" --out results/calibration
  done
"""

from __future__ import annotations

import argparse
from pathlib import Path

from primed_ai.evaluation.calibration import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="results/missing_modality.json")
    parser.add_argument("--condition", default="full")
    parser.add_argument("--out", default="results/calibration", help="parent dir; per-condition")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    out = Path(args.out) / args.condition
    res = run_calibration(args.predictions, out, condition=args.condition, n_bins=args.n_bins)
    print(f"ECE={res['ece']} (n={res['n']}); wrote figure + calibration.json under {out}")


if __name__ == "__main__":
    main()
