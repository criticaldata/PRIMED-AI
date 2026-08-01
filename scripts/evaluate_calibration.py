"""E04: EF<=40% gate calibration (reliability diagram + ECE) from a results JSON.

Example:
  python scripts/evaluate_calibration.py \
    --predictions results/missing_modality.json --condition full --out results/calibration
"""

from __future__ import annotations

import argparse

from primed_ai.evaluation.calibration import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="results/missing_modality.json")
    parser.add_argument("--condition", default="full")
    parser.add_argument("--out", default="results/calibration")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    res = run_calibration(args.predictions, args.out, condition=args.condition, n_bins=args.n_bins)
    print(f"ECE={res['ece']} (n={res['n']}); wrote figure + calibration.json under {args.out}")


if __name__ == "__main__":
    main()
