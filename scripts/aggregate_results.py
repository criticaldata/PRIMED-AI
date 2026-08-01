"""E05: rebuild the unified results file and paper-ready tables from raw result JSONs.

Example:
  python scripts/aggregate_results.py --results-dir results --out-dir results
"""

from __future__ import annotations

import argparse

from primed_ai.evaluation.aggregate import aggregate_results, write_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    agg = aggregate_results(args.results_dir)
    paths = write_all(agg, args.out_dir)
    for kind, path in paths.items():
        print(f"wrote {kind}: {path}")


if __name__ == "__main__":
    main()
