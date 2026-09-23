#!/usr/bin/env python3
"""Run D07 cohort-size sensitivity funnels without downloading cohorts.

This wraps ``build_cohort.py``'s existing SQL so the 24h / 48h / admission-mode
comparison is one command instead of several ad hoc runs.

Example:
  python scripts/run_cohort_sensitivity.py --project "$GCP_PROJECT_ID"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import pandas as pd
from build_cohort import (
    DEFAULT_MAX_BYTES_BILLED,
    GIB,
    build_cte_sql,
    build_funnel_sql,
    estimate_query_bytes,
    get_client,
    run_funnel,
)

DEFAULT_MAX_TOTAL_BYTES_BILLED = 100 * GIB


def summarize_sensitivity(funnels: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize each scenario by its final funnel row."""
    rows = []
    baseline = None
    for scenario, funnel in funnels.items():
        if funnel.empty:
            raise ValueError(f"{scenario} funnel is empty")
        first = funnel.iloc[0]
        final = funnel.iloc[-1]
        final_rows = int(final["n_studies"])
        if baseline is None:
            baseline = final_rows
        gain = final_rows - baseline
        rows.append(
            {
                "scenario": scenario,
                "final_stage": final["stage"],
                "start_rows": int(first["n_studies"]),
                "final_rows": final_rows,
                "final_subjects": int(final["n_subjects"]),
                "total_excluded": int(first["n_studies"] - final["n_studies"]),
                "last_step_excluded": int(final.get("excluded_studies", 0)),
                "gain_vs_first": gain,
                "gain_vs_first_pct": round(gain / baseline * 100, 2) if baseline else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _scenario_params(windows: list[float], include_admission: bool) -> list[dict]:
    scenarios = [
        {
            "name": f"window_{hours:g}h",
            "pair_by": "window",
            "before": hours,
            "after": hours,
            "require_admission": True,
        }
        for hours in windows
    ]
    if include_admission:
        scenarios.append(
            {
                "name": "admission",
                "pair_by": "admission",
                "before": windows[0],
                "after": windows[0],
                "require_admission": True,
            }
        )
    return scenarios


def run_sensitivity(
    *,
    project: str | None,
    windows: list[float],
    output_dir: Path,
    include_admission: bool = True,
    maximum_bytes_billed: int = DEFAULT_MAX_BYTES_BILLED,
) -> pd.DataFrame:
    client = get_client(project)
    output_dir.mkdir(parents=True, exist_ok=True)
    funnels = {}

    for spec in _scenario_params(windows, include_admission):
        cte = build_cte_sql(
            spec["before"],
            spec["after"],
            0.0,
            100.0,
            spec["require_admission"],
            spec["pair_by"],
        )
        funnel = run_funnel(
            client,
            cte,
            spec["pair_by"],
            spec["require_admission"],
            maximum_bytes_billed=maximum_bytes_billed,
        )
        funnels[spec["name"]] = funnel
        funnel.to_csv(output_dir / f"{spec['name']}_funnel.csv", index=False)
        funnel.to_json(output_dir / f"{spec['name']}_funnel.json", orient="records", indent=2)

    summary = summarize_sensitivity(funnels)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2) + "\n"
    )
    return summary


def estimate_sensitivity(
    *, project: str | None, windows: list[float], include_admission: bool = True
) -> pd.DataFrame:
    """Estimate every funnel query without running or billing it."""
    client = get_client(project)
    rows = []
    for spec in _scenario_params(windows, include_admission):
        cte = build_cte_sql(
            spec["before"],
            spec["after"],
            0.0,
            100.0,
            spec["require_admission"],
            spec["pair_by"],
        )
        sql = build_funnel_sql(cte, spec["pair_by"], spec["require_admission"])
        estimated_bytes = estimate_query_bytes(client, sql)
        rows.append(
            {
                "scenario": spec["name"],
                "estimated_bytes": estimated_bytes,
                "estimated_gib": round(estimated_bytes / GIB, 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID"))
    parser.add_argument("--window-hours", type=float, nargs="+", default=[24.0, 48.0])
    parser.add_argument("--output-dir", type=Path, default=Path("logs/cohort_sensitivity"))
    parser.add_argument("--skip-admission", action="store_true")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Dry-run all SQL and report bytes processed without executing it.",
    )
    parser.add_argument(
        "--max-bytes-billed-gib",
        type=float,
        default=DEFAULT_MAX_BYTES_BILLED / GIB,
        help="Hard BigQuery limit for each query in GiB (default: 25).",
    )
    parser.add_argument(
        "--max-total-bytes-billed-gib",
        type=float,
        default=DEFAULT_MAX_TOTAL_BYTES_BILLED / GIB,
        help="Preflight limit across all scenario queries in GiB (default: 100).",
    )
    args = parser.parse_args()

    if args.max_bytes_billed_gib <= 0 or args.max_total_bytes_billed_gib <= 0:
        parser.error("byte limits must be positive")

    maximum_bytes_billed = int(args.max_bytes_billed_gib * GIB)
    maximum_total_bytes_billed = int(args.max_total_bytes_billed_gib * GIB)
    include_admission = not args.skip_admission

    estimates = estimate_sensitivity(
        project=args.project,
        windows=args.window_hours,
        include_admission=include_admission,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(args.output_dir / "query_estimates.csv", index=False)
    (args.output_dir / "query_estimates.json").write_text(
        json.dumps(estimates.to_dict(orient="records"), indent=2) + "\n"
    )
    print("BigQuery dry-run estimates (no bytes billed):")
    print(estimates.to_string(index=False))

    oversized = estimates[estimates["estimated_bytes"] > maximum_bytes_billed]
    estimated_total = int(estimates["estimated_bytes"].sum())
    if not oversized.empty:
        names = ", ".join(oversized["scenario"].astype(str))
        parser.error(
            f"estimated query exceeds {args.max_bytes_billed_gib:g} GiB cap: {names}"
        )
    if estimated_total > maximum_total_bytes_billed:
        parser.error(
            "estimated total exceeds "
            f"{args.max_total_bytes_billed_gib:g} GiB cap "
            f"({estimated_total / GIB:.3f} GiB estimated)"
        )
    if args.estimate_only:
        print(f"\nWrote estimates to: {args.output_dir}")
        return

    summary = run_sensitivity(
        project=args.project,
        windows=args.window_hours,
        output_dir=args.output_dir,
        include_admission=include_admission,
        maximum_bytes_billed=maximum_bytes_billed,
    )
    print(summary.to_string(index=False))
    print(f"\nWrote sensitivity outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
