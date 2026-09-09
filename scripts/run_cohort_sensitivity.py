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
from build_cohort import build_cte_sql, get_client, run_funnel


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
        funnel = run_funnel(client, cte, spec["pair_by"], spec["require_admission"])
        funnels[spec["name"]] = funnel
        funnel.to_csv(output_dir / f"{spec['name']}_funnel.csv", index=False)
        funnel.to_json(output_dir / f"{spec['name']}_funnel.json", orient="records", indent=2)

    summary = summarize_sensitivity(funnels)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("GCP_PROJECT_ID"))
    parser.add_argument("--window-hours", type=float, nargs="+", default=[24.0, 48.0])
    parser.add_argument("--output-dir", type=Path, default=Path("logs/cohort_sensitivity"))
    parser.add_argument("--skip-admission", action="store_true")
    args = parser.parse_args()

    summary = run_sensitivity(
        project=args.project,
        windows=args.window_hours,
        output_dir=args.output_dir,
        include_admission=not args.skip_admission,
    )
    print(summary.to_string(index=False))
    print(f"\nWrote sensitivity outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
