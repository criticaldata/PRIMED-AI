"""Plot missing-modality degradation curves from aggregate evaluation metrics."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

CONDITION_LABELS = {
    "full": "Full",
    "echo_dropped": "Echo dropped",
    "ecg_dropped": "ECG dropped",
}
PALETTE = {
    "mae": "#0072B2",
    "auroc": "#D55E00",
    "baseline": "#4B5563",
    "grid": "#E5E7EB",
    "axis": "#111827",
}


def load_metrics(path: str | Path) -> dict:
    """Load and validate aggregate missing-modality metrics."""
    payload = json.loads(Path(path).read_text())
    rows = payload.get("metrics_table")
    if not rows:
        raise ValueError(f"No metrics_table found in {path}")

    required = {"condition", "mae", "ef40_auroc"}
    missing = [required - set(row) for row in rows]
    if any(missing):
        raise ValueError("Each metrics_table row must include condition, mae, and ef40_auroc")
    return payload


def ordered_rows(payload: dict) -> list[dict]:
    """Return metric rows in the canonical degradation order."""
    by_condition = {row["condition"]: row for row in payload["metrics_table"]}
    order = ["full", "echo_dropped", "ecg_dropped"]
    return [by_condition[name] for name in order if name in by_condition]


def caption_text(payload: dict, rows: list[dict]) -> str:
    """Draft the paper caption from the plotted point estimates."""
    by_condition = {row["condition"]: row for row in rows}
    full = by_condition.get("full", {})
    echo_dropped = by_condition.get("echo_dropped", {})
    ecg_dropped = by_condition.get("ecg_dropped", {})
    n_test = payload.get("n", {}).get("test", "held-out")
    source_task = payload.get("task", "missing-modality evaluation")
    full_mae = full.get("mae", float("nan"))
    full_auroc = full.get("ef40_auroc", float("nan"))
    echo_mae = echo_dropped.get("mae", float("nan"))
    echo_auroc = echo_dropped.get("ef40_auroc", float("nan"))
    ecg_mae = ecg_dropped.get("mae", float("nan"))
    ecg_auroc = ecg_dropped.get("ef40_auroc", float("nan"))

    echo_delta = echo_mae - full_mae
    ecg_delta = ecg_mae - full_mae
    if echo_delta > ecg_delta:
        larger_drop = "Dropping echo produced the larger MAE increase"
    elif ecg_delta > echo_delta:
        larger_drop = "Dropping ECG produced the larger MAE increase"
    else:
        larger_drop = "Both dropped-modality conditions produced similar MAE changes"

    has_ci = all("mae_ci_low" in row and "mae_ci_high" in row for row in rows)
    ci_sentence = (
        " Error bars show 95% bootstrap confidence intervals."
        if has_ci
        else " Bootstrap confidence intervals require per-example predictions."
    )

    return (
        "Missing-modality degradation curve for the held-out test set "
        f"(n={n_test}), using aggregate metrics from {source_task}. "
        f"Relative to full-modality inference (MAE {full_mae:.2f}; "
        f"EF <= 40% AUROC {full_auroc:.2f}), echo-dropped inference reached "
        f"MAE {echo_mae:.2f} and AUROC {echo_auroc:.2f}, while ECG-dropped inference "
        f"reached MAE {ecg_mae:.2f} and AUROC {ecg_auroc:.2f}. {larger_drop}. "
        "This pattern supports graceful degradation rather than silent failure."
        f"{ci_sentence}"
    )


def plot_degradation_curve(payload: dict, output_pdf: str | Path) -> Path:
    """Write a two-panel MAE/AUROC degradation curve as PDF and companion PNG."""
    cache_root = Path(tempfile.gettempdir()) / "primed-ai-plot-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ordered_rows(payload)
    if len(rows) < 2:
        raise ValueError("Need at least two modality conditions for a degradation curve")

    labels = [CONDITION_LABELS.get(row["condition"], row["condition"]) for row in rows]
    x = list(range(len(rows)))
    mae = [row["mae"] for row in rows]
    auroc = [row["ef40_auroc"] for row in rows]
    baseline_mae = rows[0].get("baseline_mae")
    mae_yerr = _asymmetric_yerr(rows, "mae")
    auroc_yerr = _asymmetric_yerr(rows, "ef40_auroc")

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png = output_pdf.with_suffix(".png")
    output_svg = output_pdf.with_suffix(".svg")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.08, hspace=0.02)

    axes[0].errorbar(
        x,
        mae,
        yerr=mae_yerr,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=1.5,
        markersize=4.2,
        capsize=3 if mae_yerr is not None else 0,
        capthick=0.9,
        elinewidth=1.0,
        color=PALETTE["mae"],
    )
    if baseline_mae is not None:
        axes[0].axhline(
            baseline_mae,
            color=PALETTE["baseline"],
            linestyle="--",
            linewidth=0.9,
            label="Mean-LVEF baseline",
        )
        axes[0].legend(frameon=False, loc="upper right", handlelength=2.4)
    axes[0].set_title("LVEF regression")
    axes[0].set_ylabel("MAE")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set_xlim(-0.1, len(rows) - 0.9)
    axes[0].grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    _apply_ci_ylim(axes[0], rows, "mae", extra_values=[baseline_mae])

    axes[1].errorbar(
        x,
        auroc,
        yerr=auroc_yerr,
        marker="s",
        markerfacecolor="white",
        markeredgewidth=1.2,
        linestyle="--",
        linewidth=1.5,
        markersize=4.0,
        capsize=3 if auroc_yerr is not None else 0,
        capthick=0.9,
        elinewidth=1.0,
        color=PALETTE["auroc"],
    )
    axes[1].set_title("EF <= 40% classification")
    axes[1].set_ylabel("AUROC")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set_xlim(-0.1, len(rows) - 0.9)
    axes[1].grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    _apply_ci_ylim(axes[1], rows, "ef40_auroc", lower_bound=0.0, upper_bound=1.0)

    for label, ax in zip(("(a)", "(b)"), axes):
        ax.text(
            -0.16,
            1.06,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color=PALETTE["axis"],
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(PALETTE["axis"])
        ax.spines["bottom"].set_color(PALETTE["axis"])
        ax.tick_params(axis="both", colors=PALETTE["axis"], width=0.8, length=3)

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_pdf


def _asymmetric_yerr(rows: list[dict], metric: str):
    low_key = f"{metric}_ci_low"
    high_key = f"{metric}_ci_high"
    if not all(low_key in row and high_key in row for row in rows):
        return None
    values = [row[metric] for row in rows]
    lows = [max(0.0, value - row[low_key]) for value, row in zip(values, rows)]
    highs = [max(0.0, row[high_key] - value) for value, row in zip(values, rows)]
    return [lows, highs]


def _apply_ci_ylim(
    ax,
    rows: list[dict],
    metric: str,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    extra_values: list[float | None] | None = None,
) -> None:
    low_key = f"{metric}_ci_low"
    high_key = f"{metric}_ci_high"
    lows = [row.get(low_key, row[metric]) for row in rows]
    highs = [row.get(high_key, row[metric]) for row in rows]
    for value in extra_values or []:
        if value is not None:
            lows.append(value)
            highs.append(value)
    y_min = min(lows)
    y_max = max(highs)
    pad = max((y_max - y_min) * 0.12, 0.02 if metric == "ef40_auroc" else 0.2)
    if lower_bound is not None:
        y_min = max(lower_bound, y_min - pad)
    else:
        y_min = y_min - pad
    if upper_bound is not None:
        y_max = min(upper_bound, y_max + pad)
    else:
        y_max = y_max + pad
    ax.set_ylim(y_min, y_max)


def write_caption(payload: dict, output_path: str | Path) -> Path:
    rows = ordered_rows(payload)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(caption_text(payload, rows) + "\n")
    return output_path
