"""Publication figures for the modality-failure harness."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _setup_mpl():
    cache = Path(tempfile.gettempdir()) / "primed-ai-plot-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42})
    return plt


def plot_complementarity_matrix(report, output_pdf):
    """Heatmap of MAE by available-modality subset (solo on diagonal, pair off-diagonal)."""
    plt = _setup_mpl()
    mods = report.complementarity["matrix"]["modalities"]
    vals = report.complementarity["matrix"]["values"]
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.2, 2.9), constrained_layout=True)
    im = ax.imshow(vals, cmap="RdYlGn_r", aspect="equal")
    ax.set_xticks(range(len(mods)), mods, rotation=20, ha="right")
    ax.set_yticks(range(len(mods)), mods)
    for i in range(len(mods)):
        for j in range(len(mods)):
            ax.text(j, i, f"{vals[i][j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("MAE by available modalities\n(diagonal = solo, off-diagonal = pair)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="MAE")
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_pdf


def plot_dropout_profile(report, output_pdf):
    """Per dropped modality: stacked silent/loud induced-critical failures + MAE increase."""
    plt = _setup_mpl()
    items = list(report.dropout.items())
    labels = [k.replace("drop_", "drop ") for k, _ in items]
    silent = [v["silent"] for _, v in items]
    loud = [v["loud"] for _, v in items]
    dmae = [v["mae_increase"] for _, v in items]
    x = range(len(items))
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    ax.bar(x, silent, color="#D55E00", label="silent (confident, wrong)")
    ax.bar(x, loud, bottom=silent, color="#E69F00", label="loud (monitorable)")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("induced critical failures")
    ax.set_title("Dropout profile: loud vs. silent")
    for i, d in enumerate(dmae):
        top = silent[i] + loud[i]
        ax.text(i, top, f"+{d:.1f} MAE", ha="center", va="bottom", fontsize=7)
    ax.legend(frameon=False, fontsize=7, loc="upper center")
    ax.margins(y=0.18)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_pdf


def plot_taxonomy(report, output_pdf):
    """Stacked failure-taxonomy bars across conditions (full + each dropout)."""
    plt = _setup_mpl()
    conds = list(report.conditions.items())
    labels = [k.replace("drop_", "drop ") for k, _ in conds]
    correct = [v["taxonomy"]["correct"] for _, v in conds]
    imprecise = [v["taxonomy"]["imprecise"] for _, v in conds]
    critical = [v["taxonomy"]["critical"] for _, v in conds]
    x = range(len(conds))
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.6, 2.9), constrained_layout=True)
    ax.bar(x, correct, color="#009E73", label="correct")
    ax.bar(x, imprecise, bottom=correct, color="#E69F00", label="imprecise")
    ax.bar(x, critical, bottom=[c + i for c, i in zip(correct, imprecise)],
           color="#D55E00", label="critical (gate wrong)")
    ax.set_xticks(list(x), labels, rotation=15, ha="right")
    ax.set_ylabel("test examples")
    ax.set_title("Failure taxonomy by condition")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_pdf
