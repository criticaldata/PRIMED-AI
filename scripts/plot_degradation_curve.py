"""Create the E02 missing-modality degradation figure and caption.

Example:
  python scripts/plot_degradation_curve.py \
    --input results/missing_modality.json \
    --output results/figures/degradation_curve.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

from primed_ai.evaluation.degradation_curve import (
    load_metrics,
    plot_degradation_curve,
    write_caption,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/missing_modality.json")
    parser.add_argument("--output", default="results/figures/degradation_curve.pdf")
    parser.add_argument(
        "--caption-output",
        default="results/figures/degradation_curve_caption.txt",
    )
    args = parser.parse_args()

    payload = load_metrics(args.input)
    figure_path = plot_degradation_curve(payload, args.output)
    caption_path = write_caption(payload, args.caption_output)

    png_path = Path(figure_path).with_suffix(".png")
    svg_path = Path(figure_path).with_suffix(".svg")
    print(f"Wrote figure PDF: {figure_path}")
    print(f"Wrote figure SVG: {svg_path}")
    print(f"Wrote figure PNG: {png_path}")
    print(f"Wrote caption draft: {caption_path}")


if __name__ == "__main__":
    main()
