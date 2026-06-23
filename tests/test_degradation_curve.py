from primed_ai.evaluation.degradation_curve import (
    caption_text,
    ordered_rows,
    plot_degradation_curve,
)


def test_ordered_rows_and_caption_describe_degradation():
    payload = {
        "task": "E01_missing_modality_evaluation",
        "n": {"test": 245},
        "metrics_table": [
            {"condition": "ecg_dropped", "mae": 15.1344, "ef40_auroc": 0.7501},
            {"condition": "full", "mae": 10.2846, "ef40_auroc": 0.7661},
            {"condition": "echo_dropped", "mae": 18.5736, "ef40_auroc": 0.693},
        ],
    }

    rows = ordered_rows(payload)
    caption = caption_text(payload, rows)

    assert [row["condition"] for row in rows] == ["full", "echo_dropped", "ecg_dropped"]
    assert "graceful degradation rather than silent failure" in caption
    assert "n=245" in caption


def test_plot_degradation_curve_uses_bootstrap_ci(tmp_path):
    payload = {
        "task": "E01_missing_modality_evaluation",
        "n": {"test": 245},
        "metrics_table": [
            {
                "condition": "full",
                "mae": 10.0,
                "mae_ci_low": 9.1,
                "mae_ci_high": 10.9,
                "ef40_auroc": 0.78,
                "ef40_auroc_ci_low": 0.72,
                "ef40_auroc_ci_high": 0.83,
            },
            {
                "condition": "echo_dropped",
                "mae": 14.0,
                "mae_ci_low": 13.0,
                "mae_ci_high": 15.2,
                "ef40_auroc": 0.69,
                "ef40_auroc_ci_low": 0.61,
                "ef40_auroc_ci_high": 0.76,
            },
            {
                "condition": "ecg_dropped",
                "mae": 11.5,
                "mae_ci_low": 10.6,
                "mae_ci_high": 12.4,
                "ef40_auroc": 0.75,
                "ef40_auroc_ci_low": 0.69,
                "ef40_auroc_ci_high": 0.8,
            },
        ],
    }

    rows = ordered_rows(payload)
    caption = caption_text(payload, rows)
    output = plot_degradation_curve(payload, tmp_path / "degradation_curve.pdf")

    assert "Error bars show 95% bootstrap confidence intervals" in caption
    assert output.is_file()
    assert output.with_suffix(".png").is_file()
