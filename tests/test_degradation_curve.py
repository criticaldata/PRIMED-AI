from primed_ai.evaluation.degradation_curve import caption_text, ordered_rows


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
