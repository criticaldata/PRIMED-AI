import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_kfold_cv as kfold_cv  # noqa: E402

CONDITIONS = ("full", "echo_dropped", "ecg_dropped")


def _payload(lvef, prediction):
    ef_le_40 = [value <= 40 for value in lvef]

    return {
        "test": {
            condition: {
                "mae": 5.0,
                "ef40_auroc": 1.0,
            }
            for condition in CONDITIONS
        },
        "bootstrap": {
            condition: {
                "mae_ci_low": 4.0,
                "mae_ci_high": 6.0,
            }
            for condition in CONDITIONS
        },
        "predictions": {
            condition: {
                "lvef": lvef,
                "prediction": prediction,
                "ef_le_40": ef_le_40,
            }
            for condition in CONDITIONS
        },
    }


def test_aggregate_results_combines_out_of_fold_predictions(tmp_path):
    fold_0 = tmp_path / "fold_0.json"
    fold_1 = tmp_path / "fold_1.json"

    fold_0.write_text(
        json.dumps(_payload([30.0, 60.0], [35.0, 55.0])),
        encoding="utf-8",
    )
    fold_1.write_text(
        json.dumps(_payload([20.0, 70.0], [25.0, 65.0])),
        encoding="utf-8",
    )

    result = kfold_cv.aggregate_results([fold_0, fold_1])

    assert result["n_folds"] == 2

    for condition in CONDITIONS:
        condition_result = result["conditions"][condition]

        assert len(condition_result["per_fold"]) == 2

        across = condition_result["across_fold"]
        assert across["mae_mean"] == 5.0
        assert across["mae_std"] == 0.0
        assert across["ef40_auroc_mean"] == 1.0
        assert across["ef40_auroc_std"] == 0.0

        pooled = condition_result["pooled_out_of_fold"]
        assert pooled["n"] == 4
        assert pooled["mae"] == 5.0
        assert pooled["ef40_auroc"] == 1.0


def test_main_runs_each_fold_without_real_training(tmp_path, monkeypatch):
    folds_dir = tmp_path / "folds"
    out_dir = tmp_path / "results"
    folds_dir.mkdir()

    for fold_idx in range(2):
        (folds_dir / f"fold_{fold_idx}.parquet").touch()

    trained = []
    evaluated = []

    def fake_train_fold(
        fold_manifest,
        fold_dir,
        *,
        epochs,
        seed,
        fusion_dim,
    ):
        trained.append(
            {
                "manifest": fold_manifest.name,
                "fold_dir": fold_dir.name,
                "epochs": epochs,
                "seed": seed,
                "fusion_dim": fusion_dim,
            }
        )
        return fold_dir / "probes" / "fused" / "cross_attn_fused.pt"

    def fake_evaluate_fold(
        fold_manifest,
        checkpoint,
        fold_dir,
        *,
        fusion_dim,
        seed,
        n_bootstrap,
    ):
        evaluated.append(
            {
                "manifest": fold_manifest.name,
                "fold_dir": fold_dir.name,
                "seed": seed,
                "fusion_dim": fusion_dim,
                "n_bootstrap": n_bootstrap,
            }
        )
        return fold_dir / "results" / "missing_modality.json"

    fake_summary = {
        "n_folds": 2,
        "conditions": {
            condition: {
                "per_fold": [],
                "across_fold": {
                    "mae_mean": 0.0,
                    "mae_std": 0.0,
                    "ef40_auroc_mean": 0.0,
                    "ef40_auroc_std": 0.0,
                },
                "pooled_out_of_fold": {
                    "n": 0,
                    "mae": 0.0,
                    "ef40_auroc": 0.0,
                },
            }
            for condition in kfold_cv.CONDITIONS
        },
    }

    monkeypatch.setattr(kfold_cv, "train_fold", fake_train_fold)
    monkeypatch.setattr(kfold_cv, "evaluate_fold", fake_evaluate_fold)
    monkeypatch.setattr(
        kfold_cv,
        "aggregate_results",
        lambda result_paths: fake_summary,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_kfold_cv.py",
            "--folds-dir",
            str(folds_dir),
            "--out-dir",
            str(out_dir),
            "--n-folds",
            "2",
            "--epochs",
            "3",
            "--fusion-dim",
            "16",
            "--seed",
            "42",
            "--n-bootstrap",
            "10",
        ],
    )

    kfold_cv.main()

    assert [row["manifest"] for row in trained] == [
        "fold_0.parquet",
        "fold_1.parquet",
    ]
    assert len(evaluated) == 2

    assert all(row["seed"] == 42 for row in trained)
    assert all(row["seed"] == 42 for row in evaluated)

    summary_path = out_dir / "kfold_results.json"
    assert summary_path.is_file()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["config"]["n_folds"] == 2
    assert summary["config"]["seed"] == 42
