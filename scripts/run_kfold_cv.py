#!/usr/bin/env python3
"""Run patient-level k-fold cross-validation for E13.

For each fold, this script will:
1. Train the existing ECG, echo, concat, and fused probes.
2. Evaluate the fused checkpoint under full, echo_dropped, and ecg_dropped conditions.
3. Save fold-specific metrics and predictions.
4. Aggregate results across folds.

The existing train_probes.py and evaluate_missing_modality.py paths are reused so
the k-fold experiment stays comparable with the canonical single-split run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from primed_ai.probes import manifest as manifest_io
from primed_ai.probes.common import auroc, regression_metrics

CONDITIONS = ("full", "echo_dropped", "ecg_dropped")


def run_command(command: list[str]) -> None:
    """Run one pipeline command and stop immediately if it fails."""
    print("\nRunning:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON result file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def train_fold(
    fold_manifest: Path,
    fold_dir: Path,
    *,
    epochs: int,
    seed: int,
    fusion_dim: int,
) -> Path:
    """Train all four probes for one fold and return the fused checkpoint path."""
    probes_dir = fold_dir / "probes"

    command = [
        sys.executable,
        "scripts/train_probes.py",
        "--manifest",
        str(fold_manifest),
        "--probe",
        "all",
        "--out-dir",
        str(probes_dir),
        "--epochs",
        str(epochs),
        "--fusion-dim",
        str(fusion_dim),
        "--seed",
        str(seed),
    ]

    run_command(command)

    checkpoint = probes_dir / "fused" / "cross_attn_fused.pt"

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Expected fused checkpoint was not created: {checkpoint}")

    return checkpoint


def evaluate_fold(
    fold_manifest: Path,
    checkpoint: Path,
    fold_dir: Path,
    *,
    fusion_dim: int,
    seed: int,
    n_bootstrap: int,
) -> Path:
    """Evaluate one fold's fused checkpoint under all modality conditions."""
    results_dir = fold_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "missing_modality.json"

    manifest_df = manifest_io.load(fold_manifest)
    echo_dim, ecg_dim = manifest_io.dims(manifest_df)

    command = [
        sys.executable,
        "scripts/evaluate_missing_modality.py",
        "--manifest",
        str(fold_manifest),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_path),
        "--embed-dim",
        str(fusion_dim),
        "--echo-dim",
        str(echo_dim),
        "--ecg-dim",
        str(ecg_dim),
        "--seed",
        str(seed),
        "--n-bootstrap",
        str(n_bootstrap),
    ]

    run_command(command)

    if not output_path.is_file():
        raise FileNotFoundError(f"Expected missing-modality result was not created: {output_path}")

    return output_path


def aggregate_results(result_paths: list[Path]) -> dict[str, Any]:
    """Combine per-fold metrics and out-of-fold predictions."""
    fold_payloads = [read_json(path) for path in result_paths]

    summary: dict[str, Any] = {
        "n_folds": len(fold_payloads),
        "conditions": {},
    }

    for condition in CONDITIONS:
        per_fold = []
        pooled_lvef = []
        pooled_prediction = []
        pooled_ef40 = []

        for fold_idx, payload in enumerate(fold_payloads):
            metrics = payload["test"][condition]
            predictions = payload["predictions"][condition]

            per_fold.append(
                {
                    "fold": fold_idx,
                    "n_test": len(predictions["lvef"]),
                    "mae": metrics["mae"],
                    "ef40_auroc": metrics["ef40_auroc"],
                    "bootstrap": payload["bootstrap"].get(condition, {}),
                }
            )

            pooled_lvef.extend(predictions["lvef"])
            pooled_prediction.extend(predictions["prediction"])
            pooled_ef40.extend(predictions["ef_le_40"])

        fold_mae = np.asarray([row["mae"] for row in per_fold], dtype=float)
        fold_auroc = np.asarray(
            [row["ef40_auroc"] for row in per_fold],
            dtype=float,
        )

        pooled_lvef_array = np.asarray(pooled_lvef, dtype=float)
        pooled_prediction_array = np.asarray(pooled_prediction, dtype=float)
        pooled_ef40_array = np.asarray(pooled_ef40, dtype=bool)

        pooled_metrics = regression_metrics(
            pooled_lvef_array,
            pooled_prediction_array,
        )
        pooled_auroc = auroc(
            pooled_ef40_array,
            -pooled_prediction_array,
        )

        summary["conditions"][condition] = {
            "per_fold": per_fold,
            "across_fold": {
                "mae_mean": round(float(np.mean(fold_mae)), 4),
                "mae_std": round(float(np.std(fold_mae, ddof=1)), 4),
                "ef40_auroc_mean": round(float(np.mean(fold_auroc)), 4),
                "ef40_auroc_std": round(float(np.std(fold_auroc, ddof=1)), 4),
            },
            "pooled_out_of_fold": {
                "n": len(pooled_lvef_array),
                "mae": pooled_metrics["mae"],
                "ef40_auroc": round(float(pooled_auroc), 4),
            },
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds-dir",
        type=Path,
        required=True,
        help="Directory containing fold_0.parquet, fold_1.parquet, etc.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/kfold_cv"),
        help="Directory for fold checkpoints and evaluation results.",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    result_paths: list[Path] = []

    for fold_idx in range(args.n_folds):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx}")
        print(f"{'=' * 60}")

        fold_manifest = args.folds_dir / f"fold_{fold_idx}.parquet"

        if not fold_manifest.is_file():
            raise FileNotFoundError(f"Fold manifest not found: {fold_manifest}")

        fold_dir = args.out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_seed = args.seed

        checkpoint = train_fold(
            fold_manifest,
            fold_dir,
            epochs=args.epochs,
            seed=fold_seed,
            fusion_dim=args.fusion_dim,
        )

        result_path = evaluate_fold(
            fold_manifest,
            checkpoint,
            fold_dir,
            fusion_dim=args.fusion_dim,
            seed=fold_seed,
            n_bootstrap=args.n_bootstrap,
        )

        result_paths.append(result_path)

    summary = aggregate_results(result_paths)

    summary["config"] = {
        "folds_dir": str(args.folds_dir),
        "out_dir": str(args.out_dir),
        "n_folds": args.n_folds,
        "epochs": args.epochs,
        "fusion_dim": args.fusion_dim,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
    }

    summary_path = args.out_dir / "kfold_results.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote k-fold results to: {summary_path}")

    for condition in CONDITIONS:
        results = summary["conditions"][condition]
        across = results["across_fold"]
        pooled = results["pooled_out_of_fold"]

        print(f"\n{condition}")
        print(f"  Across-fold MAE: {across['mae_mean']:.4f} ± {across['mae_std']:.4f}")
        print(
            f"  Across-fold AUROC: {across['ef40_auroc_mean']:.4f} ± {across['ef40_auroc_std']:.4f}"
        )
        print(f"  Pooled OOF MAE: {pooled['mae']:.4f}")
        print(f"  Pooled OOF AUROC: {pooled['ef40_auroc']:.4f}")


if __name__ == "__main__":
    main()
