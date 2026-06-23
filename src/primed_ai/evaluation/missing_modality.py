"""Checkpoint-only missing-modality evaluation for the M09 fused probe."""
from __future__ import annotations

from pathlib import Path

import torch

from primed_ai.probes.common import git_sha, save_results
from primed_ai.probes.cross_attn import (
    CrossAttnFusedProbe,
    evaluate_missing_modality,
    fused_probe_loader,
    prepare_fused_probe_data,
)


def run(
    cohort_path,
    echo_embedding_path,
    ecg_embedding_path,
    checkpoint_path,
    output_path="results/missing_modality.json",
    *,
    embed_dim: int = 16,
    echo_dim: int | None = None,
    ecg_dim: int | None = None,
    hidden: int = 256,
    batch_size: int = 64,
    seed: int = 42,
    device: str | None = None,
) -> dict:
    """Load one full-modality M09 checkpoint and evaluate held-out test conditions."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    echo_dim = echo_dim or embed_dim
    ecg_dim = ecg_dim or embed_dim
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"M09 checkpoint not found: {checkpoint}")

    parts = prepare_fused_probe_data(cohort_path, echo_embedding_path, ecg_embedding_path)
    test_loader = fused_probe_loader(parts["test"], embed_dim=echo_dim, batch_size=batch_size)

    model = CrossAttnFusedProbe(
        embed_dim=embed_dim,
        hidden=hidden,
        echo_dim=echo_dim,
        ecg_dim=ecg_dim,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)

    test_metrics = evaluate_missing_modality(model, test_loader, device)
    payload = {
        "task": "E01_missing_modality_evaluation",
        "issue": 7,
        "source_task": "M09_cross_attn_fused_probe",
        "seed": seed,
        "git_sha": git_sha(),
        "checkpoint": str(checkpoint),
        "config": {
            "cohort_path": str(cohort_path),
            "echo_embedding_path": str(echo_embedding_path),
            "ecg_embedding_path": str(ecg_embedding_path),
            "fusion_dim": embed_dim,
            "echo_dim": echo_dim,
            "ecg_dim": ecg_dim,
            "hidden": hidden,
            "batch_size": batch_size,
            "device": device,
        },
        "n": {"test": len(parts["test"])},
        "test": test_metrics,
        "metrics_table": [
            {
                "condition": condition,
                "mae": metrics["mae"],
                "ef40_auroc": metrics["ef40_auroc"],
                "baseline_mae": metrics["baseline_mae"],
            }
            for condition, metrics in test_metrics.items()
        ],
        "notes": "Dropped conditions mask one branch at inference only; no unimodal retraining.",
    }
    save_results(Path(output_path), payload)
    return payload
