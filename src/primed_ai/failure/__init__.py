"""Model-agnostic modality-failure analysis (MFA) harness.

Public API:
    analyze_modality_failure(embeddings, lvef, ef_le_40, predict_fn, ...) -> FailureReport
    classify_taxonomy(...)                      # per-example error categories
    make_synthetic_multimodal(...)             # planted-structure data for validation/demo
    masked_ridge_predict_fn(...)               # a simple maskable multimodal predict_fn
    plot_complementarity_matrix / plot_dropout_profile / plot_taxonomy
"""

from __future__ import annotations

from .core import CATEGORIES, FailureReport, analyze_modality_failure, classify_taxonomy
from .demo import (
    make_synthetic_modalities,
    make_synthetic_multimodal,
    masked_ridge_predict_fn,
    stratified_train_mask,
)

__all__ = [
    "analyze_modality_failure",
    "classify_taxonomy",
    "FailureReport",
    "CATEGORIES",
    "make_synthetic_multimodal",
    "make_synthetic_modalities",
    "masked_ridge_predict_fn",
    "stratified_train_mask",
]
