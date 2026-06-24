"""Evaluation helpers for PRIMED-AI (E01 / E03 tasks).

Expose programmatic entrypoints for CLI scripts under `scripts/`.
"""

from .fairness import run_fairness

__all__ = ["run_fairness"]
"""Evaluation entry points for PRIMED-AI experiments."""
