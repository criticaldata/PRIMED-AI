"""Shared helpers for frozen foundation-model encoders."""

from __future__ import annotations

import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    """Put ``module`` in eval mode and disable gradients on all parameters."""
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def assert_frozen(module: nn.Module) -> None:
    """Raise if any parameter still requires grad (used in unit tests)."""
    if any(p.requires_grad for p in module.parameters()):
        raise AssertionError("encoder is not frozen")
