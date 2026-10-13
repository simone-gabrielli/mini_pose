"""Utilities for validating and loading complete EMA model states."""

from __future__ import annotations

from typing import Mapping

import torch


def ema_shadow_issues(
    model: torch.nn.Module,
    shadow: Mapping[str, object],
) -> tuple[list[str], list[str], list[str]]:
    """Return missing, unexpected, and incompatible EMA state keys.

    A usable EMA state must contain the complete model ``state_dict``, including
    buffers such as BatchNorm running statistics. Parameter-only EMA shadows
    are unsafe for models containing stateful normalization layers.
    """
    expected = model.state_dict()
    shadow_keys = set(shadow.keys())
    expected_keys = set(expected.keys())
    missing = sorted(expected_keys - shadow_keys)
    unexpected = sorted(shadow_keys - expected_keys)
    incompatible: list[str] = []

    for name in sorted(expected_keys & shadow_keys):
        source = shadow[name]
        target = expected[name]
        if not isinstance(source, torch.Tensor):
            incompatible.append(name)
            continue
        if source.shape != target.shape:
            incompatible.append(name)

    return missing, unexpected, incompatible


def complete_ema_shadow(model: torch.nn.Module, shadow: object) -> bool:
    if not isinstance(shadow, Mapping) or not shadow:
        return False
    missing, unexpected, incompatible = ema_shadow_issues(model, shadow)
    return not missing and not unexpected and not incompatible
