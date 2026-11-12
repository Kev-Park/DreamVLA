"""CPU-testable SONIC token and hand interfaces; no simulator imports."""

from __future__ import annotations

import numpy as np


def fsq(token):
    token = np.asarray(token, dtype=np.float32)
    if token.ndim < 1 or token.shape[-1] != 64 or not np.isfinite(token).all():
        raise ValueError("Expected finite 64-channel SONIC tokens.")
    return np.clip(np.rint(token * 16) / 16, -1, 15 / 16)


def expert_token(reference, residual, scale=0.5):
    reference = np.asarray(reference, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    if reference.shape != residual.shape or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid reference/residual shape or scale.")
    return fsq(reference * (1 + np.float32(scale) * residual))


def previous_action_proxy(reference, full_token, scale=0.5):
    """Approximate residual history, NOT an exact expert-history reconstruction.

    FSQ loses pre-quantization information. A nonzero student token at a zero
    reference is unrepresentable; return its mask, never silently hide it.
    See docs/limitations.md before using this for expert recovery supervision.
    """
    reference = np.asarray(reference, dtype=np.float32)
    token = np.asarray(full_token, dtype=np.float32)
    if (
        reference.shape != token.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(token).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Invalid token history input.")
    out = np.zeros_like(reference)
    nonzero = reference != 0
    out[nonzero] = (token[nonzero] / reference[nonzero] - 1) / scale
    return out, (~nonzero) & (token != 0)


def nearest_binary(prediction, open_pose, closed_pose):
    prediction = np.asarray(prediction, dtype=np.float32)
    opened, closed = np.asarray(open_pose), np.asarray(closed_pose)
    if any(
        x.shape != (7,) or not np.isfinite(x).all()
        for x in (prediction, opened, closed)
    ):
        raise ValueError("Expected finite seven-joint hand commands.")
    return (
        1.0
        if np.mean((prediction - opened) ** 2) <= np.mean((prediction - closed) ** 2)
        else -1.0
    )
