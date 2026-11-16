"""Explicit, one-time initialization experiments for custom action adapters.

These modify selected model weights in memory, never runtime action values.
Source checkpoints are not written. Caller must record provenance and train anew.
"""

import math
import torch


def scale_action_encoder_output(head, scale: float, row: int | None = None):
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Action feature initialization scale must be finite and positive"
        )
    projection = head.action_encoder.W3
    if row is not None and not 0 <= row < projection.W.shape[0]:
        raise ValueError(f"Invalid action encoder row {row}")
    weight = projection.W if row is None else projection.W[row]
    bias = projection.b if row is None else projection.b[row]
    with torch.no_grad():
        before = float(weight.float().square().mean().sqrt())
        weight.mul_(scale)
        bias.mul_(scale)
        after = float(weight.float().square().mean().sqrt())
    return {
        "scale": scale,
        "row": row,
        "weight_rms_before": before,
        "weight_rms_after": after,
        "scope": "one-time action_encoder.W3 initialization; not runtime actions",
    }


def scale_action_encoder_input(
    head, scale: float, active_dims: int = 78, row: int | None = None
):
    """Amplify only active action inputs, not time embeddings or padded noise."""
    scale = float(scale)
    projection = head.action_encoder.W1
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Action input initialization scale must be finite and positive"
        )
    if not 0 < active_dims <= projection.W.shape[1]:
        raise ValueError("Invalid active action dimension")
    if row is not None and not 0 <= row < projection.W.shape[0]:
        raise ValueError(f"Invalid action encoder row {row}")
    weight = (
        projection.W[:, :active_dims]
        if row is None
        else projection.W[row, :active_dims]
    )
    with torch.no_grad():
        before = float(weight.float().square().mean().sqrt())
        weight.mul_(scale)
        after = float(weight.float().square().mean().sqrt())
    return {
        "scale": scale,
        "row": row,
        "active_dims": active_dims,
        "weight_rms_before": before,
        "weight_rms_after": after,
        "scope": "one-time action_encoder.W1 active input rows; time/padding/bias untouched",
    }
