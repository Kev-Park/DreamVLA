"""Canonical SONIC FSQ (Finite Scalar Quantization) constants + the idempotent
lattice snap, single-sourced so every script uses identical grid parameters.

Two DISTINCT operations are commonly both called "FSQ" — do not conflate them:

* **Full FSQ** (``round(z.tanh()*15.48 - 0.5)/16``): maps an *unbounded* latent
  onto the grid. NON-idempotent — it squashes already-on-grid values toward zero
  (1.0 -> 0.75, 0.5 -> 0.44). It is baked into the encoder ONNX and is correct
  only for a from-scratch *unbounded* policy latent. It lives in
  ``token_action_wrapper.FSQ`` and must NEVER be re-applied to an
  already-quantized token.

* **Lattice snap** (``round(z*16)/16``, clamp ``[-1, 15/16]``): maps *near-grid*
  values to the exact grid. IDEMPOTENT — a no-op on values already on the grid.
  Use this, and only this, to return a token to the grid after adding a residual
  (the adapter) or after a VLA emits a token. Encoder-sourced tokens are already
  on the grid (the encoder ONNX bakes in full FSQ), so the snap is a no-op there
  and the encoder/convert paths may skip it.

These constants are the single source of truth; importers must not redefine them.
"""

from __future__ import annotations

# FSQ geometry for the SONIC release models: 32 levels per channel, 2 tokens.
NUM_LEVELS = 32
MAX_NUM_TOKENS = 2
TOKEN_DIM = NUM_LEVELS * MAX_NUM_TOKENS  # 64-D flattened token
HALF_WIDTH = float(NUM_LEVELS // 2)      # 16.0 — grid step is 1/HALF_WIDTH
SNAP_MIN = -1.0
SNAP_MAX = (HALF_WIDTH - 1.0) / HALF_WIDTH  # 15/16


def fsq_lattice_snap(token):
    """Snap near-grid token values onto the FSQ lattice (idempotent).

    Accepts a NumPy array or a torch.Tensor and returns the same type. This is the
    *lattice snap*, NOT full FSQ — see module docstring. At residual=0 (or on an
    encoder-sourced token already on the grid) it is an exact passthrough.
    """
    try:  # torch path (adapters / live policies) without making torch a hard dep
        import torch

        if isinstance(token, torch.Tensor):
            return torch.clamp(
                torch.round(token * HALF_WIDTH) / HALF_WIDTH,
                min=SNAP_MIN,
                max=SNAP_MAX,
            )
    except ImportError:
        pass

    import numpy as np

    arr = np.asarray(token, dtype=np.float32)
    return np.clip(np.round(arr * HALF_WIDTH) / HALF_WIDTH, SNAP_MIN, SNAP_MAX).astype(
        np.float32
    )
