"""Opt-in DreamVLA-style residual experiment; no Isaac imports.

Reference: Kev-Park/DreamVLA g1 commit
426333ec2fcc359a1e637b5a3a6aab47f2667d5c, token_adapter_wrapper.py.
This is lattice rounding of an ALREADY encoded token, not another FSQ tanh.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

SOURCE_COMMIT = "426333ec2fcc359a1e637b5a3a6aab47f2667d5c"
POSTPROCESS = "fsq32_lattice_round"
CONTRACT = {
    "residual_transform": "multiplicative_free",
    "residual_scale": 0.1,
    "latent_clip": [-1.0, 15.0 / 16.0],
    "token_postprocess": POSTPROCESS,
    "fsq_levels": 32,
    "fsq_grid_step": 1.0 / 16.0,
}


def snap_fsq32(token: torch.Tensor) -> torch.Tensor:
    """Match DreamVLA's round-to-even, step 1/16, asymmetric 32-level grid."""
    return torch.clamp(torch.round(token * 16.0) / 16.0, min=-1.0, max=15.0 / 16.0)


def compose_dreamvla_token(reference: torch.Tensor, residual: torch.Tensor, scale=0.1):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Residual scale must be finite and positive")
    return snap_fsq32(reference * (1.0 + scale * residual))


def validate_dreamvla_checkpoint(checkpoint: Path) -> dict:
    """Never reinterpret an old continuous-token or differently initialized run."""
    import yaml

    params = Path(checkpoint).resolve().parent / "params"
    with (params / "env.yaml").open() as stream:
        env = yaml.load(stream, Loader=yaml.BaseLoader)
    with (params / "agent.yaml").open() as stream:
        agent = yaml.load(stream, Loader=yaml.BaseLoader)
    body = env["actions"]["joint_pos"]
    actual = {
        "residual_transform": body.get("residual_transform"),
        "residual_scale": float(body["residual_scale"]),
        "latent_clip": [float(v) for v in body["latent_clip"]],
        "token_postprocess": body.get("token_postprocess"),
        "fsq_levels": int(body.get("fsq_levels", 0)),
        "fsq_grid_step": 1.0 / 16.0,
    }
    if (
        actual != CONTRACT
        or "DreamVlaSonicLatentAction" not in body.get("class_type", "")
        or agent["policy"]["class_name"] != "DreamVlaResidualActorCritic"
    ):
        raise ValueError(f"DreamVLA-style checkpoint contract mismatch: {actual}")
    return actual
