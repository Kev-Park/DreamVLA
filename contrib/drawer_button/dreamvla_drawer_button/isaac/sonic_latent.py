"""SONIC residual composition and checkpoint compatibility (no Isaac imports)."""

from __future__ import annotations

import math
from pathlib import Path

import torch


def action_contract(cfg) -> dict:
    """Describe the PPO action semantics, including legacy additive defaults."""
    mode = getattr(cfg, "residual_transform", "additive")
    scale = float(cfg.residual_scale)
    bounds = [float(x) for x in cfg.latent_clip]
    if mode not in ("additive", "multiplicative", "multiplicative_free"):
        raise ValueError(f"Unknown SONIC residual transform: {mode}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("SONIC residual_scale must be finite and positive")
    if (
        len(bounds) != 2
        or not all(math.isfinite(x) for x in bounds)
        or bounds[0] >= bounds[1]
    ):
        raise ValueError("SONIC latent_clip must contain two finite increasing bounds")
    return {"residual_transform": mode, "residual_scale": scale, "latent_clip": bounds}


def compose_latent(
    reference: torch.Tensor, residual: torch.Tensor, cfg
) -> torch.Tensor:
    """Return a full decoder token; preserve the existing post-composition clipping.

    `additive` is the project's original *unsquashed* additive action mapping.
    `multiplicative` bounds the multiplier with tanh; `multiplicative_free` does
    not. No additional FSQ snapping is introduced by this change.
    """
    mode = getattr(cfg, "residual_transform", "additive")
    if mode == "additive":
        token = reference + cfg.residual_scale * residual
    elif mode == "multiplicative":
        token = reference * (1.0 + cfg.residual_scale * torch.tanh(residual))
    elif mode == "multiplicative_free":
        token = reference * (1.0 + cfg.residual_scale * residual)
    else:
        raise ValueError(f"Unknown SONIC residual transform: {mode}")
    return torch.clamp(token, min=cfg.latent_clip[0], max=cfg.latent_clip[1])


def project_to_multiplicative_envelope(
    reference: torch.Tensor, token: torch.Tensor, scale=0.1
):
    """Diagnostic projection only; production composition does not call this."""
    if not math.isfinite(scale) or not 0 < scale < 1:
        raise ValueError("Envelope scale must be finite and between zero and one")
    a, b = reference * (1 - scale), reference * (1 + scale)
    lower, upper = torch.minimum(a, b).clamp(-1, 1), torch.maximum(a, b).clamp(-1, 1)
    return torch.maximum(lower, torch.minimum(upper, token))


def validate_checkpoint_action_contract(checkpoint: str | Path, cfg) -> dict:
    """Reject loading a teacher with different action semantics.

    Isaac's YAML includes Python tuple tags. BaseLoader reads those as plain
    data and never constructs Python objects. Old runs without a mode field
    are explicitly treated as additive, not as the new runtime default.
    """
    from types import SimpleNamespace
    import yaml

    expected = action_contract(cfg)
    config_path = Path(checkpoint).resolve().parent / "params" / "env.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Cannot verify SONIC checkpoint action semantics: {config_path}"
        )
    data = yaml.load(config_path.read_text(), Loader=yaml.BaseLoader)
    saved = data.get("actions", {}).get("joint_pos", {})
    if "SonicLatentAction" not in saved.get("class_type", ""):
        raise ValueError(
            f"Checkpoint was not saved with a SONIC action term: {config_path}"
        )
    actual = action_contract(
        SimpleNamespace(
            residual_transform=saved.get("residual_transform", "additive"),
            residual_scale=saved["residual_scale"],
            latent_clip=saved["latent_clip"],
        )
    )
    if actual != expected:
        raise ValueError(
            f"SONIC action contract mismatch for {checkpoint}: saved={actual}, requested={expected}. "
            "Use the matching task or train a new teacher; do not reinterpret an existing policy."
        )
    return actual
