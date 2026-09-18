"""Shared helpers for eval_vla_sonic.py / play_vla_sonic.py.

* ``demo_motion_ids(hdf5_root)``      -- motion ids that have a collected demonstration under a
                                        collect_sonic_adapter.py output root (``*__motion_XXX.hdf5``),
                                        so evaluations start only from FILTERED reference init states.
* ``ReplaySource(hdf5_file)``          -- the recorded executed token + finger scalar of one demo, to be
                                        fed through the SAME decoder wrapper instead of the VLA
                                        (does the closed-loop stack reproduce the demonstration?).
* ``apply_episode_scale(env_cfg, s)``  -- run episodes s x longer than the reference clip: scales
                                        ``episode_length_s`` and replaces the reference-exhausted
                                        ``time_out`` term (tracking_time_out) by the flat cap; past the
                                        clip end the reference simply holds its last frame.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_MOTION_RE = re.compile(r"__motion_(\d+)\.hdf5$")


def demo_motion_ids(hdf5_root: str | Path) -> list[int]:
    root = Path(hdf5_root).expanduser()
    ids = sorted({int(m.group(1)) for p in root.rglob("*.hdf5") for m in [_MOTION_RE.search(p.name)] if m})
    if not ids:
        raise FileNotFoundError(f"no '*__motion_XXX.hdf5' under {root}")
    return ids


class ReplaySource:
    """Recorded executed token (64) + right-finger scalar per step of one collected demo."""

    def __init__(self, hdf5_file: str | Path):
        import h5py

        self.path = Path(hdf5_file).expanduser()
        with h5py.File(self.path, "r") as h:
            g = h["data/demo_0"]
            self.tokens = np.asarray(g["obs/motion_token"][()], dtype=np.float32)          # (T, 64) FSQ-snapped
            self.fingers = np.asarray(g["actions"][()][:, 64], dtype=np.float32)           # (T,) <0 closes
            meta = json.loads(h.attrs["metadata_json"]) if "metadata_json" in h.attrs else {}
        self.motion_id = int(meta.get("motion_id", -1))
        self.num_steps = int(self.tokens.shape[0])

    def latent(self, step: int) -> tuple[np.ndarray, float]:
        t = min(step, self.num_steps - 1)                                                   # hold the last one
        return self.tokens[t], float(self.fingers[t])


def apply_episode_scale(env_cfg, scale: float, tag: str) -> None:
    if scale == 1.0:
        return
    from isaaclab.envs import mdp as _mdp
    from isaaclab.managers import TerminationTermCfg as DoneTerm

    env_cfg.episode_length_s = float(env_cfg.episode_length_s) * float(scale)
    if getattr(env_cfg.terminations, "time_out", None) is not None:
        env_cfg.terminations.time_out = DoneTerm(func=_mdp.time_out, time_out=True)
    print(f"[{tag}] --episode-scale {scale}: episode_length_s -> {env_cfg.episode_length_s:.1f} s; "
          f"reference-exhausted time_out replaced by the flat cap (reference holds its last frame afterwards)")
