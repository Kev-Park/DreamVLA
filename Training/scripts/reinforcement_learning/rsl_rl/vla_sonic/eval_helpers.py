"""Shared helpers for eval_vla_sonic.py / play_vla_sonic.py.

* ``demo_motion_ids(hdf5_root)``      -- motion ids that have a collected demonstration under a
                                        collect_sonic_adapter.py output root (``*__motion_XXX.hdf5``),
                                        so evaluations start only from FILTERED reference init states.
* ``ReplaySource(hdf5_or_npz)``        -- the recorded executed token + finger scalar of one demo, to be
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


_REPLAY_RE = re.compile(r"__motion_(\d+)_replay\.npz$")


def demo_motion_ids(source: str | Path) -> list[int]:
    """Motion ids with a demonstration: from a collector HDF5 root, a dir of ``*_replay.npz`` extracts
    (see ``extract_replay``), or a text file of ids (the ``motion_ids.txt`` manifest)."""
    src = Path(source).expanduser()
    if src.is_file():
        return sorted({int(t) for t in re.split(r"[\s,]+", src.read_text().strip()) if t})
    ids = {int(m.group(1)) for p in src.rglob("*.hdf5") for m in [_MOTION_RE.search(p.name)] if m}
    ids |= {int(m.group(1)) for p in src.rglob("*_replay.npz") for m in [_REPLAY_RE.search(p.name)] if m}
    if not ids:
        raise FileNotFoundError(f"no '*__motion_XXX.hdf5' / '*_replay.npz' under {src}")
    return sorted(ids)


def extract_replay(hdf5_root: str | Path, out_dir: str | Path) -> list[Path]:
    """Write ``<stem>_replay.npz`` (tokens, fingers, motion_id) + ``motion_ids.txt`` for every demo under
    ``hdf5_root`` -- a few hundred KB per motion, so replay/filtered-sweep can run on a box that does
    not hold the 0.5 GB HDF5s."""
    import h5py

    out = Path(out_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)
    written = []
    for f in sorted(Path(hdf5_root).expanduser().rglob("*.hdf5")):
        with h5py.File(f, "r") as h:
            g = h["data/demo_0"]
            meta = json.loads(h.attrs["metadata_json"]) if "metadata_json" in h.attrs else {}
            np.savez(out / (f.stem + "_replay.npz"), tokens=g["obs/motion_token"][()].astype(np.float32),
                     fingers=g["actions"][()][:, 64].astype(np.float32), motion_id=int(meta.get("motion_id", -1)))
        written.append(out / (f.stem + "_replay.npz"))
    (out / "motion_ids.txt").write_text(" ".join(str(i) for i in demo_motion_ids(out)) + "
")
    return written


class ReplaySource:
    """Recorded executed token (64) + right-finger scalar per step of one collected demo."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if self.path.suffix == ".npz":
            d = np.load(self.path)
            self.tokens = np.asarray(d["tokens"], dtype=np.float32)                        # (T, 64) FSQ-snapped
            self.fingers = np.asarray(d["fingers"], dtype=np.float32)                      # (T,) <0 closes
            self.motion_id = int(d["motion_id"])
        else:
            import h5py

            with h5py.File(self.path, "r") as h:
                g = h["data/demo_0"]
                self.tokens = np.asarray(g["obs/motion_token"][()], dtype=np.float32)
                self.fingers = np.asarray(g["actions"][()][:, 64], dtype=np.float32)
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
