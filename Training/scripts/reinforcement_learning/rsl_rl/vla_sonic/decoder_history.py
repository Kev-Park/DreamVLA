"""Rolling history buffer for the UTM decoder's ``his_*_10frame_step1`` slots.

The decoder's obs_dict includes 10 past frames each of:
    - his_base_angular_velocity       (shape 10x3 → 30 dims flat)
    - his_body_joint_positions        (shape 10x29 → 290 dims flat)
    - his_body_joint_velocities       (shape 10x29 → 290 dims flat)
    - his_last_actions                (shape 10x29 → 290 dims flat)
    - his_gravity_dir                 (shape 10x3 → 30 dims flat)

The eval loop owns this buffer and ``push()``es one observation per env step.
Before `N_FRAMES` pushes have accumulated, we pad by repeating the oldest
frame we have (equivalent to assuming the robot started in the current pose
and has been stationary). The planner's ``context_mujoco_qpos`` (4 past
robot qpos frames) is also maintained here to keep all history state in one
place.

Frame ordering: oldest at index 0, newest at index N-1 — matches the
convention used by the deploy C++ gather functions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


N_HISTORY_FRAMES = 10
N_BODY_JOINTS = 29
N_PLANNER_CONTEXT_FRAMES = 4
PLANNER_QPOS_DIM = 36  # root_pos(3) + root_quat(4) + 29 joints


@dataclass
class DecoderHistory:
    """Flat tensors in the shapes ``build_decoder_obs`` expects."""

    his_body_joint_positions: np.ndarray       # (290,)
    his_body_joint_velocities: np.ndarray      # (290,)
    his_last_actions: np.ndarray               # (290,)
    his_base_angular_velocity: np.ndarray      # (30,)
    his_gravity_dir: np.ndarray                # (30,)

    def as_kwargs(self) -> dict[str, np.ndarray]:
        return {
            "history_body_joint_positions":  self.his_body_joint_positions,
            "history_body_joint_velocities": self.his_body_joint_velocities,
            "history_last_actions":          self.his_last_actions,
            "history_base_angular_velocity": self.his_base_angular_velocity,
            "history_gravity_dir":           self.his_gravity_dir,
        }


class HistoryBuffer:
    """Rolling per-frame buffers for decoder history + planner context.

    Usage:
        hist = HistoryBuffer()
        for step in range(max_steps):
            obs = env.step(action)
            hist.push(
                joint_pos=...,           # (29,) float32
                joint_vel=...,           # (29,) float32
                last_action=prev_action, # (29,) float32 (zeros on step 0)
                base_ang_vel=...,        # (3,)  float32
                gravity_dir=...,         # (3,)  float32
                mujoco_qpos=...,         # (36,) float32 — current frame
            )
            dec_hist = hist.decoder_history()
            planner_ctx = hist.planner_context()  # (1, 4, 36)
            # ... run UTM decoder + planner ...

    Before ``N_HISTORY_FRAMES`` pushes have accumulated, the missing prefix
    is padded by repeating the oldest observed frame.
    """

    def __init__(
        self,
        n_history_frames: int = N_HISTORY_FRAMES,
        n_body_joints: int = N_BODY_JOINTS,
        n_planner_context: int = N_PLANNER_CONTEXT_FRAMES,
    ) -> None:
        self.n_history_frames = n_history_frames
        self.n_body_joints = n_body_joints
        self.n_planner_context = n_planner_context

        self._joint_pos: deque[np.ndarray] = deque(maxlen=n_history_frames)
        self._joint_vel: deque[np.ndarray] = deque(maxlen=n_history_frames)
        self._last_actions: deque[np.ndarray] = deque(maxlen=n_history_frames)
        self._base_ang_vel: deque[np.ndarray] = deque(maxlen=n_history_frames)
        self._gravity_dir: deque[np.ndarray] = deque(maxlen=n_history_frames)
        self._mujoco_qpos: deque[np.ndarray] = deque(maxlen=n_planner_context)

    # ---------- mutation ----------

    def push(
        self,
        *,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        last_action: np.ndarray,
        base_ang_vel: np.ndarray,
        gravity_dir: np.ndarray,
        mujoco_qpos: np.ndarray,
    ) -> None:
        """Append one step's observations to every rolling buffer."""
        self._joint_pos.append(_as_float32_shape(joint_pos, (self.n_body_joints,)))
        self._joint_vel.append(_as_float32_shape(joint_vel, (self.n_body_joints,)))
        self._last_actions.append(_as_float32_shape(last_action, (self.n_body_joints,)))
        self._base_ang_vel.append(_as_float32_shape(base_ang_vel, (3,)))
        self._gravity_dir.append(_as_float32_shape(gravity_dir, (3,)))
        self._mujoco_qpos.append(_as_float32_shape(mujoco_qpos, (PLANNER_QPOS_DIM,)))

    def reset(self) -> None:
        """Drop all history; call at episode boundaries."""
        for dq in (
            self._joint_pos, self._joint_vel, self._last_actions,
            self._base_ang_vel, self._gravity_dir, self._mujoco_qpos,
        ):
            dq.clear()

    # ---------- readout ----------

    def decoder_history(self) -> DecoderHistory:
        """Build the five flat tensors ``build_decoder_obs`` consumes."""
        return DecoderHistory(
            his_body_joint_positions  = _stack_padded(self._joint_pos,    self.n_history_frames),
            his_body_joint_velocities = _stack_padded(self._joint_vel,    self.n_history_frames),
            his_last_actions          = _stack_padded(self._last_actions, self.n_history_frames),
            his_base_angular_velocity = _stack_padded(self._base_ang_vel, self.n_history_frames),
            his_gravity_dir           = _stack_padded(self._gravity_dir,  self.n_history_frames),
        )

    def planner_context(self) -> np.ndarray:
        """Build the planner's context_mujoco_qpos tensor: shape (1, 4, 36) float32."""
        stacked = _stack_padded(self._mujoco_qpos, self.n_planner_context, flatten=False)
        # _stack_padded with flatten=False returns (n_frames, D). Add batch dim.
        return stacked[None, ...].astype(np.float32)

    @property
    def is_warmed_up(self) -> bool:
        """True once ``N_HISTORY_FRAMES`` real observations have been pushed."""
        return len(self._joint_pos) == self.n_history_frames


def _as_float32_shape(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if out.shape != shape:
        raise ValueError(f"expected shape {shape}, got {out.shape}")
    return out


def _stack_padded(
    dq: Iterable[np.ndarray], n_frames: int, *, flatten: bool = True
) -> np.ndarray:
    """Stack a deque into a (n_frames, D) array, left-padding by repeating oldest.

    If flatten=True, returns (n_frames * D,). If flatten=False, (n_frames, D).
    Empty deque raises — caller should push at least once before readout.
    """
    frames = list(dq)
    if not frames:
        raise RuntimeError(
            "HistoryBuffer has no frames yet — push at least once before reading."
        )
    # Pad prefix with oldest frame to reach n_frames.
    while len(frames) < n_frames:
        frames.insert(0, frames[0])
    stacked = np.stack(frames, axis=0)  # (n_frames, D)
    return stacked.reshape(-1).astype(np.float32) if flatten else stacked.astype(np.float32)
