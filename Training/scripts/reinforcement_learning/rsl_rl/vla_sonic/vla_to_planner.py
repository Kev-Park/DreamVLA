"""Glue module: VLA action dict + env context → kinematic planner inputs.

At eval time, ``Gr00tPolicy.get_action(obs)`` returns a dict of predicted
action tensors shaped ``(B, T, D)``. This module picks the correct time slice,
derives the planner ``mode`` from predicted speed, and packs the tensors into
the exact keyword arguments ``PlannerWrapper.run`` consumes.

The planner also needs ``context_mujoco_qpos`` — 4 past frames of the robot's
own 36-D qpos — which is **not** in the VLA output. The caller is responsible
for maintaining that rolling buffer from env state and passing it in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .frame_transforms import speed_to_mode


# Keys the VLA's action dict must contain (matches examples/G1/gear_sonic_config.py).
# All values are expected to have shape (B, T_action, D).
_REQUIRED_VLA_ACTION_KEYS = (
    "planner_movement",   # (B, T, 3)
    "planner_facing",     # (B, T, 3)
    "planner_speed",      # (B, T, 1)
    "planner_height",     # (B, T, 1)
)


@dataclass
class PlannerInputs:
    """Structured planner input — pass to ``PlannerWrapper.run(**inputs.as_kwargs())``."""

    context_mujoco_qpos: np.ndarray   # (1, 4, 36) float32
    target_vel: np.ndarray            # (1,)      float32
    mode: np.ndarray                  # (1,)      int64
    movement_direction: np.ndarray    # (1, 3)    float32
    facing_direction: np.ndarray      # (1, 3)    float32
    height: np.ndarray                # (1,)      float32

    def as_kwargs(self) -> dict[str, np.ndarray]:
        return {
            "context_mujoco_qpos": self.context_mujoco_qpos,
            "target_vel":          self.target_vel,
            "mode":                self.mode,
            "movement_direction":  self.movement_direction,
            "facing_direction":    self.facing_direction,
            "height":              self.height,
        }


def build_planner_inputs(
    vla_action: Mapping[str, np.ndarray],
    context_mujoco_qpos: np.ndarray,
    *,
    t_index: int = 0,
    batch_index: int = 0,
    clip_negative_speed: bool = True,
) -> PlannerInputs:
    """Build planner kwargs from one time-slice of the VLA's action dict.

    Args:
        vla_action: output of ``Gr00tPolicy.get_action(obs)[0]`` (the action
            dict). Each value must be shape ``(B, T, D)``.
        context_mujoco_qpos: the rolling buffer of past robot qpos, shape
            ``(1, 4, 36)`` float32. Built by the eval loop from env state.
        t_index: which of the T predicted steps to use. Default 0 = execute
            the VLA's immediate-next prediction. For action chunking you'd
            loop over 0..T-1 between replans.
        batch_index: which env in the batch to extract (default 0 for a
            single-env eval).
        clip_negative_speed: if True, clamp predicted speed to >= 0 before
            computing ``mode``. VLA can over- or undershoot slightly; a
            negative speed is meaningless and would make ``speed_to_mode``
            still pick IDLE but it's tidier to clamp.
    """
    missing = [k for k in _REQUIRED_VLA_ACTION_KEYS if k not in vla_action]
    if missing:
        raise KeyError(
            f"VLA action dict missing expected keys: {missing}. "
            f"Got: {sorted(vla_action.keys())}"
        )

    if context_mujoco_qpos.shape != (1, 4, 36):
        raise ValueError(
            f"context_mujoco_qpos must be shape (1, 4, 36), got {context_mujoco_qpos.shape}"
        )
    context = np.asarray(context_mujoco_qpos, dtype=np.float32)

    # Pick the time-slice; VLA arrays are (B, T, D).
    def _pick(key: str) -> np.ndarray:
        arr = np.asarray(vla_action[key], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"vla_action['{key}'] must be 3D (B, T, D); got shape {arr.shape}")
        return arr[batch_index, t_index]  # (D,)

    movement  = _pick("planner_movement")   # (3,) VLA's raw root velocity (m/s, world frame)
    facing    = _pick("planner_facing")     # (3,) unit vector, world frame
    speed_arr = _pick("planner_speed")      # (1,) horizontal scalar speed
    height_arr= _pick("planner_height")     # (1,) root z, meters

    speed_scalar = float(speed_arr.reshape(-1)[0])
    if clip_negative_speed:
        speed_scalar = max(0.0, speed_scalar)
    mode_int = speed_to_mode(speed_scalar)

    # Build planner's `movement_direction`: per localmotion_kplanner.hpp:62 it's
    # a horizontal unit vector. VLA's `planner_movement` is a 3D velocity that
    # includes the root's Z-component (crouching while picking → big -Z). Passing
    # the raw value as "direction" confuses the planner since the interpretation
    # is a unit-length direction, not a velocity. Fix:
    #   - Drop Z, keep only horizontal components.
    #   - Zero out if IDLE (mode 0) — the direction is irrelevant when not moving.
    #   - Normalize to unit length if a meaningful direction remains.
    movement_dir = np.array([movement[0], movement[1], 0.0], dtype=np.float32)
    if mode_int == 0:
        movement_dir[:] = 0.0
    else:
        horiz_norm = float(np.linalg.norm(movement_dir[:2]))
        if horiz_norm > 1e-6:
            movement_dir[:2] /= horiz_norm
        else:
            movement_dir[:] = 0.0

    # Facing: VLA emits near-unit horizontal vector per the converter convention,
    # but defensively project to xy and renormalize if length is off.
    facing_xy = np.array([facing[0], facing[1], 0.0], dtype=np.float32)
    facing_norm = float(np.linalg.norm(facing_xy[:2]))
    if facing_norm > 1e-6:
        facing_xy[:2] /= facing_norm
    else:
        facing_xy[0] = 1.0  # fallback: forward

    return PlannerInputs(
        context_mujoco_qpos=context,
        target_vel=np.array([speed_scalar], dtype=np.float32),
        mode=np.array([mode_int], dtype=np.int64),
        movement_direction=movement_dir.reshape(1, 3),
        facing_direction=facing_xy.reshape(1, 3),
        height=np.array([float(height_arr.reshape(-1)[0])], dtype=np.float32),
    )
