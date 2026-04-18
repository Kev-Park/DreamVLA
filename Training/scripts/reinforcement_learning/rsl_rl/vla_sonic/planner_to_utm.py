"""Build the flat obs_dict tensors the UTM encoder / decoder ONNX consume.

The ONNX models take single concat'd tensors, not named inputs. This module
exposes two builder functions that lay out named observations in the exact
order the release observation_config.yaml (and the deploy-side C++ registry)
prescribe.

Encoder total: 1762 dims (14 observations).
Decoder total:  994 dims ( 6 observations).

The layout tables live as module-level constants so callers can introspect
offsets (useful for debugging or for building partial inputs with mocked
components).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from .frame_transforms import (
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    world_to_anchor_local_position,
    world_to_anchor_local_orientation,
)


# =========================================================================
# ENCODER LAYOUT (1762 dims)
# -------------------------------------------------------------------------
# Order and dims taken from
#   GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml
# with per-entry dimensions confirmed against
#   gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
#   (lines 1707-1772).
# =========================================================================

@dataclass(frozen=True)
class _EncSlot:
    name: str
    dim: int
    teleop_active: bool
    """True if the teleop encoder mode reads this slot. False slots are
    zero-filled at inference (they're for g1 or smpl modes)."""


ENCODER_LAYOUT: tuple[_EncSlot, ...] = (
    _EncSlot("encoder_mode_4",                                  4,   True),
    _EncSlot("motion_joint_positions_10frame_step5",          290,   False),
    _EncSlot("motion_joint_velocities_10frame_step5",         290,   False),
    _EncSlot("motion_root_z_position_10frame_step5",           10,   False),
    _EncSlot("motion_root_z_position",                          1,   False),
    _EncSlot("motion_anchor_orientation",                       6,   True),
    _EncSlot("motion_anchor_orientation_10frame_step5",        60,   False),
    _EncSlot("motion_joint_positions_lowerbody_10frame_step5",120,   True),
    _EncSlot("motion_joint_velocities_lowerbody_10frame_step5",120,  True),
    _EncSlot("vr_3point_local_target",                          9,   True),
    _EncSlot("vr_3point_local_orn_target",                     12,   True),
    _EncSlot("smpl_joints_10frame_step1",                     720,   False),
    _EncSlot("smpl_anchor_orientation_10frame_step1",          60,   False),
    _EncSlot("motion_joint_positions_wrists_10frame_step1",    60,   False),
)

ENCODER_TOTAL_DIM = sum(s.dim for s in ENCODER_LAYOUT)
assert ENCODER_TOTAL_DIM == 1762, f"Encoder layout mismatch: {ENCODER_TOTAL_DIM}"

# Name → (start, end) index pair for fast slice assignment.
ENCODER_SLICES: dict[str, slice] = {}
_cursor = 0
for _slot in ENCODER_LAYOUT:
    ENCODER_SLICES[_slot.name] = slice(_cursor, _cursor + _slot.dim)
    _cursor += _slot.dim


# =========================================================================
# DECODER LAYOUT (994 dims)
# =========================================================================

@dataclass(frozen=True)
class _DecSlot:
    name: str
    dim: int


DECODER_LAYOUT: tuple[_DecSlot, ...] = (
    _DecSlot("token_state",                              64),
    _DecSlot("his_base_angular_velocity_10frame_step1",  30),
    _DecSlot("his_body_joint_positions_10frame_step1",  290),
    _DecSlot("his_body_joint_velocities_10frame_step1", 290),
    _DecSlot("his_last_actions_10frame_step1",          290),
    _DecSlot("his_gravity_dir_10frame_step1",            30),
)

DECODER_TOTAL_DIM = sum(s.dim for s in DECODER_LAYOUT)
assert DECODER_TOTAL_DIM == 994, f"Decoder layout mismatch: {DECODER_TOTAL_DIM}"

DECODER_SLICES: dict[str, slice] = {}
_cursor = 0
for _d in DECODER_LAYOUT:
    DECODER_SLICES[_d.name] = slice(_cursor, _cursor + _d.dim)
    _cursor += _d.dim


# =========================================================================
# Encoder input builder (teleop mode)
# =========================================================================

# Number of lower-body joints (kept consistent with
# lower_body_joint_mujoco_order_in_isaaclab_index in the cpp registry).
N_LOWER_BODY_JOINTS = 12
ENCODER_MODE_TELEOP = 1  # mode_id for teleop (see observation_config.yaml)


def rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    """Convert SONIC-style 6D rotation → quaternion (wxyz scalar-first).

    Input convention (matches gear_sonic's
    ``utils/data_collection/transforms.py::quat_to_rot6d``):

        rot6d[..., 0:3]  = first column of R  (R[:, 0])
        rot6d[..., 3:6]  = second column of R (R[:, 1])

    So identity has rot6d = [1, 0, 0, 0, 1, 0]. The third column is
    reconstructed as the cross product of the first two (after Gram-Schmidt
    orthonormalization to guarantee a proper right-handed rotation matrix).

    Works on any leading batch shape: input (..., 6) → output (..., 4).
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    col1 = rot6d[..., 0:3]
    col2 = rot6d[..., 3:6]
    # Gram-Schmidt: normalize col1, then orthogonalize col2, then cross.
    col1 = col1 / (np.linalg.norm(col1, axis=-1, keepdims=True) + 1e-12)
    col2 = col2 - (col1 * col2).sum(axis=-1, keepdims=True) * col1
    col2 = col2 / (np.linalg.norm(col2, axis=-1, keepdims=True) + 1e-12)
    col3 = np.cross(col1, col2, axis=-1)
    mat = np.stack([col1, col2, col3], axis=-1)  # (..., 3, 3)
    # Flatten leading dims for scipy, then restore.
    flat_mat = mat.reshape(-1, 3, 3)
    quat_xyzw = R.from_matrix(flat_mat).as_quat().reshape(*mat.shape[:-2], 4)
    return quat_xyzw_to_wxyz(quat_xyzw).astype(np.float32)


def build_encoder_obs(
    *,
    # Anchor frame for VR 3-point transforms. At VLA inference, this comes
    # from the kinematic planner's first predicted frame.
    anchor_pos_world: np.ndarray,       # (3,)
    anchor_quat_wxyz: np.ndarray,       # (4,)
    anchor_rot6d: np.ndarray,           # (6,) — same anchor in rot6d form
    # Lower-body motion from the planner's mujoco_qpos output.
    lower_body_positions_future: np.ndarray,   # (10, 12)
    lower_body_velocities_future: np.ndarray,  # (10, 12)
    # VR 3-point targets in WORLD frame (as the VLA emits them).
    vr_3pt_position_world: np.ndarray,    # (9,)   [3 points × (x,y,z)]
    vr_3pt_rot6d: np.ndarray,             # (18,)  [3 points × rot6d]
) -> np.ndarray:
    """Assemble the (1, 1762) float32 encoder input for teleop mode.

    Non-teleop slots (g1-mode, smpl-mode observations) are zero-filled
    because the encoder's mode flag selects which subnetwork attends to them.
    """
    buf = np.zeros((ENCODER_TOTAL_DIM,), dtype=np.float32)

    # encoder_mode_4: [mode_id, 0, 0, 0].
    buf[ENCODER_SLICES["encoder_mode_4"]] = np.array(
        [ENCODER_MODE_TELEOP, 0.0, 0.0, 0.0], dtype=np.float32
    )

    # motion_anchor_orientation: 6D rot6d of anchor.
    buf[ENCODER_SLICES["motion_anchor_orientation"]] = np.asarray(anchor_rot6d, dtype=np.float32)

    # Lower-body future trajectories from the planner.
    lb_pos = np.asarray(lower_body_positions_future, dtype=np.float32).reshape(-1)
    lb_vel = np.asarray(lower_body_velocities_future, dtype=np.float32).reshape(-1)
    assert lb_pos.shape[0] == 120 and lb_vel.shape[0] == 120, \
        f"lower_body tensors must flatten to 120 dims; got {lb_pos.shape}, {lb_vel.shape}"
    buf[ENCODER_SLICES["motion_joint_positions_lowerbody_10frame_step5"]] = lb_pos
    buf[ENCODER_SLICES["motion_joint_velocities_lowerbody_10frame_step5"]] = lb_vel

    # VR 3-point position: transform world → anchor-local.
    pts_world = np.asarray(vr_3pt_position_world, dtype=np.float32).reshape(3, 3)
    pts_local = world_to_anchor_local_position(
        pts_world, np.asarray(anchor_pos_world, dtype=np.float32), anchor_quat_wxyz
    ).astype(np.float32)
    buf[ENCODER_SLICES["vr_3point_local_target"]] = pts_local.reshape(-1)

    # VR 3-point orientation: rot6d → quat_wxyz (world) → anchor-local quat_wxyz.
    rot6d = np.asarray(vr_3pt_rot6d, dtype=np.float32).reshape(3, 6)
    quats_world_wxyz = rot6d_to_quat_wxyz(rot6d)  # (3, 4)
    quats_local_wxyz = np.stack([
        world_to_anchor_local_orientation(q, anchor_quat_wxyz) for q in quats_world_wxyz
    ]).astype(np.float32)
    buf[ENCODER_SLICES["vr_3point_local_orn_target"]] = quats_local_wxyz.reshape(-1)

    return buf[None, :]  # (1, 1762)


# =========================================================================
# Decoder input builder
# =========================================================================

def build_decoder_obs(
    *,
    token_state: np.ndarray,                          # (64,) — from UTM encoder
    history_base_angular_velocity: np.ndarray,        # (30,) = 10 frames × 3
    history_body_joint_positions: np.ndarray,         # (290,) = 10 × 29
    history_body_joint_velocities: np.ndarray,        # (290,) = 10 × 29
    history_last_actions: np.ndarray,                 # (290,) = 10 × 29
    history_gravity_dir: np.ndarray,                  # (30,) = 10 × 3
) -> np.ndarray:
    """Assemble the (1, 994) float32 decoder input."""
    parts = {
        "token_state":                              token_state,
        "his_base_angular_velocity_10frame_step1":  history_base_angular_velocity,
        "his_body_joint_positions_10frame_step1":   history_body_joint_positions,
        "his_body_joint_velocities_10frame_step1":  history_body_joint_velocities,
        "his_last_actions_10frame_step1":           history_last_actions,
        "his_gravity_dir_10frame_step1":            history_gravity_dir,
    }
    buf = np.zeros((DECODER_TOTAL_DIM,), dtype=np.float32)
    for name, arr in parts.items():
        sl = DECODER_SLICES[name]
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        if flat.shape[0] != sl.stop - sl.start:
            raise ValueError(
                f"decoder input {name}: got {flat.shape[0]} dims, expected {sl.stop - sl.start}"
            )
        buf[sl] = flat
    return buf[None, :]  # (1, 994)
