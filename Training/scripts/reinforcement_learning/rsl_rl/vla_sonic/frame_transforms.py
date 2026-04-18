"""Coordinate transforms + mode derivation shared across the VLA→SONIC bridge.

All quaternions in this module follow **wxyz** convention (scalar first), which
matches Isaac Lab's `robot.data.root_quat_w` and the dataset's feature schema.
SciPy's Rotation class uses xyzw; conversions handled by the helpers below.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def speed_to_mode(speed: float) -> int:
    """Horizontal speed (m/s) → locomotion mode int for planner_sonic.onnx.

    Threshold table from gear_sonic_deploy's localmotion_kplanner.hpp:
        0 = IDLE          (static)
        1 = SLOW_WALK     (0.1 ~ 0.8 m/s)
        2 = WALK          (0.8 ~ 2.5 m/s)
        3 = RUN           (2.5 ~ 7.5 m/s)
    """
    s = abs(float(speed))
    if s < 0.1:
        return 0
    if s < 0.8:
        return 1
    if s < 2.5:
        return 2
    return 3


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    """Scalar-first → scalar-last quaternion. Accepts (..., 4)."""
    q = np.asarray(q_wxyz)
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    """Scalar-last → scalar-first quaternion. Accepts (..., 4)."""
    q = np.asarray(q_xyzw)
    return np.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1)


def world_to_anchor_local_position(
    pos_world: np.ndarray,
    anchor_pos_world: np.ndarray,
    anchor_quat_wxyz: np.ndarray,
) -> np.ndarray:
    """Express world-frame positions in the anchor's local frame.

    Mirrors gear_sonic's observation function at
    envs/manager_env/mdp/observations.py:1348-1363:
        diff = pos_world - anchor_pos_world
        local = quat_apply(quat_inv(anchor_quat), diff)

    Shapes:
        pos_world:          (..., 3)
        anchor_pos_world:   (..., 3)   — broadcasts against pos_world
        anchor_quat_wxyz:   (..., 4)   — broadcasts against pos_world
    """
    diff = np.asarray(pos_world) - np.asarray(anchor_pos_world)
    rot_inv = R.from_quat(quat_wxyz_to_xyzw(anchor_quat_wxyz)).inv()
    return rot_inv.apply(diff)


def world_to_anchor_local_orientation(
    quat_world_wxyz: np.ndarray, anchor_quat_wxyz: np.ndarray
) -> np.ndarray:
    """Express a world-frame orientation in the anchor's local frame (wxyz out).

    Composition:  q_local = q_anchor^{-1} * q_world
    """
    rot_world = R.from_quat(quat_wxyz_to_xyzw(quat_world_wxyz))
    rot_anchor_inv = R.from_quat(quat_wxyz_to_xyzw(anchor_quat_wxyz)).inv()
    rot_local = rot_anchor_inv * rot_world
    return quat_xyzw_to_wxyz(rot_local.as_quat())


def body_vel_to_world(v_body: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate a body-frame velocity into world frame. For Stage 3 planner input.

    v_world = R(root_quat) · v_body
    """
    return R.from_quat(quat_wxyz_to_xyzw(root_quat_wxyz)).apply(v_body)


def world_vel_to_body(v_world: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate a world-frame velocity into body frame. Inverse of body_vel_to_world."""
    return R.from_quat(quat_wxyz_to_xyzw(root_quat_wxyz)).inv().apply(v_world)


def pelvis_relative_pose(
    body_pos_w: np.ndarray,
    body_quat_w_wxyz: np.ndarray,
    pelvis_pos_w: np.ndarray,
    pelvis_quat_w_wxyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a world-frame body pose relative to the pelvis frame.

    Matches the convention used by WBCBenchmark's recorder (see
    collect_pick_cam.py — body poses stored in HDF5 are pelvis-relative).

    Returns (pos_pelvis_local, quat_pelvis_local_wxyz).
    """
    body_pos_w = np.asarray(body_pos_w)
    pelvis_pos_w = np.asarray(pelvis_pos_w)
    R_pelvis_inv = R.from_quat(quat_wxyz_to_xyzw(pelvis_quat_w_wxyz)).inv()
    pos_local = R_pelvis_inv.apply(body_pos_w - pelvis_pos_w)
    R_body = R.from_quat(quat_wxyz_to_xyzw(body_quat_w_wxyz))
    R_local = R_pelvis_inv * R_body
    return pos_local, quat_xyzw_to_wxyz(R_local.as_quat())
