"""UTM body action + VLA finger output → env action.

The `G1PickCamContinuousFingersEnvCfg` env expects a flat 41-D action:
    [27 body joints] ++ [7 left fingers] ++ [7 right fingers]

UTM's decoder outputs 29 body joints. The env's 27-DoF `JointNamesOrder`
skips waist_pitch (UTM index 14) and waist_roll (UTM index 13) — see
`motion_lib_base.py::JointNamesOrder` and
`g1_29dof_sonic_model12.yaml::DEFAULT_DOF_ANGLES`. After dropping those two,
the remaining ordering lines up 1:1:

    env[0:13]  = UTM[0:13]   (legs + waist_yaw)
    env[13:27] = UTM[15:29]  (arms)

VLA finger outputs are in the canonical 7-joint order matching the continuous
finger env cfg — see `G1PickCamContinuousFingersEnvCfg` in
`motion_tracking_pick_env.py`.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


# UTM 29-DoF indices to drop (0-indexed).
WAIST_ROLL_IDX = 13
WAIST_PITCH_IDX = 14


# =========================================================================
# Per-joint action transformation — copied from
# GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/
#   policy_parameters.hpp
# The UTM decoder outputs "scaled deltas" in the trained RL policy's action
# space — the deploy code applies:
#
#     q_target[i] = DEFAULT_ANGLES[i] + utm_output[i] * G1_ACTION_SCALE[i]
#
# Skipping this transform is why raw UTM output looks like random radians.
# =========================================================================

# Motor armatures (kg·m²)
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_NATURAL_FREQ = 10.0 * 2.0 * np.pi  # 10 Hz control bandwidth

# stiffness = armature * natural_freq²
_K_5020 = _ARMATURE_5020 * _NATURAL_FREQ * _NATURAL_FREQ
_K_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ * _NATURAL_FREQ
_K_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ * _NATURAL_FREQ
_K_4010 = _ARMATURE_4010 * _NATURAL_FREQ * _NATURAL_FREQ

# Effort limits (N·m)
_E_5020 = 25.0
_E_7520_14 = 88.0
_E_7520_22 = 139.0
_E_4010 = 5.0

# action_scale = 0.25 * effort_limit / stiffness, per-motor-type.
_S_5020 = 0.25 * _E_5020 / _K_5020
_S_7520_14 = 0.25 * _E_7520_14 / _K_7520_14
_S_7520_22 = 0.25 * _E_7520_22 / _K_7520_22
_S_4010 = 0.25 * _E_4010 / _K_4010

# Per-joint action scale in UTM's 29-DoF order (= MuJoCo / DEFAULT_DOF_ANGLES order).
G1_ACTION_SCALE = np.array([
    _S_7520_22,  # 0  left_hip_pitch
    _S_7520_22,  # 1  left_hip_roll
    _S_7520_14,  # 2  left_hip_yaw
    _S_7520_22,  # 3  left_knee
    _S_5020,     # 4  left_ankle_pitch
    _S_5020,     # 5  left_ankle_roll
    _S_7520_22,  # 6  right_hip_pitch
    _S_7520_22,  # 7  right_hip_roll
    _S_7520_14,  # 8  right_hip_yaw
    _S_7520_22,  # 9  right_knee
    _S_5020,     # 10 right_ankle_pitch
    _S_5020,     # 11 right_ankle_roll
    _S_7520_14,  # 12 waist_yaw
    _S_5020,     # 13 waist_roll
    _S_5020,     # 14 waist_pitch
    _S_5020,     # 15 left_shoulder_pitch
    _S_5020,     # 16 left_shoulder_roll
    _S_5020,     # 17 left_shoulder_yaw
    _S_5020,     # 18 left_elbow
    _S_5020,     # 19 left_wrist_roll
    _S_4010,     # 20 left_wrist_pitch
    _S_4010,     # 21 left_wrist_yaw
    _S_5020,     # 22 right_shoulder_pitch
    _S_5020,     # 23 right_shoulder_roll
    _S_5020,     # 24 right_shoulder_yaw
    _S_5020,     # 25 right_elbow
    _S_5020,     # 26 right_wrist_roll
    _S_4010,     # 27 right_wrist_pitch
    _S_4010,     # 28 right_wrist_yaw
], dtype=np.float32)

# Standing pose, UTM's 29-DoF order (from policy_parameters.hpp::default_angles).
G1_DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,        # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,        # right leg
    0.0, 0.0, 0.0,                               # waist yaw, roll, pitch
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,           # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,          # right arm
], dtype=np.float32)


def utm_body_29_to_q_target_29(utm_body_29: np.ndarray) -> np.ndarray:
    """Apply the deploy-canonical delta-from-default transform."""
    arr = np.asarray(utm_body_29, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 29:
        raise ValueError(f"utm_body_29 must be 29-D, got {arr.shape}")
    return (G1_DEFAULT_ANGLES + arr * G1_ACTION_SCALE).astype(np.float32)


def utm_body_29_to_env_27(utm_body_29: np.ndarray, *, apply_default_offset: bool = True) -> np.ndarray:
    """Drop waist_roll (13) and waist_pitch (14) from a 29-D UTM body output.

    When ``apply_default_offset`` is True (default), first apply the
    deploy-canonical ``q_target = default + utm * scale`` transform so the
    result is an absolute joint-position target in radians.
    """
    arr = np.asarray(utm_body_29, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 29:
        raise ValueError(f"utm_body_29 must be 29-D, got {arr.shape}")
    if apply_default_offset:
        arr = utm_body_29_to_q_target_29(arr)
    return np.delete(arr, [WAIST_ROLL_IDX, WAIST_PITCH_IDX]).astype(np.float32)


def utm_plus_vla_to_env_action(
    utm_body_29: np.ndarray,
    vla_action: Mapping[str, np.ndarray],
    *,
    t_index: int = 0,
    batch_index: int = 0,
) -> np.ndarray:
    """Assemble the 41-D env action from UTM body output + VLA finger predictions.

    Args:
        utm_body_29: UTM decoder output, shape ``(29,)`` or ``(1, 29)``.
        vla_action: VLA action dict from ``Gr00tPolicy.get_action(obs)[0]``.
            Must contain ``left_hand_joints`` and ``right_hand_joints``, each
            shape ``(B, T, 7)``.
        t_index: which of the T predicted steps to use (default 0).
        batch_index: which env in the batch (default 0).

    Returns:
        ``np.ndarray`` of shape ``(41,)``, dtype float32. Caller is responsible
        for batching / converting to torch before ``env.step``.
    """
    body_27 = utm_body_29_to_env_27(utm_body_29)

    def _pick_fingers(key: str) -> np.ndarray:
        if key not in vla_action:
            raise KeyError(f"vla_action missing '{key}'; got {sorted(vla_action.keys())}")
        arr = np.asarray(vla_action[key], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"vla_action['{key}'] must be (B, T, 7); got {arr.shape}")
        slice_ = arr[batch_index, t_index]  # (7,)
        if slice_.shape != (7,):
            raise ValueError(
                f"vla_action['{key}'] last dim must be 7 (fingers); got shape {slice_.shape}"
            )
        return slice_.astype(np.float32)

    left = _pick_fingers("left_hand_joints")
    right = _pick_fingers("right_hand_joints")

    return np.concatenate([body_27, left, right], axis=0).astype(np.float32)
