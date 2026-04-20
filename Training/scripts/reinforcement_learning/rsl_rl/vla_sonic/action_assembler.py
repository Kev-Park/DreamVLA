"""UTM body action + VLA finger output → env action, with correct joint-order handling.

Two very similar-looking but distinct joint orderings are involved:

- **MuJoCo order**: `g1_29dof_sonic_model12.yaml::DEFAULT_DOF_ANGLES`. Left-leg-
  all-6-joints, then right-leg-all-6, then waist_yaw/roll/pitch, then left-arm-
  all-7, then right-arm-all-7. This is the ordering of `default_angles`,
  `g1_action_scale`, and the planner's `mujoco_qpos` output.

- **SONIC-IsaacLab order**: the order the UTM ONNX models use on **both** input
  (history joint positions, velocities, last actions) and **output** (decoder's
  29-D body action). Interleaves left/right pairs at each kinematic level:
  left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, right_hip_roll,
  waist_roll, left_hip_yaw, right_hip_yaw, waist_pitch, left_knee, right_knee,
  left_shoulder_pitch, right_shoulder_pitch, left_ankle_pitch, ...
  Reconstructed from `policy_parameters.hpp::isaaclab_to_mujoco`.

- **Env JointNamesOrder** (27 joints): our Isaac Lab env's actual articulation
  layout after dropping waist_roll/pitch. Defined in
  ``motion_lib_base.py::JointNamesOrder``.

The deploy reference applies ``q_target = default + utm * scale`` with both
``default`` and ``scale`` in MuJoCo order while ``utm`` is in SONIC-IsaacLab
order, using the permutation ``floatarr[isaaclab_to_mujoco[mujoco_i]]`` to
bridge the two at element access time (policy_parameters.hpp line 3120).

We flatten that here by pre-permuting ``default`` and ``scale`` into SONIC-
IsaacLab order so the arithmetic is vectorisable.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


# =========================================================================
# Joint-order bridges (from policy_parameters.hpp)
# =========================================================================

# "mujoco order in isaaclab index": for each MuJoCo index, the SONIC-IsaacLab idx.
# Source: policy_parameters.hpp:100-101.
MUJOCO_TO_ISAACLAB = np.array([
    0,  3,  6,  9, 13, 17,  1,  4,  7, 10, 14, 18,  2,  5,  8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
], dtype=np.int64)

# Inverse: "isaaclab order in mujoco index" — for each SONIC-IsaacLab idx, the MuJoCo idx.
# Source: policy_parameters.hpp:103-104.
ISAACLAB_TO_MUJOCO = np.array([
    0,  6, 12,  1,  7, 13,  2,  8, 14,  3,  9, 15, 22,  4, 10,
    16, 23,  5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
], dtype=np.int64)


# =========================================================================
# Per-joint scale & default — in MuJoCo order per policy_parameters.hpp
# =========================================================================

# Motor armatures (kg·m²)
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425
_NATURAL_FREQ = 10.0 * 2.0 * np.pi  # 10 Hz

_K_5020 = _ARMATURE_5020 * _NATURAL_FREQ * _NATURAL_FREQ
_K_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ * _NATURAL_FREQ
_K_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ * _NATURAL_FREQ
_K_4010 = _ARMATURE_4010 * _NATURAL_FREQ * _NATURAL_FREQ

_E_5020 = 25.0
_E_7520_14 = 88.0
_E_7520_22 = 139.0
_E_4010 = 5.0

_S_5020 = 0.25 * _E_5020 / _K_5020
_S_7520_14 = 0.25 * _E_7520_14 / _K_7520_14
_S_7520_22 = 0.25 * _E_7520_22 / _K_7520_22
_S_4010 = 0.25 * _E_4010 / _K_4010

# Per-joint action scale in MuJoCo order (= DEFAULT_DOF_ANGLES order).
_G1_ACTION_SCALE_MUJOCO = np.array([
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

# Standing pose in MuJoCo order (from policy_parameters.hpp::default_angles).
_G1_DEFAULT_ANGLES_MUJOCO = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,        # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,        # right leg
    0.0, 0.0, 0.0,                               # waist yaw, roll, pitch
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,           # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,          # right arm
], dtype=np.float32)

# Empirical global action-scale multiplier. The deploy reference's formula
# ``0.25 * effort_limit / stiffness`` produces scales in [0.075, 0.55] which,
# when applied to our UTM ONNX output, yields joint-target deltas 10× larger
# than the expected "hold current pose" deltas derived by inverse-mapping the
# dataset's observation.state through the same formula. Diagnostic heredocs
# (feeding dataset step-0 inputs through the UTM and grid-searching the
# multiplier) showed that ×0.1 makes the UTM's output match the expected
# small-delta magnitudes (~0.3 rad max vs ~6 rad). Best current hypothesis:
# SONIC's training used an additional 0.1× scalar on top of the formula that
# the deploy C++ implicitly applies elsewhere.
#
# TODO: confirm against SONIC training config (look for `action_scale` or
# `action_scale_multiplier` in `gear_sonic/config/actor_critic/*.yaml`).
_SONIC_EMPIRICAL_ACTION_SCALE_MULT = 0.1

# Pre-permute into SONIC-IsaacLab order so we can do element-wise arithmetic
# with the UTM decoder's (SONIC-ordered) output.
# Given MUJOCO_TO_ISAACLAB[mj_idx] = sonic_idx, the inverse ISAACLAB_TO_MUJOCO[s_idx] = mj_idx
# tells us which MuJoCo-ordered value to place at each SONIC index.
G1_ACTION_SCALE_SONIC = (
    _G1_ACTION_SCALE_MUJOCO[ISAACLAB_TO_MUJOCO] * _SONIC_EMPIRICAL_ACTION_SCALE_MULT
).astype(np.float32)
G1_DEFAULT_ANGLES_SONIC = _G1_DEFAULT_ANGLES_MUJOCO[ISAACLAB_TO_MUJOCO].astype(np.float32)


# =========================================================================
# Waist-joint indices to drop when mapping UTM 29-D → env 27-D body.
# =========================================================================

# In SONIC-IsaacLab order, waist_roll is 5 and waist_pitch is 8.
WAIST_ROLL_SONIC_IDX = 5
WAIST_PITCH_SONIC_IDX = 8

# Permutation from the post-drop SONIC-IsaacLab-27 order to the env's
# JointNamesOrder-27 (motion_lib_base.py). Hand-derived by matching names:
# env[j] = sonic27[PERM_SONIC_27_TO_ENV_27[j]].
PERM_SONIC27_TO_ENV27 = np.array([
    0,   # env[0]  left_hip_pitch      ← sonic27[0]
    3,   # env[1]  left_hip_roll       ← sonic27[3]
    5,   # env[2]  left_hip_yaw        ← sonic27[5]  (was SONIC29[6], shifted -1 after drop-5)
    7,   # env[3]  left_knee           ← sonic27[7]  (was SONIC29[9], shifted -2 after drop-5,8)
    11,  # env[4]  left_ankle_pitch    ← sonic27[11] (was SONIC29[13])
    15,  # env[5]  left_ankle_roll     ← sonic27[15] (was SONIC29[17])
    1,   # env[6]  right_hip_pitch     ← sonic27[1]
    4,   # env[7]  right_hip_roll      ← sonic27[4]
    6,   # env[8]  right_hip_yaw       ← sonic27[6]  (was SONIC29[7])
    8,   # env[9]  right_knee          ← sonic27[8]  (was SONIC29[10])
    12,  # env[10] right_ankle_pitch   ← sonic27[12] (was SONIC29[14])
    16,  # env[11] right_ankle_roll    ← sonic27[16] (was SONIC29[18])
    2,   # env[12] waist_yaw           ← sonic27[2]
    9,   # env[13] left_shoulder_pitch ← sonic27[9]  (was SONIC29[11])
    13,  # env[14] left_shoulder_roll  ← sonic27[13] (was SONIC29[15])
    17,  # env[15] left_shoulder_yaw   ← sonic27[17] (was SONIC29[19])
    19,  # env[16] left_elbow          ← sonic27[19] (was SONIC29[21])
    21,  # env[17] left_wrist_roll     ← sonic27[21] (was SONIC29[23])
    23,  # env[18] left_wrist_pitch    ← sonic27[23] (was SONIC29[25])
    25,  # env[19] left_wrist_yaw      ← sonic27[25] (was SONIC29[27])
    10,  # env[20] right_shoulder_pitch← sonic27[10] (was SONIC29[12])
    14,  # env[21] right_shoulder_roll ← sonic27[14] (was SONIC29[16])
    18,  # env[22] right_shoulder_yaw  ← sonic27[18] (was SONIC29[20])
    20,  # env[23] right_elbow         ← sonic27[20] (was SONIC29[22])
    22,  # env[24] right_wrist_roll    ← sonic27[22] (was SONIC29[24])
    24,  # env[25] right_wrist_pitch   ← sonic27[24] (was SONIC29[26])
    26,  # env[26] right_wrist_yaw     ← sonic27[26] (was SONIC29[28])
], dtype=np.int64)


# =========================================================================
# Public helpers.
# =========================================================================

def utm_body_29_to_q_target_29_sonic(utm_body_29_sonic: np.ndarray) -> np.ndarray:
    """Apply the deploy transform ``q = default + utm * scale`` in SONIC-IsaacLab order.

    Input and output are both (29,) in SONIC-IsaacLab order. Waist_roll/pitch
    slots are included here (they'll be dropped downstream).
    """
    arr = np.asarray(utm_body_29_sonic, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 29:
        raise ValueError(f"utm_body_29 must be 29-D, got {arr.shape}")
    return (G1_DEFAULT_ANGLES_SONIC + arr * G1_ACTION_SCALE_SONIC).astype(np.float32)


def utm_body_29_to_env_27(utm_body_29_sonic: np.ndarray, *, apply_default_offset: bool = True) -> np.ndarray:
    """UTM decoder output (SONIC-IsaacLab 29-D) → env body action (JointNamesOrder 27-D).

    Pipeline:
      1. (optional) Apply ``q = default + utm * scale`` element-wise, all in
         SONIC-IsaacLab order.
      2. Drop waist_roll (idx 5) and waist_pitch (idx 8) — not present in the
         env's 27-DoF articulation.
      3. Permute remaining 27 into ``JointNamesOrder``.
    """
    arr = np.asarray(utm_body_29_sonic, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 29:
        raise ValueError(f"utm_body_29 must be 29-D, got {arr.shape}")
    if apply_default_offset:
        arr = utm_body_29_to_q_target_29_sonic(arr)
    sonic27 = np.delete(arr, [WAIST_ROLL_SONIC_IDX, WAIST_PITCH_SONIC_IDX])
    return sonic27[PERM_SONIC27_TO_ENV27].astype(np.float32)


def utm_plus_vla_to_env_action(
    utm_body_29_sonic: np.ndarray,
    vla_action: Mapping[str, np.ndarray],
    *,
    t_index: int = 0,
    batch_index: int = 0,
) -> np.ndarray:
    """Assemble the 41-D env action from UTM body output + VLA finger predictions.

    Args:
        utm_body_29_sonic: UTM decoder output in SONIC-IsaacLab order, shape
            ``(29,)`` or ``(1, 29)``.
        vla_action: VLA action dict from ``Gr00tPolicy.get_action(obs)[0]``.
            Must contain ``left_hand_joints`` and ``right_hand_joints``, each
            shape ``(B, T, 7)``.
        t_index: which of the T predicted steps to use (default 0).
        batch_index: which env in the batch (default 0).

    Returns:
        ``np.ndarray`` of shape ``(41,)``, dtype float32: 27 body (in env
        JointNamesOrder) + 7 left fingers + 7 right fingers.
    """
    body_27 = utm_body_29_to_env_27(utm_body_29_sonic)

    def _pick_fingers(key: str) -> np.ndarray:
        if key not in vla_action:
            raise KeyError(f"vla_action missing '{key}'; got {sorted(vla_action.keys())}")
        arr = np.asarray(vla_action[key], dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"vla_action['{key}'] must be (B, T, 7); got {arr.shape}")
        slice_ = arr[batch_index, t_index]
        if slice_.shape != (7,):
            raise ValueError(
                f"vla_action['{key}'] last dim must be 7 (fingers); got shape {slice_.shape}"
            )
        return slice_.astype(np.float32)

    left = _pick_fingers("left_hand_joints")
    right = _pick_fingers("right_hand_joints")

    return np.concatenate([body_27, left, right], axis=0).astype(np.float32)


# =========================================================================
# Backwards-compat re-exports so eval_vla_sonic.py imports don't break.
# =========================================================================

# Older code imported these names; preserve them, but point at SONIC-side values.
WAIST_ROLL_IDX = WAIST_ROLL_SONIC_IDX
WAIST_PITCH_IDX = WAIST_PITCH_SONIC_IDX
G1_ACTION_SCALE = G1_ACTION_SCALE_SONIC
G1_DEFAULT_ANGLES = G1_DEFAULT_ANGLES_SONIC
