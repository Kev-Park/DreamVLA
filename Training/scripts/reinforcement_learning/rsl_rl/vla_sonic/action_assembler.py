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


def utm_body_29_to_env_27(utm_body_29: np.ndarray) -> np.ndarray:
    """Drop waist_roll (13) and waist_pitch (14) from a 29-D UTM body output."""
    arr = np.asarray(utm_body_29, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 29:
        raise ValueError(f"utm_body_29 must be 29-D, got {arr.shape}")
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
