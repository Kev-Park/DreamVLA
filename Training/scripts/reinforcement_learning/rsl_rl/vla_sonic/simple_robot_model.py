"""Pinocchio-free shim providing the minimum ``RobotModel`` surface obs_to_policy needs.

The real ``gear_sonic.data.robot_model.RobotModel`` instantiates pinocchio to
parse the URDF. On some setups pinocchio's pybind11 bindings fail to register
(``No Python class registered for C++ class std::vector<std::string>``). We
don't actually need the kinematic model at eval time — only:

  * ``joint_names``: list[str] in gear_sonic's order (used to permute Isaac
    joint readings into the order the VLA was trained on).
  * ``get_joint_group_indices(name)``: list[int] of indices into ``joint_names``
    for each joint group. Used to slice per-group ``observation.state.<group>``
    tensors.

Both come from static data in
``gear_sonic/data/robot_model/supplemental_info/g1/g1_supplemental_info.py``.
The natural ordering below places every joint group contiguously in
``joint_names``, which is what obs_to_policy's ``slice(start, end+1)`` readout
requires.
"""

from __future__ import annotations

from dataclasses import dataclass


# =========================================================================
# Authoritative joint-ordering table (mirrors
# ``G1SupplementalInfo.joint_groups`` primitive groups).
# =========================================================================

_PRIMITIVE_GROUPS: list[tuple[str, list[str]]] = [
    ("left_leg", [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ]),
    ("right_leg", [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]),
    ("waist", [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]),
    ("left_arm", [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]),
    ("left_hand", [
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
    ]),
    ("right_arm", [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]),
    ("right_hand", [
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
    ]),
]

# Composite groups defined as unions of primitive groups.
_COMPOSITE_GROUPS: dict[str, list[str]] = {
    "legs": ["left_leg", "right_leg"],
    "arms": ["left_arm", "right_arm"],
    "hands": ["left_hand", "right_hand"],
    "lower_body": ["waist", "legs"],
    "upper_body_no_hands": ["arms"],
    "body": ["lower_body", "upper_body_no_hands"],
    "upper_body": ["upper_body_no_hands", "hands"],
}


@dataclass
class SimpleG1RobotModel:
    """Pinocchio-free replacement exposing the two methods obs_to_policy needs.

    ``joint_names`` and the primitive-group offsets mirror
    ``G1SupplementalInfo.joint_groups``; composite groups are resolved by
    flattening their member primitive groups.
    """

    joint_names: list[str]
    _primitive_indices: dict[str, list[int]]

    @classmethod
    def build(cls) -> "SimpleG1RobotModel":
        joint_names: list[str] = []
        primitive_indices: dict[str, list[int]] = {}
        for group, names in _PRIMITIVE_GROUPS:
            start = len(joint_names)
            joint_names.extend(names)
            primitive_indices[group] = list(range(start, len(joint_names)))
        return cls(joint_names=joint_names, _primitive_indices=primitive_indices)

    def get_joint_group_indices(self, group: str) -> list[int]:
        """Return indices into ``joint_names`` for ``group``.

        Resolves composite groups by recursively flattening primitives, then
        sorts + dedupes so obs_to_policy can build a ``slice(start, end+1)``.
        """
        if group in self._primitive_indices:
            return list(self._primitive_indices[group])
        if group in _COMPOSITE_GROUPS:
            acc: set[int] = set()
            for sub in _COMPOSITE_GROUPS[group]:
                acc.update(self.get_joint_group_indices(sub))
            return sorted(acc)
        raise KeyError(f"Unknown joint group: {group}")
