# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""43-DoF G1 articulation with the released SONIC controller gains."""

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.unitree import (
    G1_CFG,
    STIFFNESS_5020,
    DAMPING_5020,
    ARMATURE_5020,
    STIFFNESS_7520_14,
    DAMPING_7520_14,
    ARMATURE_7520_14,
    STIFFNESS_7520_22,
    DAMPING_7520_22,
    ARMATURE_7520_22,
    STIFFNESS_4010,
    DAMPING_4010,
    ARMATURE_4010,
)
from ..paths import ASSET_ROOT

_G1_SONIC_USD = ASSET_ROOT / "sonic_g1/g1_29dof_with_hand.usd"

G1_SONIC_CFG = G1_CFG.copy()
G1_SONIC_CFG.spawn.usd_path = str(_G1_SONIC_USD)
G1_SONIC_CFG.spawn.articulation_props.enabled_self_collisions = False
G1_SONIC_CFG.init_state.pos = (0.0, 0.0, 0.76)
G1_SONIC_CFG.init_state.joint_pos = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    "waist_.*_joint": 0.0,
    ".*_hand_.*": 0.0,
}

# These gains and limits match the public SONIC g1_model_12 checkpoint.  The
# hand actuator remains the same one used by the existing Open Drawer task.
G1_SONIC_CFG.actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_yaw_joint": 88.0,
            ".*_hip_roll_joint": 139.0,
            ".*_hip_pitch_joint": 139.0,
            ".*_knee_joint": 139.0,
        },
        velocity_limit_sim={
            ".*_hip_yaw_joint": 32.0,
            ".*_hip_roll_joint": 20.0,
            ".*_hip_pitch_joint": 20.0,
            ".*_knee_joint": 20.0,
        },
        stiffness={
            ".*_hip_pitch_joint": STIFFNESS_7520_22,
            ".*_hip_roll_joint": STIFFNESS_7520_22,
            ".*_hip_yaw_joint": STIFFNESS_7520_14,
            ".*_knee_joint": STIFFNESS_7520_22,
        },
        damping={
            ".*_hip_pitch_joint": DAMPING_7520_22,
            ".*_hip_roll_joint": DAMPING_7520_22,
            ".*_hip_yaw_joint": DAMPING_7520_14,
            ".*_knee_joint": DAMPING_7520_22,
        },
        armature={
            ".*_hip_pitch_joint": ARMATURE_7520_22,
            ".*_hip_roll_joint": ARMATURE_7520_22,
            ".*_hip_yaw_joint": ARMATURE_7520_14,
            ".*_knee_joint": ARMATURE_7520_22,
        },
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
    ),
    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        effort_limit_sim=50.0,
        velocity_limit_sim=37.0,
        stiffness=2.0 * STIFFNESS_5020,
        damping=2.0 * DAMPING_5020,
        armature=2.0 * ARMATURE_5020,
    ),
    "waist_yaw": ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        effort_limit_sim=88.0,
        velocity_limit_sim=32.0,
        stiffness=STIFFNESS_7520_14,
        damping=DAMPING_7520_14,
        armature=ARMATURE_7520_14,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
            ".*_wrist_yaw_joint",
        ],
        effort_limit_sim={
            ".*_shoulder_.*": 25.0,
            ".*_elbow_joint": 25.0,
            ".*_wrist_roll_joint": 25.0,
            ".*_wrist_pitch_joint": 5.0,
            ".*_wrist_yaw_joint": 5.0,
        },
        velocity_limit_sim={
            ".*_shoulder_.*": 37.0,
            ".*_elbow_joint": 37.0,
            ".*_wrist_roll_joint": 37.0,
            ".*_wrist_pitch_joint": 22.0,
            ".*_wrist_yaw_joint": 22.0,
        },
        stiffness={
            ".*_shoulder_.*": STIFFNESS_5020,
            ".*_elbow_joint": STIFFNESS_5020,
            ".*_wrist_roll_joint": STIFFNESS_5020,
            ".*_wrist_pitch_joint": STIFFNESS_4010,
            ".*_wrist_yaw_joint": STIFFNESS_4010,
        },
        damping={
            ".*_shoulder_.*": DAMPING_5020,
            ".*_elbow_joint": DAMPING_5020,
            ".*_wrist_roll_joint": DAMPING_5020,
            ".*_wrist_pitch_joint": DAMPING_4010,
            ".*_wrist_yaw_joint": DAMPING_4010,
        },
        armature={
            ".*_shoulder_.*": ARMATURE_5020,
            ".*_elbow_joint": ARMATURE_5020,
            ".*_wrist_roll_joint": ARMATURE_5020,
            ".*_wrist_pitch_joint": ARMATURE_4010,
            ".*_wrist_yaw_joint": ARMATURE_4010,
        },
    ),
    "hands": G1_CFG.actuators["hands"],
}
