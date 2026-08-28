# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.motion_tracking.g1.motion_tracking_env import keypts_deviation_ref_l2, joint_deviation_ref_l1, position_tracking_error, orientation_tracking_error, right_hand_state_target_reward, right_hand_binary_match_reward, target_ref, target_ref_slim, root_below_threshold, root_angle_below_threshold, current_time_enc, anchor_pos_tracking_exp, anchor_ori_tracking_exp, relative_keypts_tracking_exp, relative_body_ori_tracking_exp, global_keypts_tracking_exp, global_body_ori_tracking_exp, lower_body_keypt_vel_tracking, body_linvel_tracking_exp, body_angvel_tracking_exp, _FULL_BODY_NAMES, _FULL_BODY_KEYPT_IDXS
import numpy as np
import os
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sensors import CameraCfg
from isaac_utils.rotations import(
    slerp,
)
from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_interactive_base import G1InteractiveBaseEnvCfg, hand_state_target, hand_state_target_1, rel_pose_object_w_link, object_above_threshold, object_lift_reward, reset_object_state, rel_pose_object, hand_pose, object_approach_reward_right, G1Rewards as G1RewardsBase, TerminationsCfg as TerminationsCfgBase, ActionsCfg as ActionsCfgBase, MySceneCfg as MySceneCfgBase, EventCfg as EventCfgBase
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder


# =========================================================================
# SONIC-matched actuator configuration.
# Physical motor constants copied verbatim from
# ``GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots/g1.py:10-26``,
# which in turn match the deploy reference at
# ``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp``.
# The SONIC UTM policy was trained against these gains. Using them here keeps
# the sim's joint PD response matched to training — critical when applying the
# canonical action transform ``q_target = default + utm * 0.25 * effort/K``
# (see vla_sonic/action_assembler.py). Isaac's default G1_MINIMAL_CFG uses
# 2-5x stiffer gains, which turn the same commanded delta into 2-5x the torque
# and makes the robot overshoot violently.
# =========================================================================

_SONIC_ARMATURE_5020 = 0.003609725
_SONIC_ARMATURE_7520_14 = 0.010177520
_SONIC_ARMATURE_7520_22 = 0.025101925
_SONIC_ARMATURE_4010 = 0.00425
_SONIC_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10 Hz
_SONIC_DAMPING_RATIO = 2.0

_SONIC_STIFFNESS_5020 = _SONIC_ARMATURE_5020 * _SONIC_NATURAL_FREQ ** 2
_SONIC_STIFFNESS_7520_14 = _SONIC_ARMATURE_7520_14 * _SONIC_NATURAL_FREQ ** 2
_SONIC_STIFFNESS_7520_22 = _SONIC_ARMATURE_7520_22 * _SONIC_NATURAL_FREQ ** 2
_SONIC_STIFFNESS_4010 = _SONIC_ARMATURE_4010 * _SONIC_NATURAL_FREQ ** 2
_SONIC_DAMPING_5020 = 2.0 * _SONIC_DAMPING_RATIO * _SONIC_ARMATURE_5020 * _SONIC_NATURAL_FREQ
_SONIC_DAMPING_7520_14 = 2.0 * _SONIC_DAMPING_RATIO * _SONIC_ARMATURE_7520_14 * _SONIC_NATURAL_FREQ
_SONIC_DAMPING_7520_22 = 2.0 * _SONIC_DAMPING_RATIO * _SONIC_ARMATURE_7520_22 * _SONIC_NATURAL_FREQ
_SONIC_DAMPING_4010 = 2.0 * _SONIC_DAMPING_RATIO * _SONIC_ARMATURE_4010 * _SONIC_NATURAL_FREQ


def _build_sonic_matched_actuators() -> dict:
    """Return an ``actuators`` dict replacing Isaac's default gains with SONIC's.

    Covers every actuated joint in our 27-DoF env plus the 14 continuous
    finger joints (finger actuator gains are left at Isaac's defaults; SONIC
    training doesn't specify hand gains).
    """
    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 88.0,   # MuJoCo XML actuatorfrcrange="-88 88"
                ".*_hip_pitch_joint": 88.0,  # MuJoCo XML actuatorfrcrange="-88 88"
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 20.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": _SONIC_STIFFNESS_7520_22,
                ".*_hip_roll_joint": _SONIC_STIFFNESS_7520_22,
                ".*_hip_yaw_joint": _SONIC_STIFFNESS_7520_14,
                ".*_knee_joint": _SONIC_STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": _SONIC_DAMPING_7520_22,
                ".*_hip_roll_joint": _SONIC_DAMPING_7520_22,
                ".*_hip_yaw_joint": _SONIC_DAMPING_7520_14,
                ".*_knee_joint": _SONIC_DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": _SONIC_ARMATURE_7520_22,
                ".*_hip_roll_joint": _SONIC_ARMATURE_7520_22,
                ".*_hip_yaw_joint": _SONIC_ARMATURE_7520_14,
                ".*_knee_joint": _SONIC_ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            # 1× STIFFNESS/DAMPING/ARMATURE_5020. EMPIRICAL: although gear_sonic training
            # (g1.py:281-283) uses 2× for the ankles, the 2× variant was A/B-tested here and
            # produced worse, more rigid/oscillatory motion under PhysX implicit drives at
            # 500 Hz; 1× tracks contact more stably and was the best-performing setting.
            # (Paired with the 500 Hz substep in physics_overrides.py.)
            stiffness=_SONIC_STIFFNESS_5020,
            damping=_SONIC_DAMPING_5020,
            armature=_SONIC_ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=_SONIC_STIFFNESS_7520_14,
            damping=_SONIC_DAMPING_7520_14,
            armature=_SONIC_ARMATURE_7520_14,
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
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": _SONIC_STIFFNESS_5020,
                ".*_shoulder_roll_joint": _SONIC_STIFFNESS_5020,
                ".*_shoulder_yaw_joint": _SONIC_STIFFNESS_5020,
                ".*_elbow_joint": _SONIC_STIFFNESS_5020,
                ".*_wrist_roll_joint": _SONIC_STIFFNESS_5020,
                ".*_wrist_pitch_joint": _SONIC_STIFFNESS_4010,
                ".*_wrist_yaw_joint": _SONIC_STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": _SONIC_DAMPING_5020,
                ".*_shoulder_roll_joint": _SONIC_DAMPING_5020,
                ".*_shoulder_yaw_joint": _SONIC_DAMPING_5020,
                ".*_elbow_joint": _SONIC_DAMPING_5020,
                ".*_wrist_roll_joint": _SONIC_DAMPING_5020,
                ".*_wrist_pitch_joint": _SONIC_DAMPING_4010,
                ".*_wrist_yaw_joint": _SONIC_DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": _SONIC_ARMATURE_5020,
                ".*_shoulder_roll_joint": _SONIC_ARMATURE_5020,
                ".*_shoulder_yaw_joint": _SONIC_ARMATURE_5020,
                ".*_elbow_joint": _SONIC_ARMATURE_5020,
                ".*_wrist_roll_joint": _SONIC_ARMATURE_5020,
                ".*_wrist_pitch_joint": _SONIC_ARMATURE_4010,
                ".*_wrist_yaw_joint": _SONIC_ARMATURE_4010,
            },
        ),
        # SONIC training doesn't actuate hands. Keep Isaac's defaults for the
        # 14 finger joints — numbers copied from G1_MINIMAL_CFG.
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hand_index_.*",
                "left_hand_middle_.*",
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "right_hand_index_.*",
                "right_hand_middle_.*",
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
            ],
            effort_limit_sim=3.0,
            velocity_limit_sim=1.0,
            stiffness=5.0,
            damping=1.25,
            armature={
                "left_hand_index_.*": 0.001,
                "left_hand_middle_.*": 0.001,
                "left_hand_thumb_0_joint": 0.001,
                "left_hand_thumb_1_joint": 0.001,
                "left_hand_thumb_2_joint": 0.001,
                "right_hand_index_.*": 0.001,
                "right_hand_middle_.*": 0.001,
                "right_hand_thumb_0_joint": 0.001,
                "right_hand_thumb_1_joint": 0.001,
                "right_hand_thumb_2_joint": 0.001,
            },
        ),
    }

VISUALIZE_MARKERS = True
TRACKING = True
TASK_SPARSE = True
TASK_DENSE = False

# =========================================================================
# REWORK (env-gated): the reward/termination/obs/contact rework of REWARD_REWORK_PLAN.md.
# HS_REWORK=1 switches the pick task from the current hand-tuned grasp reward stack to:
#   #2 equal whole-body tracking (all-ones mask, base SONIC weights),
#   #4 object-as-hand reference tracking (replaces object_lift),
#   #5 ResMimic contact reward + right-hand ContactSensor (replaces wrist-pointing),
#   #6 ref-deviation + object-deviation + contact-loss>10f terminations + a smaller height backstop,
#   #7 privileged object pose + reference obj_traj in obs.
# Default OFF => the existing tuned config is byte-identical. Kept as a clean separate spec so it's
# reviewable and revertible before/after training. See REWARD_REWORK_PLAN.md.
# =========================================================================
REWORK = os.environ.get("HS_REWORK", "0") == "1"

# ResMimic contact reward / contact-loss termination knobs (#5/#6). Tunable; see plan.
REWORK_CONTACT_LAMBDA = float(os.environ.get("HS_REWORK_CONTACT_LAMBDA", "5.0"))   # r^c = c_hat * exp(-lambda / f)
REWORK_CONTACT_LOSS_FRAMES = int(os.environ.get("HS_REWORK_CONTACT_LOSS_FRAMES", "10"))  # loss-of-contact termination window
REWORK_REF_DEV_TAU = float(os.environ.get("HS_REWORK_REF_DEV_TAU", "0.5"))         # whole-body tracking-error termination (m, mean-keypt)
REWORK_OBJ_DEV_TAU = float(os.environ.get("HS_REWORK_OBJ_DEV_TAU", "0.3"))
REWORK_OBJ_DEV_ORI_TAU = float(os.environ.get("HS_REWORK_OBJ_DEV_ORI_TAU", "0.8"))
REWORK_OBJ_PC = os.environ.get("HS_REWORK_OBJ_PC", "0") == "1"  # ResMimic point-cloud object reward (replaces pos*ori product)
REWORK_OBJ_PC_LAMBDA = float(os.environ.get("HS_REWORK_OBJ_PC_LAMBDA", "10.0"))
REWORK_OBJ_PC_DEV_TAU = float(os.environ.get("HS_REWORK_OBJ_PC_DEV_TAU", "0.30"))  # ResMimic object-far: point-cloud dist (m) termination
REWORK_OBJ_PC_N = int(os.environ.get("HS_REWORK_OBJ_PC_N", "256"))  # sampled mesh points
_OBJ_MESH_PATH = os.environ.get("HS_REWORK_OBJ_MESH", "/bluesclues-data/home/sastrygrp-dvij/kevin/holosoma/src/holosoma_retargeting/holosoma_retargeting/models/mustard/mustard.obj")
# Object-frame points for the point-cloud reward: 8 bbox corners of the ~5x5x20cm bottle (half-extents).
_OBJ_LOCAL_PTS = torch.tensor([[sx * 0.025, sy * 0.025, sz * 0.10] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)], dtype=torch.float32)  # object ORIENTATION deviation (rad) from reference for termination (captures topple)         # object-tracking deviation termination (m)
REWORK_HEIGHT_BACKSTOP = float(os.environ.get("HS_REWORK_HEIGHT_BACKSTOP", "0.2")) # smaller root-height backstop (<0.3)
# Forward-projection factor for the grasp PALM (= render CYAN, = refine_al_29 HAND_FWD). The synthesized
# object reference tracks this palm post-grasp. MUST match refine_al_29.HAND_FWD baked into the dataset.
REWORK_HAND_FWD = float(os.environ.get("HS_REWORK_HAND_FWD", "1.5"))
REWORK_OBJ_W = float(os.environ.get("HS_REWORK_OBJ_W", "2.0"))  # #4 object_tracking weight (env-tunable; 2.0 matches ResMimic)
REWORK_HYBRID_RARM = os.environ.get("HS_REWORK_HYBRID_RARM", "0") == "1"  # #3 LadderMan hybrid: relax right-arm tracking so the task limb can deviate
# GLOBAL (world-frame) whole-body keypoint tracking instead of the SONIC-native root-relative term.
# The relative term is drift-invariant by construction (keypoints rotated into the robot root frame),
# so root drift is only resisted by the gentle anchor_pos kernel; this flag makes every body keypoint
# pay for drift at the global level, giving the residual a dense anti-drift gradient.
REWORK_GLOBAL_TRACK = os.environ.get("HS_REWORK_GLOBAL_TRACK", "0") == "1"
# Companion flag: also track per-link ORIENTATIONS in the world frame (root rotation composed in),
# so root heading/tilt drift is priced per link. Separate flag so mult010glob/mult010nh (which
# trained with GLOBAL_TRACK=1 + relative ori) stay exactly reproducible from their env vars.
REWORK_GLOBAL_ORI = os.environ.get("HS_REWORK_GLOBAL_ORI", "0") == "1"
REWORK_RARM_W = float(os.environ.get("HS_REWORK_RARM_W", "0.5"))  # right-arm relative-tracking weight when hybrid on (LadderMan omega/2)
REWORK_OBJ_GATE = os.environ.get("HS_REWORK_OBJ_GATE", "0") == "1"  # #2 gate object reward by reference contact (is_closed): c_hat*r + (1-c_hat)
REWORK_CONTACT_POS = os.environ.get("HS_REWORK_CONTACT_POS", "0") == "1"  # LadderMan position contact reward + pos contact-loss termination
REWORK_CONTACT_POS_W = float(os.environ.get("HS_REWORK_CONTACT_POS_W", "2.0"))
REWORK_CONTACT_POS_STD = float(os.environ.get("HS_REWORK_CONTACT_POS_STD", "0.12"))
REWORK_CONTACT_POS_OFFX = float(os.environ.get("HS_REWORK_CONTACT_POS_OFFX", "0.12"))  # wrist->grasp forward projection (m)
REWORK_CONTACT_POS_LOSS_TOL = float(os.environ.get("HS_REWORK_CONTACT_POS_LOSS_TOL", "0.25"))  # grasp-lost distance (m)
REWORK_CONTACT_POS_LOSS_FRAMES = int(os.environ.get("HS_REWORK_CONTACT_POS_LOSS_FRAMES", "10"))
# Contact reward (#5) + contact-loss termination (#6c) gate. Default ON, but they depend on the
# right-hand contact sensor actually reporting finger<->object force. Set HS_REWORK_CONTACT=0 to
# disable both if the sensor reads ~0 (else contact_loss spuriously terminates the grasp).
REWORK_CONTACT = os.environ.get("HS_REWORK_CONTACT", "1") == "1"

JOINTS_MASK = [
    1, # left_hip_pitch_joint
    1, # left_hip_roll_joint
    1, # left_hip_yaw_joint
    1, # left_knee_joint
    0, # left_ankle_pitch_joint
    0, # left_ankle_roll_joint
    1, # right_hip_pitch_joint
    1, # right_hip_roll_joint
    1, # right_hip_yaw_joint
    1, # right_knee_joint
    0, # right_ankle_pitch_joint
    0, # right_ankle_roll_joint
    1, # waist_yaw_joint
    1, # waist_roll_joint
    1, # waist_pitch_joint
    1, # left_shoulder_pitch_joint
    1, # left_shoulder_roll_joint
    1, # left_shoulder_yaw_joint
    1, # left_elbow_joint
    0, # left_wrist_roll_joint
    1, # left_wrist_pitch_joint
    1, # left_wrist_yaw_joint
    1, # right_shoulder_pitch_joint
    1, # right_shoulder_roll_joint
    1, # right_shoulder_yaw_joint
    1, # right_elbow_joint
    0, # right_wrist_roll_joint
    0, # right_wrist_pitch_joint
    0, # right_wrist_yaw_joint
]

KEYPTS_MASK = [
    1, # pelvis
    1, # pelvis_contour_link
    1, # left_hip_pitch_link
    1, # left_hip_roll_link
    1, # left_hip_yaw_link
    1, # left_knee_link
    0, # left_ankle_pitch_link
    1, # left_ankle_roll_link
    1, # right_hip_pitch_link
    1, # right_hip_roll_link
    1, # right_hip_yaw_link
    1, # right_knee_link
    0, # right_ankle_pitch_link
    1, # right_ankle_roll_link
    1, # waist_yaw_link
    1, # waist_roll_link
    1, # torso_link
    1, # logo_link
    1, # head_link
    1, # waist_support_link
    1, # imu_link
    1, # d435_link
    1, # mid360_link
    1, # left_shoulder_pitch_link
    1, # left_shoulder_roll_link
    1, # left_shoulder_yaw_link
    1, # left_elbow_link
    1, # left_wrist_roll_link
    1, # left_wrist_pitch_link
    1, # left_wrist_yaw_link
    1, # left_rubber_hand
    1, # right_shoulder_pitch_link
    1, # right_shoulder_roll_link
    1, # right_shoulder_yaw_link
    1, # right_elbow_link
    1, # right_wrist_roll_link
    1, # right_wrist_pitch_link
    0, # right_wrist_yaw_link
    0, # right_rubber_hand
]

# Right arm (right_shoulder_pitch .. right_rubber_hand, indices >=31) REMOVED from body
# tracking. The right arm is the task-execution limb: tracking it toward the reference pose
# fights the actual grasp (empirically the right arm got strong tracking reward + ~0 lift).
# Driven instead by the lift + pointing (target_orientation_error) + finger rewards, with the
# frozen base still providing the gross reach as a structural prior. Used by the relative-body
# pos/ori tracking terms only; lower-body velocity + anchor terms don't touch the arm anyway.
KEYPTS_MASK_NO_RARM = [m if i < 31 else 0 for i, m in enumerate(KEYPTS_MASK)]

# Right-arm keypoints ONLY (the complement of NO_RARM). Used for a LIGHT positional-supervision
# term on the task arm: re-adds a reach reference (the reference arm pose lands the hand at the
# grasp spot, which is exactly where the object sits) so the hand arrives before the finger closes
# -- without it the arm closes short and shoves the bottle over. Weighted well below the lift so
# it guides the reach but never vetoes the grasp.
KEYPTS_MASK_RARM_ONLY = [m if i >= 31 else 0 for i, m in enumerate(KEYPTS_MASK)]

# #2 equal whole-body tracking: track ALL 39 FK links equally (right arm included), no grasp-driven
# masking. All-ones so the relative body pos/ori terms cover the full body incl. the task arm/hand.
KEYPTS_MASK_ALL = [1 for _ in KEYPTS_MASK]


        
def reset_object_state_rework(env: ManagerBasedRLEnv, env_ids: torch.Tensor,
                              offset=[0.0, 0.0], height: float = 1.0):
    """#4 object reset onto the reference object trajectory (so object-tracking starts consistent).
    Places the object at the reference object_poses at each env's start_motion_time (world frame),
    instead of forcing z=height. Falls back to the base reset if object_poses is unavailable."""
    if not hasattr(env, "start_motion_times") or getattr(env.motion_lib, "object_poses", None) is None:
        return reset_object_state(env, env_ids, offset=offset, height=height)
    motion_times = env.start_motion_times[env_ids]
    motion_ids = env.motion_ids[env_ids]
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
    ref = motion_res["object_poses"]                                             # (M,7) env-local, offset-applied
    object: RigidObject = env.scene["object"]
    object_pos = ref[:, :3] + env.scene.env_origins[env_ids]                     # -> world
    object_pos[:, 0] += offset[0]
    object_pos[:, 1] += offset[1]
    object_quat = ref[:, 3:7]
    velocities = torch.zeros((env.scene.num_envs, 6), device=env.device)[env_ids]
    object.write_root_pose_to_sim(torch.cat([object_pos, object_quat], dim=-1), env_ids=env_ids)
    object.write_root_velocity_to_sim(velocities, env_ids=env_ids)


@configclass
class EventCfg(EventCfgBase):
    """Configuration for events."""

    add_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.0, 0.4),
            "operation": "add",
        },
    )


    # #4: under REWORK, spawn the object ON the reference object trajectory (object-tracking start
    # is consistent); otherwise the original fixed-height reset.
    reset_object = EventTerm(
        func=(reset_object_state_rework if REWORK else reset_object_state),
        params={
            "height": 1.0,
            "offset": [0.0, 0.0],
        },
        mode="reset"
    )



def target_orientation_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins  # type: ignore
    root_rot_link = math_utils.quat_unique(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :].clone())

    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32) - 1.
    motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
    root_pos = motion_res["grab_pos"] + motion_res["offsets"]  # world-frame grab target

    # Wrist orientation enforces TWO things at once (sum of angular penalties), both phases:
    #   (1) LEVELNESS — wrist local z-axis -> world up, so the hand is held flat/horizontal.
    #   (2) POINTING  — wrist local x-axis (the "forward" axis, per this file's convention)
    #       points horizontally at the grab target. Horizontal bearing only: levelness already
    #       pins z up, which forces x into the horizontal plane, so the two are consistent and
    #       don't fight (a full-3D aim WOULD fight levelness when reaching down).
    n = root_pos_link.shape[0]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=root_rot_link.device).unsqueeze(0).repeat(n, 1)
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=root_rot_link.device).unsqueeze(0).repeat(n, 1)
    z_w = math_utils.quat_apply(root_rot_link, z_axis)
    x_w = math_utils.quat_apply(root_rot_link, x_axis)

    # (1) levelness: angle of wrist-z off world up
    level_angle = torch.acos(torch.clamp(z_w[:, 2], -1.0, 1.0))

    # (2) pointing: angle between wrist-x (horizontal projection) and the horizontal bearing to target
    x_h = x_w.clone(); x_h[:, 2] = 0.0
    dir_h = (root_pos - root_pos_link).clone(); dir_h[:, 2] = 0.0
    x_h = x_h / (torch.norm(x_h, dim=1, keepdim=True) + 1e-6)
    dir_norm = torch.norm(dir_h, dim=1, keepdim=True)
    dir_h = dir_h / (dir_norm + 1e-6)
    point_angle = torch.acos(torch.clamp(torch.sum(x_h * dir_h, dim=1), -1.0, 1.0))
    # gate pointing off once essentially on top of the target (horizontal bearing ill-defined)
    point_mask = (dir_norm.squeeze(1) > 0.05).float()

    return torch.abs(level_angle) + torch.abs(point_angle) * point_mask


# =========================================================================
# REWORK (#4/#5/#6) reward + termination helpers (ResMimic-style). All read
# motion_lib.get_motion_state, which under the regenerated dataset returns
# "object_poses" (N,7 = pos+quat, offset-applied, env-local) — the reference
# object trajectory (candidate (i)). Candidate (ii) = right-hand FK is
# global_keypts[:, -1, :] (right_rubber_hand). See REWARD_REWORK_PLAN.md.
# =========================================================================
def _rework_motion_res(env):
    """(motion_res, ok) at the current motion time; ok=False if motion_lib absent."""
    if not hasattr(env, "motion_lib"):
        return None, False
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(
        device=env.device, dtype=torch.float32)
    return env.motion_lib.get_motion_state(env.motion_ids, motion_times), True


def _synth_object_ref_pos(motion_res, hand_fwd: float = REWORK_HAND_FWD):
    """Synthesized object-reference POSITION (N,3), env-local — the exact reference the approved #4
    render draws as YELLOW: static at the holosoma object rest until the grasp (is_closed), then the
    forward-projected PALM = global_keypts[-1] + hand_fwd*(global_keypts[-1] - global_keypts[-2])
    (= render CYAN, = where refine_al_29 drives the palm). Returns None if fields are missing."""
    gk = motion_res["global_keypts"]                                             # (N,39,3) env-local
    palm = gk[:, -1, :] + hand_fwd * (gk[:, -1, :] - gk[:, -2, :])               # forward-projected grasp point
    if "object_poses" in motion_res:
        rest = motion_res["object_poses"][:, :3]                                 # object rest (static pre-grab)
    elif motion_res.get("grab_pos") is not None:
        # Legacy pkls (original DreamControl motions) carry no object trajectory; their static
        # grab_pos (wrist-offset heuristic, env-local after +offsets) IS the object rest.
        rest = motion_res["grab_pos"] + motion_res["offsets"]
    else:
        return None
    is_closed = motion_res["is_closed"].float().unsqueeze(-1)                    # (N,1) 1 = post-grasp
    return torch.where(is_closed > 0.5, palm, rest)


def _get_object_pc_points(env, n=None):
    """Object-frame point cloud = vertices of the object MESH (.obj: centered, meters, z-up),
    sampled to n and cached on first call. Falls back to bbox corners if the file is unavailable."""
    if hasattr(env, "_obj_pc_pts"):
        return env._obj_pc_pts
    n = REWORK_OBJ_PC_N if n is None else n
    pts = None
    try:
        verts = []
        with open(_OBJ_MESH_PATH) as _f:
            for _ln in _f:
                if _ln.startswith("v "):
                    _c = _ln.split()
                    verts.append((float(_c[1]), float(_c[2]), float(_c[3])))
        if verts:
            if len(verts) > n:
                _step = len(verts) / float(n)
                verts = [verts[int(_i * _step)] for _i in range(n)]
            pts = torch.tensor(verts, dtype=torch.float32, device=env.device)
            _mn = pts.min(0).values.tolist(); _mx = pts.max(0).values.tolist()
            print("[obj-pc] loaded %d MESH vertices; size %s" % (pts.shape[0], [round(_mx[i]-_mn[i],3) for i in range(3)]), flush=True)
    except Exception as e:
        print("[obj-pc] mesh read failed (%s: %s); using bbox corners" % (type(e).__name__, e), flush=True)
    if pts is None:
        pts = _OBJ_LOCAL_PTS.to(env.device)
    env._obj_pc_pts = pts
    return pts


def _object_pc_dist(env, motion_res, obj_pos, ref_pos):
    """Mean point-cloud distance between the sim object and its reference pose (ResMimic
    object_point_cloud_dist): object-frame points transformed by current vs reference pose."""
    obj = env.scene["object"]
    lp = _get_object_pc_points(env)                                   # (K,3)
    nn = obj_pos.shape[0]; kk = lp.shape[0]
    v = lp.unsqueeze(0).expand(nn, kk, 3).reshape(-1, 3)
    oq = obj.data.root_quat_w.unsqueeze(1).expand(nn, kk, 4).reshape(-1, 4)
    rq = motion_res["object_poses"][:, 3:7].unsqueeze(1).expand(nn, kk, 4).reshape(-1, 4)
    cur = math_utils.quat_apply(oq, v).reshape(nn, kk, 3) + obj_pos.unsqueeze(1)
    ref = math_utils.quat_apply(rq, v).reshape(nn, kk, 3) + ref_pos.unsqueeze(1)
    return torch.norm(cur - ref, dim=-1).mean(dim=1)                  # (N,)


def object_tracking_reward(env: ManagerBasedRLEnv, pos_std: float = 0.1, ori_std: float = 0.5) -> torch.Tensor:
    """#4 ResMimic-style object-as-hand reference tracking (replaces object_lift).

    Rewards the SIM object for tracking the SYNTHESIZED object reference (rest until grasp, then the
    forward-projected palm), so a held object that follows the hand IS the grasp signal. NOT contact-
    gated (ResMimic object reward is always active). exp kernel in (0,1]."""
    obj: RigidObject = env.scene["object"]
    n = obj.data.root_pos_w.shape[0]
    motion_res, ok = _rework_motion_res(env)
    if not ok:
        return torch.zeros(n, device=env.device)
    ref_pos = _synth_object_ref_pos(motion_res)
    if ref_pos is None:
        return torch.zeros(n, device=env.device)
    obj_pos = obj.data.root_pos_w - env.scene.env_origins                       # (N,3) env-local
    if REWORK_OBJ_PC:
        # ResMimic point-cloud reward: exp(-lambda * mean object-MESH point distance) — pos+ori coupled.
        r = torch.exp(-REWORK_OBJ_PC_LAMBDA * _object_pc_dist(env, motion_res, obj_pos, ref_pos))
    else:
        d2 = torch.sum((obj_pos - ref_pos) ** 2, dim=1)
        pos_r = torch.exp(-d2 / (pos_std * pos_std))
        if "object_poses" in motion_res:
            angle = math_utils.quat_error_magnitude(obj.data.root_quat_w, motion_res["object_poses"][:, 3:7])
            ori_r = torch.exp(-(angle ** 2) / (ori_std * ori_std))
            r = pos_r * ori_r
        else:
            # Legacy pkls: no reference object orientation — position-only tracking.
            r = pos_r
    if REWORK_OBJ_GATE:
        # LadderMan-style contact gating: only require object tracking when the reference says
        # contact is required (is_closed = post-grab hold phase); free (=1) otherwise so the
        # approach phase is not spuriously shaped. c_hat is the ORACLE reference indicator.
        c_hat = motion_res["is_closed"].float()
        r = c_hat * r + (1.0 - c_hat)
    return r


def _right_hand_contact_force(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Total contact-force magnitude on the right-hand finger links (N,). Uses the existing
    contact_forces sensor (net force = contact vs anything; during the grasp/lift the fingers
    contact the object). A dedicated object-filtered sensor could sharpen this later."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]                 # (N,K,3)
    return torch.norm(forces, dim=-1).sum(dim=1)                                 # (N,)


def object_contact_reward(env: ManagerBasedRLEnv,
                          sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["right_hand_.*"]),
                          lam: float = 5.0) -> torch.Tensor:
    """#5 ResMimic contact reward r^c = c_hat * exp(-lam / f). c_hat = oracle reference contact
    (is_closed, the post-grab hold phase where the hand SHOULD be holding the object); f = measured
    right-hand contact force. 0 with no contact, -> 1 as the grip presses. Replaces wrist-pointing."""
    motion_res, ok = _rework_motion_res(env)
    n = env.scene.num_envs
    if not ok:
        return torch.zeros(n, device=env.device)
    c_hat = motion_res["is_closed"].float()
    f = _right_hand_contact_force(env, sensor_cfg)
    return c_hat * torch.exp(-lam / (f + 1e-3))


def _right_hand_grasp_point(env, asset_cfg, offset_x):
    """Robot right-hand grasp point = wrist link position forward-projected offset_x along its x-axis."""
    robot = env.scene[asset_cfg.name]
    bid = asset_cfg.body_ids[0]
    hand = robot.data.body_pos_w[:, bid, :].clone() - env.scene.env_origins
    hand_quat = robot.data.body_quat_w[:, bid, :]
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(hand.shape[0], 1)
    return hand + math_utils.quat_apply(hand_quat, x_axis) * offset_x


def object_contact_pos_reward(env: ManagerBasedRLEnv,
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"]),
                              std: float = 0.12, offset_x: float = 0.12) -> torch.Tensor:
    """LadderMan position contact reward: c_hat*exp(-||palm - object||^2/std^2) + (1-c_hat). Rewards the
    robot right hand staying ON the sim object during the required-contact (is_closed) phase; free before.
    Robust replacement for the disabled force-based contact reward (dead right-hand force sensor)."""
    obj: RigidObject = env.scene["object"]
    palm = _right_hand_grasp_point(env, asset_cfg, offset_x)
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    d2 = torch.sum((palm - obj_pos) ** 2, dim=1)
    r = torch.exp(-d2 / (std * std))
    motion_res, ok = _rework_motion_res(env)
    if ok:
        c_hat = motion_res["is_closed"].float()
        r = c_hat * r + (1.0 - c_hat)
    return r


def contact_loss_pos_termination(env: ManagerBasedRLEnv,
                                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"]),
                                 tol: float = 0.25, frames: int = 10, offset_x: float = 0.12) -> torch.Tensor:
    """#6c ResMimic loss-of-contact, POSITION-based (robust to dead force sensor): terminate when the
    reference requires contact (is_closed) but the robot hand is > tol (m) from the object for > frames
    consecutive steps."""
    obj: RigidObject = env.scene["object"]
    motion_res, ok = _rework_motion_res(env)
    n = env.scene.num_envs
    if not ok:
        return torch.zeros(n, dtype=torch.bool, device=env.device)
    palm = _right_hand_grasp_point(env, asset_cfg, offset_x)
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    d = torch.norm(palm - obj_pos, dim=1)
    lost = motion_res["is_closed"] & (d > tol)
    if not hasattr(env, "contact_pos_loss_steps") or env.contact_pos_loss_steps.shape[0] != n:
        env.contact_pos_loss_steps = torch.zeros(n, device=env.device)
    env.contact_pos_loss_steps = (env.contact_pos_loss_steps + 1.0) * lost.float()
    return env.contact_pos_loss_steps > float(frames)


def root_deviation_termination(env: ManagerBasedRLEnv, tau: float = 0.5) -> torch.Tensor:
    """#6a reference-motion deviation: terminate when the robot root has drifted > tau (m) from the
    reference root position. Practical whole-body-deviation proxy (root anchors the whole pose)."""
    robot: Articulation = env.scene["robot"]
    motion_res, ok = _rework_motion_res(env)
    if not ok:
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
    root_pos = robot.data.root_pos_w - env.scene.env_origins
    err = torch.norm(root_pos - motion_res["root_pos"], dim=1)
    return err > tau


def object_deviation_termination(env: ManagerBasedRLEnv, tau: float = 0.3, ori_tau: float = 0.8) -> torch.Tensor:
    """#6b object-tracking deviation: terminate when the sim object is > tau (m) from the reference
    object trajectory (ResMimic object-far). Also fires on grasp-loss (object drifts off the hand)."""
    obj: RigidObject = env.scene["object"]
    motion_res, ok = _rework_motion_res(env)
    if not ok:
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
    ref_pos = _synth_object_ref_pos(motion_res)
    if ref_pos is None:
        return torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
    obj_pos = obj.data.root_pos_w - env.scene.env_origins
    if REWORK_OBJ_PC:
        # ResMimic object-far: single MESH point-cloud distance threshold (position + orientation).
        return _object_pc_dist(env, motion_res, obj_pos, ref_pos) > REWORK_OBJ_PC_DEV_TAU
    err = torch.norm(obj_pos - ref_pos, dim=1)
    if "object_poses" not in motion_res:
        # Legacy pkls: no reference object orientation — position-only deviation.
        return err > tau
    ang = math_utils.quat_error_magnitude(obj.data.root_quat_w, motion_res["object_poses"][:, 3:7])
    return (err > tau) | (ang > ori_tau)


def contact_loss_termination(env: ManagerBasedRLEnv,
                             sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["right_hand_.*"]),
                             frames: int = 10, force_thresh: float = 1.0) -> torch.Tensor:
    """#6c ResMimic loss-of-contact: terminate when the reference REQUIRES contact (is_closed) but the
    measured right-hand contact force is absent for > `frames` consecutive steps."""
    motion_res, ok = _rework_motion_res(env)
    n = env.scene.num_envs
    if not ok:
        return torch.zeros(n, dtype=torch.bool, device=env.device)
    c_hat = motion_res["is_closed"].bool()
    f = _right_hand_contact_force(env, sensor_cfg)
    required_but_absent = c_hat & (f <= force_thresh)
    if not hasattr(env, "contact_loss_steps") or env.contact_loss_steps.shape[0] != n:
        env.contact_loss_steps = torch.zeros(n, device=env.device)
    env.contact_loss_steps = (env.contact_loss_steps + 1.0) * required_but_absent.float()
    return env.contact_loss_steps > float(frames)


@configclass
class G1Rewards(G1RewardsBase):
    """Reward terms for the MDP."""

    if TRACKING and not REWORK:
        # ---- SONIC / holosoma-matched whole-body tracking (exp Gaussian kernels) ----
        # Replaces the previous raw-L1/L2 penalties (joint_deviation_ref, keypts_deviation_ref,
        # position/orientation_tracking_error). Weights + stds copied verbatim from
        # gear_sonic/config/manager_env/rewards/tracking/base.yaml -- the SAME reward the frozen
        # SONIC base (encoder->decoder) was trained against. Each term is exp(-err/std^2) in (0,1].
        # Nominal budget = anchor_pos 0.5 + anchor_ori 0.5 + relative_body_pos 1.0 +
        # relative_body_ori 1.0 + body_linvel 1.0 = 4.0/step (relative body 2x global anchor, as
        # in SONIC). At launch --tracking-scale 0.5 halves all five -> ~2.0/step, so tracking is a
        # GAIT/POSE REGULARIZER, not the objective (the base already tracks; saturated tracking
        # gave ~0 learning gradient while drowning the grasp). The relative-body terms EXCLUDE the
        # right arm (KEYPTS_MASK_NO_RARM) so the task owns it. joint_deviation_ref is dropped:
        # SONIC tracks body keypoints (via FK), not joint angles; relative_keypts/ori subsume it.
        tracking_anchor_pos = RewTerm(
            func=anchor_pos_tracking_exp,
            weight=0.5,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.3, "eps": 0.20})  # 20cm root deadband (8->15->20) -- more slack so the robot SETTLES near the grasp root instead of micro-correcting/stumbling at the exact target, which jostled the arm into the bottle

        tracking_anchor_ori = RewTerm(
            func=anchor_ori_tracking_exp,
            weight=0.5,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.4})

        tracking_relative_body_pos = RewTerm(
            func=relative_keypts_tracking_exp,
            weight=1.5,  # 1.0->2.0->1.5: whole-body (non-right-arm) gait/pose tracking. Eased 2.0->1.5
                         # to give the grasp/wrist more room (2.0 over-competed with the task). Root
                         # anchor + deadband unchanged. Also pins the FROZEN reference left arm's static pose.
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
                    "std": 0.3, "keypts_mask": KEYPTS_MASK_NO_RARM})  # NO deadband: keep body pose/orientation tight (orientation was still off); deadband only on root

        tracking_relative_body_ori = RewTerm(
            func=relative_body_ori_tracking_exp,
            weight=1.5,  # 1.0->2.0->1.5: whole-body (non-right-arm) orientation tracking, eased to 1.5
                         # alongside the position term to give the grasp/wrist more room. Root/deadband unchanged.
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
                    "std": 0.4, "keypts_mask": KEYPTS_MASK_NO_RARM})

        tracking_body_linvel = RewTerm(
            func=lower_body_keypt_vel_tracking,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot",
                        body_names=["left_knee_link", "left_ankle_roll_link", "right_knee_link", "right_ankle_roll_link"],
                        preserve_order=True),
                    "sigma": 1.0})

        # GLOBAL (world-frame) tracking for the TASK arm (stripped from the root-relative body
        # terms above). The right-arm keypoints (shoulder->elbow->wrist, indices 31-37) track the
        # reference's ABSOLUTE world positions, not root-relative -- so the hand lands at the
        # reference's global hand pose (= the bottle) REGARDLESS of root drift. NOW INCLUDES
        # wrist_yaw (37) -- the grasp-point proxy, previously untracked (the term tracked the
        # forearm to wrist_pitch but let the wrist/hand dangle off the bottle). rubber_hand (38) is
        # the original-G1 hand link, absent on the dex-hand robot, so wrist_yaw is the closest
        # shared grasp point. Root-relative tracking can't do this: a root drift puts a root-
        # relative-correct hand at the wrong global spot. body_names order must match keypt_idxs.
        tracking_right_arm_pos = RewTerm(
            func=global_keypts_tracking_exp,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot",
                        body_names=["right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
                                    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
                                    "right_wrist_yaw_link"],
                        preserve_order=True),
                    "std": 0.3, "keypt_idxs": [31, 32, 33, 34, 35, 36, 37]})  # wrist_yaw(37)=grasp-point proxy; rubber_hand(38) absent on dex robot

        # PRECISION term for the grasp point: tight-sigma (0.1) GLOBAL track of wrist_yaw (the dex
        # hand base -- fingers branch off it; no separate palm link) to the reference grasp-wrist
        # pose (keypt 37). The wide arm term above (sigma 0.3) does the gross reach but tops out
        # ~9cm off; this kicks in for the last few cm to put the hand precisely on the bottle.
        # Tight sigma => near-dead far away, so the two layer cleanly (gross then fine).
        tracking_hand_precise = RewTerm(
            func=global_keypts_tracking_exp,
            weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"], preserve_order=True),
                    "std": 0.1, "keypt_idxs": [37]})

        # ORIENTATION baseline for the TASK arm. relative_body_ori (above) masks the right arm
        # out (KEYPTS_MASK_NO_RARM); this re-adds it on its OWN mask for distinct weighting.
        # Tracks right-arm LINK rotations to the reference, exp kernel, over KEYPTS_MASK_RARM_ONLY
        # = idx 31-36 (shoulder_pitch -> wrist_pitch); wrist_yaw(37) is masked 0 in KEYPTS_MASK so
        # the hand base is left entirely to target_orientation_error. LOW weight (0.5, well under
        # the wrist level+point penalty's -1.0 scale): a good-behavior prior on forearm/upper-arm
        # roll -- the DOF keypoint positions don't pin -- that the bespoke wrist term overrides at
        # the hand. Same rationale as tracking_right_arm_pos: reference guides, task owns the grasp.
        # Fixed weight (not in --tracking-scale loop), like the other task-arm terms.
        tracking_right_arm_ori = RewTerm(
            func=relative_body_ori_tracking_exp,
            weight=0.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
                    "std": 0.4, "keypts_mask": KEYPTS_MASK_RARM_ONLY})

        right_hand_state_target_reward_val = RewTerm(
            func=right_hand_state_target_reward,
            weight=0.3)

        # NOTE: two gait-shaping terms were tried here and REMOVED after both backfired
        # into smaller, quasi-static motion (run 2026-06-11_01-28-38):
        #   - feet_air_time (threshold=0.4 s, weight +0.1): the policy's exploratory steps
        #     have air time << 0.4 s, so EVERY touchdown earned negative reward — the term
        #     taxed stepping itself (logged negative in every run). The locomotion envs it
        #     comes from force stepping via a velocity command; this env doesn't.
        #   - lower_body_keypt_vel_tracking (exp kernel, sigma=0.5): standing phases match
        #     the near-zero reference velocity trivially while a mistimed swing scores the
        #     same ~0 as not swinging at all — net effect was a leg-motion damper, not a
        #     gait shaper. Both functions remain in motion_tracking_env.py for reference.
        # If shuffling is re-attacked, prefer contact-SCHEDULE matching (reward agreement
        # between reference foot-contact phase, from ankle keypoint height, and the actual
        # contact sensor): positive-shaped, threshold-free, and unsatisfiable by stillness.

        # NOTE: right_hand_object_proximity reward was removed after a failed experiment —
        # gaussian std=0.15 had dead gradient at typical wrist-bottle distances (~40 cm),
        # disturbing body tracking without producing a learning signal for the reach. The
        # function is kept in motion_tracking_env.py for later reuse (would need a wider
        # std or a delta-distance formulation to actually work).

        target_orientation_error = RewTerm(func=target_orientation_error,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])},
            weight=-1.5)  # -1.0->-1.5: moderate wrist up-weight (between safe -1.0 and the -2.0 that
                          # killed the grasp) w/ residual_scale 0.3 + lift 16. (hist: -3.0, -5.0 VETOED, -1.0, -2.0 killed, -1.5.)
                          # Empirically -5.0 VETOED the grasp: the policy bootstrapped a grasp
                          # (lift 0.028 -> 0.54 by iter ~697) but the grasp pose sits ~0.28 rad off
                          # this aim target, costing ~-1.4 of penalty vs only +0.54 of lift, so it
                          # dropped the grasp to realign the wrist. At -1.0 the grasp's wrist penalty
                          # (~0.28) stays well under the lift, so lift wins and the grasp survives
                          # while a light aim still guides the reach. Penalty = levelness (wrist-z ->
                          # world up) + horizontal pointing (wrist-x -> bearing to grab target), both
                          # enforced every step; pointing gated off when on top of the target.


    if TASK_DENSE and not REWORK:
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
        object_approach_reward = RewTerm(func=object_approach_reward_right,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])},
            weight=100.0)
        right_hand_state_target_reward_val = RewTerm(
            func=right_hand_state_target_reward,
            weight=0.3)
    
    
    if TASK_SPARSE and not REWORK:
        # Continuous saturating-exp, hold-accumulating lift reward (replaces dead-zoned object_above_threshold).
        # Object rests at z=0.9; a successful grasp peaks at ~0.976 (~7.6 cm reachable).
        #   * SATURATING EXP from rest: 1-exp(-(z-0.90)/tau), tau=0.03 -> 0 at rest, ~0.81 at the
        #     +5 cm success bar, ~0.92 at apex. Strongest gradient at lift-off (bootstrap), bounded
        #     (0,1] -- same currency as the exp tracking terms.
        #   * is_closed-GATED (anti-knockover) and continuous through the hold (not re-zeroed).
        #   * ACCUMULATES with hold: 1x -> 2x over a 1 s grip, self-resets on drop.
        # n_successes (z>0.95 & closed) unchanged so eval/collection stay comparable.
        # Weight 8.0 (bumped from 3.0): the grasp DOES get discovered (object_lift peaked ~0.96
        # mid-run) but then FADES because grasping costs more in stability/tracking than lift 3.0
        # repaid -> net-negative -> policy abandons it. 8.0 pushes the successful-grasp lift well
        # above that cost so the policy KEEPS the grasp it finds. Compensates for SCARCITY too --
        # lift only pays in the is_closed window and only when actually lifting, while tracking pays
        # every step. It is 0 in the pre-grasp walk (tracking owns that); the right arm has no
        # competing tracking, so a large lift here doesn't fight the gait.
        object_lift = RewTerm(
            func=object_lift_reward,
            weight=16.0,  # 8.0->16.0 (2x): push grasp discovery/rate harder relative to tracking, now that
                          # the wrist term is back to -1.0. Still upright-gated (2x of ~0 when tipped = ~0).
            params={"rest_z": 0.90, "success_z": 0.95, "tau": 0.03, "hold_rate": 0.02, "hold_cap": 50.0}
        )

    if REWORK:
        # HS_REWORK_NO_FEET_SHAPING=1: drop the two inherited gait-shaping terms that conflict with
        # SONIC-faithful tracking — feet_parallel_to_ground (-1.0) penalizes the non-flat ankle
        # orientations the ori-tracking term rewards during swing (heel-strike/toe-off), and
        # feet_slide (-0.1) taxes the corrective stepping the anchor term asks for. Neither exists
        # in native SONIC. Leaves alive/termination and the (negligible) torque/acc terms in place.
        if os.environ.get("HS_REWORK_NO_FEET_SHAPING", "0") == "1":
            feet_slide = None
            feet_parallel_to_ground = None
        # ===================== REWARD REWORK (#2 equal tracking / #4 object / #5 contact) =====================
        # #2 EQUAL whole-body tracking — base SONIC weights, ALL-ONES mask (right arm tracked equally),
        # no root deadband. Drops the grasp-driven right-arm masking + per-limb/wrist/hand-precise terms.
        tracking_anchor_pos = RewTerm(func=anchor_pos_tracking_exp, weight=0.5,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.3, "eps": 0.0})
        tracking_anchor_ori = RewTerm(func=anchor_ori_tracking_exp, weight=0.5,
            params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.4})
        # #3 hybrid tracking (LadderMan): HS_REWORK_HYBRID_RARM=1 relaxes the right arm so the task
        # limb can deviate. Body terms drop to NO_RARM (full weight); a SEPARATE right-arm term re-adds
        # the arm at reduced weight (LadderMan omega/2). Separate terms also stop the arm error from
        # saturating the whole-body mean-then-exp kernel. Default off => MASK_ALL (unchanged).
        if REWORK_GLOBAL_TRACK:
            # HS_REWORK_GLOBAL_TRACK=1: whole-body keypoints tracked in the WORLD (env-local) frame,
            # same kernel/weight/std as the relative term it replaces. Every keypoint now carries the
            # root-drift error, so drift is penalized ~26x more densely than anchor_pos alone. Uses the
            # full-body set regardless of HYBRID_RARM (the arm error is dominated by reach anyway).
            # Orientation stays relative below: body rotations are near-invariant to a translation
            # drift, and heading error is already owned by anchor_ori.
            tracking_global_body_pos = RewTerm(func=global_keypts_tracking_exp, weight=1.0,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=_FULL_BODY_NAMES, preserve_order=True),
                        "std": 0.3, "keypt_idxs": _FULL_BODY_KEYPT_IDXS})
        else:
            tracking_relative_body_pos = RewTerm(func=relative_keypts_tracking_exp, weight=1.0,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
                        "std": 0.3, "keypts_mask": (KEYPTS_MASK_NO_RARM if REWORK_HYBRID_RARM else KEYPTS_MASK_ALL)})
        if REWORK_GLOBAL_ORI:
            # HS_REWORK_GLOBAL_ORI=1: per-link orientations in the WORLD frame (root rotation
            # composed into both sides) — angular-drift analog of tracking_global_body_pos.
            tracking_global_body_ori = RewTerm(func=global_body_ori_tracking_exp, weight=1.0,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
                        "std": 0.4, "keypts_mask": (KEYPTS_MASK_NO_RARM if REWORK_HYBRID_RARM else KEYPTS_MASK_ALL)})
        else:
            tracking_relative_body_ori = RewTerm(func=relative_body_ori_tracking_exp, weight=1.0,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
                        "std": 0.4, "keypts_mask": (KEYPTS_MASK_NO_RARM if REWORK_HYBRID_RARM else KEYPTS_MASK_ALL)})
        if REWORK_HYBRID_RARM:
            tracking_rarm_pos = RewTerm(func=relative_keypts_tracking_exp, weight=REWORK_RARM_W,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
                        "std": 0.3, "keypts_mask": KEYPTS_MASK_RARM_ONLY})
            tracking_rarm_ori = RewTerm(func=relative_body_ori_tracking_exp, weight=REWORK_RARM_W,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True),
                        "std": 0.4, "keypts_mask": KEYPTS_MASK_RARM_ONLY})
        # #1 SONIC-format velocity tracking: full-body LINEAR (std 1.0) + ANGULAR (std 3.14),
        # consuming reference velocities stored in motion_lib (FD at load). Replaces legs-only term.
        tracking_body_linvel = RewTerm(func=body_linvel_tracking_exp, weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=_FULL_BODY_NAMES, preserve_order=True),
                    "keypt_idxs": _FULL_BODY_KEYPT_IDXS, "std": 1.0})
        tracking_body_angvel = RewTerm(func=body_angvel_tracking_exp, weight=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=_FULL_BODY_NAMES, preserve_order=True),
                    "keypt_idxs": _FULL_BODY_KEYPT_IDXS, "std": 3.14})
        # ============= NATIVE SONIC REGULARIZATION (gear_sonic rewards/tracking/base.yaml) =============
        # The frozen base was trained under these three penalty terms; without them the residual can
        # inject jerky actions, ride joint limits, and brush the scene at zero cost. Native weights
        # verbatim. (Tracking terms incl. body lin/angvel were already ported above.)
        action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
        joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-10.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])})
        # Native body_names regex + the dex-hand FINGER links additionally whitelisted: the fingers
        # MUST contact the object (native SONIC trained a rubber-hand G1 with no object, so its
        # regex predates articulated fingers).
        undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-0.1,
            params={"sensor_cfg": SceneEntityCfg("contact_forces",
                        body_names=["^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
                                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                                    "(?!left_elbow_link$)(?!right_elbow_link$)"
                                    "(?!left_rubber_hand$)(?!right_rubber_hand$)"
                                    "(?!left_hand_)(?!right_hand_).+$"]),
                    "threshold": 1.0})
        # finger open/close tracking retained (grasp actuation signal; NOT the wrist-pointing penalty).
        right_hand_state_target_reward_val = RewTerm(func=right_hand_state_target_reward, weight=0.3)
        # #4 object-as-hand reference tracking — REPLACES object_lift. Tracks the sim object to the
        # holosoma reference object trajectory (glued to the hand in the reference). Weight/std initial;
        # user to tune. Set source="hand_fk" to instead track the reference right-hand FK.
        object_tracking = RewTerm(func=object_tracking_reward, weight=REWORK_OBJ_W,
            params={"pos_std": 0.1, "ori_std": 0.5})
        # #5 ResMimic contact reward — REPLACES target_orientation_error (wrist pointing). c_hat*exp(-lam/f).
        # Gated: disabled when HS_REWORK_CONTACT=0 (right-hand contact sensor unreliable).
        if REWORK_CONTACT:
            object_contact = RewTerm(func=object_contact_reward, weight=2.0,
                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_hand_.*"]),
                        "lam": REWORK_CONTACT_LAMBDA})
        # LadderMan position-based contact reward (robust to the dead force sensor): reward the right
        # hand staying ON the object during the required-contact phase. HS_REWORK_CONTACT_POS=1.
        if REWORK_CONTACT_POS:
            object_contact_pos = RewTerm(func=object_contact_pos_reward, weight=REWORK_CONTACT_POS_W,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"]),
                        "std": REWORK_CONTACT_POS_STD, "offset_x": REWORK_CONTACT_POS_OFFX})
        # HS_REWORK_SONIC_ONLY=1: strip everything that is NOT in gear_sonic's native reward set
        # (rewards/tracking/base.yaml = 6 tracking terms + action_rate/joint_limit/undesired_contacts).
        # Drops the pick-task rewards AND the inherited survival/shaping terms. Combine with
        # HS_REWORK_CONTACT=0 HS_REWORK_CONTACT_POS=0 so the conditional contact terms don't exist.
        if os.environ.get("HS_REWORK_SONIC_ONLY", "0") == "1":
            object_tracking = None
            right_hand_state_target_reward_val = None
            alive_reward = None
            termination_penalty = None
            dof_torques_l2 = None
            dof_acc_l2 = None
            feet_slide = None
            feet_parallel_to_ground = None

@configclass
class TerminationsCfg(TerminationsCfgBase):
    """Termination terms for the MDP."""

    if TRACKING and not REWORK:
        base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis","torso_link","waist_yaw_link","waist_roll_link","left_shoulder_pitch_link","right_shoulder_pitch_link",
                                                    ]), "threshold": 1.0},
        )
    if not REWORK:
        torso_below_threshold = DoneTerm(
            func=root_below_threshold, params={"thres": 0.3})
        torso_angle_below_threshold = DoneTerm(
            func=root_angle_below_threshold, params={"thres": 0.5})

    if REWORK:
        # ===================== TERMINATION REWORK (#6) =====================
        # Drop fall-based tilt-angle + base-contact. Keep a SMALLER root-height backstop as a safety
        # (triggered too often at 0.3). Add reference-deviation, object-deviation, contact-loss.
        torso_below_threshold = DoneTerm(
            func=root_below_threshold, params={"thres": REWORK_HEIGHT_BACKSTOP})
        ref_deviation = DoneTerm(
            func=root_deviation_termination, params={"tau": REWORK_REF_DEV_TAU})
        object_deviation = DoneTerm(
            func=object_deviation_termination, params={"tau": REWORK_OBJ_DEV_TAU, "ori_tau": REWORK_OBJ_DEV_ORI_TAU})
        if REWORK_CONTACT:
            contact_loss = DoneTerm(
                func=contact_loss_termination,
                params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_hand_.*"]),
                        "frames": REWORK_CONTACT_LOSS_FRAMES, "force_thresh": 1.0})
        # Position-based contact-loss termination (ResMimic contact-loss>Nf, robust to dead force sensor).
        if REWORK_CONTACT_POS:
            contact_loss_pos = DoneTerm(
                func=contact_loss_pos_termination,
                params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"]),
                        "tol": REWORK_CONTACT_POS_LOSS_TOL, "frames": REWORK_CONTACT_POS_LOSS_FRAMES,
                        "offset_x": REWORK_CONTACT_POS_OFFX})


    
def target_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    if not hasattr(env, 'motion_lib'):
        return_val = torch.zeros((env.scene.num_envs, 4), device=env.device)
        return_val[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        return return_val
    else :
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32) - 1.
        motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        root_pos = motion_res["grab_pos"] + motion_res["offsets"]
        time_mask = 1. - motion_res["is_closed"].float()
        time_init = torch.clip((motion_times+1.)/1.5, 0., 1.)  # Ensure the time mask is between 0 and 1
        
        x_axis = torch.tensor([0.0, 0.0, -1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1).float()
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1).float()
        z_axis = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1).float()

        target_rot_mat_init = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
        target_rot_quat_init = math_utils.quat_from_matrix(target_rot_mat_init)

        a_axis = root_pos - root_pos_link # (2*x_axis + y_axis)/sqrt(5)
        a_axis = a_axis / torch.norm(a_axis, dim=1, keepdim=True)
        
        b_axis = torch.zeros_like(a_axis)
        b_axis[:,0] = -a_axis[:,1]
        b_axis[:,1] = a_axis[:,0]
        b_axis[:,2] = 0.0
        b_axis = b_axis / torch.norm(b_axis, dim=1, keepdim=True)

        z_axis = torch.cross(a_axis, b_axis, dim=1)

        x_axis = 2*a_axis - b_axis
        x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)

        y_axis = torch.cross(z_axis, x_axis, dim=1)

        target_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
        target_rot_quat = math_utils.quat_from_matrix(target_rot_mat)
        
        target_rot_quat = slerp(target_rot_quat_init, target_rot_quat, time_init.unsqueeze(1))

        x_axis_post = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        y_axis_post = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        z_axis_post = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        
        target_post_rot_mat = torch.stack([x_axis_post, y_axis_post, z_axis_post], dim=2)  # shape (N, 3, 3)
        target_post_rot_quat = math_utils.quat_from_matrix(target_post_rot_mat)        
        return target_rot_quat * time_mask.unsqueeze(1) + target_post_rot_quat * (1. - time_mask.unsqueeze(1))


def rigid_body_mass(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """The mass of the rigid body."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()
    return masses.to(env.device)


def object_state_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """#7 privileged CURRENT object state: env-local pose (pos+quat) + world lin/ang velocity. (N,13)."""
    obj: RigidObject = env.scene["object"]
    pos = obj.data.root_pos_w - env.scene.env_origins
    quat = obj.data.root_quat_w
    return torch.cat([pos, quat, obj.data.root_lin_vel_w, obj.data.root_ang_vel_w], dim=-1)


def object_ref_obs(env: ManagerBasedRLEnv, time_offset: float = 0.0) -> torch.Tensor:
    """#7 SYNTHESIZED object TARGET pose (the reference the reward tracks) at current+time_offset,
    env-local. (N,7) = pos (rest->palm) + object orientation. Identity fallback when unavailable."""
    n = env.scene.num_envs
    fallback = torch.zeros((n, 7), device=env.device); fallback[:, 3] = 1.0
    if not hasattr(env, "motion_lib") or getattr(env.motion_lib, "object_poses", None) is None:
        return fallback
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(
        device=env.device, dtype=torch.float32) + time_offset
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    ref_pos = _synth_object_ref_pos(motion_res)
    if ref_pos is None:
        return fallback
    return torch.cat([ref_pos, motion_res["object_poses"][:, 3:7]], dim=-1)      # synth pos + object ori


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)})
        
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)})
        actions = ObsTerm(func=mdp.last_action)
        
        if TRACKING:
            target_ref_curr = ObsTerm(func=target_ref, params={"visualize_markers": VISUALIZE_MARKERS})
            target_ref_next = ObsTerm(func=target_ref, params={"time_offset": .1})
            target_ref_next_next = ObsTerm(func=target_ref, params={"time_offset": .2})
            # Far-horizon slim lookahead (t+0.3 … t+0.9 s, 34 dims each: ref joints + root
            # pose, no keypoints). Extends the reference window to 1.0 s at 0.1 s spacing —
            # mirroring SONIC's G1 encoder (num_future_frames=10, dt_future_ref_frames=0.1).
            # A stride is 0.4–0.6 s; with only 0.2 s visibility the policy saw <half a
            # reference step before having to commit, favoring shuffles over committed steps.
            target_ref_slim_03 = ObsTerm(func=target_ref_slim, params={"time_offset": .3})
            target_ref_slim_04 = ObsTerm(func=target_ref_slim, params={"time_offset": .4})
            target_ref_slim_05 = ObsTerm(func=target_ref_slim, params={"time_offset": .5})
            target_ref_slim_06 = ObsTerm(func=target_ref_slim, params={"time_offset": .6})
            target_ref_slim_07 = ObsTerm(func=target_ref_slim, params={"time_offset": .7})
            target_ref_slim_08 = ObsTerm(func=target_ref_slim, params={"time_offset": .8})
            target_ref_slim_09 = ObsTerm(func=target_ref_slim, params={"time_offset": .9})
            target_orientation_hand = ObsTerm(
                func=target_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])})
        
        if TASK_DENSE:
            current_time = ObsTerm(func=current_time_enc)
        
        right_hand_state_target_val = ObsTerm(
            func=hand_state_target)
        right_hand_state_target_val_1 = ObsTerm(
            func=hand_state_target_1)


        # Task specific observations:
        rel_pose_object = ObsTerm(func=rel_pose_object)
        rel_pose_object_w_link_val = ObsTerm(func=rel_pose_object_w_link, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])})
        right_hand_pose_val = ObsTerm(func=hand_pose, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])})
        object_mass = ObsTerm(func=rigid_body_mass, params={"asset_cfg": SceneEntityCfg("object")})

        if REWORK:
            # #7 privileged object state (current pose+vel) + reference object target (obj_traj) now + t+0.2s.
            # Single policy obs group => available to BOTH actor and critic (symmetric), per plan.
            object_state = ObsTerm(func=object_state_obs)
            object_ref_now = ObsTerm(func=object_ref_obs, params={"time_offset": 0.0})
            object_ref_next = ObsTerm(func=object_ref_obs, params={"time_offset": 0.2})

        def __post_init__(self):
            self.enable_corruption = False # improves real world robustness
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg(ActionsCfgBase):
    """Action specifications for the MDP."""
    left_hand_action = mdp.BinaryJointPositionActionCfg(asset_name="robot",
            joint_names=["left_hand.*"],
            open_command_expr={"left_hand_index_.*": -np.pi / 2.0, 
                                "left_hand_middle_.*": -np.pi / 2.0,
                                "left_hand_thumb_0_joint": 0.0,
                                "left_hand_thumb_1_joint": np.pi / 3.0,
                                "left_hand_thumb_2_joint": np.pi/2.,
                                },
            close_command_expr={"left_hand_index_.*": -np.pi / 2.0, 
                                "left_hand_middle_.*": -np.pi / 2.0,
                                "left_hand_thumb_0_joint": 0.0,
                                "left_hand_thumb_1_joint": np.pi / 3.0,
                                "left_hand_thumb_2_joint": np.pi/2.,
                                })

    right_hand_action = mdp.BinaryJointPositionActionCfg(asset_name="robot",
            joint_names=["right_hand.*"],
            open_command_expr={"right_hand.*": 0.0},
            close_command_expr={"right_hand_index_.*": np.pi / 2.0, 
                                "right_hand_middle_.*": np.pi / 2.0,
                                "right_hand_thumb_0_joint": 0.0,
                                "right_hand_thumb_1_joint": -np.pi / 3.0,
                                "right_hand_thumb_2_joint": -np.pi/2.,
                                })

@configclass
class MySceneCfg(MySceneCfgBase):
    """Configuration for the terrain scene with a legged robot."""

    kitchen = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Kitchen",
        spawn=sim_utils.CuboidCfg(
            size=(1., 2., 0.8),collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0.6, 0.2), metallic=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            ))
    )
    
    # Object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            size=(.05, .05, 0.2),collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0., 0.2, 0.6), metallic=0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=1.2,
                dynamic_friction=1.2,
                restitution=0.0,
            )
        ),  
    )


@configclass
class G1PickEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Green-box "table" pos. Box size=(1,2,0.8) → top = pos_z + 0.4. Table top kept at
        # 0.8 (pos_z=0.4) so the object rests at 0.8 + 0.1 (half-height) = 0.9 m. The
        # reference hand grabs at ~0.86 m (TrajGen refine_motions.py OFFSET_Z), so a 0.8
        # table top leaves ~6 cm of clearance between the hand and the table — deliberately
        # below the refinement's exact 0.86 table top to avoid the hand scraping/colliding
        # with the surface during the grasp. Object rest height = 0.9 (reward thresholds set
        # accordingly below).
        self.scene.kitchen.init_state.pos = (2.55, 0, 0.4)

        enable_cameras = bool(getattr(self, "enable_cameras", False))
        if enable_cameras:
            rot = np.array([0.7538, 0.61221, -0.1505, -0.1853])
            rot_mat = np.array(math_utils.matrix_from_quat(torch.tensor(rot)))
            theta = -np.pi*0.75
            rot_z_theta = np.array([[np.cos(theta), -np.sin(theta), 0.0], \
                                    [np.sin(theta), np.cos(theta), 0.0], \
                                    [0.0, 0.0, 1.0]])
            rot_mat = rot_z_theta @ rot_mat
            rot_quat = tuple(math_utils.quat_from_matrix(torch.tensor(rot_mat)).tolist())
            self.scene.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera_new",
                                          spawn=PinholeCameraCfg(
                                              focal_length=18.1476,
                                              focus_distance=400.,
                                              horizontal_aperture=20.955,
                                              clipping_range=(0.1, 10000.0),
                                          ),
                                          data_types=["rgb"],
                                          height=1920,
                                          width=2560,
                                          offset=CameraCfg.OffsetCfg(
                                              pos=(-1.03+2.1-0.034, 4.05-0.9, 1.31),
                                              rot=rot_quat,
                                              convention="opengl"
                                          ),)
        self.ref_motions_path = "../TrajGen/sample/Pick_sim2"

@configclass
class G1PickCamEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # Enable cameras only when a video-capable run requests them.
    enable_cameras_for_collection: bool = False
    ref_motions_path: str = "../TrajGen/sample/Pick_sim2"
    kitchen_usd_path: str = "assets/HQ Kitchen/Collected_kitchen_flat/kitchen_flat3.usd"
    object_usd_path: str = "assets/mustard_bottle.usd"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.ref_motions_path = getattr(self, "ref_motions_path", "../TrajGen/sample/Pick_sim2")
        self.kitchen_usd_path = getattr(
            self, "kitchen_usd_path", "assets/HQ Kitchen/Collected_kitchen_flat/kitchen_flat3.usd"
        )
        self.object_usd_path = getattr(self, "object_usd_path", "assets/mustard_bottle.usd")
        self.scene.kitchen = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen",
            spawn=sim_utils.UsdFileCfg(usd_path=self.kitchen_usd_path, scale=(1.0, 1.0, 0.89)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(2.1 - 0.06, 1.0, 0.0), rot=(1, 0, 0, 0)),
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.object_usd_path,
                scale=(1.0, 1.0, 1.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            ),
        )

        enable_cameras = bool(getattr(self, "enable_cameras", False) or self.enable_cameras_for_collection)
        if enable_cameras:
            rot = np.array([0.7538, 0.61221, -0.1505, -0.1853])
            rot_mat = np.array(math_utils.matrix_from_quat(torch.tensor(rot)))
            theta = -np.pi*0.75
            rot_z_theta = np.array([[np.cos(theta), -np.sin(theta), 0.0], \
                                    [np.sin(theta), np.cos(theta), 0.0], \
                                    [0.0, 0.0, 1.0]])
            rot_mat = rot_z_theta @ rot_mat
            rot_quat = tuple(math_utils.quat_from_matrix(torch.tensor(rot_mat)).tolist())
            self.scene.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera_new",
                                      spawn=PinholeCameraCfg(
                                          focal_length=18.1476,
                                          focus_distance=400.,
                                          horizontal_aperture=20.955,
                                          clipping_range=(0.1, 10000.0),
                                      ),
                                      data_types=["rgb"],
                                      height=1920,
                                      width=2560,
                                          offset=CameraCfg.OffsetCfg(
                                              pos=(-1.03+2.1-0.034, 4.05-0.9, 1.31),
                                              rot=rot_quat,
                                              convention="opengl"
                                          ),)
            self.scene.camera_robot = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/Camera_robot",
                                          spawn=PinholeCameraCfg(
                                              focal_length=7.6,
                                              focus_distance=400.0,
                                              horizontal_aperture=20.0,
                                              clipping_range=(0.01, 100.0),
                                          ),
                                          data_types=["rgb"],
                                          height=720,
                                          width=1280,
                                          offset=CameraCfg.OffsetCfg(
                                              pos=(0.05, 0., 0.36),
                                              rot=(0.568, 0.421, -0.421, -0.568),
                                              convention="opengl"
                                          ),
                                        )
        self.ref_motions_path = "../TrajGen/sample/Pick_sim2"

@configclass
class G1PickPlayEnvCfg(G1InteractiveBaseEnvCfg):
    rewards: G1Rewards = G1Rewards()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=100.)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.ref_motions_path = "../TrajGen/sample/Pick_sim2"
        self.scene.terrain = None
        self.scene.sky_light = None
        
        self.scene.kitchen = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Kitchen",
            spawn=sim_utils.UsdFileCfg(usd_path="assets/HQ Kitchen/Collected_kitchen_flat/kitchen_flat3.usd",scale=(1.,1.,0.89)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(2.1-0.06, 1.0, 0.), rot=(1, 0, 0, 0)),
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path="assets/mustard_bottle.usd",
                scale=(1., 1., 1.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=.1),
            ),
        )
        # self.scene.terrain = None
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        print("Robot added")
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005


@configclass
class ContinuousFingersActionsCfg(ActionsCfg):
    """Continuous 7-DoF finger control per hand, replacing the binary open/close.

    Action values are absolute joint position targets (``use_default_offset=False``,
    ``offset=0.0``) so the VLA's predicted finger positions, which match the
    dataset's ``observation.state`` convention, map 1:1 to action slots.

    The body ``joint_pos`` action is ALSO overridden to pass-through (scale=1.0,
    no default offset). The base env used ``scale=0.5, use_default_offset=True``
    for RL training; at VLA+SONIC eval we apply the canonical
    ``q = default + utm * g1_action_scale`` transform in Python (see
    ``vla_sonic.action_assembler``) and feed absolute joint targets to the env,
    so the env must not re-apply its own default+scale.
    """
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JointNamesOrder,
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
        offset=0.0,
    )
    left_hand_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "left_hand_index_0_joint", "left_hand_index_1_joint",
            "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        ],
        preserve_order=True, scale=1.0, use_default_offset=False, offset=0.0,
    )
    right_hand_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_index_1_joint",
            "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        ],
        preserve_order=True, scale=1.0, use_default_offset=False, offset=0.0,
    )


@configclass
class G1PickCamContinuousFingersEnvCfg(G1PickCamEnvCfg):
    actions: ContinuousFingersActionsCfg = ContinuousFingersActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Replace Isaac's default G1 actuator gains with SONIC-training-matched
        # values so that the canonical ``q_target = default + utm * scale``
        # transform produces the motion the UTM was trained against. Without
        # this, Isaac's stiffer gains (2-5x SONIC's) cause violent overshoot.
        self.scene.robot.actuators = _build_sonic_matched_actuators()
        # Verification trace — spot-check a few gains so we can confirm the
        # override actually landed at env init time (cfg-level override can be
        # silently clobbered downstream by USD-baked gains or SceneCfg copies).
        legs = self.scene.robot.actuators["legs"]
        print(f"[SONIC-gains] legs.stiffness = {legs.stiffness}")
        print(f"[SONIC-gains] legs.damping   = {legs.damping}")
        arms = self.scene.robot.actuators["arms"]
        print(f"[SONIC-gains] arms.stiffness = {arms.stiffness}")
        feet = self.scene.robot.actuators["feet"]
        print(f"[SONIC-gains] feet.stiffness = {feet.stiffness} damping = {feet.damping}")


@configclass
class G1PickContinuousFingersEnvCfg(G1PickEnvCfg):
    """No-camera pick env with continuous fingers + SONIC-matched actuators.

    Same scene / rewards / observations / motion as ``G1PickEnvCfg`` (no camera
    unless ``enable_cameras`` is set), but swaps in ``ContinuousFingersActionsCfg``
    so the body action is pass-through (absolute joint targets) — matching the
    SONIC decoder convention used by ``train_sonic.py`` and ``eval_parquet_sonic.py``.
    Used for RL-training a custom encoder against the frozen SONIC decoder.
    """
    actions: ContinuousFingersActionsCfg = ContinuousFingersActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        # SONIC-training-matched PD gains so ``q = default + utm * scale`` produces
        # the motion the UTM was trained against (Isaac's default gains overshoot).
        self.scene.robot.actuators = _build_sonic_matched_actuators()
        legs = self.scene.robot.actuators["legs"]
        print(f"[SONIC-gains] legs.stiffness = {legs.stiffness}")
        print(f"[SONIC-gains] legs.damping   = {legs.damping}")
        arms = self.scene.robot.actuators["arms"]
        print(f"[SONIC-gains] arms.stiffness = {arms.stiffness}")
        feet = self.scene.robot.actuators["feet"]
        print(f"[SONIC-gains] feet.stiffness = {feet.stiffness} damping = {feet.damping}")


@configclass
class BinaryFingersActionsCfg(ActionsCfg):
    """Hybrid actions: passthrough body (for SONIC decoder) + binary fingers (for RL).

    Body action mirrors ``ContinuousFingersActionsCfg`` (absolute joint targets) so
    the SONIC decoder's output maps 1:1 to env command. Finger actions revert to
    ``BinaryJointPositionActionCfg`` — exactly what the original ``train.py`` used.

    This gives the RL policy a single 1-D scalar per hand (close/open), which the env's
    action manager internally expands to 7-D joint targets using the open/close
    command_expr dicts. Trivially dense binary-match reward + no PD lag in the reward
    signal. The continuous-finger layout is *only* needed at VLA deployment (where the
    VLA writes 7-D continuous finger targets) — see ``G1PickContinuousFingersEnvCfg``.
    """
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JointNamesOrder,
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
        offset=0.0,
    )
    # Inherits ``left_hand_action`` and ``right_hand_action`` (both
    # BinaryJointPositionActionCfg) from ``ActionsCfg`` unchanged.


@configclass
class G1PickBinaryFingersEnvCfg(G1PickEnvCfg):
    """No-camera pick env with binary fingers + passthrough body + SONIC-matched gains.

    Used by ``train_sonic.py``. Action layout is [27 body | 1 left binary | 1 right binary]
    = 29 dims. Trained encoder ONNX-exports to body-tokens-only, so at deployment it slots
    into ``G1PickContinuousFingersEnvCfg`` (where the VLA writes continuous fingers) with
    zero layout coupling — the encoder only ever bridges body actions, which are passthrough
    in both envs.

    Also swaps the right-hand reward function: under binary fingers there's a clean 1-D
    binary action slot to read, so we use the original train.py-style binary match against
    ``motion_lib["is_closed"]`` instead of the joint-space ``exp(-L1/SCALE)`` reward
    (which was designed around the 7-D continuous action layout).
    """
    actions: BinaryFingersActionsCfg = BinaryFingersActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = _build_sonic_matched_actuators()
        # Manipuland: mustard bottle USD by DEFAULT (the object holosoma retargeted the references
        # around; realistic collision geometry). HS_OBJ=cuboid reverts to the legacy 5x5x20cm blue
        # cuboid for reproducing pre-mustard checkpoints/evals. CAVEATS vs the cuboid: (1) UsdFileCfg
        # has no physics_material field, so friction comes from mustard_bottle.usd (not the cuboid's
        # 1.2/1.2 grip); (2) the rest height on the 0.8 table top depends on the bottle's collision
        # mesh, not the cuboid's exact 0.9 — VALIDATE the settled object z in sim and re-pin the
        # TrajGen re-table target (holosoma_to_pkl.py) if it differs from 0.900.
        if os.environ.get("HS_OBJ", "mustard") != "cuboid":
            self.scene.object = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Object",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
                spawn=sim_utils.UsdFileCfg(
                    usd_path="assets/mustard_bottle.usd",
                    scale=(1.0, 1.0, 1.5),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                ),
            )
            print("[G1PickBinaryFingers] object = mustard_bottle.usd (HS_OBJ=cuboid reverts to legacy cuboid)")
        # Swap the right-hand reward to the binary-match version (uses action_manager.action
        # directly; no PD-tracking lag, no sharpness parameter, dense {0,1} signal).
        # (Term is None under HS_REWORK_SONIC_ONLY=1 — nothing to swap then.)
        if self.rewards.right_hand_state_target_reward_val is not None:
            self.rewards.right_hand_state_target_reward_val.func = right_hand_binary_match_reward
        legs = self.scene.robot.actuators["legs"]
        print(f"[SONIC-gains] legs.stiffness = {legs.stiffness}")
        print(f"[SONIC-gains] legs.damping   = {legs.damping}")
        arms = self.scene.robot.actuators["arms"]
        print(f"[SONIC-gains] arms.stiffness = {arms.stiffness}")
        feet = self.scene.robot.actuators["feet"]
        print(f"[SONIC-gains] feet.stiffness = {feet.stiffness} damping = {feet.damping}")
        print("[BinaryFingers] right_hand reward swapped to binary-match against is_closed")


def hide_unwanted_visuals(env, env_ids=None):
    """Startup event: hide the RENDER of unwanted prims while KEEPING any colliders.

    The kitchen-visuals collection env (G1PickCamBinaryFingersEnvCfg) deliberately keeps the
    training physics — the green collision-box "table" (``/World/envs/env_*/Kitchen``, so the
    object rests at 0.9 m) and the ``/World/ground`` terrain plane (so the robot stands) — and
    layers the HQ kitchen USD (``KitchenVisual``) on top purely as a visual backdrop. Without
    this hook the green box and the marble ground plane ALSO render, so the ego camera sees
    assets from BOTH the RL-training scene and the kitchen at once. The kitchen USD also ships
    a decorative glass beer bottle prop (``aBottle``, from Corona.usd) sitting on the counter,
    which we don't want in the recorded footage either.

    ``UsdGeom.MakeInvisible`` toggles only the USD ``visibility`` token; the PhysX collision
    APIs are untouched, so physics (rest height, ground contact) is identical to training. We
    hide the green box, the ground plane, and the kitchen's glass-bottle prop — but NOT
    ``KitchenVisual`` itself, nor ``Object`` (the mustard-bottle manipuland, which lives under
    ``/World/envs/env_*/Object``, not under ``KitchenVisual``).

    NOTE on the red grab-location marker: ``env.goal_marker`` is created in
    ``ManagerBasedRLEnv.__init__`` AFTER the startup events run, so it can't be hidden here —
    it is disabled instead via the ``target_ref`` ``visualize_markers=False`` obs params (see
    G1PickCamBinaryFingersEnvCfg.__post_init__).
    """
    import re

    import omni.usd
    from pxr import UsdGeom

    # The green collision box is the per-env prim "/World/envs/env_<N>/Kitchen" — a DIRECT
    # child of the env namespace. Match that exact shape (segment before "/Kitchen" is the
    # env), NOT a bare name=="Kitchen", because the referenced kitchen USD also contains a
    # prim named "Kitchen" deeper under ".../KitchenVisual/...". A bare name match hid the
    # entire kitchen backdrop.
    green_box_re = re.compile(r"/env_\d+/Kitchen$")

    stage = omni.usd.get_context().get_stage()
    hidden = []
    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        # Never touch the kitchen backdrop subtree (except its glass-bottle prop below).
        in_kitchen_visual = "/KitchenVisual" in path
        # (1) green collision-box table, (2) the terrain ground plane ("/World/ground"),
        # (3) the kitchen's decorative glass bottle prop (any "bottle"-named prim WITHIN the
        #     KitchenVisual subtree; the mustard-bottle Object under /Object is never touched).
        is_collision_box = bool(green_box_re.search(path)) and not in_kitchen_visual
        # Terrain ground plane. For terrain_type="plane" the importer spawns the DEFAULT
        # grid-world ground plane (default_environment.usd → a grid texture) at
        # "/World/ground/terrain", not at "/World/ground" itself. Hide the whole subtree so the
        # grid mesh prim is set invisible directly (don't rely on ancestor-visibility pruning).
        is_ground = path == "/World/ground" or path.startswith("/World/ground/")
        is_kitchen_bottle = in_kitchen_visual and "bottle" in name.lower()
        if is_collision_box or is_ground or is_kitchen_bottle:
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(path)
    print(f"[CamBinaryFingers] hid {len(hidden)} unwanted visual prim(s) (green box + ground "
          f"plane + kitchen glass bottle); colliders kept, kitchen backdrop remains: {hidden}")


@configclass
class G1PickCamBinaryFingersEnvCfg(G1PickBinaryFingersEnvCfg):
    """Kitchen-VISUALS pick env with binary fingers + the EXACT training physics.

    For ego-view data collection of the SONIC adapter (collect_sonic_adapter.py). Inherits
    EVERYTHING physics-relevant from G1PickBinaryFingersEnvCfg unchanged — the green-box
    collision table (top 0.8 → object rests at 0.9), the cuboid object, passthrough body +
    binary fingers, SONIC-matched actuators, and the binary-match reward — so the policy the
    adapter learned at the 0.9 grasp height behaves identically. On top of that it adds:
      - the HQ kitchen USD as a visual backdrop, and
      - the third-person + torso-ego (d435) cameras,
    so the recorded ego frames show a realistic scene.

    IMPORTANT CAVEAT (verify in sim): the cuboid object + green-box collision table are kept
    for grasp-physics fidelity (rest = 0.9 m, matching training). The kitchen USD is a
    pure visual; its counter will NOT necessarily line up with the 0.8 collision top, so the
    object may appear to float/clip relative to the kitchen counter. The POLICY works (it
    only sees physics), but if you need the cuboid to visually sit on the kitchen counter,
    nudge ``self.scene.kitchen.init_state.pos`` z so the counter visual top reaches ~0.8.
    A fully realistic kitchen+bottle collection at 0.9 would require re-training the adapter
    on that geometry; this env prioritizes policy fidelity + kitchen backdrop.
    """

    kitchen_usd_path: str = "assets/HQ Kitchen/Collected_kitchen_flat/kitchen_flat3.usd"
    object_usd_path: str = "assets/mustard_bottle.usd"

    def __post_init__(self):
        super().__post_init__()  # green-box physics @ 0.9 + SONIC gains + binary reward
        # Kitchen USD visual backdrop (no rigid body; AssetBaseCfg). Positioned as in
        # G1PickCamEnvCfg. The green-box collision table (from the parent) remains the
        # physics surface so the 0.9 rest height is preserved.
        self.scene.kitchen_visual = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/KitchenVisual",
            spawn=sim_utils.UsdFileCfg(usd_path=self.kitchen_usd_path, scale=(1.0, 1.0, 0.89)),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(2.1 - 0.06, 1.0, 0.0), rot=(1, 0, 0, 0)),
        )
        # Swap the blue cuboid manipuland for the mustard bottle USD so the recorded footage
        # shows a realistic object (same init pose / mass / scale as G1PickCamEnvCfg's bottle).
        # CAVEAT (grasp fidelity): the adapter was trained to grasp the 5x5x20cm cuboid, so the
        # bottle's different collision geometry can lower the grasp success rate — the
        # collector's success filter then simply writes fewer trajectories. Also note UsdFileCfg
        # has NO `physics_material` field (unlike CuboidCfg), so the bottle's friction comes from
        # mustard_bottle.usd, not the cuboid's static/dynamic 1.2 grip. If grasps slip badly,
        # the fidelity-preserving alternative is to keep the cuboid as an INVISIBLE collider and
        # parent the bottle mesh under /Object as a visual-only child so it tracks the rigid body.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.35, 0.40, 1.0413], rot=[1, 0, 0, 0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.object_usd_path,
                scale=(1.0, 1.0, 1.5),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            ),
        )
        # Third-person + torso-ego cameras (same configs as G1PickCamEnvCfg).
        rot = np.array([0.7538, 0.61221, -0.1505, -0.1853])
        rot_mat = np.array(math_utils.matrix_from_quat(torch.tensor(rot)))
        theta = -np.pi * 0.75
        rot_z_theta = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                                [np.sin(theta), np.cos(theta), 0.0],
                                [0.0, 0.0, 1.0]])
        rot_mat = rot_z_theta @ rot_mat
        rot_quat = tuple(math_utils.quat_from_matrix(torch.tensor(rot_mat)).tolist())
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera_new",
            spawn=PinholeCameraCfg(focal_length=18.1476, focus_distance=400.0,
                                   horizontal_aperture=20.955, clipping_range=(0.1, 10000.0)),
            data_types=["rgb"], height=1920, width=2560,
            offset=CameraCfg.OffsetCfg(pos=(-1.03 + 2.1 - 0.034, 4.05 - 0.9, 1.31),
                                       rot=rot_quat, convention="opengl"),
        )
        # Ego camera rendered at 1280x960 (4:3) — a clean 2x of the 640x480 SONIC ego-view
        # feature. The previous 1280x720 (16:9) forced the converter's _resize_ego_view to
        # squish 16:9 into 4:3 (non-uniform: 0.5x horizontal, 0.667x vertical), which BLURS and
        # horizontally distorts every frame. Rendering at the target 4:3 aspect makes the
        # downscale a uniform 2x anti-aliased shrink → crisp, undistorted, and the stored
        # intrinsics finally match the 4:3 dataset spec. Higher native res also helps the RTX
        # denoiser. (FOV is set by aperture, not pixel count, so this only adds detail.)
        self.scene.camera_robot = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/Camera_robot",
            spawn=PinholeCameraCfg(focal_length=7.6, focus_distance=400.0,
                                   horizontal_aperture=20.0, clipping_range=(0.01, 100.0)),
            data_types=["rgb"], height=960, width=1280,
            offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 0.36),
                                       rot=(0.568, 0.421, -0.421, -0.568), convention="opengl"),
        )
        # Hide the unwanted visuals (green collision box + marble ground plane + kitchen glass
        # bottle prop) so the ego camera sees only the kitchen backdrop + the mustard bottle.
        # Startup event → runs once after the scene spawns; colliders are kept so physics is
        # unchanged. Without this the footage shows training-scene assets overlapping the kitchen.
        self.events.hide_unwanted_visuals = EventTerm(
            func=hide_unwanted_visuals,
            mode="startup",
        )
        # Remove the red grab-location marker (env.goal_marker) from the footage. target_ref
        # draws it at the grab point when visualize_markers=True; set False on all three
        # target_ref obs terms so it is parked at z=-0.1 (below the floor, out of the ego view)
        # every step instead. This gates ONLY marker drawing — the returned obs tensor is
        # identical, so the policy is unaffected. (Guarded getattr in case TRACKING disabled them.)
        for _term_name in ("target_ref_curr", "target_ref_next", "target_ref_next_next"):
            _term = getattr(self.observations.policy, _term_name, None)
            if _term is not None:
                _term.params = {**_term.params, "visualize_markers": False}
        print("[CamBinaryFingers] kitchen visual + third-person/ego cameras added; ego cam at "
              "1280x960 (4:3, crisp); manipuland swapped to mustard bottle (grasp physics now "
              "from the bottle USD, not the trained cuboid); table-0.9 + SONIC gains inherited; "
              "green box + ground plane + glass bottle hidden; red grab marker disabled")
