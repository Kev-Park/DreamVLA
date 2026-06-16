# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.motion_tracking.g1.motion_tracking_env import keypts_deviation_ref_l2, joint_deviation_ref_l1, position_tracking_error, orientation_tracking_error, target_orientation_error, right_hand_state_target_reward, right_hand_binary_match_reward, target_ref, target_ref_slim, root_below_threshold, root_angle_below_threshold, current_time_enc
import numpy as np
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
from isaaclab_tasks.manager_based.interactive_motion_tracking.g1.motion_tracking_interactive_base import G1InteractiveBaseEnvCfg, hand_state_target, hand_state_target_1, rel_pose_object_w_link, object_above_threshold, reset_object_state, rel_pose_object, hand_pose, object_approach_reward_right, G1Rewards as G1RewardsBase, TerminationsCfg as TerminationsCfgBase, ActionsCfg as ActionsCfgBase, MySceneCfg as MySceneCfgBase, EventCfg as EventCfgBase
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


    reset_object = EventTerm(
        func=reset_object_state,
        params={
            "height": 1.0,
            "offset": [0.0, 0.0],
        },
        mode="reset"
    )



def target_orientation_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    root_rot_link = math_utils.quat_unique(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :].clone())
    
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

    # z_axis is a_axis x b_axis
    z_axis = torch.cross(a_axis, b_axis, dim=1)

    x_axis = 2*a_axis - b_axis
    x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)

    y_axis = torch.cross(z_axis, x_axis, dim=1)

    target_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
    target_rot_quat = math_utils.quat_from_matrix(target_rot_mat)
    
    target_rot_quat = slerp(target_rot_quat_init, target_rot_quat, time_init.unsqueeze(1))
    angle = math_utils.quat_error_magnitude(target_rot_quat, root_rot_link)

    z_axis_post = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    z_axis_w = math_utils.quat_apply(root_rot_link, z_axis_post)
    angle_post = torch.acos(torch.clamp(z_axis_w[:, 2], -1.0, 1.0))
    return torch.abs(angle) * time_mask + torch.abs(angle_post) * (1. - time_mask)


@configclass
class G1Rewards(G1RewardsBase):
    """Reward terms for the MDP."""

    if TRACKING:
        joint_deviation_ref = RewTerm(
            func=joint_deviation_ref_l1,
            weight=-0.4,  # bumped from -0.2: stronger reference tracking to suppress
                          # stance-phase foot oscillation during reach (hips+knees are
                          # in JOINTS_MASK; tighter hip tracking constrains foot position
                          # downstream of the kinematic chain).
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "joint_mask": JOINTS_MASK})

        keypts_deviation_ref = RewTerm(
            func=keypts_deviation_ref_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True), "keypts_mask": KEYPTS_MASK})


        position_tracking_error = RewTerm(
            func=position_tracking_error,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        orientation_tracking_error = RewTerm(
            func=orientation_tracking_error,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

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
            weight=-3.0)  # bumped from -1.0: keeps wrist-orientation penalty larger than
                          # the finger reward (~0.26 weighted) so PPO's credit assignment
                          # around the grab frame doesn't reinforce incidental inward wrist
                          # rotation. Diagnosis: wrist tracked perfectly early in training
                          # when finger reward was ~0.003 (33x smaller); drift emerged once
                          # finger reward grew ~85x and overtook the orientation penalty.
                          # Note: target_orientation_error's `time_mask` switches from
                          # `angle` (bottle-pointing) to `angle_post` (just-vertical) at
                          # is_closed=1, so the closed-phase term is structurally weaker —
                          # the -3.0 weight compensates for both magnitude crossover AND
                          # the time-mask attenuation at the grab transition.


    if TASK_DENSE:
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
        object_approach_reward = RewTerm(func=object_approach_reward_right,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])},
            weight=100.0)
        right_hand_state_target_reward_val = RewTerm(
            func=right_hand_state_target_reward,
            weight=0.3)
    
    
    if TASK_SPARSE:
        # Strict lift reward calibrated to the ACTUAL achievable lift. The object rests at
        # z=0.9; an observed *successful* grasp peaks at ~0.976 (only ~7.6 cm of lift is
        # reachable given the reference motion). has_grasped ramps 0→1 over
        # (fall_thres, height_thres):
        #   fall_thres = 0.92  → 2 cm above rest: resting/jitter scores exactly 0, gradient
        #                        starts as soon as the bottle lifts off.
        #   height_thres = 0.95 → full reward at a 5 cm lift, comfortably below the observed
        #                        success apex (~0.976) so genuine pickups reliably earn full
        #                        credit; the gradient spans 0.92→0.95. n_successes keys off
        #                        z>0.95, so eval/collection "success" = a real ~5 cm lift.
        # (Earlier 1.05 was unreachable — a successful grasp scored only ~0.43, giving the
        # policy no signal that it had actually succeeded; 0.97 was tight against the apex.)
        object_above_the_ground = RewTerm(
            func=object_above_threshold,
            weight=.5,
            params={"height_thres": 0.95, "fall_thres": 0.92}
        )
@configclass
class TerminationsCfg(TerminationsCfgBase):
    """Termination terms for the MDP."""

    if TRACKING :
        base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis","torso_link","waist_yaw_link","waist_roll_link","left_shoulder_pitch_link","right_shoulder_pitch_link",
                                                    ]), "threshold": 1.0},
        )
    torso_below_threshold = DoneTerm(
        func=root_below_threshold, params={"thres": 0.3})
    torso_angle_below_threshold = DoneTerm(
        func=root_angle_below_threshold, params={"thres": 0.5})
    
    
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
        # Swap the right-hand reward to the binary-match version (uses action_manager.action
        # directly; no PD-tracking lag, no sharpness parameter, dense {0,1} signal).
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
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    hidden = []
    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        # (1) green collision-box table (per-env "/World/envs/env_*/Kitchen") and (2) the
        #     terrain ground plane ("/World/ground") — exact match avoids hiding "KitchenVisual".
        # (3) the kitchen's decorative glass bottle prop: any "bottle"-named prim WITHIN the
        #     KitchenVisual subtree (so the mustard-bottle Object under /Object is never touched).
        is_collision_box = name == "Kitchen"
        is_ground = path == "/World/ground"
        is_kitchen_bottle = "/KitchenVisual" in path and "bottle" in name.lower()
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
