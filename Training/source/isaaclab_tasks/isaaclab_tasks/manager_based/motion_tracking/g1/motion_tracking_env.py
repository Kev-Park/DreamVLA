# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from ast import List
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from dataclasses import MISSING
import numpy as np
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
import torch
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg

from dataclasses import MISSING
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder

VISUALIZE_MARKERS = True

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
    1, # left_wrist_roll_joint
    1, # left_wrist_pitch_joint
    1, # left_wrist_yaw_joint
    1, # right_shoulder_pitch_joint
    1, # right_shoulder_roll_joint
    1, # right_shoulder_yaw_joint
    1, # right_elbow_joint
    1, # right_wrist_roll_joint
    1, # right_wrist_pitch_joint
    1, # right_wrist_yaw_joint
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
    1, # right_wrist_yaw_link
    1, # right_rubber_hand
]

def reset_joints_for_motion(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints to first pose in the motion library.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, 'start_motion_times'):
        joint_pos_new = torch.zeros((len(env_ids), 41), device=env.device)
        joint_vel = torch.zeros((len(env_ids), 41), device=env.device)
    else :
        env.motion_ids[env_ids] = torch.randint(0, env.total_motions, (len(env_ids),), device=env.device)
        motion_times = env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)[env_ids]
        motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)[env_ids]
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        joint_pos: torch.tensor = motion_res["dof_pos"]
        joint_ids = torch.tensor(asset_cfg.joint_ids, device=env.device)
        inv_joint_ids = torch.zeros_like(joint_ids)
        inv_joint_ids[joint_ids] = torch.arange(len(joint_ids), device=env.device)
        joint_pos_new = torch.zeros((len(env_ids), 41), device=env.device)
        joint_pos_new[:, joint_ids] = joint_pos.clone()
        joint_vel = asset.data.default_joint_vel[env_ids].clone()

    asset.write_joint_state_to_sim(joint_pos_new, joint_vel, env_ids=env_ids)

def reset_root_state_for_motion(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset_z: float = 0.15,
):
    """Reset the asset root state to the root state of the first pose in the motion library.

    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    if not hasattr(env, 'start_motion_times'):
        positions = torch.zeros((env.scene.num_envs, 3), device=env.device)
        orientations = torch.zeros((env.scene.num_envs, 4), device=env.device)
        orientations[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        velocities = torch.zeros((env.scene.num_envs, 6), device=env.device)
    else :
        motion_times = env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)[env_ids]
        motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)[env_ids]
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        velocities = torch.zeros((env.scene.num_envs, 6), device=env.device)[env_ids]
        root_states = torch.cat([motion_res["root_pos"], motion_res["root_rot"], velocities], dim=-1)
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
        orientations = root_states[:, 3:7]
        velocities = root_states[:, 7:13]
        positions[:, 2] += offset_z

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    
    reset_robot_joints = EventTerm(
        func=reset_joints_for_motion,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)
        }
    )

    reset_base = EventTerm(
        func=reset_root_state_for_motion,
        mode="reset"
    )

    
def get_keypts(joint_angles, joint_names, pk2_robot):
    
    q_dict = {name: joint_angles[:, i] for i, name in enumerate( joint_names ) }

    tf_dict = pk2_robot.forward_kinematics( q_dict )

    keypts = torch.zeros((len(joint_angles), len(tf_dict), 3), device=joint_angles.device)
    cntr = 0 
    
    for name in tf_dict.keys():
        tf_val = tf_dict[name].get_matrix()  
        t = tf_val[:, :3, -1 ]
        keypts[:, cntr , :] = t 
        cntr += 1 
    return keypts


def keypts_deviation_ref_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), keypts_mask: List = KEYPTS_MASK) -> torch.Tensor:
    """Penalize deviation of keypoints from the reference keypoints."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    joints = asset.data.joint_pos[:, asset_cfg.joint_ids]
    keypts_robot = get_keypts(joints, env.joint_names, env.pk2_robot)
    
    global_keypts = motion_res["global_keypts"] + env.scene.env_origins.unsqueeze(1)

    robot_pos = asset.data.root_pos_w
    robot_quat = asset.data.root_quat_w
    ref_keypts_robot = math_utils.quat_apply(
        math_utils.quat_conjugate(robot_quat).unsqueeze(1).repeat_interleave(repeats=global_keypts.shape[1],dim=1), global_keypts - robot_pos.unsqueeze(1))
    return torch.sum(torch.norm((ref_keypts_robot-keypts_robot)*torch.tensor(keypts_mask, device=env.device).unsqueeze(0).unsqueeze(-1),dim=2), dim=1)

def joint_deviation_ref_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joint_mask: List = JOINTS_MASK) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    ref_joint_pos = motion_res["dof_pos"]
    angle = (asset.data.joint_pos[:, asset_cfg.joint_ids] - ref_joint_pos)*torch.tensor(joint_mask, device=env.device).unsqueeze(0)
    return torch.sum(torch.abs(angle), dim=1)


def position_tracking_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    root_pos = motion_res["root_pos"]
    root_pos_robot = asset.data.root_pos_w - env.scene.env_origins
    
    return torch.norm(root_pos - root_pos_robot, dim=1)

def orientation_tracking_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize tracking of the root orientation using L2-norm.

    The function computes the root orientation error between the desired root orientation (from the command) and the
    current root orientation of the asset's body (in world frame). The root orientation error is computed as the L2-norm
    of the difference between the desired and current root orientations.
    """
    # extract the asset (to enable type hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    root_rot = motion_res["root_rot"]
    return math_utils.quat_error_magnitude(root_rot, asset.data.root_quat_w)

# Distance (m) below which the geometric "point at the bottle" orientation target fades
# out. The target frame is built from the normalized wrist→grab_pos direction, which is
# singular as the wrist reaches the grab point: at 5 cm a 1 cm wrist move swings the
# target ~11°, at 2 cm ~28° — the target spins, the policy chases it, and the result is
# inward wrist rotation + high-frequency arm corrections concentrated at final approach.
# Inside the fade radius we blend toward the closed-phase constraint (wrist z-axis
# vertical), which is what the reward becomes after the grab anyway — making the
# orientation objective continuous through the grasp instead of singular at it.
_ORIENT_FADE_RADIUS = 0.15


def target_orientation_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    root_rot_link = math_utils.quat_unique(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :].clone())

    # NOTE: this used to sample motion state at `motion_times - 1.` (one second in the
    # past). grab_pos is static per motion so only is_closed was affected — the open→closed
    # mask switch fired a full second AFTER the actual grab, so for the first second of the
    # lift this reward still demanded the wrist point at the bottle's original resting
    # position. As the wrist rises, that direction tilts down/inward — the reward was
    # actively fighting the lift (dragging the grasped wrist back toward the table) and
    # feeding the inward-rotation behavior. Now sampled at the true current motion time,
    # consistent with every other reward term.
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
    root_pos = motion_res["grab_pos"] + motion_res["offsets"]
    time_mask = 1. - motion_res["is_closed"].float()

    a_axis = root_pos - root_pos_link # (2*x_axis + y_axis)/sqrt(5)
    wrist_to_grab_dist = torch.norm(a_axis, dim=1, keepdim=True)
    a_axis = a_axis / torch.clamp_min(wrist_to_grab_dist, 1e-6)
    
    b_axis = torch.zeros_like(a_axis)
    b_axis[:,0] = -a_axis[:,1]
    b_axis[:,1] = a_axis[:,0]
    b_axis[:,2] = 0.0
    b_axis = b_axis / torch.norm(b_axis, dim=1, keepdim=True)

    # z_axis is a_axis x b_axis
    z_axis = torch.cross(a_axis, b_axis, dim=1)

    y_axis = a_axis + 2*b_axis
    y_axis = y_axis / torch.norm(y_axis, dim=1, keepdim=True)

    x_axis = torch.cross(y_axis, z_axis, dim=1)

    target_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
    target_rot_quat = math_utils.quat_from_matrix(target_rot_mat)
    
    # angle between target and current orientation (root_rot_link)
    angle = math_utils.quat_error_magnitude(target_rot_quat, root_rot_link)

    z_axis_post = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    z_axis_w = math_utils.quat_apply(root_rot_link, z_axis_post)
    angle_post = torch.acos(torch.clamp(z_axis_w[:, 2], -1.0, 1.0))

    # Fade the geometric target out inside _ORIENT_FADE_RADIUS of the grab point:
    # fade = 1 (pure geometric target) when far, → 0 (pure vertical constraint) at the
    # bottle. Avoids the normalized-direction singularity at the grasp; the open-phase
    # objective transitions smoothly into the closed-phase objective.
    fade = torch.clamp(wrist_to_grab_dist.squeeze(1) / _ORIENT_FADE_RADIUS, 0.0, 1.0)
    open_phase_err = fade * torch.abs(angle) + (1.0 - fade) * torch.abs(angle_post)
    return open_phase_err * time_mask + torch.abs(angle_post) * (1. - time_mask)


# Canonical per-joint open/closed pose for the right hand. Values come from the original
# BinaryJointPositionActionCfg in motion_tracking_pick_env.py:575-583 — the source of truth
# for "what 'open hand' and 'closed hand' mean on this G1 model." Joints are listed in the
# same order as ContinuousFingersActionsCfg.right_hand_action so the indices line up.
_RIGHT_FINGER_NAMES = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
_RIGHT_FINGER_OPEN_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_RIGHT_FINGER_CLOSED_POSE = (
    0.0,                       # thumb_0
    -float(np.pi) / 3.0,       # thumb_1  (close direction: negative)
    -float(np.pi) / 2.0,       # thumb_2  (close direction: negative)
    float(np.pi) / 2.0,        # index_0  (close direction: positive)
    float(np.pi) / 2.0,        # index_1
    float(np.pi) / 2.0,        # middle_0
    float(np.pi) / 2.0,        # middle_1
)
_RIGHT_HAND_REWARD_SCALE = 1.0  # rad; sharpened from 2.0 — "halfway" reward drops to 0.011
                                # (was 0.107) and "always wrong target" drops to 5e-5 (was 0.011).
                                # Steeper landscape kills "always closed" and "blend=0.5"
                                # averaging strategies that exploited the smoother reward.
# Number of motion frames over which the finger target linearly interpolates from the open
# pose to the closed pose, ending at the grab_idx frame. At the env's 50 Hz control rate,
# 10 frames = 200 ms — matches the timescale of a human hand closing on an object. Before
# the window: target = open pose (avoids pre-closing on empty air during reach). After
# grab_idx: target = closed pose (drives the actual grasp).
_RIGHT_HAND_TRANSITION_FRAMES = 10.0


def right_hand_object_proximity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"]),
    std: float = 0.15,
) -> torch.Tensor:
    """Reward right-wrist proximity to the bottle during the OPEN phase of the reference motion.

    Provides an explicit gradient toward the bottle during reach — body-tracking rewards
    alone are too coarse (one sum over 27 joints) to land the right hand precisely at the
    grasp pose. Time-masked to the open phase (``is_closed = 0``) so post-grasp lifting
    isn't penalized for moving the wrist away from the bottle's resting spot.

    Returns ``exp(-(dist/std)^2) * (1 - is_closed)`` — gaussian-ish proximity in [0, 1]
    that drops off sharply outside ``std`` meters of the bottle (default 15 cm). With a
    moderate weight in the cfg, the max contribution sits comparable to the alive reward
    so it provides clear gradient without overwhelming body tracking.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wrist_pos = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] - env.scene.env_origins  # (N, 3)

    bottle: RigidObject = env.scene["object"]
    bottle_pos = bottle.data.root_pos_w - env.scene.env_origins  # (N, 3)

    dist = torch.norm(wrist_pos - bottle_pos, dim=1)  # (N,)
    proximity = torch.exp(-(dist / std) ** 2)         # (N,) in [0, 1]

    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(
        device=env.device, dtype=torch.float32
    )
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    is_open = 1.0 - motion_res["is_closed"].float()   # (N,)

    return proximity * is_open


def right_hand_state_target_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward right-hand finger joints matching the smoothly-interpolated target pose.

    Replaces the legacy reward that read ``env.action_manager.action[:, -1]`` (binary-action
    artifact; broken for ContinuousFingersActionsCfg).

    Target pose interpolates linearly between open and closed around the reference grab_idx
    so the policy isn't asked to *jump* finger pose in a single frame at the transition:

        current_frame < grab_idx - W   →  target = open pose
        current_frame ∈ [grab_idx-W, grab_idx]
                                       →  target = lerp(open, closed, alpha)
        current_frame ≥ grab_idx       →  target = closed pose

    where ``W = _RIGHT_HAND_TRANSITION_FRAMES`` (10 frames ≈ 200 ms at 50 Hz) and
    ``alpha = clamp((current_frame - (grab_idx - W)) / W, 0, 1)``.

    The grab_idx comes from ``env.motion_lib.switch_idxs[env.motion_ids]`` — the per-motion
    manually-annotated grasp frame from the SMPL retargeting pipeline. Smoothly closing
    around it (rather than at it) lets the bottle stay graspable: a fully-closed fist at
    the *start* of the reach blocks the bottle from entering the hand envelope when the
    wrist arrives. Open → gradual close → closed mirrors actual grasp dynamics.

    Return is exp(-L1_sum / SCALE) so the value stays in [0, 1] — +0.3 cfg weight matches.
    """
    asset = env.scene["robot"]

    # Cache joint indices + pose tensors on first call.
    if not hasattr(env, "_right_finger_joint_ids"):
        name_to_idx = {n: i for i, n in enumerate(asset.data.joint_names)}
        env._right_finger_joint_ids = torch.tensor(
            [name_to_idx[n] for n in _RIGHT_FINGER_NAMES],
            device=env.device, dtype=torch.long,
        )
        env._right_finger_open_pose = torch.tensor(
            _RIGHT_FINGER_OPEN_POSE, device=env.device, dtype=torch.float32,
        )
        env._right_finger_closed_pose = torch.tensor(
            _RIGHT_FINGER_CLOSED_POSE, device=env.device, dtype=torch.float32,
        )

    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(
        device=env.device, dtype=torch.float32
    )

    # Smooth open→closed interpolation alpha. motion_lib.switch_idxs is in *frame* units
    # (= grab_idx + idx_shift, set at motion-file load time); each motion has its own per-
    # motion timestep ``_motion_dt[motion_id]`` (different source recordings can have
    # different frame rates), so converting motion_times to frames is per-motion.
    motion_lib = env.motion_lib
    motion_dt = motion_lib._motion_dt[env.motion_ids]                              # (N,) float, s/frame
    current_frame = motion_times / motion_dt                                        # (N,) float
    switch_frame = motion_lib.switch_idxs[env.motion_ids]                          # (N,) float
    alpha = torch.clamp(
        (current_frame - (switch_frame - _RIGHT_HAND_TRANSITION_FRAMES))
        / _RIGHT_HAND_TRANSITION_FRAMES,
        0.0, 1.0,
    ).unsqueeze(1)                                                                  # (N, 1)

    target = (
        alpha * env._right_finger_closed_pose.unsqueeze(0)
        + (1.0 - alpha) * env._right_finger_open_pose.unsqueeze(0)
    )                                                                               # (N, 7)
    actual = asset.data.joint_pos[:, env._right_finger_joint_ids]                   # (N, 7)

    l1_dev = torch.sum(torch.abs(actual - target), dim=1)                           # (N,) in rad
    return torch.exp(-l1_dev / _RIGHT_HAND_REWARD_SCALE)                            # (N,) in [0, 1]

def right_hand_binary_match_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Original train.py-style binary-match reward for the right hand.

    Reads the right-hand binary action slot (last dim of ``action_manager.action`` under
    BinaryFingersActionsCfg = [27 body | 1 left | 1 right]) and compares its sign to the
    motion library's ``is_closed`` flag for the current frame. Returns 1 if the commanded
    direction agrees with the reference, 0 otherwise.

    Convention from BinaryJointPositionActionCfg: action < 0 → closed pose, action >= 0 →
    open pose. So we want action < 0 when is_closed = 1 and action > 0 when is_closed = 0.

    Replaces the joint-space ``exp(-L1/SCALE)`` reward (which is still used by the
    ContinuousFingers env). This version has no PD-tracking lag, no sharpness parameter,
    and a uniformly dense {0, 1} signal — well-suited to a 1-D binary action surface.
    """
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(
        device=env.device, dtype=torch.float32
    )
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    action_right_hand = env.action_manager.action[:, -1].clone()
    is_closed = motion_res["is_closed"].float()
    return ((action_right_hand < 0.0).float() * is_closed
            + (action_right_hand > 0.0).float() * (1.0 - is_closed))


def feet_air_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    Mirrors ``isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards.feet_air_time``
    but drops the velocity-command gate — the motion-tracking pick env has an empty
    ``CommandsCfg``, so the original ``command_manager.get_command("base_velocity")``
    lookup would error. Without the gate, the reward fires on every first-contact event;
    during the standing/reaching phase the feet should rarely lift, so ``first_contact``
    is mostly 0 and the term contributes only when the policy commits to a step (which
    is exactly the behavior we want to encourage during walking, and exactly the
    behavior we want to NOT happen during reaching).

    Reward at each first-contact event = ``last_air_time − threshold``. Below threshold:
    negative contribution (the air time was too short — rapid stepping). Above threshold:
    positive contribution (committed step). With a positive weight, this rewards committed
    steps and penalizes rapid foot tapping.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward


def left_hand_state_target_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The left hand state target of the robot in the environment.

    """
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)

    action_left_hand = env.action_manager.action[:,-2].clone()
    reward = (action_left_hand<0.)*motion_res["is_closed"].float() + (action_left_hand>0.)*(1. - motion_res["is_closed"].float()).float()
    
    return reward


@configclass
class G1Rewards():
    """Reward terms for the MDP."""

    # task terms
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    alive_reward = RewTerm(func=mdp.is_alive, weight=1.0)
    
    joint_deviation_ref = RewTerm(
        func=joint_deviation_ref_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)})
    
    keypts_deviation_ref = RewTerm(
        func=keypts_deviation_ref_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JointNamesOrder, preserve_order=True)})
    

    # pose_deviation
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

    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7,params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7,params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )
    

    right_hand_state_target_reward_val = RewTerm(
        func=right_hand_state_target_reward,
        weight=0.3)
    
    
    feet_parallel_to_ground = RewTerm(
        func=mdp.feet_parallel_to_ground,
        weight=-1.,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )


def root_below_threshold(env: ManagerBasedRLEnv, thres=0.3) -> torch.Tensor:
    """Terminate the episode when the root z is below threshold."""
    robot: RigidObject = env.scene["robot"]
    root_pos = robot.data.root_pos_w - env.scene.env_origins
    
    return root_pos[:,2] <= thres

def root_angle_below_threshold(env: ManagerBasedRLEnv, thres=0.5) -> torch.Tensor:
    """Terminate the episode when the root z is below threshold."""
    robot: RigidObject = env.scene["robot"]
    root_rot_quat = math_utils.quat_unique(robot.data.root_quat_w)
    root_rot_mat = math_utils.matrix_from_quat(root_rot_quat)
    root_z_axis = root_rot_mat[:, 2]  # Get the z-axis of the root orientation
    cos_angle_z = root_z_axis[:, 2]  # Extract the z-component of the z-axis
    # import pdb; pdb.set_trace()
    return cos_angle_z <= thres
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis","torso_link","waist_yaw_link","waist_roll_link","left_shoulder_pitch_link","right_shoulder_pitch_link",
    #                                             ]), "threshold": 1.0},
    # )
    torso_below_threshold = DoneTerm(
        func=root_below_threshold)
    torso_angle_below_threshold = DoneTerm(
        func=root_angle_below_threshold)
    # torso_angle_below_threshold = DoneTerm(
    #     func=root_angle_below_threshold)

    
@configclass
class CommandsCfg:
    """Command terms for the MDP."""



def target_ref(env: ManagerBasedRLEnv, time_offset: float = 0., visualize_markers: bool = VISUALIZE_MARKERS) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if not hasattr(env, 'motion_lib'):
        ref_joint_pos = torch.zeros((env.scene.num_envs, 27), device=env.device)
        ref_joint_vel = torch.zeros((env.scene.num_envs, 27), device=env.device)
        root_pos = torch.zeros((env.scene.num_envs, 3), device=env.device)
        root_rot = torch.zeros((env.scene.num_envs, 4), device=env.device)
        global_keypts = torch.zeros((env.scene.num_envs, 39, 3), device=env.device) + env.scene.env_origins.unsqueeze(1)
        root_rot[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
    else:
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32) + time_offset
        motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
        ref_joint_pos = motion_res["dof_pos"]
        ref_joint_vel = torch.zeros((env.scene.num_envs, 27), device=env.device)
        root_pos = motion_res["root_pos"]
        root_rot = motion_res["root_rot"]
        offset = motion_res["offsets"]
        local_keypts = motion_res["local_keypts"]
        global_keypts = motion_res["global_keypts"] + env.scene.env_origins.unsqueeze(1)
        # Translate global keypoints to the local frame of the robot
        goal_pos = motion_res["grab_pos"]
        if goal_pos is not None and visualize_markers:
            env.goal_marker.visualize(goal_pos + env.scene.env_origins + offset)
        else :
            marker_pos = torch.zeros((env.scene.num_envs, 3), device=env.device)
            marker_pos[:, 2] = -0.1
            env.goal_marker.visualize(marker_pos)
        # Update the visualization markers in the environment
        if time_offset == 0. and visualize_markers and hasattr(env, "update_visualization_markers"):
            env.update_visualization_markers(global_keypts)
        if not visualize_markers and hasattr(env, "update_visualization_markers"):
            marker_pos = torch.zeros_like(global_keypts)
            marker_pos[:, :, 2] = -0.1
            env.update_visualization_markers(marker_pos)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_robot = asset.data.root_pos_w - env.scene.env_origins
    root_rot_robot = math_utils.quat_unique(asset.data.root_quat_w)
    
    robot_pos = asset.data.root_pos_w
    robot_quat = asset.data.root_quat_w
    
    local_robot_keypts = math_utils.quat_apply(
        math_utils.quat_conjugate(robot_quat).unsqueeze(1).repeat_interleave(repeats=global_keypts.shape[1],dim=1), global_keypts - robot_pos.unsqueeze(1)).reshape(
        env.scene.num_envs, -1)
        
    # Convert the root position (root_pos) to the robot's local frame with root_pos_robot as the origin and root_rot_robot as the rotation.
    root_pos_robot_local = math_utils.quat_apply(
        math_utils.quat_conjugate(root_rot_robot), root_pos - root_pos_robot)
    root_rot_robot_local = math_utils.quat_mul(
        math_utils.quat_conjugate(root_rot_robot), root_rot)
    
    # Create the target reference tensor
    target_ref_tensor = torch.cat([ref_joint_pos, ref_joint_vel, root_pos_robot_local, root_rot_robot_local, local_robot_keypts], dim=-1)
    
    return target_ref_tensor

def target_ref_slim(env: ManagerBasedRLEnv, time_offset: float = 0.) -> torch.Tensor:
    """Slim future-reference sample: ref joint pos + ref root pose in robot frame (34 dims).

    Far-horizon companion to ``target_ref`` (178 dims). Drops the 117-D keypoint block
    (FK of joints+root — needed for precision reaching at the near horizon, redundant for
    gait planning further out) and the 27 always-zero joint-vel padding. Content mirrors
    the joint/root-level ``command_multi_future`` input of SONIC's G1 encoder, which sees
    10 future frames at 0.1 s spacing (1.0 s horizon, sonic_release.yaml:48-49) — long
    enough to cover a full reference stride before the policy must commit to a step.
    No visualization-marker side effects (unlike ``target_ref``).
    """
    if not hasattr(env, 'motion_lib'):
        out = torch.zeros((env.scene.num_envs, 34), device=env.device)
        out[:, 30] = 1.0  # identity quaternion w-component (dims 30:34 are root_rot)
        return out
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(
        device=env.device, dtype=torch.float32
    ) + time_offset
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    ref_joint_pos = motion_res["dof_pos"]                                   # (N, 27)
    root_pos = motion_res["root_pos"]                                       # (N, 3)
    root_rot = motion_res["root_rot"]                                       # (N, 4)

    asset: RigidObject = env.scene["robot"]
    root_pos_robot = asset.data.root_pos_w - env.scene.env_origins
    root_rot_robot = math_utils.quat_unique(asset.data.root_quat_w)
    root_pos_robot_local = math_utils.quat_apply(
        math_utils.quat_conjugate(root_rot_robot), root_pos - root_pos_robot)
    root_rot_robot_local = math_utils.quat_mul(
        math_utils.quat_conjugate(root_rot_robot), root_rot)

    return torch.cat([ref_joint_pos, root_pos_robot_local, root_rot_robot_local], dim=-1)  # (N, 34)


def current_time_enc(env: ManagerBasedRLEnv, n_freqs = 1) -> torch.Tensor:
    """The current time in the environment."""
    if not hasattr(env, 'motion_lib'):
        curr_time = torch.zeros((env.scene.num_envs, ), device=env.device)
    else:
        curr_time = (env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)).clone().detach()
    curr_time_norm = curr_time / env.cfg.episode_length_s
    encodings = curr_time_norm.unsqueeze(1)  # Start with the normalized time as the first encoding
    for i in range(n_freqs):
        freq = 2 ** i
        curr_time_enc = torch.sin(freq * curr_time_norm * 2 * np.pi)
        encodings = torch.cat([encodings, curr_time_enc.unsqueeze(1)], dim=1)
    return encodings



    
def right_hand_state_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The right hand state target of the robot in the environment.
    """
    if not hasattr(env, 'motion_lib'):
        obs = torch.zeros((env.scene.num_envs, 1), device=env.device)
        return obs
    else :
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
        motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
        return motion_res["is_closed"].unsqueeze(1).float()  # Ensure the output is a float tensor



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
        
        a_axis = root_pos_link - root_pos # (2*x_axis + y_axis)/sqrt(5)
        a_axis = a_axis / torch.norm(a_axis, dim=1, keepdim=True)
        
        b_axis = torch.zeros_like(a_axis)
        b_axis[:,0] = -a_axis[:,1]
        b_axis[:,1] = a_axis[:,0]
        b_axis[:,2] = 0.0
        b_axis = b_axis / torch.norm(b_axis, dim=1, keepdim=True)

        # z_axis is a_axis x b_axis
        z_axis = torch.cross(a_axis, b_axis, dim=1)

        x_axis = a_axis + 2*b_axis
        x_axis = x_axis / torch.norm(x_axis, dim=1, keepdim=True)

        y_axis = torch.cross(z_axis, x_axis, dim=1)

        target_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
        target_rot_quat = math_utils.quat_from_matrix(target_rot_mat)
        
        
        z_axis_post = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        x_axis_post = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        y_axis_post = torch.tensor([1.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        
        target_post_rot_mat = torch.stack([x_axis_post, y_axis_post, z_axis_post], dim=2)  # shape (N, 3, 3)
        target_post_rot_quat = math_utils.quat_from_matrix(target_post_rot_mat)
        return target_rot_quat * time_mask.unsqueeze(1) + target_post_rot_quat * (1. - time_mask.unsqueeze(1))



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
        
        target_ref = ObsTerm(func=target_ref)

        target_orientation = ObsTerm(
            func=target_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names=["right_wrist_yaw_link"])})

        right_hand_state_target_val = ObsTerm(
            func=right_hand_state_target)


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=JointNamesOrder, preserve_order=True, scale=0.5, use_default_offset=True)
    
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
            open_command_expr={"right_hand_index_.*": np.pi / 2.0, 
                                "right_hand_middle_.*": np.pi / 2.0,
                                "right_hand_thumb_0_joint": 0.0,
                                "right_hand_thumb_1_joint": -np.pi / 3.0,
                                "right_hand_thumb_2_joint": -np.pi/2.,
                                },
            close_command_expr={"right_hand_index_.*": np.pi / 2.0, 
                                "right_hand_middle_.*": np.pi / 2.0,
                                "right_hand_thumb_0_joint": 0.0,
                                "right_hand_thumb_1_joint": -np.pi / 3.0,
                                "right_hand_thumb_2_joint": -np.pi/2.,
                                })

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1., 1., 1.), metallic=0),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=400.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class G1MotionTrackingEnvCfg(ManagerBasedRLEnvCfg):
    rewards: G1Rewards = G1Rewards()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=1024, env_spacing=2.5)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.ref_motions_path = "../TrajGen/sample/follow2"
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        
