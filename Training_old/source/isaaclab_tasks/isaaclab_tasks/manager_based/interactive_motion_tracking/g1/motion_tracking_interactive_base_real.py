# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from ast import List
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.motion_tracking.g1.motion_tracking_env import reset_joints_for_motion, reset_root_state_for_motion, get_keypts
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.assets import Articulation, RigidObject
import isaaclab.utils.math as math_utils
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg

from dataclasses import MISSING
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip
from isaaclab_tasks.utils.motion_lib.motion_lib_base import JointNamesOrder


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
    0, # left_wrist_pitch_joint
    0, # left_wrist_yaw_joint
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
    0, # left_rubber_hand
    1, # right_shoulder_pitch_link
    1, # right_shoulder_roll_link
    1, # right_shoulder_yaw_link
    1, # right_elbow_link
    1, # right_wrist_roll_link
    1, # right_wrist_pitch_link
    1, # right_wrist_yaw_link
    0, # right_rubber_hand
]


def reset_object_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    offset = [0.0, 0.0],
    height = 0.67,
):
    """Reset the object root state appropriately.
    """
    # get default root state
    if hasattr(env, 'start_motion_times'):
        motion_times = env.start_motion_times[env_ids]
        motion_ids = env.motion_ids[env_ids]
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        velocities = torch.zeros((env.scene.num_envs, 6), device=env.device)[env_ids]
        motion_offset = motion_res["offsets"]
        goal_pos = motion_res["grab_pos"]
        root_states = torch.cat([motion_res["root_pos"], motion_res["root_rot"], velocities], dim=-1)
        orientations = root_states[:, 3:7]
        object: RigidObject = env.scene["object"]

        if 'object_pose' in env.command_manager.active_terms :
            _goal_pos = env.command_manager.get_command('object_pose')[:, :3]
            goal_pos = math_utils.quat_apply(orientations, _goal_pos[env_ids]) + root_states[:, 0:3]
        
        object_pos = goal_pos + motion_offset + env.scene.env_origins[env_ids]
        object_pos[:,2] = height
        object_pos[:,0] += offset[0]
        object_pos[:,1] += offset[1]
        object_quat = torch.zeros((env.scene.num_envs, 4), device=env.device)[env_ids]
        object_quat[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        object.write_root_pose_to_sim(torch.cat([object_pos, object_quat], dim=-1), env_ids=env_ids)
        object.write_root_velocity_to_sim(velocities, env_ids=env_ids)
        
def reset_root_state_for_motion_w_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset_z: float = 0.15,
    offset_xy: tuple[float, float] = (0.0, 0.0),
    wait_until: float = 0.0,
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
        if env.common_step_counter < wait_until and env.num_envs > 1001:
            positions[:, 0] += offset_xy[0]
            positions[:, 1] += offset_xy[1]

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

        # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
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
        func=reset_root_state_for_motion_w_offset,
        mode="reset",
        params={
            "offset_z": 0.05,
            "offset_xy": (-1.0, 0.0),
            "wait_until": 40000,
        }
    )

def keypts_deviation_ref_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), keypts_mask: List = KEYPTS_MASK) -> torch.Tensor:
    """Penalize deviation of keypoints from the reference keypoints."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    joints = asset.data.joint_pos[:, asset_cfg.joint_ids]
    keypts_robot = get_keypts(joints, env.joint_names, env.pk2_robot)
    
    local_keypts = motion_res["local_keypts"].clone()
    ref_rot = motion_res["root_rot"].clone()
    ref_pos = motion_res["root_pos"].clone()
    ref_euler_roll, ref_euler_pitch, ref_euler_yaw = math_utils.euler_xyz_from_quat(ref_rot)
    robot_pos = asset.data.root_pos_w
    robot_quat = asset.data.root_quat_w
    robot_euler_roll, robot_euler_pitch, robot_euler_yaw = math_utils.euler_xyz_from_quat(robot_quat)
    quat = math_utils.quat_from_euler_xyz(ref_euler_roll, ref_euler_pitch, robot_euler_yaw)
    global_keypts = math_utils.quat_apply(
        quat.unsqueeze(1).repeat_interleave(repeats=local_keypts.shape[1], dim=1), local_keypts) 
    global_keypts[:, :, :2] += asset.data.root_pos_w.unsqueeze(1)[:, :, :2]
    global_keypts[:, :, 2] += ref_pos[:, 2].unsqueeze(1)
    if "object_pose" not in env.command_manager.active_terms:
        env.update_visualization_markers(global_keypts)
    
    ref_keypts_robot = math_utils.quat_apply(
        math_utils.quat_conjugate(robot_quat).unsqueeze(1).repeat_interleave(repeats=global_keypts.shape[1],dim=1), global_keypts - robot_pos.unsqueeze(1))
    return torch.sum(torch.norm((ref_keypts_robot-keypts_robot)*torch.tensor(keypts_mask, device=env.device).unsqueeze(0).unsqueeze(-1),dim=2), dim=1)

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
    
    return torch.abs(root_pos[:,2] - root_pos_robot[:,2])

def velocity_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: RigidObject = env.scene[asset_cfg.name]
    root_vel_lin = robot.data.root_lin_vel_b
    root_vel_angular = robot.data.root_ang_vel_b
    return torch.norm(root_vel_lin[:, :2], dim=1) + torch.abs(root_vel_angular[:, 2])

def target_orientation_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    root_rot_link = math_utils.quat_unique(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :].clone())
    
    
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times - 1.
    motion_ids = env.motion_ids
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
    time_mask = 1. - motion_res["is_closed"].float()
    
    
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    
    target_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
    target_rot_quat = math_utils.quat_from_matrix(target_rot_mat)
    
    # angle between target and current orientation (root_rot_link)
    angle = math_utils.quat_error_magnitude(target_rot_quat, root_rot_link)

    x_axis_post = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    y_axis_post = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    z_axis_post = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    # z_axis_w = math_utils.quat_apply(root_rot_link, z_axis_post)
    target_post_rot_mat = torch.stack([x_axis_post, y_axis_post, z_axis_post], dim=2)  # shape (N, 3, 3)
    target_post_rot_quat = math_utils.quat_from_matrix(target_post_rot_mat)
    angle_post = math_utils.quat_error_magnitude(target_post_rot_quat, root_rot_link)
    return torch.abs(angle) * time_mask + torch.abs(angle_post) * (1. - time_mask)

def target_orientation_error_rev(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    root_rot_link = math_utils.quat_unique(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :].clone())
    

    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times - 1.
    motion_ids = env.motion_ids
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
    
    time_mask = 1. - motion_res["is_closed"].float()
    
    
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    y_axis = torch.tensor([0.0, -1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    z_axis = torch.tensor([0.0, 0.0, -1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    
    target_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
    target_rot_quat = math_utils.quat_from_matrix(target_rot_mat)
    
    # angle between target and current orientation (root_rot_link)
    angle = math_utils.quat_error_magnitude(target_rot_quat, root_rot_link)

    x_axis_post = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    y_axis_post = torch.tensor([0.0, -1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    z_axis_post = torch.tensor([0.0, 0.0, -1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    # z_axis_w = math_utils.quat_apply(root_rot_link, z_axis_post)
    target_post_rot_mat = torch.stack([x_axis_post, y_axis_post, z_axis_post], dim=2)  # shape (N, 3, 3)
    target_post_rot_quat = math_utils.quat_from_matrix(target_post_rot_mat)
    angle_post = math_utils.quat_error_magnitude(target_post_rot_quat, root_rot_link)
    return torch.abs(angle) * time_mask + torch.abs(angle_post) * (1. - time_mask)


def object_above_threshold(env: ManagerBasedRLEnv, height_thres = 0.7, fall_thres = 0.66) -> torch.Tensor:
    object: RigidObject = env.scene["object"]
    root_pos = object.data.root_pos_w
    # print(root_pos[:,2])
    has_grasped = (root_pos[:,2] > height_thres) * 1. + (root_pos[:,2] < height_thres) * (root_pos[:,2] > fall_thres) * (root_pos[:,2]-fall_thres) / (height_thres-fall_thres)
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    if env.num_envs < 1001 :
        env.n_successes += (root_pos[:,2] > height_thres) * motion_res["is_closed"]
    return (has_grasped)*motion_res["is_closed"].float()


def object_approach_reward_left(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0, offset_x: float = 0., offset_y: float = 0.) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    root_rot_link = math_utils.quat_unique(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :].clone())
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    x_axis_w = math_utils.quat_apply(root_rot_link, x_axis)
    root_pos_link += x_axis_w * offset_x
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    rot_mat = math_utils.matrix_from_quat(root_rot_link)
    y_axis_w = math_utils.quat_apply(root_rot_link, y_axis)
    root_pos_link += y_axis_w * offset_y

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    z_axis_w = math_utils.quat_apply(root_rot_link, z_axis)
    
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32) + 1.
    motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
    root_pos = motion_res["grab_pos"] + motion_res["offsets"]
    
    motion_times_back = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times_back)
    
    time_mask = (1. - motion_res["is_closed"].float()) * (motion_times_back > 0.1)
    rel_vector = root_pos_link - root_pos
    dist = torch.norm(rel_vector, dim=1)
    reward = (env.prev_dist-dist)*time_mask
    env.prev_dist = dist
    return reward

def object_approach_reward_right(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0, offset_x: float = 0., offset_y: float = 0.) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    root_rot_link = math_utils.quat_unique(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :].clone())
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    x_axis_w = math_utils.quat_apply(root_rot_link, x_axis)
    root_pos_link += x_axis_w * offset_x
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    rot_mat = math_utils.matrix_from_quat(root_rot_link)
    y_axis_w = math_utils.quat_apply(root_rot_link, y_axis)
    root_pos_link += y_axis_w * offset_y

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    z_axis_w = math_utils.quat_apply(root_rot_link, z_axis)
    
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32) + 1.
    motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
    root_pos = motion_res["grab_pos"] + motion_res["offsets"]
    
    motion_times_back = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times_back)
    
    time_mask = (1. - motion_res["is_closed"].float()) * (motion_times_back > 0.1)
    rel_vector = root_pos_link - root_pos
    dist = torch.norm(rel_vector, dim=1)
    reward = (env.prev_dist1-dist)*time_mask
    env.prev_dist1 = dist
    return reward

def touch_goal(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, thres=0.05, offset_x: float = 0.0, offset_y: float = 0.0, offset_z: float = 0.0) -> torch.Tensor:
    """Reward if the right hand index finger is close to the goal.
    """
    pos_right_hand = env.scene["robot"].data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32) + 0.1
    motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
    root_rot_link = math_utils.quat_unique(env.scene["robot"].data.body_quat_w[:, asset_cfg.body_ids[0], :].clone())
    
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=pos_right_hand.device).unsqueeze(0).repeat(pos_right_hand.shape[0], 1)
    x_axis_w = math_utils.quat_apply(root_rot_link, x_axis)
    pos_right_hand += x_axis_w * offset_x
    
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=pos_right_hand.device).unsqueeze(0).repeat(pos_right_hand.shape[0], 1)
    y_axis_w = math_utils.quat_apply(root_rot_link, y_axis)
    pos_right_hand += y_axis_w * offset_y

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=pos_right_hand.device).unsqueeze(0).repeat(pos_right_hand.shape[0], 1)
    z_axis_w = math_utils.quat_apply(root_rot_link, z_axis)
    pos_right_hand += z_axis_w * offset_z

    is_closed = motion_res["is_closed"].float()
    pos_goal = motion_res["grab_pos"] + motion_res["offsets"]
    reward = torch.norm(pos_right_hand - pos_goal, dim=1) < thres
    
    motion_times_next = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32) - 0.1
    motion_res_next = env.motion_lib.get_motion_state(env.motion_ids, motion_times_next)
    is_closed_next = motion_res_next["is_closed"].float()
    if env.num_envs < 1001 :
        env.n_successes += (reward * is_closed * (1 - is_closed_next)) > 0.
    return reward * is_closed * (1. - is_closed_next)

def goal_approach_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), offset_x: float = 0.21, offset_y: float = 0.0, offset_z: float = 0.03) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    root_rot_link = math_utils.quat_unique(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :].clone())
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    x_axis_w = math_utils.quat_apply(root_rot_link, x_axis)
    root_pos_link += x_axis_w * offset_x
    y_axis = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    rot_mat = math_utils.matrix_from_quat(root_rot_link)
    y_axis_w = math_utils.quat_apply(root_rot_link, y_axis)
    root_pos_link += y_axis_w * offset_y

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
    z_axis_w = math_utils.quat_apply(root_rot_link, z_axis)
    root_pos_link += z_axis_w * offset_z

    motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.long)
    motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
    motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
    root_pos = motion_res["grab_pos"] + motion_res["offsets"]

    time_mask = (1. - motion_res["is_closed"].float()) * (motion_times > 0.1)
    rel_vector = root_pos_link - root_pos
    dist = torch.norm(rel_vector, dim=1)
    reward = (env.prev_dist-dist)*time_mask
    env.prev_dist = dist
    
    return reward

@configclass
class G1Rewards():
    """Reward terms for the MDP."""

    # task terms
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    alive_reward = RewTerm(func=mdp.is_alive, weight=1.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7,params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7,params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_parallel_to_ground = RewTerm(
        func=mdp.feet_parallel_to_ground,
        weight=-1.,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
        },
    )


    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

def target_z(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_robot = asset.data.root_pos_w - env.scene.env_origins
    root_rot_robot = math_utils.quat_unique(asset.data.root_quat_w)
    if not hasattr(env, 'motion_lib'):
        # If motion library is not available, return a tensor of zeros
        obs = torch.zeros((env.scene.num_envs, 7), device=env.device)
        obs[:, 4] = 1.0  # set the w component to 1.0 for identity quaternion
        return obs
    else :
    
        root_rot = torch.zeros((env.scene.num_envs, 4), device=env.device)
        root_rot[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32) + 1.
        motion_ids = env.motion_ids.clone().detach().to(device=env.device, dtype=torch.int32)
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        root_pos = motion_res["grab_pos"] + motion_res["offsets"]
        init_root_pos_robot = root_pos_robot.clone()*0.
        init_root_pos_robot[:, 2] = 0.  # Initial position of the robot's root in the local frame
        root_pos_robot_local = root_pos - init_root_pos_robot
        if "object_pose" in env.command_manager.active_terms:
            root_pos_robot_local = env.command_manager.get_command("object_pose")[:, :3]
        
        
        root_pos_robot_local[:, :2] = 0.
        
        return root_pos_robot_local


def rel_pose_object(env: ManagerBasedRLEnv, action_name: str | None = None, fix_height: bool = False, fix_rel_base_height: bool = False, base_height: float = 0.793, only_pos: bool = False) -> torch.Tensor:
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_robot = asset.data.root_pos_w - env.scene.env_origins
    # root_rot_robot = math_utils.quat_unique(asset.data.root_quat_w)
    if not hasattr(env, 'motion_lib'):
        # If motion library is not available, return a tensor of zeros
        obs = torch.zeros((env.scene.num_envs, 7), device=env.device)
        obs[:, 4] = 1.0  # set the w component to 1.0 for identity quaternion
        return obs
    else :
        root_rot = torch.zeros((env.scene.num_envs, 4), device=env.device)
        root_rot[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times + 1.
        motion_ids = env.motion_ids
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        root_pos = motion_res["grab_pos"] + motion_res["offsets"]
        if fix_rel_base_height:
            init_root_pos_robot = root_pos_robot.clone()*0.
            init_root_pos_robot[:, 2] = base_height
            root_pos_robot_local = root_pos - init_root_pos_robot
        else :
            init_root_pos_robot = root_pos_robot.clone()*0.
            root_pos_robot_local = root_pos - init_root_pos_robot
        root_rot_robot_local = root_rot
        
        if fix_height:
            root_pos_robot_local[:, 2] = 0.213
        # Create the target reference tensor
        if "object_pose" in env.command_manager.active_terms:
            root_pos_robot_local = env.command_manager.get_command("object_pose")[:, :3]
        
        if only_pos:
            target_ref_tensor = root_pos_robot_local
        else:
            target_ref_tensor = torch.cat([root_pos_robot_local, root_rot_robot_local], dim=-1)
        
        return target_ref_tensor

def global_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_robot = asset.data.root_pos_w - env.scene.env_origins
    root_rot_robot = math_utils.quat_unique(asset.data.root_quat_w)
    _, _, yaw = math_utils.euler_xyz_from_quat(root_rot_robot)
    x = root_pos_robot[:, 0]
    y = root_pos_robot[:, 1]
    # z = root_pos_robot[:, 2]
    # import pdb; pdb.set_trace()
    target_ref_tensor = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1), yaw.unsqueeze(-1)], dim=-1)
    return target_ref_tensor

def rel_pose_goal(env: ManagerBasedRLEnv, action_name: str | None = None, fix_height: bool = False, fix_rel_base_height: bool = False) -> torch.Tensor:
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_robot = asset.data.root_pos_w - env.scene.env_origins
    root_rot_robot = math_utils.quat_unique(asset.data.root_quat_w)
    if not hasattr(env, 'motion_lib'):
        # If motion library is not available, return a tensor of zeros
        obs = torch.zeros((env.scene.num_envs, 7), device=env.device)
        obs[:, 4] = 1.0  # set the w component to 1.0 for identity quaternion
        return obs
    else :
        root_rot = torch.zeros((env.scene.num_envs, 4), device=env.device)
        root_rot[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times + 1.
        motion_ids = env.motion_ids
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        root_pos = motion_res["grab_pos"] + motion_res["offsets"]
        if fix_rel_base_height:
            init_root_pos_robot = root_pos_robot.clone()*0.
            init_root_pos_robot[:, 2] = 0.793
            root_pos_robot_local = math_utils.quat_apply(
                math_utils.quat_conjugate(root_rot_robot), root_pos - init_root_pos_robot)
        else :
            root_pos_robot_local = math_utils.quat_apply(
                math_utils.quat_conjugate(root_rot_robot), root_pos - root_pos_robot)
        root_rot_robot_local = math_utils.quat_mul(
            math_utils.quat_conjugate(root_rot_robot), root_rot)
        
        if fix_height:
            root_pos_robot_local[:, 2] = 0.213

        if "goal_pose" in env.command_manager.active_terms:
            root_pos_robot_local = env.command_manager.get_command("goal_pose")[:, :3]
        
        # Create the target reference tensor
        target_ref_tensor = torch.cat([root_pos_robot_local, root_rot_robot_local], dim=-1)
        
        return target_ref_tensor

def rel_pos_goal(env: ManagerBasedRLEnv, action_name: str | None = None, fix_height: bool = False, fix_rel_base_height: bool = False) -> torch.Tensor:
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_robot = asset.data.root_pos_w - env.scene.env_origins
    root_rot_robot = math_utils.quat_unique(asset.data.root_quat_w)
    if not hasattr(env, 'motion_lib'):
        # If motion library is not available, return a tensor of zeros
        obs = torch.zeros((env.scene.num_envs, 7), device=env.device)
        obs[:, 4] = 1.0  # set the w component to 1.0 for identity quaternion
        return obs
    else :
        root_rot = torch.zeros((env.scene.num_envs, 4), device=env.device)
        root_rot[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times + 1.
        motion_ids = env.motion_ids
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        root_pos = motion_res["grab_pos"] + motion_res["offsets"]
        if fix_rel_base_height:
            init_root_pos_robot = root_pos_robot.clone()*0.
            init_root_pos_robot[:, 2] = 0.793
            root_pos_robot_local = math_utils.quat_apply(
                math_utils.quat_conjugate(root_rot_robot), root_pos - init_root_pos_robot)
        else :
            root_pos_robot_local = math_utils.quat_apply(
                math_utils.quat_conjugate(root_rot_robot), root_pos - root_pos_robot)
        root_rot_robot_local = math_utils.quat_mul(
            math_utils.quat_conjugate(root_rot_robot), root_rot)
        
        if fix_height:
            root_pos_robot_local[:, 2] = 0.213

        if "goal_pose" in env.command_manager.active_terms:
            root_pos_robot_local = env.command_manager.get_command("goal_pose")[:, :3]
        
        return root_pos_robot_local


def hand_pose(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: SceneEntityCfg = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    root_rot_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7].clone()
    
    target_ref_tensor = torch.cat([root_pos_link, root_rot_link], dim=-1)
    
    return target_ref_tensor


def target_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    if not hasattr(env, 'motion_lib'):
        return_val = torch.zeros((env.scene.num_envs, 4), device=env.device)
        return_val[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        return return_val
    else :
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times - 1.
        motion_ids = env.motion_ids
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        # root_pos = motion_res["grab_pos"] + motion_res["offsets"]
        time_mask = 1. - motion_res["is_closed"].float()

        x_axis = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)

        target_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
        target_rot_quat = math_utils.quat_from_matrix(target_rot_mat)
        
        
        x_axis_post = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        y_axis_post = torch.tensor([0.0, 1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        z_axis_post = torch.tensor([0.0, 0.0, 1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        
        target_post_rot_mat = torch.stack([x_axis_post, y_axis_post, z_axis_post], dim=2)  # shape (N, 3, 3)
        target_post_rot_quat = math_utils.quat_from_matrix(target_post_rot_mat)
        return target_rot_quat * time_mask.unsqueeze(1) + target_post_rot_quat * (1. - time_mask.unsqueeze(1))

def target_orientation_rev(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
    if not hasattr(env, 'motion_lib'):
        return_val = torch.zeros((env.scene.num_envs, 4), device=env.device)
        return_val[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        return return_val
    else :
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times - 1.
        motion_ids = env.motion_ids
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        time_mask = 1. - motion_res["is_closed"].float()

        x_axis = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        y_axis = torch.tensor([0.0, -1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        z_axis = torch.tensor([0.0, 0.0, -1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)

        target_rot_mat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # shape (N, 3, 3)
        target_rot_quat = math_utils.quat_from_matrix(target_rot_mat)
        
        
        x_axis_post = torch.tensor([1.0, 0.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        y_axis_post = torch.tensor([0.0, -1.0, 0.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        z_axis_post = torch.tensor([0.0, 0.0, -1.0], device=root_pos_link.device).unsqueeze(0).repeat(root_pos_link.shape[0], 1)
        
        target_post_rot_mat = torch.stack([x_axis_post, y_axis_post, z_axis_post], dim=2)  # shape (N, 3, 3)
        target_post_rot_quat = math_utils.quat_from_matrix(target_post_rot_mat)
        # import pdb; pdb.set_trace()
        return target_rot_quat * time_mask.unsqueeze(1) + target_post_rot_quat * (1. - time_mask.unsqueeze(1))

def hand_state_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The hand state target of the robot in the environment.
    """
    if not hasattr(env, 'motion_lib'):
        obs = torch.zeros((env.scene.num_envs, 1), device=env.device)
        return obs
    else :
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times.clone().detach().to(device=env.device, dtype=torch.float32)
        motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
        return motion_res["is_closed"].unsqueeze(1).float()  # Ensure the output is a float tensor

def hand_state_target_1(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The hand state target of the robot in the environment.

    """
    if not hasattr(env, 'motion_lib'):
        # If motion library is not available, return a tensor of zeros
        obs = torch.zeros((env.scene.num_envs, 1), device=env.device)
        return obs
    else :
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times - .7
        motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
        return_val = 1. - motion_res["is_closed"].unsqueeze(1).float()  # Ensure the output is a float tensor
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times + .7
        motion_res = env.motion_lib.get_motion_state(env.motion_ids, motion_times)
        return_val *= motion_res["is_closed"].unsqueeze(1).float()  # Ensure the output is a float tensor
        return return_val  # Ensure the output is a float tensor

def rel_pose_object_w_link(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    if not hasattr(env, 'motion_lib'):
        obs = torch.zeros((env.scene.num_envs, 7), device=env.device)
        obs[:, 4] = 1.0  # set the w component to 1.0 for identity quaternion
        return obs
    else :
    
        asset: SceneEntityCfg = env.scene[asset_cfg.name]
        root_pos_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone() - env.scene.env_origins # type: ignore
        root_rot_link = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7].clone()
        root_rot = torch.zeros((env.scene.num_envs, 4), device=env.device)
        root_rot[:, 0] = 1.0  # set the w component to 1.0 for identity quaternion
        
        motion_times = env.episode_length_buf * env.step_dt + env.start_motion_times + 1.
        motion_ids = env.motion_ids
        motion_res = env.motion_lib.get_motion_state(motion_ids, motion_times)
        root_pos = motion_res["grab_pos"] + motion_res["offsets"]
        
        root_pos_link_local = math_utils.quat_apply(
            math_utils.quat_conjugate(root_rot_link), root_pos - root_pos_link)
        root_rot_link_local = math_utils.quat_mul(
            math_utils.quat_conjugate(root_rot_link), root_rot)
        
        # Create the target reference tensor
        target_ref_tensor = torch.cat([root_pos_link_local, root_rot_link_local], dim=-1)
    
    return target_ref_tensor

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
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=JointNamesOrder, preserve_order=True, scale=0.5, use_default_offset=True)
    
    
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
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    
    camera = None 

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class G1InteractiveBaseEnvCfg(ManagerBasedRLEnvCfg):
    rewards: G1Rewards = G1Rewards()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    observations: ObservationsCfg = ObservationsCfg()
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    terminations: TerminationsCfg = TerminationsCfg()
    actions: ActionsCfg = ActionsCfg()

    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        

