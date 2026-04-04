# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_apply
from isaaclab.assets import RigidObject, Articulation
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def position_command_error(env: ManagerBasedRLEnv, command_name: str, shift: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3].clone()  # clone to avoid modifying the command tensor
    des_quat_b = command[:, 3:7]
    # Shift des_pos_b by shift amount along x-axis of des_quat_w
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=des_pos_b.device).unsqueeze(0).repeat(des_pos_b.shape[0], 1)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7].clone()
    x_axis_w = quat_apply(curr_quat_w, x_axis)
    # des_pos_b += x_axis_b * shift
    # import pdb; pdb.set_trace()
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone()  # type: ignore
    curr_pos_w += x_axis_w * shift
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, shift: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3].clone()  # clone to avoid modifying the command tensor
    des_quat_b = command[:, 3:7]
    # Shift des_pos_b by shift amount along x-axis of des_quat_w
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=des_pos_b.device).unsqueeze(0).repeat(des_pos_b.shape[0], 1)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7].clone()
    x_axis_w = quat_apply(curr_quat_w, x_axis)
    # des_pos_b += x_axis_b * shift
    # import pdb; pdb.set_trace()
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3].clone()  # type: ignore
    curr_pos_w += x_axis_w * shift
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, tol: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return torch.maximum(quat_error_magnitude(curr_quat_w, des_quat_w) - tol, torch.zeros_like(curr_quat_w[:, 0]))


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_contact_reward(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Compute a binary reward based on feet contact and command activity.

    For each sample in the batch:
      - All four contact points (specified by sensor_cfg.body_ids) must be in contact.
        A foot is considered in contact if the maximum net force over history exceeds 1.0.
      - There must be no active command, i.e. the norm of the first two command dimensions is < 0.06.

    The reward is 1 if both conditions are met (all feet are in contact and no command is issued);
    otherwise, the reward is 0.
    """
    # Retrieve the contact sensor from the scene using the sensor configuration name.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Compute the contact status for the specified body parts:
    # - Calculate the norm of the net forces over the last history,
    # - Take the maximum over the history dimension,
    # - Check if it exceeds 1.0.
    # The resulting tensor has shape [batch, num_feet] (expected to be 4).
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    
    # Check if all feet are in contact for each sample.
    feet_all_contact = contacts.all(dim=1)
    
    # Retrieve the command tensor, assumed to be of shape [batch, command_dim].
    command = env.command_manager.get_command(command_name)
    
    # Check for "no command" situation: 
    # the command is inactive if the norm of its first two components is less than 0.06.
    no_command = torch.norm(command[:, :2], dim=1) < 0.06
    
    # The final binary reward is 1 if all feet are in contact AND there is no command, else 0.
    reward = (no_command & feet_all_contact).float()
    # print ('feet contact reward ', reward)
    
    return reward

def feet_slide(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # import pdb; pdb.set_trace()
    asset = env.scene[asset_cfg.name]
    # print(contacts)
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # env.net_movement += asset.data.body_lin_vel_w[0, asset_cfg.body_ids, :]
    # print('net movement: ', env.net_movement)

    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_parallel_to_ground(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet not being parallel to the ground.

    """
    # Penalize feet sliding
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=body_quat.device).unsqueeze(0).repeat(body_quat.shape[0], 1)
    z_axis_w_left = quat_apply(body_quat[:,0], z_axis)
    angle_left = torch.acos(torch.clamp(z_axis_w_left[:, 2], -1.0, 1.0))
    if body_quat.shape[1] > 1:
        z_axis_w_right = quat_apply(body_quat[:,1], z_axis)
        angle_right = torch.acos(torch.clamp(z_axis_w_right[:, 2], -1.0, 1.0))
        reward = angle_left + angle_right
    else :
        reward = angle_left
    
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)
