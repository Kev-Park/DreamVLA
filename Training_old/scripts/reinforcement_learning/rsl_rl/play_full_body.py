# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
# from isaaclab.assets import Articulation, RigidObject

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=700, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# exit(0)

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

# from omni.isaac.core.utils.prims import set_prim_pose
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
# cfg = UrdfConverterCfg(
#     asset_path="/home/azureuser/IsaacLab/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf",     # Absolute path to your URDF file
#     usd_dir="/home/azureuser/IsaacLab/HumanoidVerse/humanoidverse/data/robots/g1",       # Directory to store the generated USD
#     usd_file_name="g1_27dof.usd",                # Optional: name for the USD file
#     force_usd_conversion=True,                # Optional: force re-generation
#     make_instanceable=True,                   # Optional: for memory efficiency
#     fix_base=True                             # Optional: fix the base link
# )

# converter = UrdfConverter(cfg)
# print("USD file generated at:", converter.usd_path)
# exit(0)
# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    # exit(0)
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # policy_left1.pt is in the same location as where this script is, just change the global path
    upper_body_policy_path = "/home/azureuser/IsaacLab/scripts/reinforcement_learning/rsl_rl/policy_left1.pt"

    # policy.pt is in the same location as where this script is, just change the global path
    policy_path = '/home/azureuser/IsaacLab/scripts/reinforcement_learning/rsl_rl/policy.pt' 
    log_dir = os.path.dirname(resume_path)
    # print(log_dir)
    # exit(0)
    # create isaac environment
    # import pdb; pdb.set_trace()  # noqa: E702
    # print(env_cfg)
    # env_cfg.viewer.origin_type = "asset_root"
    # env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (5.,5.,5.)
    print(env_cfg.viewer)
    # exit(0)
    # env_cfg.viewer.body_name = "right_back_foot"
    # env_cfg.viewer.asset_name = "robot"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    print(ISAACLAB_NUCLEUS_DIR)
    # exit(0)
    
    # from pxr import Usd
    # print(f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd")

    # exit(0)
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    
    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    ppo_runner = OnPolicyRunner(env, self.agent_cfg.to_dict(), log_dir=None, device=self.agent_cfg.device, \
                                    obs_space_dim=self.obs_space_dim, action_space_dim=self.action_space_dim)
    ppo_runner.load(self.resume_path)

        # obtain the trained policy for inference
    self.policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    policy = torch.jit.load(policy_path).to(device=env.unwrapped.device, dtype=torch.float32)
    print(policy)
    # exit(0)
    upper_body_policy = torch.jit.load(upper_body_policy_path).to(device=env.unwrapped.device, dtype=torch.float32)
    
    # print(camera_sensor)
    # exit(0)
    # initial_pos, initial_orient = camera_sensor.get_world_poses(convention="ros")
    # print(f"[INFO] Initial camera position: {initial_pos}, orientation: {initial_orient}")
    # exit(0)
    # simulate environment
    print(policy)
    upper_body_default_pos = torch.tensor([0., 0., 0, 0., 0, 0, 0, 0., -0., 0, 0., 0, 0, 0])
    # asset: Articulation = env.scene[asset_cfg.name]
    # joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # exit(0)
    left_arm_ids = [9, 13, 17, 19, 21, 23, 25]
    STOP_AT = 4.
    ARM_POLICY_AT = 5.
    NEW_POSE_AT = 7.
    OLD_POSE_AT = 8.
    BACK_TO_DEFAULT_AT = 9.
    START_MOVING_AT = 10.5
    curr_t = 0.
    target_left_ee_pose = torch.tensor([0.27, 0.15, 0.3, 1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        curr_t += dt
        with torch.inference_mode():
            print(f"[INFO] Current time: {curr_t:.2f} s")
            # agent stepping
            # obs[:,3] = 0.
            # obs[:,6] = 1.
            # print("Obs: ", obs[0,:])
            body_pose = obs[:, 12:12 + 27].clone()
            left_arm_pose = body_pose[:, left_arm_ids]
            body_pose_vel = obs[:, 39:39 + 27].clone()
            left_arm_pose_vel = body_pose_vel[:, left_arm_ids]
            if curr_t > 0.1 :
                left_arm_actions = actions_upper_body.clone()[:, :7]
            actions_upper_body = torch.zeros((obs.shape[0], 14), device=env.unwrapped.device)
            actions_upper_body[:, :] = upper_body_default_pos.unsqueeze(0)
            if curr_t > STOP_AT and curr_t < BACK_TO_DEFAULT_AT:
                env.unwrapped.command_manager.get_command("base_velocity")[:,:] = 0.
            if curr_t > ARM_POLICY_AT and curr_t < BACK_TO_DEFAULT_AT:
                obs_left_arm = torch.cat((target_left_ee_pose.unsqueeze(0).repeat(obs.shape[0], 1), left_arm_pose, left_arm_pose_vel, left_arm_actions), dim=-1)
                actions_left_arm = upper_body_policy(obs_left_arm)
                actions_upper_body[:, :7] = actions_left_arm
            if curr_t > NEW_POSE_AT:
                target_left_ee_pose = torch.tensor([0.37, 0.15, 0.3, 1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
            if curr_t > OLD_POSE_AT:
                target_left_ee_pose = torch.tensor([0.27, 0.15, 0.3, 1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
            if curr_t > BACK_TO_DEFAULT_AT:
                actions_left_arm = left_arm_pose * 2 * 0.9
                actions_upper_body[:, :7] = actions_left_arm
            if curr_t > START_MOVING_AT:
                env.unwrapped.command_manager.get_command("base_velocity")[:, 0] = 0.5
                env.unwrapped.command_manager.get_command("base_velocity")[:, 1] = -0.1
                env.unwrapped.command_manager.get_command("base_velocity")[:, 2] = 0.0
            actions_lower_body = policy(obs[:,:-14])

            actions = torch.cat((actions_lower_body, actions_upper_body), dim=-1)
            # print("Actions: ", actions[0,:])
            # print(obs[0,12:12+27])
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # print(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
