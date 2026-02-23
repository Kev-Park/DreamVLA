# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.assets import Articulation, RigidObject

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--baseline", type=str, default="a", help="Name of the baseline.")
parser.add_argument("--id", type=str, default="0", help="Name of the id.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--path", type=str, default=None, help="Path to the task.")
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
        print(resume_path)
        import pdb; pdb.set_trace()  # noqa: E702
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        # import pdb; pdb.set_trace()  # noqa: E702
    else:
        # print(log_root_path)
        if args_cli.path:
            # if path is provided, use it to get the checkpoint
            resume_path = args_cli.path
        else :
            task_names_ = args_cli.task.split("-")[3:-1]
            task_name = ""
            for task_name_ in task_names_:
                task_name += task_name_ + "-"
            task_name = task_name[:-1]
            if args_cli.baseline is not None:
                task_name += "-" + args_cli.baseline
            # import pdb; pdb.set_trace()  # noqa: E702
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint, task_name=task_name)
        
    log_dir = os.path.dirname(resume_path)
    env_cfg.viewer.eye = (2.5,-5.,5.)
    env_cfg.viewer.lookat = (2., 0., 0.)
    env_cfg.observations.policy.enable_corruption = False
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    print(ISAACLAB_NUCLEUS_DIR)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": "eval_"+args_cli.baseline+"_"+args_cli.id,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt
    print("dt:", dt)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # print("Here???")
    joint_angless = []
    root_poss = []
    root_quatss = []
    # print("Whats happening???")
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs).clone()
            joint_angles = env.unwrapped.scene["robot"].data.joint_pos.clone()
            # print(joint_angles[0])
            joint_angless.append(joint_angles)
            root_poss.append(env.unwrapped.scene["robot"].data.root_pos_w.clone() - env.unwrapped.scene.env_origins.clone())
            root_quatss.append(env.unwrapped.scene["robot"].data.root_quat_w.clone())
            obs, _, dones, _ = env.step(actions)
            
        if args_cli.video:
            timestep += 1
            print(timestep)
            
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    joint_angless = torch.cat(joint_angless, dim=0)
    root_poss = torch.cat(root_poss, dim=0)
    root_quatss = torch.cat(root_quatss, dim=0)
    # Save as pkl
    import pickle
    os.makedirs(log_dir + "/eval", exist_ok=True)
    with open(os.path.join(log_dir, "eval", "motions.pkl"), "wb") as f:
        pickle.dump({"joint_angles": joint_angless, "root_pos": root_poss, "root_quats": root_quatss}, f)
    print("Success rate: ", 100*(torch.sum(env.unwrapped.n_successes > 0)/int(args_cli.num_envs)).detach().cpu().item(), "%")

    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
