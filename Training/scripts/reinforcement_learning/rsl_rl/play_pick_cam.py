# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import builtins
from functools import partial

from isaaclab.app import AppLauncher
import cv2
import cli_args  # isort: skip

print = partial(builtins.print, flush=True)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
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
parser.add_argument("--name", type=str, default="test.mp4", help="Name of the video file.")
parser.add_argument("--object_name", type=str, default=None, help="Name of the object.")
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
import numpy as np
import os
import time
import torch

# from omni.isaac.core.utils.prims import set_prim_pose
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import isaaclab.utils.math as math_utils
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
OUTPUT_VIDEO = "pick"

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    # exit(0)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        enable_cameras=bool(args_cli.enable_cameras),
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        print(resume_path)
        #import pdb; pdb.set_trace()  # noqa: E702
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        if args_cli.path:
            # if path is provided, use it to get the checkpoint
            resume_path = args_cli.path
        else :
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    log_dir = os.path.dirname(resume_path)
    env_cfg.viewer.eye = (1.,-2.,2.)
    env_cfg.viewer.lookat = (2., 0., 0.)
    if args_cli.object_name is not None and args_cli.object_name != "none":
        env_cfg.scene.object.spawn.usd_path = "assets/"+args_cli.object_name+".usd"
    # Ensure robot camera exists whenever camera mode is enabled, even for non-cam task variants.
    if bool(args_cli.enable_cameras) and not hasattr(env_cfg.scene, "camera_robot"):
        env_cfg.scene.camera_robot = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/Camera_robot",
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
                pos=(0.05, 0.0, 0.36),
                rot=(0.568, 0.421, -0.421, -0.568),
                convention="opengl",
            ),
        )

    # Ensure third-person camera exists whenever camera mode is enabled.
    _existing_cam = getattr(env_cfg.scene, "camera", None)
    print(f"[DEBUG] env_cfg.scene.camera before injection: {_existing_cam!r}")
    if bool(args_cli.enable_cameras) and not isinstance(_existing_cam, CameraCfg):
        import numpy as _np
        _rot = _np.array([0.7538, 0.61221, -0.1505, -0.1853])
        _rot_mat = _np.array(math_utils.matrix_from_quat(torch.tensor(_rot)))
        _theta = -_np.pi * 0.75
        _rot_z = _np.array([
            [_np.cos(_theta), -_np.sin(_theta), 0.0],
            [_np.sin(_theta),  _np.cos(_theta), 0.0],
            [0.0,              0.0,              1.0],
        ])
        _rot_quat = tuple(math_utils.quat_from_matrix(torch.tensor(_rot_z @ _rot_mat)).tolist())
        env_cfg.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera_new",
            spawn=PinholeCameraCfg(
                focal_length=18.1476,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            data_types=["rgb"],
            height=1920,
            width=2560,
            offset=CameraCfg.OffsetCfg(
                pos=(-1.03 + 2.1 - 0.034, 4.05 - 0.9, 1.31),
                rot=_rot_quat,
                convention="opengl",
            ),
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    print(ISAACLAB_NUCLEUS_DIR)
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # camera = Camera(
    #     CameraCfg(prim_path="/world/Camera")
    # )
    # camera.initialize()
    # We intentionally do not use gym.wrappers.RecordVideo here because it records
    # the env render (third-person), while this script writes robot-camera video directly.

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

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt
    print("dt:", dt)

    # reset environment
    obs_out = env.get_observations()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
    timestep = 0

    scene_keys = list(env.unwrapped.scene.keys())
    if "camera_robot" not in scene_keys:
        raise RuntimeError(
            "Required robot camera is missing. "
            f"Found: {scene_keys}. "
            "Use a task config that defines camera_robot "
            "(e.g. Isaac-Motion-Tracking-Pick-v0)."
        )

    cam_robot = env.unwrapped.scene["camera_robot"]
    cam_tp = env.unwrapped.scene["camera"] if "camera" in scene_keys else None
    if cam_tp is not None:
        print("[INFO] Third-person camera found — will record a second video.")
    else:
        print("[INFO] No third-person camera in scene — skipping third-person video.")

    video_folder = os.path.join(log_dir, "videos", "play")
    os.makedirs(video_folder, exist_ok=True)
    robot_video_path = os.path.join(video_folder, args_cli.name)
    tp_video_path = os.path.join(video_folder, os.path.splitext(args_cli.name)[0] + "_tp.mp4")
    print(f"[INFO] Writing robot camera video to: {robot_video_path}")
    if cam_tp is not None:
        print(f"[INFO] Writing third-person camera video to: {tp_video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer_robot = None
    video_writer_tp = None
    video_fps = max(1, int(round(1.0 / env.unwrapped.step_dt)))

    def _raw_to_bgr(raw_tensor):
        frame = raw_tensor.cpu().numpy()
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = np.nan_to_num(frame, nan=0.0, posinf=1.0, neginf=0.0)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.float32)
            if np.max(frame) > 1.5:
                frame = frame / 255.0
            frame = np.clip(frame, 0.0, 1.0)
            frame = np.power(frame, 1.0 / 2.2)
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    printed_frame_stats = False

    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode

        # import pdb; pdb.set_trace()  # noqa: E702
        image_robot = cam_robot.data.output["rgb"] # robot face camera color data
        # Read object location directly from Isaac scene state (world -> robot local frame).
        object_pos_world = env.unwrapped.scene["object"].data.root_pos_w
        robot_pos_world = env.unwrapped.scene["robot"].data.root_pos_w
        robot_quat_world = math_utils.quat_unique(env.unwrapped.scene["robot"].data.root_quat_w)
        object_pos_local = math_utils.quat_apply(
            math_utils.quat_conjugate(robot_quat_world), object_pos_world - robot_pos_world
        )
        x_new = object_pos_local[0, 0].item()
        y_new = object_pos_local[0, 1].item()


        for i in range(1):
            frame_rgb = image_robot[i].cpu().numpy()
            if not printed_frame_stats:
                print(
                    f"[INFO] camera_robot frame stats: shape={frame_rgb.shape}, "
                    f"dtype={frame_rgb.dtype}, min={float(np.min(frame_rgb)):.4f}, "
                    f"max={float(np.max(frame_rgb)):.4f}"
                )
                printed_frame_stats = True

            # Drop alpha if present.
            if frame_rgb.ndim == 3 and frame_rgb.shape[2] == 4:
                frame_rgb = frame_rgb[:, :, :3]

            # Sanitize non-finite values first.
            frame_rgb = np.nan_to_num(frame_rgb, nan=0.0, posinf=1.0, neginf=0.0)

            if frame_rgb.dtype != np.uint8:
                frame_rgb = frame_rgb.astype(np.float32)
                # Normalize to [0, 1] regardless of whether source is [0,1] or [0,255].
                if np.max(frame_rgb) > 1.5:
                    frame_rgb = frame_rgb / 255.0
                frame_rgb = np.clip(frame_rgb, 0.0, 1.0)
                # Convert linear-like camera output to display-friendly sRGB.
                frame_rgb = np.power(frame_rgb, 1.0 / 2.2)
                frame_rgb = (frame_rgb * 255.0).clip(0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if video_writer_robot is None:
                frame_h, frame_w = frame_bgr.shape[:2]
                video_writer_robot = cv2.VideoWriter(robot_video_path, fourcc, video_fps, (frame_w, frame_h))
                print(f"[INFO] Initialized robot video writer at {frame_w}x{frame_h} @ {video_fps} FPS")

            # Annotate frame with object height
            try:
                object_z = float(env.unwrapped.scene["object"].data.root_pos_w[0, 2].item())
                label = f"Object z: {object_z:.3f} m"
            except Exception:
                label = "Object z: n/a"
            cv2.putText(
                frame_bgr,
                label,
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                label,
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            video_writer_robot.write(frame_bgr)

        # Third-person camera recording
        if cam_tp is not None:
            frame_bgr_tp = _raw_to_bgr(cam_tp.data.output["rgb"][0])
            if video_writer_tp is None:
                h, w = frame_bgr_tp.shape[:2]
                video_writer_tp = cv2.VideoWriter(tp_video_path, fourcc, video_fps, (w, h))
                print(f"[INFO] Initialized third-person video writer at {w}x{h} @ {video_fps} FPS")
            video_writer_tp.write(frame_bgr_tp)

        # image = camera.get_image("rgb")
        with torch.inference_mode():
            obs[:,52] = x_new
            obs[:,53] = y_new
            actions = policy(obs).clone()
            obs, _, dones, _ = env.step(actions)
        if timestep == args_cli.video_length:
            break
        timestep += 1
        print(timestep)


        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # release video writers
    if video_writer_robot is not None:
        print(f"[INFO] Releasing video writer: {video_writer_robot}")
        video_writer_robot.release()
    if video_writer_tp is not None:
        print(f"[INFO] Releasing third-person video writer: {video_writer_tp}")
        video_writer_tp.release()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
