# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a SONIC-encoder checkpoint (trained by train_sonic.py) and record a single video.

The trained policy is the SONIC *encoder*: it outputs 64-D tokens, NOT env joint actions.
So this MUST route through the same ``TokenActionDecoderVecEnvWrapper`` + frozen ONNX decoder
as training — the stock play.py/play_pick_cam.py would feed the 64-D token straight into the
41-D action manager and produce garbage.

Records ONE third-person video (the body view — best for judging stand-vs-track behavior).
Actions are DETERMINISTIC (actor mean via ``get_inference_policy``), which matters a lot here:
training samples a ~N(mean, std) Gaussian for exploration, and with a large learned std the
sampled actions are mostly tanh-saturation noise — only the mean reflects the policy's intent.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import builtins
from functools import partial

from isaaclab.app import AppLauncher
import cv2
import cli_args  # isort: skip

print = partial(builtins.print, flush=True)

parser = argparse.ArgumentParser(description="Play a SONIC-encoder RSL-RL checkpoint and record a video.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (1 for a clean video).")
parser.add_argument("--task", type=str, default="Isaac-Motion-Tracking-Pick-ContFingers-v0", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--name", type=str, default="sonic_play.mp4", help="Output video file name.")
parser.add_argument("--path", type=str, default=None, help="Explicit checkpoint path (overrides auto-discovery).")
parser.add_argument("--object_name", type=str, default=None, help="Override the object USD.")
parser.add_argument(
    "--sonic-decoder-onnx", type=str,
    default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx",
    help="Path to the frozen SONIC decoder ONNX (must match what train_sonic.py used).",
)
# append RSL-RL cli arguments (gives --checkpoint, --load_run, etc.)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# this script always renders → cameras must be on
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import isaaclab.utils.math as math_utils
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# SONIC token-action wrapper (same wiring as train_sonic.py)
from vla_sonic.token_action_wrapper import TokenActionDecoderVecEnvWrapper, load_frozen_decoder


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
        frame = np.power(frame, 1.0 / 2.2)  # linear → sRGB
        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        enable_cameras=True,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # ---- resolve checkpoint (optional weight specification) ----
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif args_cli.path:
        resume_path = args_cli.path
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # ---- camera framing ----
    env_cfg.viewer.eye = (1.0, -2.0, 2.0)
    env_cfg.viewer.lookat = (2.0, 0.0, 0.0)
    if args_cli.object_name is not None and args_cli.object_name != "none":
        env_cfg.scene.object.spawn.usd_path = "assets/" + args_cli.object_name + ".usd"

    # Ensure a third-person camera exists (the ContFingers env only adds it when enable_cameras).
    if not isinstance(getattr(env_cfg.scene, "camera", None), CameraCfg):
        _rot = np.array([0.7538, 0.61221, -0.1505, -0.1853])
        _rot_mat = np.array(math_utils.matrix_from_quat(torch.tensor(_rot)))
        _theta = -np.pi * 0.75
        _rot_z = np.array([
            [np.cos(_theta), -np.sin(_theta), 0.0],
            [np.sin(_theta),  np.cos(_theta), 0.0],
            [0.0,             0.0,             1.0],
        ])
        _rot_quat = tuple(math_utils.quat_from_matrix(torch.tensor(_rot_z @ _rot_mat)).tolist())
        env_cfg.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera_new",
            spawn=PinholeCameraCfg(
                focal_length=18.1476, focus_distance=400.0, horizontal_aperture=20.955,
                clipping_range=(0.01, 100.0),
            ),
            data_types=["rgb"], height=1920, width=2560,
            offset=CameraCfg.OffsetCfg(
                pos=(-1.03 + 2.1 - 0.034, 4.05 - 0.9, 1.31), rot=_rot_quat, convention="opengl",
            ),
        )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    print("Observation space:", env.observation_space)
    print("Action space (env, pre-wrapper):", env.action_space)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Pump the Omniverse event loop BEFORE any reset so Isaac's PhysX/USD/RTX shader
    # compile (triggered by the first env.reset() inside the wrapper's super().__init__)
    # can actually progress. With enable_cameras=True this otherwise hangs the script
    # at the first reset, GPU util at 0%. Same trick eval_parquet_sonic.py uses.
    print("[play_sonic] pumping Omniverse event loop to let render/physics initialise...")
    for _ in range(60):
        simulation_app.update()

    # ---- wrap so the 64-D token policy drives the frozen decoder (same as training) ----
    device = agent_cfg.device
    print(f"[play_sonic] loading frozen SONIC decoder ONNX: {args_cli.sonic_decoder_onnx}")
    decoder = load_frozen_decoder(args_cli.sonic_decoder_onnx, device)
    env = TokenActionDecoderVecEnvWrapper(env, decoder, device, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    ppo_runner.load(resume_path)

    # DETERMINISTIC inference policy (actor mean, no Gaussian sampling) — see module docstring.
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    scene_keys = list(env.unwrapped.scene.keys())
    if "camera" not in scene_keys:
        raise RuntimeError(f"Third-person camera missing from scene. Found: {scene_keys}")
    cam_tp = env.unwrapped.scene["camera"]

    video_folder = os.path.join(log_dir, "videos", "play")
    os.makedirs(video_folder, exist_ok=True)
    video_path = os.path.join(video_folder, args_cli.name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_fps = max(1, int(round(1.0 / env.unwrapped.step_dt)))
    writer = None
    print(f"[INFO] Writing video to: {video_path} @ {video_fps} FPS")

    # Warm-up env.step with a zero token so the camera buffer fills before the first
    # read inside the loop. Without this, cam_tp.data.output["rgb"][0] returns an
    # empty / not-yet-populated tensor and the script stalls. Same trick eval_parquet_sonic
    # and eval_sonic_control use (warm-up step + two _APP.update() flushes).
    print("[play_sonic] warm-up env.step to populate camera buffers...")
    zero_token = torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float32)
    env.step(zero_token)
    simulation_app.update()
    simulation_app.update()

    obs_out = env.get_observations()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
    dt = env.unwrapped.step_dt
    timestep = 0

    while simulation_app.is_running():
        start_time = time.time()

        frame_bgr = _raw_to_bgr(cam_tp.data.output["rgb"][0])
        if writer is None:
            h, w = frame_bgr.shape[:2]
            writer = cv2.VideoWriter(video_path, fourcc, video_fps, (w, h))
            print(f"[INFO] Initialized video writer at {w}x{h}")
        # annotate object height so "did it lift?" is visible
        try:
            object_z = float(env.unwrapped.scene["object"].data.root_pos_w[0, 2].item())
            label = f"step {timestep}  object z: {object_z:.3f} m"
        except Exception:
            label = f"step {timestep}"
        for color, thick in ((( 0, 0, 0), 4), ((255, 255, 255), 2)):
            cv2.putText(frame_bgr, label, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, thick, cv2.LINE_AA)
        writer.write(frame_bgr)

        with torch.inference_mode():
            actions = policy(obs).clone()
            obs, _, dones, _ = env.step(actions)

        timestep += 1
        if timestep >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if writer is not None:
        print(f"[INFO] Releasing video writer ({timestep} frames)")
        writer.release()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
