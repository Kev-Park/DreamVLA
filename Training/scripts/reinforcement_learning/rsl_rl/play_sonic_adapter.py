# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a SONIC residual-ADAPTER checkpoint (trained by train_sonic_adapter.py) and record video.

Sister script to ``play_sonic.py`` for the frozen-encoder + residual-adapter pipeline:

    frozen G1 encoder → base token ─┬─▶ obs ─▶ adapter policy ─▶ residual ─┐
                                    └──────────────────────────── (add) ◀──┘
                                                  │
                                            FSQ → frozen decoder → env body action

MUST mirror train_sonic_adapter.py's wiring exactly:
  - ``TokenAdapterVecEnvWrapper`` (encoder runs per step; base token appended to obs)
  - ``AdapterActorCritic`` with actor [256, 128] (zero-init output layer at train time)
  - experiment dir ``<exp>_sonic_adapter`` (checkpoint auto-discovery)
  - clip_actions=None and the SAME --residual-scale used in training (the residual
    bound is part of the policy's effective action semantics).

A useful property of this script: action = 0 ⇒ pure zero-shot frozen-SONIC playback.
Pass ``--zero-residual`` to force that for the whole video — this is the zero-shot
baseline measurement (how good is frozen SONIC on our motions, no learning at all)
and needs NO checkpoint.

Camera, video, and per-step RTX-flush patterns are copied 1:1 from play_sonic.py.
Actions are DETERMINISTIC (actor mean via ``get_inference_policy``).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import builtins
from functools import partial
from pathlib import Path

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

print = partial(builtins.print, flush=True)

parser = argparse.ArgumentParser(description="Play a SONIC residual-adapter RSL-RL checkpoint and record a video.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (1 for a clean video).")
parser.add_argument("--task", type=str, default="Isaac-Motion-Tracking-Pick-BinaryFingers-v0", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--name", type=str, default="sonic_adapter_play.mp4", help="Output video file name.")
parser.add_argument("--path", type=str, default=None, help="Explicit checkpoint path (overrides auto-discovery).")
parser.add_argument(
    "--sonic-decoder-onnx", type=str,
    default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx",
    help="Path to the frozen SONIC decoder ONNX (must match training).",
)
parser.add_argument(
    "--sonic-encoder-onnx", type=str,
    default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx",
    help="Path to the frozen SONIC encoder ONNX (must match training).",
)
parser.add_argument(
    "--residual-scale", type=float, default=0.3,
    help="Residual bound — MUST match the value used by train_sonic_adapter.py.",
)
parser.add_argument(
    "--zero-residual", action="store_true", default=False,
    help="Ignore the policy and play with residual=0 (+finger open): pure zero-shot "
         "frozen-SONIC playback of the reference motion. No checkpoint required.",
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
_APP = simulation_app

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

# SONIC adapter wiring (same as train_sonic_adapter.py)
from vla_sonic.token_action_wrapper import load_frozen_decoder
from vla_sonic.token_adapter_wrapper import TokenAdapterVecEnvWrapper, load_frozen_encoder
from vla_sonic.physics_overrides import apply_sonic_physics_overrides
from vla_sonic.adapter_actor_critic import AdapterActorCritic

# Register the custom ActorCritic with rsl_rl (eval(class_name) resolves in the
# on_policy_runner module's globals — same pattern as the training script).
import rsl_rl.modules
import rsl_rl.runners.on_policy_runner as _rsl_rl_opr
rsl_rl.modules.AdapterActorCritic = AdapterActorCritic
_rsl_rl_opr.AdapterActorCritic = AdapterActorCritic


# =========================================================================
# Camera + video helpers — copied 1:1 from play_sonic.py.
# =========================================================================

def _inject_cameras(env_cfg) -> None:
    """Unconditionally set the third-person + robot-ego camera cfgs with the kitchen pose."""
    rot = np.array([0.7538, 0.61221, -0.1505, -0.1853])
    rot_mat = np.array(math_utils.matrix_from_quat(torch.tensor(rot)))
    theta = -np.pi * 0.75
    rot_z = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    rot_mat = rot_z @ rot_mat
    rot_quat = tuple(math_utils.quat_from_matrix(torch.tensor(rot_mat)).tolist())
    env_cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_new",
        spawn=PinholeCameraCfg(
            focal_length=18.1476, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.1, 10000.0),
        ),
        data_types=["rgb"], height=1920, width=2560,
        offset=CameraCfg.OffsetCfg(
            pos=(-1.03 + 2.1 - 0.034, 4.05 - 0.9, 1.31),
            rot=rot_quat, convention="opengl",
        ),
    )
    env_cfg.scene.camera_robot = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/Camera_robot",
        spawn=PinholeCameraCfg(
            focal_length=7.6, focus_distance=400.0,
            horizontal_aperture=20.0, clipping_range=(0.01, 100.0),
        ),
        data_types=["rgb"], height=480, width=640,
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.36),
            rot=(0.568, 0.421, -0.421, -0.568), convention="opengl",
        ),
    )


def _read_camera_rgb(env, key: str):
    try:
        cam = env.unwrapped.scene[key]
    except KeyError:
        return None
    try:
        output = cam.data.output
    except Exception:
        return None
    if "rgb" not in output:
        return None
    rgb = output["rgb"][0, ..., :3]
    if rgb.dtype != torch.uint8:
        rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
    return rgb.cpu().numpy()


class VideoWriter:
    """cv2-backed mp4 writer (mp4v codec — self-contained, no ffmpeg dependency)."""

    def __init__(self, path: Path, fps: int):
        import cv2
        self._cv2 = cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fps = max(1, int(fps))
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = None
        self.frame_count = 0

    def write(self, frame_rgb):
        frame_bgr = self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2BGR)
        if self._writer is None:
            h, w = frame_bgr.shape[:2]
            self._writer = self._cv2.VideoWriter(str(self.path), self._fourcc, self._fps, (w, h))
            print(f"[video] writer opened {w}x{h} @ {self._fps} fps → {self.path}")
        self._writer.write(frame_bgr)
        self.frame_count += 1

    def close(self):
        if self._writer is not None:
            self._writer.release()
        print(f"[video] {self.path.name}: {self.frame_count} frames written")


def _overlay(frame_rgb, label: str):
    import cv2
    for color, thick in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(frame_rgb, label, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, thick, cv2.LINE_AA)
    return frame_rgb


# =========================================================================
# Main.
# =========================================================================

def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        enable_cameras=True,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # ---- mirror train_sonic_adapter.py's agent overrides EXACTLY ----
    agent_cfg.experiment_name = f"{agent_cfg.experiment_name}_sonic_adapter"
    agent_cfg.policy.class_name = "AdapterActorCritic"
    agent_cfg.policy.actor_hidden_dims = [256, 128]
    agent_cfg.policy.critic_hidden_dims = [512, 256, 256]
    print(f"[play_sonic_adapter] policy = AdapterActorCritic, actor={agent_cfg.policy.actor_hidden_dims}")

    # ---- resolve checkpoint (skipped entirely in --zero-residual mode) ----
    resume_path = None
    log_dir = None
    if not args_cli.zero_residual:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        elif args_cli.path:
            resume_path = args_cli.path
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)

    # Match the SONIC decoder's training-time physics — same as train_sonic_adapter.py.
    apply_sonic_physics_overrides(env_cfg)

    # eval-style camera + viewer setup
    env_cfg.viewer.eye = (1.0, -2.0, 2.0)
    env_cfg.viewer.lookat = (2.0, 0.0, 0.0)
    _inject_cameras(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    print(f"[env] action_space (pre-wrapper) = {env.action_space}")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Pump the Omniverse event loop BEFORE the wrapper's super().__init__ does env.reset().
    print("[play_sonic_adapter] pumping Omniverse event loop to let render/physics initialise...")
    for _ in range(60):
        _APP.update()

    # ---- frozen encoder + decoder + adapter wrapper (same wiring as training) ----
    device = agent_cfg.device
    print(f"[play_sonic_adapter] loading frozen SONIC decoder ONNX: {args_cli.sonic_decoder_onnx}")
    decoder = load_frozen_decoder(args_cli.sonic_decoder_onnx, device)
    print(f"[play_sonic_adapter] loading frozen SONIC encoder ONNX: {args_cli.sonic_encoder_onnx}")
    encoder = load_frozen_encoder(args_cli.sonic_encoder_onnx, device)
    env = TokenAdapterVecEnvWrapper(
        env, decoder, encoder, device,
        residual_scale=args_cli.residual_scale,
        clip_actions=None,
    )

    # ---- policy: trained adapter, or the zero-residual baseline ----
    if args_cli.zero_residual:
        print("[play_sonic_adapter] ZERO-RESIDUAL mode: pure frozen-SONIC zero-shot playback "
              "(no checkpoint loaded; action = 0 → residual = 0, fingers open)")

        def policy(obs):
            return torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float32)
    else:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        ppo_runner.load(resume_path)
        # DETERMINISTIC inference policy (actor mean, no Gaussian sampling).
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Warm-up env.step with a zero action (= pure SONIC base behavior) so the camera
    # annotators populate before the first read.
    print("[play_sonic_adapter] warm-up env.step + flush to populate camera buffers...")
    zero_action = torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float32)
    env.step(zero_action)
    _APP.update()
    _APP.update()

    # Open the writer AFTER the warm-up so we know the camera is ready.
    if log_dir is not None:
        video_folder = Path(log_dir) / "videos" / "play"
    else:
        video_folder = Path("logs") / "rsl_rl" / agent_cfg.experiment_name / "zero_shot" / "videos"
    video_path = video_folder / args_cli.name
    video_fps = max(1, int(round(1.0 / env.unwrapped.step_dt)))
    writer = VideoWriter(video_path, fps=video_fps)
    print(f"[INFO] Writing video to: {video_path} @ {video_fps} FPS")

    obs_out = env.get_observations()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
    dt = env.unwrapped.step_dt
    timestep = 0
    _prev_frame = None

    while simulation_app.is_running() and timestep < args_cli.video_length:
        start_time = time.time()

        # 1. policy → residual → base+residual → frozen decoder → env.step (deterministic)
        with torch.inference_mode():
            actions = policy(obs).clone()
            obs, _, dones, _ = env.step(actions)

        # 2. flush RTX render pipeline so the camera annotator delivers THIS step's frame
        _APP.update()
        _APP.update()

        # 3. read + write the frame
        frame = _read_camera_rgb(env, "camera")
        if frame is None:
            print(f"[WARN] step {timestep}: third-person camera 'camera' returned no frame")
        else:
            if timestep < 5 and _prev_frame is not None:
                diff = int(np.abs(frame.astype(np.int32) - _prev_frame.astype(np.int32)).max())
                print(f"[step {timestep}] camera max_pixel_diff_from_prev={diff}")
            _prev_frame = frame.copy()

            try:
                object_z = float(env.unwrapped.scene["object"].data.root_pos_w[0, 2].item())
                label = f"step {timestep}  object z: {object_z:.3f} m"
            except Exception:
                label = f"step {timestep}"
            if args_cli.zero_residual:
                label = "ZERO-SHOT  " + label
            writer.write(_overlay(frame, label))

        timestep += 1

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
