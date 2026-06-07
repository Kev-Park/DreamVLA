# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a SONIC-encoder checkpoint (trained by train_sonic.py) and record a single video.

The trained policy is the SONIC *encoder*: it emits 64-D tokens, NOT env joint actions.
So this MUST route through the same ``TokenActionDecoderVecEnvWrapper`` + frozen ONNX
decoder as training — the stock play.py / play_pick_cam.py would feed the 64-D token
straight into the 41-D action manager and produce garbage.

Camera, video, and per-step RTX-flush patterns mirror ``eval_parquet_sonic.py`` (the same
script that frames the kitchen scene correctly and produces fresh per-step frames):
  - ``_inject_cameras`` unconditionally writes the third-person + robot-ego camera cfgs
    with the kitchen-framing pose (so we don't depend on env-cfg ordering).
  - imageio (libx264) VideoWriter — eval's pattern.
  - ``_APP.update(); _APP.update()`` AFTER each ``env.step`` to flush the RTX render
    pipeline into the camera annotators. Without this each frame is the warm-up render.

Actions are DETERMINISTIC (actor mean via ``get_inference_policy``) — training samples a
~N(mean, std) Gaussian for exploration; sampled actions at large std are mostly
tanh-saturation noise. Only the mean reflects the policy's intent.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import builtins
from functools import partial
from pathlib import Path

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

print = partial(builtins.print, flush=True)

parser = argparse.ArgumentParser(description="Play a SONIC-encoder RSL-RL checkpoint and record a video.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (1 for a clean video).")
# Default to the HQ-kitchen + mustard-bottle Cam variant (same task eval_sonic_control and
# eval_vla_sonic use). The policy was trained on Isaac-Motion-Tracking-Pick-ContFingers-v0
# (green-box proxy scene); same env structure, same obs/action layout, only the visual + USD
# object geometry differ. Kitchen pos shifts ~(2.55,0)→(2.04,1.0), so grasp targeting may
# look slightly off relative to training — fine for visualization, not for benchmarking.
parser.add_argument("--task", type=str, default="Isaac-Motion-Tracking-Pick-Cam-ContFingers-v0", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--name", type=str, default="sonic_play.mp4", help="Output video file name.")
parser.add_argument("--path", type=str, default=None, help="Explicit checkpoint path (overrides auto-discovery).")
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
_APP = simulation_app  # alias to match eval_parquet_sonic.py naming

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
from vla_sonic.physics_overrides import apply_sonic_physics_overrides
from vla_sonic.split_head_actor_critic import SplitHeadActorCritic

# Register the custom ActorCritic with rsl_rl so its `class_name` lookup finds it when
# loading the checkpoint. rsl_rl 3.0.1's OnPolicyRunner uses `eval(class_name)` inside
# _construct_algorithm (on_policy_runner.py:418); eval resolves names in the globals of
# the module where it's called, so we have to inject the class into
# rsl_rl.runners.on_policy_runner's namespace. (Also adding to rsl_rl.modules for
# completeness.) Without this, the checkpoint's split-head actor (actor.trunk.*,
# actor.body_head.*, actor.finger_head.*) cannot be loaded into a vanilla ActorCritic
# whose state_dict expects actor.0.weight etc.
import rsl_rl.modules
import rsl_rl.runners.on_policy_runner as _rsl_rl_opr
rsl_rl.modules.SplitHeadActorCritic = SplitHeadActorCritic
_rsl_rl_opr.SplitHeadActorCritic = SplitHeadActorCritic


# =========================================================================
# Camera + video helpers — copied 1:1 from eval_parquet_sonic.py so the scene
# framing and per-step flush behavior are identical to the eval script you've
# already validated frames the kitchen correctly.
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
    """cv2-backed mp4 writer (mp4v codec — self-contained, no ffmpeg dependency).

    eval_parquet_sonic.py uses imageio+libx264, but that path requires imageio-ffmpeg to
    be installed and matched. cv2's bundled libavcodec writes a universally-playable
    mp4 with no extra deps. Takes RGB in, converts to BGR for cv2 internally.
    """

    def __init__(self, path: Path, fps: int):
        import cv2
        self._cv2 = cv2
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fps = max(1, int(fps))
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = None  # lazy-init on first write so we know the frame size
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
    """Draw a label in the top-left. cv2.putText works on RGB arrays unchanged."""
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

    # Match train_sonic.py: this checkpoint was trained with the split-head actor, so the
    # runner must instantiate SplitHeadActorCritic (not the default ActorCritic) before
    # load_state_dict, or the parameter names won't line up.
    agent_cfg.policy.class_name = "SplitHeadActorCritic"
    print(f"[play_sonic] policy.class_name = {agent_cfg.policy.class_name}")

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

    # Match the SONIC decoder's training-time physics — same as train_sonic.py.
    apply_sonic_physics_overrides(env_cfg)

    # eval-style camera + viewer setup
    env_cfg.viewer.eye = (1.0, -2.0, 2.0)
    env_cfg.viewer.lookat = (2.0, 0.0, 0.0)
    _inject_cameras(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    print(f"[env] action_space (pre-wrapper) = {env.action_space}")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Pump the Omniverse event loop BEFORE the wrapper's super().__init__ does env.reset()
    # — needed under enable_cameras so RTX/PhysX/USD shader compile can finish.
    print("[play_sonic] pumping Omniverse event loop to let render/physics initialise...")
    for _ in range(60):
        _APP.update()

    # ---- wrap so the 64-D token policy drives the frozen decoder (same as training) ----
    device = agent_cfg.device
    print(f"[play_sonic] loading frozen SONIC decoder ONNX: {args_cli.sonic_decoder_onnx}")
    decoder = load_frozen_decoder(args_cli.sonic_decoder_onnx, device)
    env = TokenActionDecoderVecEnvWrapper(env, decoder, device, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    ppo_runner.load(resume_path)

    # DETERMINISTIC inference policy (actor mean, no Gaussian sampling).
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Warm-up env.step with a zero token so the camera annotators populate
    # before the first read. eval_parquet_sonic / eval_sonic_control do the same.
    print("[play_sonic] warm-up env.step + flush to populate camera buffers...")
    zero_token = torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float32)
    env.step(zero_token)
    _APP.update()
    _APP.update()

    # Open the writer AFTER the warm-up so we know the camera is ready.
    video_folder = Path(log_dir) / "videos" / "play"
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

        # 1. policy → token → frozen-decoder → env.step (deterministic)
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
            # quick diagnostic: confirm frames are actually changing (no static-frame regression)
            if timestep < 5 and _prev_frame is not None:
                diff = int(np.abs(frame.astype(np.int32) - _prev_frame.astype(np.int32)).max())
                print(f"[step {timestep}] camera max_pixel_diff_from_prev={diff}")
            _prev_frame = frame.copy()

            try:
                object_z = float(env.unwrapped.scene["object"].data.root_pos_w[0, 2].item())
                label = f"step {timestep}  object z: {object_z:.3f} m"
            except Exception:
                label = f"step {timestep}"
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
