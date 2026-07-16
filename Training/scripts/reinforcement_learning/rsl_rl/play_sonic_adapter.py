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
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for deterministic motion selection. Same seed → same "
                         "drawn motion across runs (needed for fair skip-start comparisons).")
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
parser.add_argument(
    "--start-pregrab-margin", type=float, default=None,
    help="Seconds before each motion's grab frame to start episodes at, skipping the "
         "walk approach. E.g. 0.5 starts the robot ~0.5 s before grab, already at the "
         "counter. Default None = start at motion frame 0. Drops BOTH the prepend slide "
         "and the walk — use to rule out locomotion overall.",
)
parser.add_argument(
    "--skip-start-frames", type=int, default=None,
    help="Skip the refinement's PAUSE+INTERP prepend (20 frames) — the ~1 s constant-rate "
         "'interpolate-to-initial-pose' slide glued to the front of every refined motion. "
         "Pass 20 to start at the true motion beginning while KEEPING the walk. Targeted "
         "fix for the unnatural linear slide; mutually exclusive with --start-pregrab-margin.",
)
parser.add_argument(
    "--waist-dof", type=int, default=27, choices=[27, 29],
    help="Body DOF. 29 actuates waist_roll/pitch (29-DOF + dex-hands USD) to match SONIC's "
         "training articulation. 27 = legacy welded-waist asset.",
)
parser.add_argument(
    "--reference-playback", action="store_true", default=False,
    help="KINEMATIC reference baseline: ignore the encoder/decoder/policy entirely and "
         "teleport the robot to the motion_lib reference pose every frame (no physics "
         "influence on the displayed pose). Shows the ideal target motion in the SAME "
         "scene/camera as policy videos, so it's directly comparable. No checkpoint "
         "required. Use this to confirm the reference motions themselves are clean.",
)
parser.add_argument(
    "--reference-pd", action="store_true", default=False,
    help="PHYSICS reference baseline: drive the robot's native JointPositionAction PD "
         "targets with the motion_lib reference each step (target = reference joint "
         "angles) and let Isaac's PD actuators track it under gravity/contact. NO SONIC "
         "encoder/decoder, NO kinematic teleport — this is holosoma-instead-of-SONIC in "
         "the physics sim. Right fingers close during the grab-hold window. No checkpoint "
         "required. Use to test whether the reference itself is dynamically feasible.",
)
parser.add_argument(
    "--keep-terms", action="store_true", default=False,
    help="RENDER-ONLY escape hatch: keep the env's terminations. By DEFAULT this playback script "
         "disables ALL terminations so a rollout is never reset mid-clip (tilt/height/contact/"
         "time_out). Training (train_sonic_adapter.py) and eval (eval_sonic_adapter.py) are "
         "unaffected — they use their own cfgs and always keep terminations.")
parser.add_argument(
    "--overlay-ref", action="store_true", default=False,
    help="Draw the tracked REFERENCE motion as an overlay on top of the (physics) residual "
         "playback: 39 spheres at the reference link world positions, updated every frame "
         "from motion_lib. Right-arm links (idx>=31: right_shoulder..right_wrist_yaw + "
         "right_rubber_hand) are RED, the rest GREEN, so right-arm tracking error is legible. "
         "Where the policy tracks well a sphere sits on the robot's link; a sphere floating "
         "off a link is tracking error. Render-only viz — does not touch the policy/obs/physics.",
)
parser.add_argument(
    "--overlay-obj-candidates", action="store_true", default=False,
    help="For the #4 object-reference diagnostic: draw two object-reference candidate markers per "
         "frame — (i) YELLOW = holosoma object_poses trajectory (needs a .pkl with `object_poses`), "
         "(ii) CYAN = right-hand FK (right_rubber_hand). Judge which sits on the hand through the "
         "grasp. Render-only viz; pairs well with --overlay-ref.",
)
parser.add_argument(
    "--ref-motions-path", type=str, default=None,
    help="Override the env's ref_motions_path (dir of reference .pkl files). Use to point "
         "reference-playback at an isolated set (e.g. the holosoma 29-DOF .pkl) without "
         "disturbing the task's default reference dir.",
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
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# SONIC adapter wiring (same as train_sonic_adapter.py)
from vla_sonic.token_action_wrapper import load_frozen_decoder
from vla_sonic.token_adapter_wrapper import TokenAdapterVecEnvWrapper, load_frozen_encoder
from vla_sonic.physics_overrides import apply_sonic_physics_overrides
from vla_sonic.robot_29dof import apply_29dof_waist_override
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


# =========================================================================
# Reference-motion overlay markers (--overlay-ref).
# 39 spheres at the reference link world positions (motion_lib.global_keypts),
# updated every frame. Right-arm links (idx>=31) are RED, the rest GREEN, so the
# right-arm links we un-masked for full-fidelity tracking are legible against the
# physically-simulated robot. Pure viz — never reads/writes the policy/obs/physics.
# =========================================================================

# Reference keypoint order matches get_keypts / KEYPTS_MASK (URDF FK link order, 39 links).
# idx>=31 == the right arm (right_shoulder_pitch..right_wrist_yaw + right_rubber_hand); this is
# exactly the KEYPTS_MASK_NO_RARM cutoff (`m if i<31 else 0`) used in the tracking rewards.
_N_KEYPTS = 39
_RARM_START_IDX = 31


def _make_ref_overlay_markers():
    """Create the VisualizationMarkers: a green sphere prototype (body) + a red one (right arm)."""
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ref_keypts",
        markers={
            "body": sim_utils.SphereCfg(
                radius=0.028,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.15)),
            ),
            "rarm": sim_utils.SphereCfg(
                radius=0.032,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.1, 0.1)),
            ),
        },
    )
    markers = VisualizationMarkers(cfg)
    # env 0 only (num_envs=1 for a clean video). Per-keypoint prototype index: right arm -> red.
    marker_indices = [1 if i >= _RARM_START_IDX else 0 for i in range(_N_KEYPTS)]
    return markers, marker_indices


def _update_ref_overlay_markers(env, markers, marker_indices, device):
    """Place the spheres at this step's reference link world positions."""
    unw = env.unwrapped
    with torch.inference_mode():
        motion_times = unw.episode_length_buf * unw.step_dt + unw.start_motion_times.clone().detach().to(
            device=device, dtype=torch.float32
        )
        res = unw.motion_lib.get_motion_state(unw.motion_ids, motion_times)
        # global_keypts are env-local; + env_origins -> world (same convention as the rewards).
        gk = res["global_keypts"].to(device) + unw.scene.env_origins.unsqueeze(1)   # (N, 39, 3)
    translations = gk[0]                                                             # env 0 -> (39, 3)
    markers.visualize(
        translations=translations,
        marker_indices=torch.tensor(marker_indices, device=device, dtype=torch.long),
    )


# =========================================================================
# Object-reference candidate overlay (--overlay-obj-candidates), for the #4
# "object-as-hand reference" diagnostic. Draws TWO markers per frame so the
# candidate object-reference trajectories can be judged against the ref hand:
#   (i)  YELLOW  = holosoma object_poses trajectory (motion_lib.object_poses).
#   (ii) CYAN    = right-hand FK (global_keypts[-1] = right_rubber_hand).
# Pure viz. Requires a .pkl carrying `object_poses` (Adapter B, current gen).
# =========================================================================
def _make_obj_candidate_markers():
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/obj_candidates",
        markers={
            "holosoma": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),   # yellow = (i)
            ),
            "rhand_fk": sim_utils.SphereCfg(
                radius=0.038,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 1.0)),    # cyan  = (ii)
            ),
        },
    )
    return VisualizationMarkers(cfg)


def _update_obj_candidate_markers(env, markers, device):
    """Place the two object-reference candidates at this step (env 0)."""
    unw = env.unwrapped
    with torch.inference_mode():
        motion_times = unw.episode_length_buf * unw.step_dt + unw.start_motion_times.clone().detach().to(
            device=device, dtype=torch.float32
        )
        res = unw.motion_lib.get_motion_state(unw.motion_ids, motion_times)
        origin = unw.scene.env_origins                                              # (N, 3)
        # (i) holosoma object trajectory (offset already applied in get_motion_state) -> world.
        obj_i = res["object_poses"][:, :3].to(device) + origin                      # (N, 3)
        # (ii) right-hand FK: last keypoint = right_rubber_hand.
        gk = res["global_keypts"].to(device) + origin.unsqueeze(1)                  # (N, 39, 3)
        obj_ii = gk[:, -1, :]                                                       # (N, 3)
    translations = torch.stack([obj_i[0], obj_ii[0]], dim=0)                        # (2, 3)
    markers.visualize(
        translations=translations,
        marker_indices=torch.tensor([0, 1], device=device, dtype=torch.long),
    )


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


def _build_reference_joint_map(env, device):
    """Map motion_lib's 27 reference joints → robot articulation joint indices.

    motion_lib.get_motion_state returns ``dof_pos`` in motion_lib.joint_names order
    (27 body joints, env JointNamesOrder). The robot articulation has more joints
    (body + fingers) in its own order. Returns a (27,) long tensor ``joint_map`` such
    that ``robot_q[:, joint_map] = ref_dof_pos`` places each reference joint correctly.
    """
    unw = env.unwrapped
    robot = unw.scene["robot"]
    ref_names = list(unw.motion_lib.joint_names)
    name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    missing = [n for n in ref_names if n not in name_to_idx]
    if missing:
        raise RuntimeError(f"[reference-playback] reference joints absent on robot: {missing}")
    return torch.tensor([name_to_idx[n] for n in ref_names], device=device, dtype=torch.long)


def _write_reference_pose(env, joint_map, device):
    """Teleport the robot to the motion_lib reference pose for the current frame.

    Pure kinematic: writes joint positions (reference), zero joint velocities, the
    reference root pose (+ env origins), and zero root velocity. Fingers stay at their
    default (no finger reference in motion_lib). Called AFTER env.step so the displayed
    pose is the exact reference, with physics never accumulating.
    """
    unw = env.unwrapped
    robot = unw.scene["robot"]
    # IsaacLab's write_joint_*_to_sim does in-place updates to its data buffers (e.g.
    # joint_acc), which are inference tensors — those writes must happen inside
    # inference_mode or torch raises "Inplace update to inference tensor outside
    # InferenceMode is not allowed".
    with torch.inference_mode():
        motion_times = unw.episode_length_buf * unw.step_dt + unw.start_motion_times.clone().detach().to(
            device=device, dtype=torch.float32
        )
        res = unw.motion_lib.get_motion_state(unw.motion_ids, motion_times)

        q = robot.data.default_joint_pos.clone()
        q[:, joint_map] = res["dof_pos"].to(q.dtype)
        qd = torch.zeros_like(q)
        robot.write_joint_state_to_sim(q, qd)

        root_pos = res["root_pos"].to(device) + unw.scene.env_origins
        root_quat = res["root_rot"].to(device)  # wxyz, matches IsaacLab root pose convention
        root_pose = torch.cat([root_pos, root_quat], dim=-1)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros((unw.num_envs, 6), device=device))


def _make_reference_pd_policy(env, joint_map, device):
    """Native-action policy that PD-tracks the motion_lib reference under physics.

    The env's body action term is ``JointPositionActionCfg(scale, use_default_offset=True)``
    so the PD target is ``default[joint] + action*scale``. Setting
    ``action = (ref_dof - default) / scale`` makes the PD target equal the reference joint
    angles; Isaac's SONIC-matched PD actuators then track it under gravity/contact — no
    decoder, no teleport. This is "holosoma instead of SONIC" in the physics sim.

    The reference ``dof_pos`` is returned in ``motion_lib.joint_names`` order, which is
    exactly the body action term's ``JointNamesOrder`` (preserve_order=True) — same index,
    no permutation. Action layout is ``[body(n_body), left_hand(1), right_hand(1)]`` (the
    ActionsCfg field order: inherited joint_pos, then left/right binary finger terms).
    Left fingers: open==close in the cfg, so the value is irrelevant (0). Right fingers:
    BinaryJointAction closes when action < 0, so close during the grab-hold window.
    """
    unw = env.unwrapped
    robot = unw.scene["robot"]
    scale = float(unw.cfg.actions.joint_pos.scale)
    default_body = robot.data.default_joint_pos[:, joint_map].clone()   # (N, n_body), JointNamesOrder
    n_body = int(joint_map.shape[0])
    total = int(env.num_actions)
    if total != n_body + 2:
        print(f"[reference-pd] WARN: action dim {total} != body {n_body} + 2 fingers; "
              "assuming layout [body, left_hand, right_hand] — verify ActionsCfg order.")
    print(f"[reference-pd] body DOF={n_body}, total action dim={total}, joint_pos scale={scale}")

    def policy(obs):
        with torch.inference_mode():
            motion_times = unw.episode_length_buf * unw.step_dt + unw.start_motion_times.clone().detach().to(
                device=device, dtype=torch.float32
            )
            res = unw.motion_lib.get_motion_state(unw.motion_ids, motion_times)
            ref_dof = res["dof_pos"].to(device)                        # (N, n_body), JointNamesOrder
            act = torch.zeros((env.num_envs, total), device=device, dtype=torch.float32)
            act[:, :n_body] = (ref_dof - default_body) / scale
            # right hand (index n_body+1): close (<0) during grab-hold, else open (>0)
            is_closed = res["is_closed"].to(device).reshape(-1).float()
            act[:, n_body + 1] = torch.where(is_closed > 0.5, -1.0, 1.0)
            return act

    return policy


# =========================================================================
# Main.
# =========================================================================

def main():
    # Deterministic motion selection. The env draws a RANDOM motion per reset
    # (torch.randint in reset_joints_for_motion); without a fixed seed two runs play
    # different motions, so e.g. --skip-start-frames 0 vs 20 would reach different
    # endpoints purely from drawing different motions (not from the skip). Seeding here
    # makes the drawn motion reproducible across runs so comparisons are apples-to-apples.
    _seed = getattr(args_cli, "seed", None)
    if _seed is None:
        _seed = 0
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    print(f"[play_sonic_adapter] seed = {_seed} (deterministic motion draw; "
          "same seed → same motion across runs)")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        enable_cameras=True,
    )
    env_cfg.seed = _seed
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # ---- mirror train_sonic_adapter.py's agent overrides EXACTLY ----
    agent_cfg.experiment_name = f"{agent_cfg.experiment_name}_sonic_adapter"
    agent_cfg.policy.class_name = "AdapterActorCritic"
    agent_cfg.policy.actor_hidden_dims = [256, 128]
    agent_cfg.policy.critic_hidden_dims = [512, 256, 256]
    print(f"[play_sonic_adapter] policy = AdapterActorCritic, actor={agent_cfg.policy.actor_hidden_dims}")

    # ---- resolve checkpoint (skipped in --zero-residual / --reference-playback modes) ----
    resume_path = None
    log_dir = None
    _no_checkpoint = args_cli.zero_residual or args_cli.reference_playback or args_cli.reference_pd
    if not _no_checkpoint:
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

    # 29-DOF strict-fidelity articulation (actuated waist roll/pitch) to match SONIC training.
    if args_cli.waist_dof == 29:
        apply_29dof_waist_override(env_cfg)

    # Optional: point reference-playback at an isolated .pkl set (e.g. holosoma 29-DOF).
    if args_cli.ref_motions_path is not None:
        env_cfg.ref_motions_path = args_cli.ref_motions_path
        print(f"[play_sonic_adapter] ref_motions_path override -> {args_cli.ref_motions_path}")

    # Optional: start episodes near the grab frame to skip the foot-skating walk approach
    # (reset_joints_for_motion reads env.cfg.motion_start_pregrab_margin_s). Set on the cfg
    # BEFORE gym.make so it's honored from the very first reset.
    if args_cli.start_pregrab_margin is not None:
        env_cfg.motion_start_pregrab_margin_s = args_cli.start_pregrab_margin
        print(f"[play_sonic_adapter] start_pregrab_margin = {args_cli.start_pregrab_margin}s "
              "(episodes begin near grab; walk approach skipped)")
    if args_cli.skip_start_frames is not None:
        # +10: reset 10 frames LATER so the decoder-history seed (the 10 frames PRECEDING the
        # reset) lands on REAL motion, not the skipped interpolation prepend.
        env_cfg.motion_skip_start_frames = args_cli.skip_start_frames + 10
        print(f"[play_sonic_adapter] skip_start_frames = {args_cli.skip_start_frames} "
              f"(+10 history warmup -> env resets at frame {args_cli.skip_start_frames + 10})")

    # RENDER-ONLY: drop the torso-angle (>=60deg tilt) termination so a dynamic rollout isn't cut
    # short by an early reset (lets the full base drift play out). Only edits THIS run's cfg;
    # training/eval terminations are untouched.
    # RENDER-ONLY DEFAULT: never reset a playback mid-clip. This is a render/visualization script,
    # so unless --keep-terms is passed, disable EVERY termination term (tilt, root-height,
    # base_contact, time_out, ...) so the full rollout plays out. Only edits THIS run's cfg;
    # train_sonic_adapter.py / eval_sonic_adapter.py have their own cfgs and keep terminations.
    if not args_cli.keep_terms:
        _disabled = []
        for _name in list(vars(env_cfg.terminations).keys()):
            if not _name.startswith("_") and getattr(env_cfg.terminations, _name) is not None:
                setattr(env_cfg.terminations, _name, None)
                _disabled.append(_name)
        print(f"[play_sonic_adapter] RENDER-ONLY: terminations disabled (no mid-clip reset): {_disabled}")
    else:
        print("[play_sonic_adapter] --keep-terms: env terminations kept")

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

    device = agent_cfg.device
    if args_cli.reference_pd:
        # ---- reference-PD: NO SONIC decoder/encoder. Wrap with the plain RSL-RL wrapper so
        # the env's NATIVE JointPositionAction is what env.step applies (PD physics). ----
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        env = RslRlVecEnvWrapper(env, clip_actions=None)
        print("[play_sonic_adapter] REFERENCE-PD mode: native JointPositionAction PD-driven "
              "by the motion_lib reference each step (physics tracking; no SONIC decoder, no "
              "teleport).")
    else:
        # ---- frozen encoder + decoder + adapter wrapper (same wiring as training) ----
        print(f"[play_sonic_adapter] loading frozen SONIC decoder ONNX: {args_cli.sonic_decoder_onnx}")
        decoder = load_frozen_decoder(args_cli.sonic_decoder_onnx, device)
        print(f"[play_sonic_adapter] loading frozen SONIC encoder ONNX: {args_cli.sonic_encoder_onnx}")
        encoder = load_frozen_encoder(args_cli.sonic_encoder_onnx, device)
        env = TokenAdapterVecEnvWrapper(
            env, decoder, encoder, device,
            residual_scale=args_cli.residual_scale,
            clip_actions=None,
        )

    # ---- policy: trained adapter, zero-residual baseline, reference playback, or reference-PD ----
    if args_cli.reference_pd:
        # Real policy is built after the warm-up (needs motion_lib + the joint map). Placeholder
        # returns zeros (only the warm-up uses an explicit zero action, never this).
        print("[play_sonic_adapter] REFERENCE-PD: reference→PD-target policy built after warm-up.")

        def policy(obs):
            return torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float32)
    elif args_cli.reference_playback:
        print("[play_sonic_adapter] REFERENCE-PLAYBACK mode: kinematic teleport to the "
              "motion_lib reference each frame (encoder/decoder/policy ignored). Robot is "
              "overwritten to the reference pose after every step → pure target motion.")

        def policy(obs):
            # Action is irrelevant (robot state is overwritten post-step); zeros just
            # advance the env machinery (episode_length_buf, cameras).
            return torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float32)
    elif args_cli.zero_residual:
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
    elif args_cli.reference_pd:
        video_folder = Path("logs") / "rsl_rl" / agent_cfg.experiment_name / "reference_pd" / "videos"
    elif args_cli.reference_playback:
        video_folder = Path("logs") / "rsl_rl" / agent_cfg.experiment_name / "reference" / "videos"
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

    # Reference-playback: build the motion_lib→robot joint map once (after reset, so
    # motion_lib/motion_ids exist).
    ref_joint_map = (
        _build_reference_joint_map(env, device)
        if (args_cli.reference_playback or args_cli.reference_pd) else None
    )
    if args_cli.reference_playback:
        _write_reference_pose(env, ref_joint_map, device)  # frame 0 = reference at reset
        _APP.update()
        _APP.update()
    if args_cli.reference_pd:
        policy = _make_reference_pd_policy(env, ref_joint_map, device)

    # Reference-overlay markers (created once, after reset so motion_lib/motion_ids exist).
    ref_markers, ref_marker_indices = _make_ref_overlay_markers() if args_cli.overlay_ref else (None, None)
    if args_cli.overlay_ref:
        _update_ref_overlay_markers(env, ref_markers, ref_marker_indices, device)  # frame 0
        print(f"[overlay-ref] {_N_KEYPTS} reference-keypoint spheres enabled "
              f"(idx>={_RARM_START_IDX} right arm = RED, rest = GREEN)")

    # Object-reference candidate markers (#4 diagnostic).
    obj_cand_markers = _make_obj_candidate_markers() if args_cli.overlay_obj_candidates else None
    if args_cli.overlay_obj_candidates:
        _update_obj_candidate_markers(env, obj_cand_markers, device)  # frame 0
        print("[overlay-obj-candidates] (i) holosoma object_poses = YELLOW, "
              "(ii) right-hand FK = CYAN")

    while simulation_app.is_running() and timestep < args_cli.video_length:
        start_time = time.time()

        # 1. policy → residual → base+residual → frozen decoder → env.step (deterministic)
        with torch.inference_mode():
            actions = policy(obs).clone()
            obs, _, dones, _ = env.step(actions)

        # 1b. reference-playback: overwrite the robot to the exact reference pose AFTER
        # the step, so the displayed motion is the kinematic target (physics discarded).
        if args_cli.reference_playback:
            _write_reference_pose(env, ref_joint_map, device)

        # 1c. reference overlay: move the spheres to THIS step's reference link positions
        # (before the render flush so they appear in this frame).
        if args_cli.overlay_ref:
            _update_ref_overlay_markers(env, ref_markers, ref_marker_indices, device)
        if args_cli.overlay_obj_candidates:
            _update_obj_candidate_markers(env, obj_cand_markers, device)

        # 2. flush RTX render pipeline so the camera annotator delivers THIS step's frame
        _APP.update()
        _APP.update()

        # frame diagnostic: resolve robot/object/table world frames + confirm object placement.
        if timestep in (0, 40, 80):
            _uw = env.unwrapped
            _o = _uw.scene.env_origins[0]
            def _loc(a):
                try: return [round(float(v), 2) for v in (a - _o).tolist()]
                except Exception: return None
            _rp = _loc(_uw.scene["robot"].data.root_pos_w[0])
            _op = _loc(_uw.scene["object"].data.root_pos_w[0])
            _hp = None
            for _nm in ("right_rubber_hand", "right_rubber_hand_link", "right_wrist_yaw_link"):
                try:
                    _hb = _uw.scene["robot"].find_bodies(_nm)[0]
                    if _hb: _hp = _loc(_uw.scene["robot"].data.body_pos_w[0, _hb[0]]); break
                except Exception: pass
            _kp = None
            try: _kp = _loc(_uw.scene["kitchen"].data.root_pos_w[0])
            except Exception:
                try: _kp = list(getattr(_uw.scene["kitchen"].cfg.init_state, "pos", None))
                except Exception: pass
            print(f"[frame-diag t={timestep}] robot_root={_rp}  right_hand={_hp}  object={_op}  kitchen={_kp}")

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
            if args_cli.reference_pd:
                label = "REFERENCE-PD  " + label
            elif args_cli.reference_playback:
                label = "REFERENCE  " + label
            elif args_cli.zero_residual:
                label = "ZERO-SHOT  " + label
            if args_cli.overlay_ref:
                label = label + "  [ref overlay: R-arm red]"
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
