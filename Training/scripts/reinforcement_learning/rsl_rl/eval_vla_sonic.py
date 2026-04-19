"""VLA + SONIC closed-loop eval in Isaac Lab.

Pipeline per env step (inside the chunk, see --chunk-size):

    env obs ─▶ ObsToPolicyAdapter ─▶ Gr00tPolicy.get_action ─▶ action_dict
                                                                     │
           action_dict (first chunk step only) ──▶ PlannerWrapper ───▶ mujoco_qpos
                                                                     │
           anchor_pose + lb_trajectory + vr_3pt  ──▶ build_encoder_obs
                                                                     │
                                                            UtmWrapper.run_encoder → token
                                                                     │
                                       token + HistoryBuffer ──▶ build_decoder_obs
                                                                     │
                                                            UtmWrapper.run_decoder → body_29
                                                                     │
                body_29 + vla_fingers ──▶ utm_plus_vla_to_env_action → env_action_41
                                                                     │
                                                              env.step(env_action_41)

Run:

    cd WBCBenchmark/Training && python3 scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py \\
        --vla-checkpoint /home/dvij/kevin/checkpoints/run-01 \\
        --num-episodes 1 \\
        --record-video /home/dvij/kevin/eval_videos/run01_ep0

Produces ``<prefix>_third_person.mp4`` and ``<prefix>_ego.mp4`` when --record-video is set.
"""

from __future__ import annotations

import argparse
import builtins
import os
import sys
import time
from functools import partial
from pathlib import Path

print = partial(builtins.print, flush=True)


# =========================================================================
# Phase 1: Isaac Lab AppLauncher MUST come first (before any gym/torch that
# might touch omniverse).
# =========================================================================

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLA+SONIC closed-loop eval")
    parser.add_argument("--task", default="Isaac-Motion-Tracking-Pick-Cam-ContFingers-v0")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Keep at 1 to avoid camera OOM.")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Execute first N of the VLA's 16 predicted steps before replanning.")
    parser.add_argument("--vla-checkpoint", required=True)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--language", default="pick up the mustard bottle")
    # Default paths assume DreamVLA/ and GR00T-WholeBodyControl/ are sibling repos,
    # and you run this script from DreamVLA/Training/. Override if your layout differs.
    parser.add_argument("--encoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx")
    parser.add_argument("--decoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")
    parser.add_argument("--planner-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx")
    parser.add_argument("--record-video", default=None,
                        help="Output prefix for MP4 files. Saves both third-person + ego.")
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    # AppLauncher args get appended below.
    return parser


_parser = _parse_cli()

# Lazy import AppLauncher so --help works even without isaaclab installed.
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(_parser)
_ARGS = _parser.parse_args()

# Cameras always needed for the VLA's ego view — override if not set.
_ARGS.enable_cameras = True

app_launcher = AppLauncher(_ARGS)
_APP = app_launcher.app


# =========================================================================
# Phase 2: Heavy imports (safe now that AppLauncher is up).
# =========================================================================

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  # registers tasks
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Ensure vla_sonic package is importable from its parent dir.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from vla_sonic import (  # noqa: E402
    HistoryBuffer,
    PlannerWrapper,
    UtmWrapper,
    build_decoder_obs,
    build_encoder_obs,
    build_planner_inputs,
    utm_plus_vla_to_env_action,
)
from vla_sonic.frame_transforms import quat_wxyz_to_xyzw  # noqa: E402
from vla_sonic.planner_to_utm import rot6d_to_quat_wxyz  # noqa: E402 — (unused here but kept for reference)
from vla_sonic.obs_to_policy import ObsAdapterConfig, ObsToPolicyAdapter  # noqa: E402

from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402

# gear_sonic's RobotModel provides the joint-group layout the VLA was trained with.
from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model  # noqa: E402


# =========================================================================
# Joint-order helpers: Isaac's robot → UTM's 29-DoF MuJoCo order.
# =========================================================================

UTM_29_JOINT_NAMES = [
    # Legs (0–11) — matches DEFAULT_DOF_ANGLES in g1_29dof_sonic_model12.yaml.
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # Waist (12–14)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # Left arm (15–21)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    # Right arm (22–28)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
assert len(UTM_29_JOINT_NAMES) == 29


def build_isaac_to_utm_perm(isaac_joint_names: list[str]) -> np.ndarray:
    """Return (29,) array of Isaac indices s.t. ``isaac_q[perm] == utm_q``.

    Fails loudly if any UTM joint is missing from Isaac — means the env's robot
    asset doesn't expose the expected 29 DoF.
    """
    name_to_idx = {n: i for i, n in enumerate(isaac_joint_names)}
    perm = np.full(29, -1, dtype=np.int64)
    missing = []
    for i, name in enumerate(UTM_29_JOINT_NAMES):
        idx = name_to_idx.get(name, -1)
        if idx < 0:
            missing.append(name)
        else:
            perm[i] = idx
    if missing:
        raise RuntimeError(
            f"Isaac robot is missing UTM-expected joints: {missing}. "
            "Check the G1 URDF / articulation cfg."
        )
    return perm


# =========================================================================
# Planner output decomposition.
# =========================================================================

# Planner output layout per frame (mujoco_qpos, 36-D):
#   [0:3]   root_pos  (world frame)
#   [3:7]   root_quat (wxyz scalar-first)
#   [7:36]  29 joints in UTM_29_JOINT_NAMES order
PLANNER_ROOT_POS_SLICE = slice(0, 3)
PLANNER_ROOT_QUAT_SLICE = slice(3, 7)
PLANNER_JOINTS_SLICE = slice(7, 36)

# Lower body = legs only (indices 0–11 of the 29 joints) = qpos[7:19].
LOWER_BODY_JOINT_SLICE = slice(7, 19)  # 12 joints

# Encoder expects 10 future frames at step 5 (frames [0, 5, 10, ..., 45]).
ENCODER_FUTURE_FRAME_INDICES = list(range(0, 50, 5))

# Planner raw output frame rate (before resampling, per planner_onnx.md).
PLANNER_OUTPUT_FPS = 30.0


def extract_anchor_pose(mujoco_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (anchor_pos_world, anchor_quat_wxyz, anchor_rot6d) from frame 0."""
    frame0 = mujoco_qpos[0, 0]  # (36,)
    anchor_pos = np.asarray(frame0[PLANNER_ROOT_POS_SLICE], dtype=np.float32).copy()
    anchor_quat = np.asarray(frame0[PLANNER_ROOT_QUAT_SLICE], dtype=np.float32).copy()
    # rot6d from quat: columns 0 and 1 of the rotation matrix.
    from scipy.spatial.transform import Rotation as R
    R_mat = R.from_quat(quat_wxyz_to_xyzw(anchor_quat)).as_matrix().astype(np.float32)
    rot6d = np.concatenate([R_mat[:, 0], R_mat[:, 1]]).astype(np.float32)  # (6,)
    return anchor_pos, anchor_quat, rot6d


def extract_lower_body_future(mujoco_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions (10,12), velocities (10,12)) at the encoder's subsample grid."""
    qpos = np.asarray(mujoco_qpos[0], dtype=np.float32)  # (N, 36)
    n_frames = qpos.shape[0]
    need = max(ENCODER_FUTURE_FRAME_INDICES) + 1
    if n_frames < need:
        # Pad by repeating the last frame so the slicer doesn't error.
        pad = np.repeat(qpos[-1:], need - n_frames, axis=0)
        qpos = np.concatenate([qpos, pad], axis=0)
    lb_all = qpos[:, LOWER_BODY_JOINT_SLICE]  # (N, 12)
    # Velocities at 30 Hz: central differences over the full trajectory, then subsample.
    vel_all = np.gradient(lb_all, 1.0 / PLANNER_OUTPUT_FPS, axis=0).astype(np.float32)
    pos = lb_all[ENCODER_FUTURE_FRAME_INDICES].astype(np.float32)  # (10, 12)
    vel = vel_all[ENCODER_FUTURE_FRAME_INDICES].astype(np.float32)  # (10, 12)
    return pos, vel


# =========================================================================
# VR 3-point extraction from the VLA action dict.
# =========================================================================

def extract_vr_3pt(
    vla_action: dict, *, t_index: int = 0, batch_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vr_3pt_position_world (9,), vr_3pt_rot6d (18,)) for one VLA step.

    The VLA emits these keys as (B, T, D) per ``gear_sonic_config.py``:
        vr_3pt_position    : (B, T, 9)   world-frame xyz per point
        vr_3pt_orientation : (B, T, 18)  rot6d per point (left_wrist, right_wrist, torso)
    """
    pos = np.asarray(vla_action["vr_3pt_position"], dtype=np.float32)
    orn = np.asarray(vla_action["vr_3pt_orientation"], dtype=np.float32)
    if pos.ndim != 3 or pos.shape[-1] != 9:
        raise ValueError(f"vr_3pt_position must be (B,T,9); got {pos.shape}")
    if orn.ndim != 3 or orn.shape[-1] != 18:
        raise ValueError(f"vr_3pt_orientation must be (B,T,18); got {orn.shape}")
    return pos[batch_index, t_index].copy(), orn[batch_index, t_index].copy()


# =========================================================================
# Video writer wrapping imageio-ffmpeg.
# =========================================================================

class VideoWriter:
    def __init__(self, path: Path, fps: int):
        import imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7)
        self.path = path

    def write(self, frame_rgb: np.ndarray) -> None:
        self._writer.append_data(frame_rgb)

    def close(self) -> None:
        self._writer.close()


def _read_camera_rgb(env, key: str) -> np.ndarray | None:
    cam = env.unwrapped.scene.get(key, None) if hasattr(env.unwrapped.scene, "get") else None
    if cam is None:
        try:
            cam = env.unwrapped.scene[key]
        except KeyError:
            return None
    rgb = cam.data.output["rgb"][0, ..., :3]
    if rgb.dtype != torch.uint8:
        rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
    return rgb.cpu().numpy()


# =========================================================================
# Main rollout.
# =========================================================================

def main() -> int:
    args = _ARGS

    # --- 1. Build env ---------------------------------------------------
    env_cfg = parse_env_cfg(
        args.task,
        device=f"cuda:0",
        num_envs=args.num_envs,
        enable_cameras=True,
    )
    env = gym.make(args.task, cfg=env_cfg)
    print(f"[env] {args.task}  action_space={env.action_space}")

    # --- 2. Build VLA policy -------------------------------------------
    print(f"[vla] loading {args.vla_checkpoint}")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.vla_checkpoint,
        device="cuda:0",
    )

    # --- 3. Build SONIC wrappers ---------------------------------------
    print(f"[utm] encoder={args.encoder_onnx}")
    print(f"[utm] decoder={args.decoder_onnx}")
    utm = UtmWrapper(args.encoder_onnx, args.decoder_onnx)

    print(f"[planner] {args.planner_onnx}")
    planner = PlannerWrapper(args.planner_onnx)

    # --- 4. Obs adapter & joint-order perm -----------------------------
    robot_model = instantiate_g1_robot_model(waist_location="lower_body")
    obs_adapter = ObsToPolicyAdapter(
        env,
        ObsAdapterConfig(
            language_instruction=args.language,
            robot_model=robot_model,
            camera_scene_key="camera_robot",
        ),
    )
    robot = env.unwrapped.scene["robot"]
    isaac_to_utm_perm = build_isaac_to_utm_perm(list(robot.data.joint_names))

    # --- 5. History buffer for decoder + planner context ---------------
    history = HistoryBuffer()

    # --- 6. Video writers (optional) -----------------------------------
    writers: dict[str, VideoWriter] = {}
    if args.record_video:
        prefix = Path(args.record_video)
        writers["camera"] = VideoWriter(prefix.with_name(prefix.name + "_third_person.mp4"), args.video_fps)
        writers["camera_robot"] = VideoWriter(prefix.with_name(prefix.name + "_ego.mp4"), args.video_fps)
        print(f"[video] writing {writers['camera'].path} and {writers['camera_robot'].path}")

    # --- 7. Rollout -----------------------------------------------------
    action_space_dim = env.action_space.shape[-1]
    zero_action = torch.zeros((args.num_envs, action_space_dim), device="cuda:0", dtype=torch.float32)
    prev_utm_body_29 = np.zeros(29, dtype=np.float32)

    total_successes = 0

    for ep in range(args.num_episodes):
        print(f"\n[episode {ep}]")
        obs, info = env.reset()
        history.reset()

        vla_chunk: dict | None = None
        chunk_step = 0

        for step in range(args.max_steps_per_episode):
            # 7a. Build VLA obs + refresh action chunk every `chunk_size` steps.
            if vla_chunk is None or chunk_step >= args.chunk_size:
                vla_obs = obs_adapter()
                vla_out = policy.get_action(vla_obs)
                # Gr00tPolicy.get_action returns either a dict or (dict, ...) tuple.
                vla_chunk = vla_out[0] if isinstance(vla_out, tuple) else vla_out
                chunk_step = 0

            # 7b. Extract this chunk-step's VLA slice.
            t_idx = chunk_step

            # 7c. Push current env state into history BEFORE planner so context is up to date.
            q_isaac = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            qd_isaac = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
            q_utm = q_isaac[isaac_to_utm_perm]
            qd_utm = qd_isaac[isaac_to_utm_perm]
            root_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
            root_quat_w = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)  # wxyz
            root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
            # Projected gravity in body frame.
            from scipy.spatial.transform import Rotation as R
            gravity_body = R.from_quat(quat_wxyz_to_xyzw(root_quat_w)).inv().apply(
                np.array([0.0, 0.0, -1.0], dtype=np.float32)
            ).astype(np.float32)
            mujoco_qpos = np.concatenate([root_pos_w, root_quat_w, q_utm]).astype(np.float32)

            history.push(
                joint_pos=q_utm,
                joint_vel=qd_utm,
                last_action=prev_utm_body_29,
                base_ang_vel=root_ang_vel_b,
                gravity_dir=gravity_body,
                mujoco_qpos=mujoco_qpos,
            )

            # 7d. Run planner using this chunk-step's VLA planner commands.
            planner_inputs = build_planner_inputs(
                vla_action=vla_chunk,
                context_mujoco_qpos=history.planner_context(),
                t_index=t_idx,
            )
            planner_out = planner.run(**planner_inputs.as_kwargs())

            # 7e. Extract anchor + lower-body trajectory from planner output.
            anchor_pos_w, anchor_quat_wxyz, anchor_rot6d = extract_anchor_pose(planner_out.mujoco_qpos)
            lb_pos, lb_vel = extract_lower_body_future(planner_out.mujoco_qpos)

            # 7f. VR 3-point from VLA (world frame; encoder builder does the anchor-local transform).
            vr_pos_world, vr_rot6d = extract_vr_3pt(vla_chunk, t_index=t_idx)

            # 7g. Build encoder obs → run encoder → build decoder obs → run decoder.
            enc_obs = build_encoder_obs(
                anchor_pos_world=anchor_pos_w,
                anchor_quat_wxyz=anchor_quat_wxyz,
                anchor_rot6d=anchor_rot6d,
                lower_body_positions_future=lb_pos,
                lower_body_velocities_future=lb_vel,
                vr_3pt_position_world=vr_pos_world,
                vr_3pt_rot6d=vr_rot6d,
            )
            token = utm.run_encoder({"obs_dict": enc_obs}).reshape(-1)  # (64,)

            dec_hist = history.decoder_history()
            dec_obs = build_decoder_obs(
                token_state=token,
                **dec_hist.as_kwargs(),
            )
            utm_body_29 = utm.run_decoder({"obs_dict": dec_obs}).reshape(-1)  # (29,)

            # 7h. Assemble env action.
            env_action_np = utm_plus_vla_to_env_action(
                utm_body_29=utm_body_29,
                vla_action=vla_chunk,
                t_index=t_idx,
            )  # (41,)
            env_action = torch.as_tensor(env_action_np[None, :], device="cuda:0", dtype=torch.float32)

            # 7i. Step.
            obs, rew, term, trunc, info = env.step(env_action)

            # 7j. Video frames.
            if writers:
                for key, w in writers.items():
                    frame = _read_camera_rgb(env, key)
                    if frame is not None:
                        w.write(frame)

            prev_utm_body_29 = utm_body_29
            chunk_step += 1

            if bool(term[0] if term.ndim > 0 else term):
                break

        n_succ = int(getattr(env.unwrapped, "n_successes", 0))
        total_successes = n_succ  # cumulative across episodes is tracked by env itself
        print(f"[episode {ep}] ended at step {step+1}; cumulative successes={n_succ}")

    for w in writers.values():
        w.close()

    env.close()
    _APP.close()

    print(f"\n[eval] {total_successes}/{args.num_episodes} successes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
