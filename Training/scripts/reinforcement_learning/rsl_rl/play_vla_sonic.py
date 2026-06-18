"""VLA (unitree_g1_sonic) + SONIC decoder closed-loop *recording* script in Isaac Lab.

Records a video of the VLA rolling out its policy (third-person + ego). Qualitative
sibling of ``eval_vla_sonic.py`` (which reports lift statistics headlessly); this
one renders one (or a few) rollouts so you can WATCH the policy. Same closed-loop
wiring as the eval, so the videos reflect exactly what is measured.

This VLA embodiment predicts the SONIC latent token DIRECTLY (``motion_token``,
64-D) plus the finger joints, so the pipeline is short — no kinematic planner,
no encoder, no vr_3pt teleop stage:

    env obs ─▶ ObsToPolicyAdapter ─▶ Gr00tPolicy.get_action ─▶ action_dict
                                                                     │
                          action_dict["motion_token"] (64-D)  ───────┤
                                                                     │
                          token + HistoryBuffer ──▶ build_decoder_obs
                                                                     │
                                              UtmWrapper.run_decoder → body_29
                                                                     │
        body_29 + action_dict["{left,right}_hand_joints"] ──▶ utm_plus_vla_to_env_action → env_action_41
                                                                     │
                                                              env.step(env_action_41)

Run:

    cd WBCBenchmark/Training && python3 scripts/reinforcement_learning/rsl_rl/play_vla_sonic.py \\
        --vla-checkpoint /home/dvij/kevin/checkpoints/run-01 \\
        --num-episodes 1 \\
        --record-video /home/dvij/kevin/eval_videos/run01_ep0

Produces ``<prefix>_third_person.mp4`` and ``<prefix>_ego.mp4``. ``--record-video``
defaults to ``./vla_rollout`` so a video is always written.
"""

from __future__ import annotations

import argparse
import builtins
import sys
import time
from functools import partial
from pathlib import Path

print = partial(builtins.print, flush=True)

# This VLA embodiment predicts the SONIC token directly. Its tag is baked into
# the checkpoint as ``unitree_g1_sonic`` (NOT ``new_embodiment`` — that was the
# older vr_3pt formulation). Overridable via --embodiment-tag.
DEFAULT_EMBODIMENT_TAG = "unitree_g1_sonic"


# =========================================================================
# Phase 1: Isaac Lab AppLauncher MUST come first (before any gym/torch that
# might touch omniverse).
# =========================================================================

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLA(unitree_g1_sonic)+SONIC closed-loop video recorder")
    parser.add_argument("--task", default="Isaac-Motion-Tracking-Pick-Cam-ContFingers-v0")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Keep at 1 to avoid camera OOM.")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Execute first N of the VLA's predicted steps before replanning.")
    parser.add_argument("--vla-checkpoint", required=True,
                        help="Path to the unitree_g1_sonic fine-tuned GR00T checkpoint dir "
                             "(the VLA that emits motion_token + hand joints).")
    parser.add_argument("--embodiment-tag", default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--language", default="pick up the mustard bottle")
    # Default paths assume DreamVLA/ and GR00T-WholeBodyControl/ are sibling repos,
    # and you run this script from DreamVLA/Training/. Override if your layout differs.
    # Only the decoder is used; the encoder is loaded by UtmWrapper but never run.
    parser.add_argument("--encoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx",
                        help="Loaded by UtmWrapper for consistency but NOT used in this pipeline.")
    parser.add_argument("--decoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")
    parser.add_argument("--record-video", default="vla_rollout",
                        help="Output prefix for MP4 files. Saves third-person + ego. This is a "
                             "recording script, so it defaults to './vla_rollout'.")
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--no-fsq-snap", dest="fsq_snap", action="store_false", default=True,
                        help="Disable snapping the VLA's continuous motion_token onto the FSQ "
                             "lattice (32 levels) before the decoder. Snapping is ON by default "
                             "because the decoder was trained on on-grid tokens.")
    parser.add_argument("--seed", type=int, default=0)
    # AppLauncher args get appended below.
    return parser


_parser = _parse_cli()

# Lazy import AppLauncher so --help works even without isaaclab installed.
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(_parser)
_ARGS = _parser.parse_args()

# Cameras always needed for the VLA's ego view + recording — override if not set.
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
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import PinholeCameraCfg  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

# Ensure vla_sonic package is importable from its parent dir.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from vla_sonic import (  # noqa: E402
    HistoryBuffer,
    UtmWrapper,
    build_decoder_obs,
    utm_plus_vla_to_env_action,
)
from vla_sonic.action_assembler import G1_ACTION_SCALE_SONIC, G1_DEFAULT_ANGLES_SONIC  # noqa: E402
from vla_sonic.obs_to_policy import ObsAdapterConfig, ObsToPolicyAdapter  # noqa: E402
from vla_sonic.physics_overrides import apply_sonic_physics_overrides  # noqa: E402
from vla_sonic.simple_robot_model import SimpleG1RobotModel  # noqa: E402

from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402


# =========================================================================
# Joint-order helpers: Isaac's robot → UTM's 29-DoF SONIC-IsaacLab order.
# =========================================================================

UTM_29_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "right_hip_pitch_joint",      # 1
    "waist_yaw_joint",            # 2
    "left_hip_roll_joint",        # 3
    "right_hip_roll_joint",       # 4
    "waist_roll_joint",           # 5  <-- absent on env's 27-DoF G1 (zero-filled)
    "left_hip_yaw_joint",         # 6
    "right_hip_yaw_joint",        # 7
    "waist_pitch_joint",          # 8  <-- absent on env's 27-DoF G1 (zero-filled)
    "left_knee_joint",            # 9
    "right_knee_joint",           # 10
    "left_shoulder_pitch_joint",  # 11
    "right_shoulder_pitch_joint", # 12
    "left_ankle_pitch_joint",     # 13
    "right_ankle_pitch_joint",    # 14
    "left_shoulder_roll_joint",   # 15
    "right_shoulder_roll_joint",  # 16
    "left_ankle_roll_joint",      # 17
    "right_ankle_roll_joint",     # 18
    "left_shoulder_yaw_joint",    # 19
    "right_shoulder_yaw_joint",   # 20
    "left_elbow_joint",           # 21
    "right_elbow_joint",          # 22
    "left_wrist_roll_joint",      # 23
    "right_wrist_roll_joint",     # 24
    "left_wrist_pitch_joint",     # 25
    "right_wrist_pitch_joint",    # 26
    "left_wrist_yaw_joint",       # 27
    "right_wrist_yaw_joint",      # 28
]
assert len(UTM_29_JOINT_NAMES) == 29


def build_isaac_to_utm_perm(isaac_joint_names: list[str]) -> np.ndarray:
    """Return (29,) array of Isaac indices s.t. ``isaac_q[perm] == utm_q``."""
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
        print(f"[perm] UTM joints absent on Isaac robot (zero-filling): {missing}")
    return perm


def _gather_with_mask(isaac_values: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Apply the permutation, filling ``perm < 0`` positions with 0."""
    out = np.zeros(perm.shape[0], dtype=np.float32)
    valid = perm >= 0
    out[valid] = isaac_values[perm[valid]]
    return out


def extract_motion_token(
    vla_action: dict, *, t_index: int = 0, batch_index: int = 0
) -> np.ndarray:
    """Return the (64,) SONIC latent token for one VLA step (motion_token, (B,T,64))."""
    if "motion_token" not in vla_action:
        raise KeyError(
            f"vla_action has no 'motion_token'; got {sorted(vla_action.keys())}. "
            f"Is this a unitree_g1_sonic checkpoint? (--embodiment-tag)"
        )
    tok = np.asarray(vla_action["motion_token"], dtype=np.float32)
    if tok.ndim != 3 or tok.shape[-1] != 64:
        raise ValueError(f"motion_token must be (B,T,64); got {tok.shape}")
    return tok[batch_index, t_index].copy()


def fsq_snap_token(token: np.ndarray) -> np.ndarray:
    """Snap a continuous token onto SONIC's FSQ lattice (32 levels, step 1/16).

    The decoder was trained on FSQ-quantized tokens (exact grid points k/16 in
    [-1, 15/16]); the VLA regresses a continuous approximation. Snapping recovers
    the on-grid values the decoder expects (matches token_adapter_wrapper.py).
    """
    half_width = 16.0  # 32 FSQ levels → half_width = 32 // 2
    return np.clip(
        np.round(token * half_width) / half_width,
        -1.0, (half_width - 1.0) / half_width,
    ).astype(np.float32)


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


def _read_camera_rgb(env, key: str, *, verbose: bool = False) -> np.ndarray | None:
    try:
        cam = env.unwrapped.scene[key]
    except KeyError:
        if verbose:
            print(f"[video] scene has no '{key}'")
        return None
    try:
        output = cam.data.output
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[video] {key}: cam.data.output failed: {e}")
        return None
    if "rgb" not in output:
        if verbose:
            print(f"[video] {key}: no 'rgb' key in output, have {list(output.keys())}")
        return None
    rgb = output["rgb"][0, ..., :3]
    if rgb.dtype != torch.uint8:
        rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
    return rgb.cpu().numpy()


# =========================================================================
# Camera injection — third-person `camera` + robot-mounted `camera_robot`.
# Offsets from motion_tracking_pick_env.py (3rd-person) and
# collect_pick_cam.py:683-699 (robot-mounted d435).
# =========================================================================

def _inject_cameras(env_cfg) -> None:
    """Force-set 3rd-person `camera` and robot-mounted `camera_robot` on the scene cfg."""
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
            focal_length=18.1476,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10000.0),
        ),
        data_types=["rgb"],
        height=1920, width=2560,
        offset=CameraCfg.OffsetCfg(
            pos=(-1.03 + 2.1 - 0.034, 4.05 - 0.9, 1.31),
            rot=rot_quat,
            convention="opengl",
        ),
    )
    env_cfg.scene.camera_robot = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/Camera_robot",
        spawn=PinholeCameraCfg(
            focal_length=7.6,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.01, 100.0),
        ),
        data_types=["rgb"],
        height=480, width=640,
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.36),
            rot=(0.568, 0.421, -0.421, -0.568),
            convention="opengl",
        ),
    )


# =========================================================================
# Main rollout.
# =========================================================================

def main() -> int:
    args = _ARGS

    # --- 1. Build env ---------------------------------------------------
    env_cfg = parse_env_cfg(
        args.task,
        device="cuda:0",
        num_envs=args.num_envs,
        enable_cameras=True,
    )
    # Match the SONIC decoder's training-time physics (500 Hz substep, fixed
    # friction, self-collisions, solver iters). The decoder emits ABSOLUTE joint
    # targets calibrated for this regime; without it the env's default randomized
    # friction / coarser substep / self-collisions-off make the robot unstable.
    # Same call both working SONIC scripts use (eval/play_sonic_adapter.py).
    apply_sonic_physics_overrides(env_cfg)
    _inject_cameras(env_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    print(f"[env] {args.task}  action_space={env.action_space}")

    # --- 2. Build VLA policy -------------------------------------------
    print(f"[vla] loading {args.vla_checkpoint}  (embodiment={args.embodiment_tag})")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.vla_checkpoint,
        device="cuda:0",
    )

    # --- 3. Build SONIC decoder (encoder loaded but unused) ------------
    print(f"[utm] decoder={args.decoder_onnx}")
    utm = UtmWrapper(args.encoder_onnx, args.decoder_onnx)

    # --- 4. Obs adapter & joint-order perm -----------------------------
    robot_model = SimpleG1RobotModel.build()
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

    # The pick reward's ``object_above_threshold`` only increments its success
    # counter when ``hasattr(env, "n_successes")`` AND num_envs < 1001.
    env.unwrapped.n_successes = torch.zeros(env.unwrapped.num_envs, device="cuda:0", dtype=torch.float32)

    # --- 5. Decoder proprio-history buffer -----------------------------
    history = HistoryBuffer()

    # --- 6. Video writers (opened after Isaac Lab init) ----------------
    writers: dict[str, VideoWriter] = {}
    prefix: Path | None = Path(args.record_video) if args.record_video else None

    # --- 7. Rollout -----------------------------------------------------
    action_space_dim = env.action_space.shape[-1]
    zero_action = torch.zeros((args.num_envs, action_space_dim), device="cuda:0", dtype=torch.float32)
    total_successes = 0

    for ep in range(args.num_episodes):
        print(f"\n[episode {ep}]")
        obs, info = env.reset()
        # Camera sensors populate on env.step(), not env.reset(). Warm-up step +
        # render flush so the first ego frame the VLA sees is current.
        env.step(zero_action)
        _APP.update()
        _APP.update()
        history.reset()
        prev_utm_body_29 = None  # set to (q-default)/scale on frame 0, then decoder output

        # Open video writers on first episode after Isaac Lab is fully up.
        if ep == 0 and prefix is not None and not writers:
            scene_keys = list(env.unwrapped.scene.keys()) if hasattr(env.unwrapped.scene, "keys") else []
            print(f"[video] scene entities: {scene_keys}")
            if "camera" in scene_keys:
                writers["camera"] = VideoWriter(prefix.with_name(prefix.name + "_third_person.mp4"), args.video_fps)
            else:
                print("[video] 3rd-person 'camera' missing — skipping third_person.mp4")
            if "camera_robot" in scene_keys:
                writers["camera_robot"] = VideoWriter(prefix.with_name(prefix.name + "_ego.mp4"), args.video_fps)
            else:
                print("[video] ego 'camera_robot' missing — skipping ego.mp4")
            for key, w in writers.items():
                print(f"[video] writing {w.path}")

        vla_chunk: dict | None = None
        chunk_step = 0
        step = 0

        for step in range(args.max_steps_per_episode):
            # 7a. Build VLA obs + refresh action chunk every `chunk_size` steps.
            if vla_chunk is None or chunk_step >= args.chunk_size:
                vla_obs = obs_adapter()
                vla_out = policy.get_action(vla_obs)
                vla_chunk = vla_out[0] if isinstance(vla_out, tuple) else vla_out
                chunk_step = 0
                if step == 0:
                    print("\n[VLA @ step 0] action-dict dump (t=0 slice, batch=0):")
                    for k in sorted(vla_chunk.keys()):
                        arr = np.asarray(vla_chunk[k])
                        slice_ = arr[0, 0] if arr.ndim == 3 else arr.reshape(-1)
                        prev = slice_.reshape(-1)[:8]
                        print(f"  {k} [shape {tuple(arr.shape)}] = {prev.round(4).tolist()}"
                              f"{' ...' if slice_.size > 8 else ''}")

            t_idx = chunk_step

            # 7b. Push current env state into the decoder history. Convention
            # matches the validated token_action_wrapper.py: joint positions are
            # RELATIVE to the SONIC default standing pose, gravity is IsaacLab's
            # body-frame projected gravity (NOT recomputed), and last_action is the
            # previous decoder output (seeded on frame 0 with the latent that
            # reproduces the current pose, q-default/scale). Feeding raw absolute
            # joint positions here is off-distribution for the decoder → instability.
            q_isaac = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            qd_isaac = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
            q_sonic = _gather_with_mask(q_isaac, isaac_to_utm_perm)
            qd_sonic = _gather_with_mask(qd_isaac, isaac_to_utm_perm)
            jp_sonic = (q_sonic - G1_DEFAULT_ANGLES_SONIC).astype(np.float32)
            gravity_body = robot.data.projected_gravity_b[0].detach().cpu().numpy().astype(np.float32)
            root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
            if prev_utm_body_29 is None:
                last_action = (jp_sonic / G1_ACTION_SCALE_SONIC).astype(np.float32)
            else:
                last_action = prev_utm_body_29
            history.push(
                joint_pos=jp_sonic,
                joint_vel=qd_sonic,
                last_action=last_action,
                base_ang_vel=root_ang_vel_b,
                gravity_dir=gravity_body,
                mujoco_qpos=np.zeros(36, dtype=np.float32),  # unused (no planner)
            )

            # 7c. Token straight from the VLA (no planner/encoder).
            token = extract_motion_token(vla_chunk, t_index=t_idx)  # (64,)
            if args.fsq_snap:
                token = fsq_snap_token(token)

            # 7d. Decoder: token + proprio history → 29-D SONIC body action.
            dec_hist = history.decoder_history()
            dec_obs = build_decoder_obs(token_state=token, **dec_hist.as_kwargs())
            utm_body_29 = utm.run_decoder({"obs_dict": dec_obs}).reshape(-1)  # (29,)

            # 7e. Assemble env action (body + VLA fingers).
            env_action_np = utm_plus_vla_to_env_action(
                utm_body_29_sonic=utm_body_29,
                vla_action=vla_chunk,
                t_index=t_idx,
            )  # (41,)
            if step < 3:
                print(f"[step {step}] motion_token[:8] = {token[:8].round(3).tolist()}")
                print(f"[step {step}]   env body[:12] (legs)     = {env_action_np[:12].round(3).tolist()}")
                print(f"[step {step}]   env left fingers[27:34]  = {env_action_np[27:34].round(3).tolist()}")
                print(f"[step {step}]   env right fingers[34:41] = {env_action_np[34:41].round(3).tolist()}")
            env_action = torch.as_tensor(env_action_np[None, :], device="cuda:0", dtype=torch.float32)

            # 7f. Step.
            obs, rew, term, trunc, info = env.step(env_action)
            # Flush the RTX render pipeline so the camera annotator delivers THIS
            # step's frame — both for the video below and for the ego view the
            # next chunk's obs_adapter() reads. (Same pattern as play_sonic_adapter.py.)
            _APP.update()
            _APP.update()

            # 7g. Video frames.
            if writers:
                for key, w in writers.items():
                    frame = _read_camera_rgb(env, key)
                    if frame is not None:
                        w.write(frame)

            prev_utm_body_29 = utm_body_29
            chunk_step += 1

            # End the episode on termination OR truncation (time-out).
            done = bool((term[0] if hasattr(term, "ndim") and term.ndim > 0 else term)) or bool(
                (trunc[0] if hasattr(trunc, "ndim") and trunc.ndim > 0 else trunc)
            )
            if done:
                break

        n_succ = int(env.unwrapped.n_successes.sum().item())
        total_successes = n_succ  # cumulative lift-steps, tracked by env reward
        print(f"[episode {ep}] ended at step {step+1}; cumulative lift-steps={n_succ}")

    for w in writers.values():
        w.close()

    env.close()
    _APP.close()

    print(f"\n[play] recorded {args.num_episodes} episode(s); "
          f"cumulative lift-steps={total_successes}")
    if prefix is not None:
        print(f"[play] videos written with prefix: {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
