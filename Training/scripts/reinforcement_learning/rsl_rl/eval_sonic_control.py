"""CSV-driven SONIC encoder+decoder eval in Isaac Lab.

Replaces the kinematic planner with pre-recorded planner_motion CSV data
(from gear_sonic_deploy MotionRecorder / bash deploy.sh --enable-motion-recording)
so the UTM encoder+decoder can be validated in isolation.

CSV format (one directory per recording session):
    joint_pos.csv     — (N, 29) joint angles in SONIC-IsaacLab column order
    body_pos.csv      — (N, 3)  root/pelvis world position xyz
    body_quat.csv     — (N, 4)  root/pelvis world quaternion wxyz  (w,x,y,z)
    joint_vel.csv     — (N, 29) joint velocities in SONIC-IsaacLab order
    body_ang_vel.csv  — (N, 3)  root angular velocity in body frame

Pipeline per env step:

    csv[step] ──▶ anchor pose + lower-body future trajectory (10 frames × 12 joints)
                              │
              anchor + lb_future ──▶ build_encoder_obs (1, 1762)
                              │
                       UtmWrapper.run_encoder → token (64,)
                              │
             token + HistoryBuffer ──▶ build_decoder_obs (1, 994)
                              │
                       UtmWrapper.run_decoder → body_29 (29,)
                              │
             body_29 ──▶ utm_body_29_to_env_27 + zero_fingers → env_action (41,)
                              │
                        env.step(env_action)

Run:
    cd WBCBenchmark/Training && python3 \\
        scripts/reinforcement_learning/rsl_rl/eval_sonic_control.py \\
        --motion-dir /path/to/planner_motion_175248 \\
        --record-video /path/to/output/sonic_control
"""

from __future__ import annotations

import argparse
import builtins
import sys
import time
from functools import partial
from pathlib import Path

print = partial(builtins.print, flush=True)


# =========================================================================
# Phase 1: AppLauncher first (must precede all Isaac Lab imports).
# =========================================================================

def _parse_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSV-driven SONIC encoder+decoder eval — no planner, no VLA"
    )
    parser.add_argument("--task", default="Isaac-Motion-Tracking-Pick-Cam-ContFingers-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Max env steps per episode (capped to CSV length if shorter).")
    parser.add_argument(
        "--motion-dir", required=True, type=Path,
        help="Directory produced by MotionRecorder: contains joint_pos.csv etc.",
    )
    parser.add_argument(
        "--encoder-onnx",
        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx",
    )
    parser.add_argument(
        "--decoder-onnx",
        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx",
    )
    parser.add_argument("--record-video", default=None,
                        help="Output path prefix. Saves _third_person.mp4 and _ego.mp4.")
    parser.add_argument("--video-fps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser


_parser = _parse_cli()

from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(_parser)
_ARGS = _parser.parse_args()
_ARGS.enable_cameras = True

app_launcher = AppLauncher(_ARGS)
_APP = app_launcher.app


# =========================================================================
# Phase 2: Heavy imports.
# =========================================================================

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import PinholeCameraCfg  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from vla_sonic import (  # noqa: E402
    HistoryBuffer,
    UtmWrapper,
    build_decoder_obs,
    build_encoder_obs,
)
from vla_sonic.action_assembler import (  # noqa: E402
    G1_ACTION_SCALE_SONIC,
    G1_DEFAULT_ANGLES_SONIC,
    MUJOCO_TO_ISAACLAB,
    utm_body_29_to_env_27,
)
from vla_sonic.frame_transforms import quat_wxyz_to_xyzw  # noqa: E402


# =========================================================================
# Timing constants (match C++ deploy stack).
# =========================================================================

_CTRL_HZ      = 50.0
_ENC_STEP_50HZ = 5      # 100 ms between encoder lookahead frames
_ENC_FRAMES   = 10      # 0.9 s total lookahead


# =========================================================================
# CSV loader.
# =========================================================================

class CsvMotionData:
    """Load and cache one MotionRecorder session from a directory of CSV files.

    All arrays are in SONIC-IsaacLab column order as written by the C++ recorder.
    MuJoCo-order variants are precomputed once for encoder use.
    """

    def __init__(self, motion_dir: Path) -> None:
        import pandas as pd

        def _read(name: str) -> np.ndarray:
            p = motion_dir / f"{name}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing required file: {p}")
            arr = pd.read_csv(p, header=0).values.astype(np.float32)
            print(f"  [{name}.csv] shape={arr.shape}")
            return arr

        print(f"[csv] loading from {motion_dir.name} …")
        joint_pos_sonic = _read("joint_pos")    # (N, 29) SONIC-IsaacLab order
        body_pos        = _read("body_pos")     # (N, 3)
        body_quat       = _read("body_quat")    # (N, 4)  wxyz
        joint_vel_sonic = _read("joint_vel")    # (N, 29) SONIC-IsaacLab order
        body_ang_vel    = _read("body_ang_vel") # (N, 3)  body frame

        assert joint_pos_sonic.shape[1] == 29
        assert body_pos.shape[1] == 3
        assert body_quat.shape[1] == 4

        # Align lengths — recorder may flush files at slightly different times.
        n = min(
            joint_pos_sonic.shape[0], body_pos.shape[0],
            body_quat.shape[0], joint_vel_sonic.shape[0], body_ang_vel.shape[0],
        )
        if not all(
            a.shape[0] == n
            for a in [joint_pos_sonic, body_pos, body_quat, joint_vel_sonic, body_ang_vel]
        ):
            print(f"  [warn] CSV lengths differ — using min={n}")

        self.joint_pos_sonic = joint_pos_sonic[:n]   # SONIC-IsaacLab order (for history)
        self.body_pos        = body_pos[:n, 0:3]
        self.body_quat       = body_quat[:n, 0:4]    # wxyz
        self.joint_vel_sonic = joint_vel_sonic[:n]   # SONIC-IsaacLab order (for history)
        self.body_ang_vel    = body_ang_vel[:n, 0:3]
        self.n_frames        = n

        # MuJoCo-ordered joint positions for encoder lower-body future.
        # joint_pos_mj[t, mj_idx] = joint_pos_sonic[t, MUJOCO_TO_ISAACLAB[mj_idx]]
        self.joint_pos_mj = self.joint_pos_sonic[:, MUJOCO_TO_ISAACLAB]  # (N, 29)

        # Forward-difference velocities in MuJoCo order, at 50 Hz.
        # vel[t] = (pos[t+1] - pos[t]) * _CTRL_HZ; last frame copies second-to-last.
        jv = np.empty_like(self.joint_pos_mj)
        jv[:-1] = (self.joint_pos_mj[1:] - self.joint_pos_mj[:-1]) * _CTRL_HZ
        jv[-1]  = jv[-2]
        self.joint_vel_mj = jv  # (N, 29) MuJoCo order

        print(
            f"[csv] {n} frames @ {_CTRL_HZ:.0f} Hz  "
            f"({n / _CTRL_HZ:.1f} s)  "
            f"lb_joint_range=[{self.joint_pos_mj[:, :12].min():.3f}, "
            f"{self.joint_pos_mj[:, :12].max():.3f}]  "
            f"root_z=[{self.body_pos[:, 2].min():.3f}, {self.body_pos[:, 2].max():.3f}]"
        )

    def clamp(self, step: int) -> int:
        return min(step, self.n_frames - 1)

    def lb_future(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """Return lower-body joint positions and velocities for the encoder lookahead.

        Extracts 10 frames at 5-step spacing (100 ms/step, 0.9 s total).
        Matches C++ GatherMotionJointPositionsMultiFrame(step_size=5).

        Returns:
            lb_pos: (10, 12) float32 — lower-body angles in MuJoCo order
            lb_vel: (10, 12) float32 — forward-difference velocities in MuJoCo order
        """
        n = self.n_frames
        indices = [min(step + k * _ENC_STEP_50HZ, n - 1) for k in range(_ENC_FRAMES)]
        lb_pos = self.joint_pos_mj[indices, :12].astype(np.float32)
        lb_vel = self.joint_vel_mj[indices, :12].astype(np.float32)
        return lb_pos, lb_vel


# =========================================================================
# Joint-order bridge: Isaac Lab env → SONIC-IsaacLab.
# =========================================================================

# UTM joint name order (SONIC-IsaacLab, 29 joints).
_UTM_29_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "right_hip_pitch_joint",      # 1
    "waist_yaw_joint",            # 2
    "left_hip_roll_joint",        # 3
    "right_hip_roll_joint",       # 4
    "waist_roll_joint",           # 5  (dropped from 27-D env)
    "left_hip_yaw_joint",         # 6
    "right_hip_yaw_joint",        # 7
    "waist_pitch_joint",          # 8  (dropped from 27-D env)
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
assert len(_UTM_29_JOINT_NAMES) == 29


def _build_isaac_to_utm_perm(isaac_joint_names: list[str]) -> np.ndarray:
    """Build permutation: for each SONIC-IsaacLab slot, the Isaac Lab robot joint index."""
    name_to_idx = {n: i for i, n in enumerate(isaac_joint_names)}
    perm = np.full(29, -1, dtype=np.int64)
    missing = []
    for i, name in enumerate(_UTM_29_JOINT_NAMES):
        idx = name_to_idx.get(name, -1)
        if idx < 0:
            missing.append(name)
        else:
            perm[i] = idx
    if missing:
        print(f"[perm] UTM joints absent on Isaac robot (zero-filling): {missing}")
    return perm


def _gather(isaac_values: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Gather Isaac Lab joint values into SONIC-IsaacLab order, filling missing with 0."""
    out = np.zeros(perm.shape[0], dtype=np.float32)
    valid = perm >= 0
    out[valid] = isaac_values[perm[valid]]
    return out


# =========================================================================
# VR 3-point FK helpers (replicates collect_pick_cam.py::_capture_teleop_frame).
# =========================================================================

_NECK_Z_OFFSET = np.float32(0.35)  # torso_link local-Z offset to "neck" point


def _vr_body_indices(robot) -> tuple[int, int, int]:
    """Return (left_wrist_idx, right_wrist_idx, torso_idx) for the robot asset.

    Body order matches training data: left_wrist_yaw_link, right_wrist_yaw_link,
    torso_link (neck = torso + 0.35 m local Z).
    """
    def _find(name: str) -> int:
        ids, names = robot.find_bodies([name])
        if len(ids) != 1:
            raise RuntimeError(f"Expected 1 match for body {name!r}, got {names}")
        return int(ids[0])
    return _find("left_wrist_yaw_link"), _find("right_wrist_yaw_link"), _find("torso_link")


def _vr_from_fk(
    robot,
    root_pos_w: np.ndarray,      # (3,) float32 world
    root_quat_w: np.ndarray,     # (4,) float32 wxyz world
    wrist_l_idx: int,
    wrist_r_idx: int,
    torso_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pelvis-local VR 3-point positions (9,) and rot6d (18,).

    Point order: [left_wrist, right_wrist, neck] — matches training parquet.
    rot6d convention: first two COLUMNS of local rotation matrix, concatenated.
    Neck = torso_link FK + 0.35 m along torso's local Z in pelvis frame.
    """
    R_pelvis_inv = R.from_quat(quat_wxyz_to_xyzw(root_quat_w)).inv()

    def _local(idx: int) -> tuple[np.ndarray, np.ndarray]:
        pos_w  = robot.data.body_pos_w[0, idx].detach().cpu().numpy().astype(np.float32)
        quat_w = robot.data.body_quat_w[0, idx].detach().cpu().numpy().astype(np.float32)
        pos_local = R_pelvis_inv.apply(pos_w - root_pos_w).astype(np.float32)
        R_local   = (R_pelvis_inv * R.from_quat(quat_wxyz_to_xyzw(quat_w))).as_matrix().astype(np.float32)
        return pos_local, R_local

    pos_l, R_l = _local(wrist_l_idx)
    pos_r, R_r = _local(wrist_r_idx)
    pos_t, R_t = _local(torso_idx)

    neck_pos = pos_t + R_t @ np.array([0.0, 0.0, _NECK_Z_OFFSET], dtype=np.float32)

    vr_pos  = np.concatenate([pos_l, pos_r, neck_pos]).astype(np.float32)
    # rot6d: first two columns of rotation matrix (column-major convention)
    vr_rot6d = np.concatenate([
        R_l[:, 0], R_l[:, 1],
        R_r[:, 0], R_r[:, 1],
        R_t[:, 0], R_t[:, 1],
    ]).astype(np.float32)

    return vr_pos, vr_rot6d


# =========================================================================
# Camera helpers (identical to eval_parquet_sonic.py).
# =========================================================================

class VideoWriter:
    def __init__(self, path: Path, fps: int):
        import imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7)
        self.path = path
        self.frame_count = 0

    def write(self, frame_rgb: np.ndarray) -> None:
        self._writer.append_data(frame_rgb)
        self.frame_count += 1

    def close(self) -> None:
        print(f"[video] {self.path.name}: {self.frame_count} frames written")
        self._writer.close()


def _read_camera_rgb(env, key: str) -> np.ndarray | None:
    try:
        cam = env.unwrapped.scene[key]
    except KeyError:
        return None
    try:
        output = cam.data.output
    except Exception:  # noqa: BLE001
        return None
    if "rgb" not in output:
        return None
    rgb = output["rgb"][0, ..., :3]
    if rgb.dtype != torch.uint8:
        rgb = rgb.clamp(0.0, 255.0).to(torch.uint8)
    return rgb.cpu().numpy()


def _inject_cameras(env_cfg) -> None:
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


# =========================================================================
# Main.
# =========================================================================

def main() -> int:
    args = _ARGS

    # --- 1. Load CSV motion data ---------------------------------------
    csv = CsvMotionData(args.motion_dir)
    max_steps = min(args.max_steps, csv.n_frames)
    print(f"[csv] running at most {max_steps} steps")

    # --- 2. Build env --------------------------------------------------
    env_cfg = parse_env_cfg(
        args.task, device="cuda:0", num_envs=args.num_envs, enable_cameras=True,
    )

    # --- 2a. Equilibrate physics to match MuJoCo (gear_sonic_deploy/scene_full.xml) ---
    # MuJoCo uses gravity=-7.5 m/s² (not -9.81), floor friction=0.5, body friction=0.5.
    # IsaacSim defaults differ on all three; mismatches cause systematic under-actuation.
    # NOTE: verify gravity against the actual GR00T scene file if results are unexpected.
    env_cfg.sim.gravity = (0.0, 0.0, -7.5)
    print(f"[physics] gravity overridden to {env_cfg.sim.gravity}")
    try:
        env_cfg.scene.terrain.physics_material.static_friction = 0.5
        env_cfg.scene.terrain.physics_material.dynamic_friction = 0.5
        print("[physics] terrain friction overridden to static=0.5 dynamic=0.5")
    except AttributeError:
        print("[physics] terrain.physics_material not found — skipping friction override")
    try:
        env_cfg.events.physics_material.params["static_friction_range"] = (0.5, 0.5)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.5, 0.5)
        print("[physics] robot body friction overridden to 0.5")
    except (AttributeError, KeyError):
        print("[physics] events.physics_material not found — skipping body friction override")

    _inject_cameras(env_cfg)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    print(f"[env] {args.task}  action_space={env.action_space}")

    # --- 3. Build UTM (encoder + decoder) -----------------------------
    print(f"[utm] encoder={args.encoder_onnx}")
    print(f"[utm] decoder={args.decoder_onnx}")
    utm = UtmWrapper(args.encoder_onnx, args.decoder_onnx)
    print(utm.describe())

    # --- 4. Joint-order permutation (Isaac Lab → SONIC-IsaacLab) ------
    robot = env.unwrapped.scene["robot"]
    isaac_to_utm_perm = _build_isaac_to_utm_perm(list(robot.data.joint_names))

    # --- 4b. VR 3-point body indices (left wrist, right wrist, torso) --
    _vr_l_idx, _vr_r_idx, _vr_t_idx = _vr_body_indices(robot)
    print(f"[vr_fk] body indices: left_wrist={_vr_l_idx} right_wrist={_vr_r_idx} torso={_vr_t_idx}")

    # --- 5. History buffer --------------------------------------------
    history = HistoryBuffer()

    # --- 6. Video writers (opened after first env.reset()) ------------
    writers: dict[str, VideoWriter] = {}
    prefix: Path | None = Path(args.record_video) if args.record_video else None

    # --- 7. Pump Omniverse event loop to let physics/shaders compile --
    print("[sim] pumping Omniverse event loop to let physics initialise…")
    for _ in range(60):
        _APP.update()
        time.sleep(0.5)
    print("[sim] ready")

    action_space_dim = env.action_space.shape[-1]
    zero_action = torch.zeros(
        (args.num_envs, action_space_dim), device="cuda:0", dtype=torch.float32
    )

    # --- 8. Episode loop ----------------------------------------------
    for ep in range(args.num_episodes):
        print(f"\n[episode {ep}]")
        env.reset()
        env.step(zero_action)   # warm-up camera buffer
        _APP.update()
        _APP.update()

        history.reset()
        prev_utm_body_29 = np.zeros(29, dtype=np.float32)

        # Bias-shift: align CSV world-frame positions to sim spawn position.
        # csv.body_pos[0] is wherever the MuJoCo recording started; sim resets
        # to its own spawn point.  All anchor_pos_w values are offset by this
        # constant delta so csv[0] == sim initial root position.
        _sim_init_pos = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        _pos_bias = _sim_init_pos - csv.body_pos[0]
        print(f"[episode {ep}] pos_bias={_pos_bias.round(4).tolist()}  "
              f"sim_init={_sim_init_pos.round(4).tolist()}  "
              f"csv[0]={csv.body_pos[0].round(4).tolist()}")

        # Pre-fill decoder history with CSV frames 0..9 to seed the correct
        # initial gait phase.  With all-zero (standing) history the decoder has
        # no phase signal and outputs symmetric bilateral motion (both feet lift
        # simultaneously).  Seeding from the recording's opening gait state
        # breaks that symmetry before the first real sim step.
        # last_action is approximated by inverting q = default + utm * scale.
        _scale_inv = 1.0 / G1_ACTION_SCALE_SONIC
        for _pre in range(10):
            _ci = min(_pre, csv.n_frames - 1)
            _q_pre   = csv.joint_pos_sonic[_ci]
            _qd_pre  = csv.joint_vel_sonic[_ci]
            _av_pre  = csv.body_ang_vel[_ci]
            _grav_pre = (
                R.from_quat(quat_wxyz_to_xyzw(csv.body_quat[_ci]))
                .inv()
                .apply(np.array([0.0, 0.0, -1.0], dtype=np.float32))
                .astype(np.float32)
            )
            _rp_pre  = csv.body_pos[_ci] + _pos_bias
            _mq_pre  = np.concatenate(
                [_rp_pre, csv.body_quat[_ci], _q_pre[MUJOCO_TO_ISAACLAB]]
            ).astype(np.float32)
            history.push(
                joint_pos=_q_pre - G1_DEFAULT_ANGLES_SONIC,
                joint_vel=_qd_pre,
                last_action=(_q_pre - G1_DEFAULT_ANGLES_SONIC) * _scale_inv,
                base_ang_vel=_av_pre,
                gravity_dir=_grav_pre,
                mujoco_qpos=_mq_pre,
            )
        print(f"[episode {ep}] decoder history pre-filled from CSV frames 0..9")

        # Open video writers on first episode.
        if ep == 0 and prefix is not None and not writers:
            scene_keys = list(env.unwrapped.scene.keys()) if hasattr(env.unwrapped.scene, "keys") else []
            print(f"[video] scene entities: {scene_keys}")
            for key, suffix in [("camera", "_third_person.mp4"), ("camera_robot", "_ego.mp4")]:
                if key in scene_keys:
                    w = VideoWriter(prefix.with_name(prefix.name + suffix), args.video_fps)
                    writers[key] = w
                    print(f"[video] writing {w.path}")
                else:
                    print(f"[video] '{key}' missing — skipping {suffix}")

        for step in range(max_steps):
            csv_step = csv.clamp(step)

            # ----------------------------------------------------------
            # 8a. Read robot state from sim.
            # ----------------------------------------------------------
            q_isaac      = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            qd_isaac     = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
            root_pos_w   = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
            root_quat_w  = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
            root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)

            # SONIC-IsaacLab order (for history and UTM input).
            q_sonic  = _gather(q_isaac,  isaac_to_utm_perm)
            qd_sonic = _gather(qd_isaac, isaac_to_utm_perm)

            # MuJoCo order (for mujoco_qpos history).
            q_mujoco = q_sonic[MUJOCO_TO_ISAACLAB]

            # Gravity direction in body frame.
            gravity_body = (
                R.from_quat(quat_wxyz_to_xyzw(root_quat_w))
                .inv()
                .apply(np.array([0.0, 0.0, -1.0], dtype=np.float32))
                .astype(np.float32)
            )

            mujoco_qpos = np.concatenate(
                [root_pos_w, root_quat_w, q_mujoco]
            ).astype(np.float32)  # (36,)

            # ----------------------------------------------------------
            # 8b. Push robot state into history buffer.
            # ----------------------------------------------------------
            history.push(
                joint_pos=q_sonic - G1_DEFAULT_ANGLES_SONIC,
                joint_vel=qd_sonic,
                last_action=prev_utm_body_29,
                base_ang_vel=root_ang_vel_b,
                gravity_dir=gravity_body,
                mujoco_qpos=mujoco_qpos,
            )

            # ----------------------------------------------------------
            # 8c. Build encoder observation from CSV data.
            # ----------------------------------------------------------
            # Anchor pose: sim robot's live position (anchor_pos_world is unused by
            # build_encoder_obs — accepted for backward compat only).  Using the live
            # position keeps this consistent with C++ where the anchor tracks the robot.
            # CSV orientation is still used for anchor_rot6d (encodes heading error).
            anchor_pos_w     = root_pos_w.copy()
            anchor_quat_wxyz = csv.body_quat[csv_step].copy()
            _csv_ref_pos     = csv.body_pos[csv_step] + _pos_bias  # for tracking diagnostic only

            # Anchor rotation relative to robot's current orientation.
            # Matches C++ PlannerToUtm::ComputeAnchorOrientation.
            _R_robot  = R.from_quat(quat_wxyz_to_xyzw(root_quat_w))
            _R_anchor = R.from_quat(quat_wxyz_to_xyzw(anchor_quat_wxyz))
            _R_rel    = (_R_robot.inv() * _R_anchor).as_matrix().astype(np.float32)
            # SONIC rot6d: row-wise flatten of first 2 columns (C++ line 677-683).
            # C++ stores [R₀₀,R₀₁, R₁₀,R₁₁, R₂₀,R₂₁] — row-major.
            # Must use reshape(-1) / flatten("C"), NOT flatten("F") (column-major).
            anchor_rot6d = _R_rel[:, :2].reshape(-1).astype(np.float32)

            # Lower-body future trajectory from CSV.
            lb_pos, lb_vel = csv.lb_future(csv_step)  # (10, 12) each

            # VR 3-point: FK-computed pelvis-local wrist + neck poses.
            # Matches collect_pick_cam.py::_capture_teleop_frame convention.
            # Passing zeros was out-of-distribution for the encoder.
            vr_pos, vr_rot6d = _vr_from_fk(
                robot, root_pos_w, root_quat_w,
                _vr_l_idx, _vr_r_idx, _vr_t_idx,
            )

            enc_obs = build_encoder_obs(
                anchor_pos_world=anchor_pos_w,
                anchor_quat_wxyz=anchor_quat_wxyz,
                anchor_rot6d=anchor_rot6d,
                lower_body_positions_future=lb_pos,
                lower_body_velocities_future=lb_vel,
                vr_3pt_position_anchor_local=vr_pos,
                vr_3pt_rot6d=vr_rot6d,
            )  # (1, 1762)

            # ----------------------------------------------------------
            # 8d. Run encoder → token.
            # ----------------------------------------------------------
            token = utm.run_encoder({"obs_dict": enc_obs}).reshape(-1)  # (64,)

            # ----------------------------------------------------------
            # 8e. Run decoder → body_29.
            # ----------------------------------------------------------
            dec_hist    = history.decoder_history()
            dec_obs     = build_decoder_obs(token_state=token, **dec_hist.as_kwargs())  # (1, 994)
            utm_body_29 = utm.run_decoder({"obs_dict": dec_obs}).reshape(-1)  # (29,)

            # ----------------------------------------------------------
            # 8f. Assemble env action: UTM body (27) + zero fingers (14).
            # ----------------------------------------------------------
            body_27 = utm_body_29_to_env_27(utm_body_29)  # (27,)
            env_action_np = np.concatenate(
                [body_27, np.zeros(14, dtype=np.float32)]
            ).astype(np.float32)  # (41,)

            # ----------------------------------------------------------
            # Diagnostics for first few steps.
            # ----------------------------------------------------------
            if step < 3:
                csv_lb = csv.joint_pos_mj[csv_step, :12]
                sim_lb = q_mujoco[:12]
                lb_err = np.abs(csv_lb - sim_lb)
                print(f"\n[step {step}] csv_step={csv_step}")
                print(f"  csv_ref_pos     = {_csv_ref_pos.round(4).tolist()}")
                print(f"  anchor_rot6d    = {anchor_rot6d.round(4).tolist()}")
                print(f"  lb_pos[0]       = {lb_pos[0].round(4).tolist()}")
                print(f"  token[:8]       = {token[:8].round(3).tolist()}")
                print(f"  utm_body_29[:12]= {utm_body_29[:12].round(3).tolist()}")
                print(f"  body_27[:12]    = {body_27[:12].round(3).tolist()}")
                print(f"  lb_tracking_err = max={lb_err.max():.4f} mean={lb_err.mean():.4f}")
            elif step % 100 == 0:
                csv_lb = csv.joint_pos_mj[csv_step, :12]
                sim_lb = q_mujoco[:12]
                lb_err = np.abs(csv_lb - sim_lb)
                root_err = np.linalg.norm(_csv_ref_pos - root_pos_w)
                print(
                    f"[step {step:4d}] "
                    f"lb_err max={lb_err.max():.4f} mean={lb_err.mean():.4f}  "
                    f"root_pos_err={root_err:.4f}  "
                    f"token_norm={np.linalg.norm(token):.3f}"
                )

            # ----------------------------------------------------------
            # 8g. Step sim.
            # ----------------------------------------------------------
            env_action = torch.as_tensor(
                env_action_np[None, :], device="cuda:0", dtype=torch.float32
            )
            _, rew, term, *_ = env.step(env_action)

            # Flush RTX render pipeline.
            _APP.update()
            _APP.update()

            # ----------------------------------------------------------
            # 8h. Video frames.
            # ----------------------------------------------------------
            if writers:
                _prev = getattr(main, "_prev_cam_frame", {})
                for key, w in writers.items():
                    frame = _read_camera_rgb(env, key)
                    if frame is not None:
                        if step < 5 and key in _prev:
                            diff = int(
                                np.abs(
                                    frame.astype(np.int32) - _prev[key].astype(np.int32)
                                ).max()
                            )
                            print(f"[step {step}] camera '{key}' max_pixel_diff={diff}")
                        _prev[key] = frame.copy()
                        w.write(frame)
                main._prev_cam_frame = _prev

            prev_utm_body_29 = utm_body_29

            if bool(term[0] if term.ndim > 0 else term):
                print(f"[step {step}] terminated  rew={float(rew[0]):.4f}")
                break

        print(f"[episode {ep}] ended at step {step + 1}")

    for w in writers.values():
        w.close()

    env.close()
    _APP.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
