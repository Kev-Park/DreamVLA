"""CSV-driven SONIC encoder+decoder eval in Isaac Lab.

Reads @out/sonic_input.csv — a recording of keyboard-driven locomotion commands
from the SONIC C++ MuJoCo stack — and replays them through the full
planner → encoder → decoder → env pipeline in Isaac Lab.

The planner is confirmed working (lower-body joint targets match MuJoCo output).
This script isolates the encoder+decoder for debugging by feeding the same
verified planner inputs in IsaacSim, bypassing the VLA and parquet machinery.

CSV column format (g1_deploy_onnx_ref.cpp lines 3514-3532):
  [0]  motion_index
  [1]  current_frame
  [2]  operator_state.play
  [3]  operator_state.start
  [4]  operator_state.stop
  [5]  planner_state.enabled
  [6]  planner_state.initialized
  [7]  locomotion_mode   (LocomotionMode int: 0=IDLE, 1=SLOW_WALK, ...)
  [8]  movement_direction[0]
  [9]  movement_direction[1]
  [10] movement_direction[2]
  [11] facing_direction[0]
  [12] facing_direction[1]
  [13] facing_direction[2]
  [14] movement_speed    (-1 = mode default)
  [15] height            (-1 = mode default)

Pipeline per env step:
    csv_row[step] ──▶ PlannerInputs (mode, movement, facing, speed, height)
                                │
                         PlannerWrapper.run → mujoco_qpos (lower-body traj)
                                │
         anchor + lb_traj + default_vr_3pt ──▶ build_encoder_obs
                                │
                          UtmWrapper.run_encoder → token (64-D)
                                │
              token + HistoryBuffer ──▶ build_decoder_obs
                                │
                          UtmWrapper.run_decoder → body_29
                                │
               utm_body_29_to_env_27 + zero_fingers → env.step(action_41)

Run (on remote machine):
    cd WBCBenchmark/Training && python3 \\
        scripts/reinforcement_learning/rsl_rl/eval_sonic_csv.py \\
        --csv /path/to/sonic_input.csv \\
        --record-video /path/to/out/sonic_csv_debug
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
# Phase 1: AppLauncher first (must happen before any Isaac imports).
# =========================================================================

def _parse_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CSV-driven SONIC encoder+decoder eval (no VLA checkpoint needed)"
    )
    parser.add_argument("--task", default="Isaac-Motion-Tracking-Pick-Cam-ContFingers-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=2000,
                        help="Capped to CSV length if shorter.")
    parser.add_argument("--csv", required=True, type=Path,
                        help="Path to sonic_input.csv recorded from SONIC C++ deploy.")
    parser.add_argument("--encoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx")
    parser.add_argument("--decoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")
    parser.add_argument("--planner-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx")
    parser.add_argument("--record-video", default=None,
                        help="Output path prefix.  Appends _third_person.mp4 and _ego.mp4.")
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
    PlannerWrapper,
    UtmWrapper,
    build_decoder_obs,
    build_encoder_obs,
)
from vla_sonic.action_assembler import (  # noqa: E402
    MUJOCO_TO_ISAACLAB,
    utm_body_29_to_env_27,
)
from vla_sonic.frame_transforms import quat_wxyz_to_xyzw  # noqa: E402


# =========================================================================
# Planner bootstrap constants (mirrors eval_parquet_sonic.py).
# =========================================================================

_MUJOCO_DEFAULT_ANGLES_29 = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left  leg: hp, hr, hy, knee, ap, ar
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg: hp, hr, hy, knee, ap, ar
     0.0,   0.0, 0.0,                        # waist: yaw, roll, pitch
     0.2,   0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left  arm: sp, sr, sy, elbow, wr, wp, wy
     0.2,  -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm: sp, sr, sy, elbow, wr, wp, wy
], dtype=np.float32)
_MUJOCO_STANDING_Z = 0.78874

_PLANNER_HZ      = 30.0
_CTRL_HZ         = 50.0
_REPLAN_HZ       = 10.0
_REPLAN_STEPS    = int(_CTRL_HZ / _REPLAN_HZ)        # 5 env steps per replan
_LOOK_AHEAD_50HZ = 2
_CTX_START_50HZ  = _REPLAN_STEPS + _LOOK_AHEAD_50HZ  # 7
_CTX_SPACING_50HZ= _CTRL_HZ / _PLANNER_HZ            # ≈ 1.667
_ENC_STEP_50HZ   = 5
_ENC_FRAMES      = 10

# VR 3-point defaults for keyboard locomotion (no headset → neutral arm pose).
# Positions in anchor-local (≈ pelvis-local) frame: [lw, rw, neck] × xyz.
# Identity 6D rotation: col0=[1,0,0], col1=[0,1,0] for each of the 3 points.
_DEFAULT_VR_3PT_POS   = np.array([
     0.0,  0.25, 0.0,   # left  wrist
     0.0, -0.25, 0.0,   # right wrist
     0.0,  0.0,  0.5,   # neck
], dtype=np.float32)  # (9,)
_DEFAULT_VR_3PT_ROT6D = np.tile(
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3
).astype(np.float32)  # (18,)


# =========================================================================
# Planner context helpers (identical to eval_parquet_sonic.py).
# =========================================================================

def _make_robot_planner_context(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,  # noqa: ARG001
    q_mujoco: np.ndarray,
) -> np.ndarray:
    frame = np.zeros(36, dtype=np.float32)
    frame[0:3] = np.array([0.0, 0.0, float(root_pos[2])], dtype=np.float32)
    frame[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    frame[7:36] = q_mujoco.astype(np.float32)
    return np.tile(frame[np.newaxis, np.newaxis, :], (1, 4, 1))


def _quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0.astype(np.float64)
    q1 = q1.astype(np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)).astype(np.float32)
    theta0 = float(np.arccos(dot))
    theta  = theta0 * t
    sin0   = float(np.sin(theta0))
    return ((np.sin(theta0 - theta) / sin0) * q0 + (np.sin(theta) / sin0) * q1).astype(np.float32)


def _interp_planner_frame(frames: np.ndarray, t_float: float) -> np.ndarray:
    n = frames.shape[0]
    i0 = int(t_float)
    frac = t_float - i0
    i0 = min(max(i0, 0), n - 1)
    i1 = min(i0 + 1, n - 1)
    if i0 == i1:
        frac = 0.0
    f0, f1 = frames[i0], frames[i1]
    out = np.empty(36, dtype=np.float32)
    out[0:3]  = f0[0:3]  + frac * (f1[0:3]  - f0[0:3])
    out[3:7]  = _quat_slerp_wxyz(f0[3:7], f1[3:7], frac)
    out[7:36] = f0[7:36] + frac * (f1[7:36] - f0[7:36])
    return out


def _resample_planner_to_50hz(qpos_30hz: np.ndarray) -> np.ndarray:
    frames_30 = qpos_30hz[0]
    t30 = frames_30.shape[0]
    t50 = max(1, int(t30 * _CTRL_HZ / _PLANNER_HZ))
    out = np.stack(
        [_interp_planner_frame(frames_30, i * _PLANNER_HZ / _CTRL_HZ) for i in range(t50)]
    )
    return out[np.newaxis]  # (1, T_50, 36)


# =========================================================================
# Joint-order helpers (identical to eval_parquet_sonic.py).
# =========================================================================

UTM_29_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]
assert len(UTM_29_JOINT_NAMES) == 29

PLANNER_ROOT_POS_SLICE  = slice(0, 3)
PLANNER_ROOT_QUAT_SLICE = slice(3, 7)
LOWER_BODY_QPOS_INDICES_MUJOCO_ORDER = np.array(
    [7 + i for i in range(12)], dtype=np.int64,
)


def build_isaac_to_utm_perm(isaac_joint_names: list[str]) -> np.ndarray:
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
    out = np.zeros(perm.shape[0], dtype=np.float32)
    valid = perm >= 0
    out[valid] = isaac_values[perm[valid]]
    return out


# =========================================================================
# Encoder obs helper (mirrors eval_parquet_sonic.py's per-step logic).
# =========================================================================

def _build_enc_obs_from_planner(
    cached_out_50hz: np.ndarray,
    planner_step: int,
    root_quat_w: np.ndarray,
) -> np.ndarray:
    """Build encoder obs from the current 50 Hz planner buffer frame.

    Args:
        cached_out_50hz: (1, T_50, 36) 50 Hz resampled planner output.
        planner_step:    Current 50 Hz frame index within the replan cycle [0,4].
        root_quat_w:     Current robot root quaternion (wxyz, world frame).

    Returns:
        (1, 1762) float32 encoder obs array.
    """
    from scipy.spatial.transform import Rotation as R

    _t50       = cached_out_50hz.shape[1]
    _frames_50 = cached_out_50hz[0]
    _cur       = min(planner_step, _t50 - 1)

    anchor_pos_w      = _frames_50[_cur, PLANNER_ROOT_POS_SLICE].copy()
    anchor_quat_wxyz  = _frames_50[_cur, PLANNER_ROOT_QUAT_SLICE].copy()

    _R_robot  = R.from_quat(quat_wxyz_to_xyzw(root_quat_w))
    _R_anchor = R.from_quat(quat_wxyz_to_xyzw(anchor_quat_wxyz))
    _R_rel_mat = (_R_robot.inv() * _R_anchor).as_matrix().astype(np.float32)
    anchor_rot6d = _R_rel_mat[:, :2].flatten("F").astype(np.float32)

    _lb_idx    = [min(planner_step + k * _ENC_STEP_50HZ, _t50 - 1) for k in range(_ENC_FRAMES)]
    _lb_frames = _frames_50[_lb_idx]
    lb_pos = _lb_frames[:, LOWER_BODY_QPOS_INDICES_MUJOCO_ORDER].astype(np.float32)

    _lb_dense     = _frames_50[:, LOWER_BODY_QPOS_INDICES_MUJOCO_ORDER].astype(np.float32)
    _lb_vel_dense = np.empty_like(_lb_dense)
    _lb_vel_dense[:-1] = (_lb_dense[1:] - _lb_dense[:-1]) * _CTRL_HZ
    _lb_vel_dense[-1]  = _lb_vel_dense[-2]
    lb_vel = _lb_vel_dense[_lb_idx]

    return build_encoder_obs(
        anchor_pos_world                = anchor_pos_w,
        anchor_quat_wxyz                = anchor_quat_wxyz,
        anchor_rot6d                    = anchor_rot6d,
        lower_body_positions_future     = lb_pos,
        lower_body_velocities_future    = lb_vel,
        vr_3pt_position_anchor_local    = _DEFAULT_VR_3PT_POS,
        vr_3pt_rot6d                    = _DEFAULT_VR_3PT_ROT6D,
    )


# =========================================================================
# CSV reader for sonic_input.csv.
# =========================================================================

class SonicCsvReader:
    """Parse and replay the SONIC C++ input recording.

    Each row is one 50 Hz control tick.  Columns:
      [0] motion_index   [1] current_frame
      [2] play           [3] start          [4] stop
      [5] planner_enabled   [6] planner_initialized
      [7] locomotion_mode
      [8-10] movement_direction   [11-13] facing_direction
      [14] movement_speed         [15] height
    """

    def __init__(self, path: Path) -> None:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip().rstrip(",")
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 16:
                    continue  # skip short / malformed lines
                rows.append(parts)
        self.n_rows = len(rows)
        self._rows  = rows
        print(f"[csv] loaded {self.n_rows} rows from {path.name}")

    def get_row(self, step: int) -> dict:
        """Return parsed dict for row `step` (clamped to valid range)."""
        idx = min(max(step, 0), self.n_rows - 1)
        p   = self._rows[idx]
        return {
            "motion_index":         int(p[0]),
            "current_frame":        int(p[1]),
            "play":                 bool(int(p[2])),
            "start":                bool(int(p[3])),
            "stop":                 bool(int(p[4])),
            "planner_enabled":      bool(int(p[5])),
            "planner_initialized":  bool(int(p[6])),
            "locomotion_mode":      int(p[7]),
            "movement_direction":   np.array([float(p[8]),  float(p[9]),  float(p[10])], dtype=np.float32),
            "facing_direction":     np.array([float(p[11]), float(p[12]), float(p[13])], dtype=np.float32),
            "movement_speed":       float(p[14]),
            "height":               float(p[15]),
        }

    def first_active_step(self) -> int:
        """Return index of first row where planner is enabled and initialized."""
        for i, p in enumerate(self._rows):
            if int(p[5]) == 1 and int(p[6]) == 1:
                return i
        return 0


# =========================================================================
# Video writer (identical to eval_parquet_sonic.py).
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


# =========================================================================
# Camera helpers (identical to eval_parquet_sonic.py).
# =========================================================================

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

    # --- 1. Load CSV ----------------------------------------------------
    csv_reader  = SonicCsvReader(args.csv)
    first_active = csv_reader.first_active_step()
    max_steps   = min(args.max_steps_per_episode, csv_reader.n_rows - first_active)
    print(f"[csv] first active step (planner enabled+initialized): {first_active}")
    print(f"[csv] running {max_steps} steps from active start")

    # --- 2. Build env ---------------------------------------------------
    env_cfg = parse_env_cfg(
        args.task, device="cuda:0", num_envs=args.num_envs, enable_cameras=True,
    )
    _inject_cameras(env_cfg)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    print(f"[env] {args.task}  action_space={env.action_space}")

    action_space_dim = env.action_space.shape[-1]
    zero_action      = torch.zeros((args.num_envs, action_space_dim),
                                   device="cuda:0", dtype=torch.float32)

    # --- 3. Build SONIC wrappers ----------------------------------------
    print(f"[utm] encoder={args.encoder_onnx}")
    print(f"[utm] decoder={args.decoder_onnx}")
    utm     = UtmWrapper(args.encoder_onnx, args.decoder_onnx)
    planner = PlannerWrapper(args.planner_onnx)
    print(utm.describe())

    # --- 4. Joint-order permutation -------------------------------------
    robot = env.unwrapped.scene["robot"]
    isaac_to_utm_perm = build_isaac_to_utm_perm(list(robot.data.joint_names))

    # --- 5. History buffer ----------------------------------------------
    history = HistoryBuffer()

    # --- 6. Pump Omniverse event loop before env.reset() ----------------
    print("[sim] pumping Omniverse event loop to let physics initialise...")
    for _i in range(60):
        _APP.update()
        time.sleep(0.5)
    print("[sim] ready")

    # --- 7. Video writers (created after Isaac Lab fully up) ------------
    writers: dict[str, VideoWriter] = {}
    prefix: Path | None = Path(args.record_video) if args.record_video else None

    # --- 8. Rollout -----------------------------------------------------
    prev_utm_body_29 = np.zeros(29, dtype=np.float32)

    planner_context: np.ndarray | None = None
    cached_planner_out  = None
    cached_out_50hz: np.ndarray | None = None

    from scipy.spatial.transform import Rotation as R

    for ep in range(args.num_episodes):
        print(f"\n[episode {ep}]")
        obs, info = env.reset()
        env.step(zero_action)
        _APP.update()
        _APP.update()
        history.reset()
        planner_step        = 0
        cached_planner_out  = None
        cached_out_50hz     = None

        # Bootstrap planner context from actual robot state.
        _q_b  = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        _qs_b = _gather_with_mask(_q_b, isaac_to_utm_perm)
        _qm_b = _qs_b[MUJOCO_TO_ISAACLAB]
        _rp_b = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        _rq_b = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
        planner_context = _make_robot_planner_context(_rp_b, _rq_b, _qm_b)

        # Open video writers on first episode.
        if ep == 0 and prefix is not None and not writers:
            scene_keys = list(env.unwrapped.scene.keys()) \
                if hasattr(env.unwrapped.scene, "keys") else []
            print(f"[video] scene entities: {scene_keys}")
            if "camera" in scene_keys:
                writers["camera"] = VideoWriter(
                    prefix.with_name(prefix.name + "_third_person.mp4"), args.video_fps)
            if "camera_robot" in scene_keys:
                writers["camera_robot"] = VideoWriter(
                    prefix.with_name(prefix.name + "_ego.mp4"), args.video_fps)
            for w in writers.values():
                print(f"[video] writing {w.path}")

        prev_utm_body_29 = np.zeros(29, dtype=np.float32)

        for step in range(max_steps):
            csv_step = first_active + step
            row = csv_reader.get_row(csv_step)

            # 8a. Push current env state into history.
            q_isaac       = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            qd_isaac      = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
            q_sonic       = _gather_with_mask(q_isaac,  isaac_to_utm_perm)
            qd_sonic      = _gather_with_mask(qd_isaac, isaac_to_utm_perm)
            q_mujoco      = q_sonic[MUJOCO_TO_ISAACLAB]
            root_pos_w    = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
            root_quat_w   = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
            root_ang_vel_b= robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
            gravity_body  = R.from_quat(quat_wxyz_to_xyzw(root_quat_w)).inv().apply(
                np.array([0.0, 0.0, -1.0], dtype=np.float32)
            ).astype(np.float32)
            mujoco_qpos   = np.concatenate([root_pos_w, root_quat_w, q_mujoco]).astype(np.float32)

            history.push(
                joint_pos   = q_sonic,
                joint_vel   = qd_sonic,
                last_action = prev_utm_body_29,
                base_ang_vel= root_ang_vel_b,
                gravity_dir = gravity_body,
                mujoco_qpos = mujoco_qpos,
            )

            # 8b. Replan at 10 Hz (every _REPLAN_STEPS env steps).
            if planner_step == 0:
                # Update planner context from previous 50 Hz buffer.
                if cached_out_50hz is not None:
                    ctx_frames = []
                    for n in range(4):
                        t_f = _CTX_START_50HZ + n * _CTX_SPACING_50HZ
                        ctx_frames.append(_interp_planner_frame(cached_out_50hz[0], t_f))
                    planner_context = np.stack(ctx_frames)[np.newaxis]  # (1, 4, 36)

                # Build PlannerInputs directly from CSV row (no VLA needed).
                mode     = row["locomotion_mode"]
                mov_dir  = row["movement_direction"].reshape(1, 3)
                fac_dir  = row["facing_direction"].reshape(1, 3)
                speed    = row["movement_speed"]
                height   = row["height"]

                # Normalise facing direction (defensive).
                fac_norm = float(np.linalg.norm(fac_dir[0, :2]))
                if fac_norm > 1e-6:
                    fac_dir[0, :2] /= fac_norm
                else:
                    fac_dir[0] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

                # Use mode default if speed == -1 (same as C++ behaviour).
                if speed < 0.0:
                    speed = {1: 0.2, 2: 1.2, 3: 2.5}.get(mode, 0.0)

                cached_planner_out = planner.run(
                    context_mujoco_qpos = planner_context,
                    target_vel          = np.array([speed], dtype=np.float32),
                    mode                = np.array([mode],  dtype=np.int64),
                    movement_direction  = mov_dir.astype(np.float32),
                    facing_direction    = fac_dir.astype(np.float32),
                    height              = np.array([height], dtype=np.float32),
                )
                cached_out_50hz = _resample_planner_to_50hz(cached_planner_out.mujoco_qpos)

                if step == 0:
                    out_first = cached_planner_out.mujoco_qpos[0, 0]
                    print(f"\n[PLANNER @ step 0] mode={mode}  speed={speed:.3f}")
                    print(f"  movement={mov_dir[0].round(3).tolist()}  facing={fac_dir[0].round(3).tolist()}")
                    print(f"  out[0] root_pos  = {out_first[0:3].round(4).tolist()}")
                    print(f"  out[0] root_quat = {out_first[3:7].round(4).tolist()}")
                    print(f"  out[0] legs[:12] = {out_first[7:19].round(4).tolist()}")

            # 8c. Encoder → token.
            enc_obs = _build_enc_obs_from_planner(cached_out_50hz, planner_step, root_quat_w)
            token   = utm.run_encoder({"obs_dict": enc_obs}).reshape(-1)

            # 8d. Decoder → body_29.
            dec_hist    = history.decoder_history()
            dec_obs     = build_decoder_obs(token_state=token, **dec_hist.as_kwargs())
            utm_body_29 = utm.run_decoder({"obs_dict": dec_obs}).reshape(-1)

            # 8e. Assemble env action (body only; zeros for fingers).
            body_27      = utm_body_29_to_env_27(utm_body_29)
            n_fingers    = action_space_dim - 27
            if n_fingers > 0:
                env_action_np = np.concatenate(
                    [body_27, np.zeros(n_fingers, dtype=np.float32)], axis=0
                )
            else:
                env_action_np = body_27

            if step < 3:
                print(f"[step {step}] mode={row['locomotion_mode']}  speed={row['movement_speed']:.2f}")
                print(f"[step {step}]   utm_body_29[:15]     = {utm_body_29[:15].round(3).tolist()}")
                print(f"[step {step}]   env body[:12] (legs) = {env_action_np[:12].round(3).tolist()}")

            env_action = torch.as_tensor(
                env_action_np[None, :], device="cuda:0", dtype=torch.float32
            )

            # 8f. Step env.
            obs, rew, term, trunc, info = env.step(env_action)

            # 8g. Flush RTX render pipeline.
            _APP.update()
            _APP.update()

            # 8h. Video frames.
            if writers:
                _prev_frame = getattr(main, "_prev_cam_frame", {})
                for key, w in writers.items():
                    frame = _read_camera_rgb(env, key)
                    if frame is not None:
                        _prev_frame[key] = frame.copy()
                        w.write(frame)
                main._prev_cam_frame = _prev_frame

            prev_utm_body_29 = utm_body_29
            planner_step = (planner_step + 1) % _REPLAN_STEPS

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
