"""Parquet-driven SONIC closed-loop eval in Isaac Lab.

Identical pipeline to eval_vla_sonic.py, but the VLA is replaced by a
precomputed trajectory streamed from a LeRobot parquet file.  Use this to
validate the kinematic planner + UTM encoder/decoder in isolation — no VLA
checkpoint required.

Pipeline per env step:

    parquet[step] ──▶ fake_vla_chunk (planner cmds + vr_3pt + fingers)
                                │
        fake_vla_chunk ──▶ PlannerWrapper ──▶ mujoco_qpos
                                │
        anchor + lb_trajectory + vr_3pt ──▶ build_encoder_obs
                                │
                         UtmWrapper.run_encoder → token
                                │
              token + HistoryBuffer ──▶ build_decoder_obs
                                │
                         UtmWrapper.run_decoder → body_29
                                │
        body_29 + parquet_fingers ──▶ utm_plus_vla_to_env_action → env_action_41
                                │
                          env.step(env_action_41)

Run:

    cd WBCBenchmark/Training && python3 scripts/reinforcement_learning/rsl_rl/eval_parquet_sonic.py \\
        --parquet /path/to/episode_000000.parquet \\
        --num-episodes 1 \\
        --record-video /home/dvij/kevin/eval_videos/parquet_ep0
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
# Phase 1: AppLauncher first.
# =========================================================================

def _parse_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parquet-driven SONIC closed-loop eval (no VLA checkpoint needed)"
    )
    parser.add_argument("--task", default="Isaac-Motion-Tracking-Pick-Cam-ContFingers-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=500,
                        help="Capped automatically to parquet length if shorter.")
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="How many parquet frames to read per replan (mirrors VLA chunk size).")
    parser.add_argument("--parquet", required=True, type=Path,
                        help="Path to a single LeRobot episode_*.parquet file.")
    parser.add_argument("--encoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx")
    parser.add_argument("--decoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")
    parser.add_argument("--planner-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx")
    parser.add_argument("--record-video", default=None,
                        help="Output prefix. Saves _third_person.mp4, _ego.mp4, _vla_skeleton.mp4.")
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
    build_planner_inputs,
    utm_plus_vla_to_env_action,
)
from scipy.spatial.transform import Rotation as R  # noqa: E402
from vla_sonic.action_assembler import (  # noqa: E402
    G1_ACTION_SCALE_SONIC,
    G1_DEFAULT_ANGLES_SONIC,
    MUJOCO_TO_ISAACLAB,
)
from vla_sonic.frame_transforms import quat_wxyz_to_xyzw  # noqa: E402


# =========================================================================
# Planner bootstrap constants (mirrors eval_parquet_kinematic.py).
# =========================================================================

_PLANNER_HZ           = 30.0
_PARQUET_HZ           = 50.0   # parquet recording rate — same as control rate
_CTRL_HZ              = 50.0
_REPLAN_HZ            = 10.0
_REPLAN_STEPS         = int(_CTRL_HZ / _REPLAN_HZ)      # 5 env steps per replan
# C++ resamples the planner 30 Hz output to a 50 Hz MotionSequence buffer, then
# UpdateContextFromMotion samples at gen_frame_ = replan_steps(5) + look_ahead(2) = 7,
# spacing 50/30 ≈ 1.667 frames between the 4 context frames (= 1/30 s at 50 Hz rate).
_LOOK_AHEAD_50HZ      = 2                               # C++ motion_look_ahead_steps
_CTX_START_50HZ       = _REPLAN_STEPS + _LOOK_AHEAD_50HZ  # 7 — 50 Hz start frame for context
_CTX_SPACING_50HZ     = _CTRL_HZ / _PLANNER_HZ          # ≈ 1.667 50 Hz frames between 30 Hz slots
# Encoder lookahead: step_size=5 at 50 Hz (matches C++ GatherMotionJointPositionsMultiFrame),
# giving 100 ms/step, 10 frames, 0.9 s total lookahead.
_ENC_STEP_50HZ        = 5
_ENC_FRAMES           = 10
_ENC_GRAD_DT          = _ENC_STEP_50HZ / _CTRL_HZ       # 0.1 s between selected frames


def _make_robot_planner_context(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,  # noqa: ARG001 — kept for call-site clarity
    q_mujoco: np.ndarray,
) -> np.ndarray:
    """Bootstrap: 4 identical frames of current robot state → (1, 4, 36).

    Matches C++ InitializeContext exactly: XY position clamped to origin,
    identity quaternion (yaw = 0), actual joint positions.
    """
    # C++ InitializeContext uses (x=0, y=0, z=actual_height) and identity quaternion —
    # not the actual XY position or yaw-zeroed quat. This normalises the planner frame
    # to the origin at startup regardless of where the robot spawned.
    frame = np.zeros(36, dtype=np.float32)
    frame[0:3] = np.array([0.0, 0.0, float(root_pos[2])], dtype=np.float32)
    frame[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # identity quaternion
    frame[7:36] = q_mujoco.astype(np.float32)
    return np.tile(frame[np.newaxis, np.newaxis, :], (1, 4, 1))


def _quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Slerp between two wxyz quaternions. Matches C++ quat_slerp_d."""
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
    """Lerp/slerp one frame from a (T, 36) array at fractional index t_float.

    Position (0:3) and joints (7:36) use lerp; quaternion wxyz (3:7) uses slerp.
    Out-of-range t_float clamps to [0, T-1].
    """
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
    """Resample (1, T_30, 36) planner output → (1, T_50, 36) at 50 Hz.

    Matches the C++ deploy stack: lerp for position/joints, slerp for
    quaternion. T_50 = floor(T_30 / 30 * 50) — same as C++ std::floor.
    """
    frames_30 = qpos_30hz[0]
    t30 = frames_30.shape[0]
    t50 = max(1, int(t30 * _CTRL_HZ / _PLANNER_HZ))
    out = np.stack(
        [_interp_planner_frame(frames_30, i * _PLANNER_HZ / _CTRL_HZ) for i in range(t50)]
    )  # (T_50, 36)
    return out[np.newaxis]  # (1, T_50, 36)


# =========================================================================
# Joint-order helpers (identical to eval_vla_sonic.py).
# =========================================================================

UTM_29_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "right_hip_pitch_joint",      # 1
    "waist_yaw_joint",            # 2
    "left_hip_roll_joint",        # 3
    "right_hip_roll_joint",       # 4
    "waist_roll_joint",           # 5  dropped
    "left_hip_yaw_joint",         # 6
    "right_hip_yaw_joint",        # 7
    "waist_pitch_joint",          # 8  dropped
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
# Planner output decomposition (identical to eval_vla_sonic.py).
# =========================================================================

PLANNER_ROOT_POS_SLICE  = slice(0, 3)
PLANNER_ROOT_QUAT_SLICE = slice(3, 7)
PLANNER_JOINTS_SLICE    = slice(7, 36)

LOWER_BODY_QPOS_INDICES_MUJOCO_ORDER = np.array(
    [7 + i for i in range(12)], dtype=np.int64,
)


def extract_vr_3pt(
    vla_action: dict, *, t_index: int = 0, batch_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(vla_action["vr_3pt_position"],    dtype=np.float32)
    orn = np.asarray(vla_action["vr_3pt_orientation"], dtype=np.float32)
    return pos[batch_index, t_index].copy(), orn[batch_index, t_index].copy()


# =========================================================================
# Parquet action streamer — replaces Gr00tPolicy.
# =========================================================================

# Parquet column → vla_chunk key.  All values are (B=1, T=chunk_size, D).
_PARQUET_COL_MAP: dict[str, tuple[str, type]] = {
    "teleop.planner_movement":   ("planner_movement",   np.float32),  # (3,)
    "teleop.planner_facing":     ("planner_facing",     np.float32),  # (3,)
    "teleop.planner_speed":      ("planner_speed",      np.float32),  # (1,)
    "teleop.planner_height":     ("planner_height",     np.float32),  # (1,)
    "teleop.vr_3pt_position":    ("vr_3pt_position",    np.float32),  # (9,)
    "teleop.vr_3pt_orientation": ("vr_3pt_orientation", np.float32),  # (18,)
    "teleop.left_hand_joints":   ("left_hand_joints",   np.float32),  # (7,)
    "teleop.right_hand_joints":  ("right_hand_joints",  np.float32),  # (7,)
}

# Optional columns — loaded when present, ignored when absent.
_PARQUET_OPT_MAP: dict[str, tuple[str, type]] = {
    "teleop.planner_mode": ("planner_mode", np.int64),    # explicit mode int
    "teleop.run":          ("run",          np.float32),  # boolean RUN override
}


class ParquetActionStreamer:
    """Streams ground-truth teleop actions from a LeRobot parquet file.

    Returns dicts shaped (B=1, T=chunk_size, D) — the same format that
    Gr00tPolicy.get_action() produces — so build_planner_inputs,
    extract_vr_3pt, and utm_plus_vla_to_env_action work unchanged.
    """

    def __init__(self, path: Path) -> None:
        import pandas as pd

        df = pd.read_parquet(path)
        self.n_frames = len(df)
        self._arrays: dict[str, np.ndarray] = {}

        def _cell(val, dtype):
            arr = val.astype(dtype) if isinstance(val, np.ndarray) else np.array(val, dtype=dtype)
            return np.atleast_1d(arr)

        missing = [col for col in _PARQUET_COL_MAP if col not in df.columns]
        if missing:
            raise RuntimeError(f"Parquet missing required columns: {missing}")

        for col, (key, dtype) in _PARQUET_COL_MAP.items():
            rows = [_cell(df[col].iloc[i], dtype) for i in range(self.n_frames)]
            self._arrays[key] = np.stack(rows)  # (N, D)

        for col, (key, dtype) in _PARQUET_OPT_MAP.items():
            if col in df.columns:
                rows = [_cell(df[col].iloc[i], dtype) for i in range(self.n_frames)]
                self._arrays[key] = np.stack(rows)

        task = str(df["task"].iloc[0]) if "task" in df.columns else ""
        print(f"[parquet] {self.n_frames} frames  task='{task}'  path={path.name}")

    def get_lerp_frame(self, pidx_f: float) -> dict[str, np.ndarray]:
        """Return a lerp'd single frame as (1, 1, D) — mirrors kinematic.py formula.

        pidx_f is a fractional parquet index (= step * parquet_hz / ctrl_hz).
        p0 and p1 are clamped to [0, n_frames-1]; alpha is the fractional part.
        """
        p0 = min(int(pidx_f), self.n_frames - 1)
        p1 = min(p0 + 1, self.n_frames - 1)
        alpha = pidx_f - int(pidx_f)
        chunk: dict[str, np.ndarray] = {}
        for key, arr in self._arrays.items():
            a = arr[p0].astype(np.float32)
            b = arr[p1].astype(np.float32)
            chunk[key] = ((1.0 - alpha) * a + alpha * b)[np.newaxis, np.newaxis]
        return chunk

    def get_chunk(self, frame_start: int, chunk_size: int) -> dict[str, np.ndarray]:
        """Return a fake vla_chunk shaped (1, chunk_size, D) for each key.

        Frames past the end of the trajectory are filled by repeating the last
        frame, so the pipeline never errors on short episodes.
        """
        end = frame_start + chunk_size
        chunk: dict[str, np.ndarray] = {}
        for key, arr in self._arrays.items():
            n = self.n_frames
            if frame_start >= n:
                slice_ = np.tile(arr[-1:], (chunk_size, 1))
            elif end <= n:
                slice_ = arr[frame_start:end]
            else:
                valid = arr[frame_start:]
                pad   = np.tile(arr[-1:], (end - n, 1))
                slice_ = np.concatenate([valid, pad], axis=0)
            chunk[key] = slice_[np.newaxis]   # (1, chunk_size, D)
        return chunk


def _get_planner_mode(streamer: ParquetActionStreamer, frame: int) -> int:
    """Derive planner mode for a given parquet frame index.

    Mirrors eval_parquet_kinematic.py _get_mode priority chain:
      1. teleop.planner_mode column (explicit int) if present
      2. teleop.run flag → mode 3 (RUN) when True
      3. speed_to_mode(planner_speed) as fallback
    """
    from vla_sonic.frame_transforms import speed_to_mode
    arrays = streamer._arrays
    f = min(frame, streamer.n_frames - 1)
    if "planner_mode" in arrays:
        return int(arrays["planner_mode"][f].flat[0])
    if "run" in arrays and float(arrays["run"][f].flat[0]) > 0.5:
        return 3  # RUN
    speed = max(0.0, float(arrays["planner_speed"][f].flat[0]))
    return speed_to_mode(speed)


# =========================================================================
# Video writers.
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


class VLAVisWriter:
    """Renders VLA-format skeleton frames to MP4 via matplotlib Agg."""

    _AXIS_LEN = 0.08
    _TRAIL_ALPHA_MIN = 0.15
    _BG  = "#0d1117"
    _FG  = "#e6edf3"
    _C_LW = (0.33, 0.53, 1.00)
    _C_RW = (1.00, 0.33, 0.27)
    _C_NK = (0.27, 0.87, 0.40)
    _C_RT = (1.00, 0.80, 0.00)

    def __init__(self, path: Path, fps: int, trail_len: int = 60):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        self._plt = plt

        fig = plt.figure(figsize=(14, 8), facecolor=self._BG, dpi=100)
        fig.patch.set_facecolor(self._BG)
        self._ax3d  = fig.add_axes([0.00, 0.00, 0.65, 1.00], projection="3d")
        self._ax_txt = fig.add_axes([0.65, 0.00, 0.35, 1.00])
        self._ax3d.set_facecolor(self._BG)
        self._ax_txt.set_facecolor(self._BG)
        self._ax_txt.axis("off")
        self._fig = fig

        self._trails: dict[str, list] = {"lw": [], "rw": [], "nk": []}
        self._trail_len = trail_len

        import imageio
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7)
        self.path = path

    @staticmethod
    def _rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
        col0 = r6d[:3].astype(np.float64)
        col1 = r6d[3:6].astype(np.float64)
        n0 = np.linalg.norm(col0)
        if n0 > 1e-8:
            col0 /= n0
        col1 -= np.dot(col1, col0) * col0
        n1 = np.linalg.norm(col1)
        if n1 > 1e-8:
            col1 /= n1
        return np.stack([col0, col1, np.cross(col0, col1)], axis=1)

    def write(
        self,
        vr_pos: np.ndarray,
        vr_rot6d: np.ndarray,
        root_pos: np.ndarray,
        planner_mode: int,
        planner_speed: float,
        planner_height: float,
        planner_facing: np.ndarray,
        planner_movement: np.ndarray,
        step: int,
    ) -> None:
        lw_pos = vr_pos[0:3].astype(np.float64)
        rw_pos = vr_pos[3:6].astype(np.float64)
        nk_pos = vr_pos[6:9].astype(np.float64)
        lw_R = self._rot6d_to_matrix(vr_rot6d[0:6])
        rw_R = self._rot6d_to_matrix(vr_rot6d[6:12])
        nk_R = self._rot6d_to_matrix(vr_rot6d[12:18])

        for key, pt in [("lw", lw_pos), ("rw", rw_pos), ("nk", nk_pos)]:
            self._trails[key].append(pt.copy())
            if len(self._trails[key]) > self._trail_len:
                self._trails[key].pop(0)

        ax = self._ax3d
        ax.cla()
        ax.set_facecolor(self._BG)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#2d3040")
        ax.tick_params(colors=self._FG, labelsize=7)
        for axis_obj in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis_obj.label.set_color(self._FG)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.set_title(f"Parquet actions — step {step}", color=self._FG, fontsize=10, pad=4)

        cx, cy, cz = nk_pos
        half = 1.2
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_zlim(cz - half, cz + half)
        ax.set_box_aspect([1, 1, 1])

        gs = np.linspace(-half, half, 6)
        for g in gs:
            ax.plot([cx + g, cx + g], [cy - half, cy + half], [0, 0],
                    color="#2d3040", lw=0.5, alpha=0.6)
            ax.plot([cx - half, cx + half], [cy + g, cy + g], [0, 0],
                    color="#2d3040", lw=0.5, alpha=0.6)

        for key, color in [("lw", self._C_LW), ("rw", self._C_RW), ("nk", self._C_NK)]:
            trail = self._trails[key]
            if len(trail) >= 2:
                pts = np.array(trail)
                k = len(pts)
                for i in range(k - 1):
                    alpha = self._TRAIL_ALPHA_MIN + (1.0 - self._TRAIL_ALPHA_MIN) * (i / (k - 1))
                    ax.plot(pts[i:i+2, 0], pts[i:i+2, 1], pts[i:i+2, 2],
                            color=color, lw=1.5, alpha=alpha)

        rp = root_pos.astype(np.float64)
        ax.scatter(*rp, color=self._C_RT, s=100, zorder=5, depthshade=False)
        fv = planner_facing.astype(np.float64)
        ax.quiver(*rp, *(fv * 0.4), color=self._C_RT, lw=2, arrow_length_ratio=0.25)

        for pos, color, label in [
            (lw_pos, self._C_LW, "L wrist"),
            (rw_pos, self._C_RW, "R wrist"),
            (nk_pos, self._C_NK, "Neck"),
        ]:
            ax.scatter(*pos, color=color, s=220, zorder=6, depthshade=False)
            ax.text(pos[0], pos[1], pos[2] + 0.07, label,
                    color=color, fontsize=7, ha="center")

        for start, end, color in [
            (lw_pos, nk_pos, self._C_LW),
            (rw_pos, nk_pos, self._C_RW),
        ]:
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                    color=color, lw=2.5, alpha=0.85)

        torso_est = nk_pos - nk_R[:, 2] * 0.35
        ax.plot([nk_pos[0], torso_est[0]], [nk_pos[1], torso_est[1]],
                [nk_pos[2], torso_est[2]], color="#888888", lw=2.0, alpha=0.6)

        a = self._AXIS_LEN
        for pos, R_mat in [(lw_pos, lw_R), (rw_pos, rw_R), (nk_pos, nk_R)]:
            ax.quiver(*pos, *(R_mat[:, 0] * a), color="#ff4444", lw=1.5, arrow_length_ratio=0.35)
            ax.quiver(*pos, *(R_mat[:, 1] * a), color="#44ff44", lw=1.5, arrow_length_ratio=0.35)
            ax.quiver(*pos, *(R_mat[:, 2] * a), color="#4488ff", lw=1.5, arrow_length_ratio=0.35)

        ax.view_init(elev=20, azim=-60)

        _MODE_LABELS = {0: "IDLE", 1: "SLOW_WALK", 2: "WALK", 3: "RUN"}
        mv = planner_movement.astype(np.float64)

        lines = [
            ("PARQUET ACTIONS",                                          True,  "#f7d060"),
            ("",                                                         False, self._FG),
            ("── Planner ──────────────────",                            False, "#888899"),
            (f"Mode:   {_MODE_LABELS.get(planner_mode, str(planner_mode))}",
                                                                         False, self._FG),
            (f"Speed:  {planner_speed:.3f} m/s",                        False, self._FG),
            (f"Height: {planner_height:.3f} m",                         False, self._FG),
            (f"Facing:  [{fv[0]:+.3f}, {fv[1]:+.3f}, {fv[2]:+.3f}]",  False, self._FG),
            (f"Move:    [{mv[0]:+.3f}, {mv[1]:+.3f}, {mv[2]:+.3f}]",   False, self._FG),
            ("",                                                         False, self._FG),
            ("── VR 3-pt Positions ─────────",                          False, "#888899"),
            (f"L wrist  [{lw_pos[0]:+.3f}, {lw_pos[1]:+.3f}, {lw_pos[2]:+.3f}]",
                                                                         False, "#5588ff"),
            (f"R wrist  [{rw_pos[0]:+.3f}, {rw_pos[1]:+.3f}, {rw_pos[2]:+.3f}]",
                                                                         False, "#ff5544"),
            (f"Neck     [{nk_pos[0]:+.3f}, {nk_pos[1]:+.3f}, {nk_pos[2]:+.3f}]",
                                                                         False, "#44dd66"),
            ("",                                                         False, self._FG),
            ("── Arm lengths ───────────────",                           False, "#888899"),
            (f"L arm:  {np.linalg.norm(lw_pos - nk_pos):.3f} m",        False, self._FG),
            (f"R arm:  {np.linalg.norm(rw_pos - nk_pos):.3f} m",        False, self._FG),
            (f"Wrists: {np.linalg.norm(lw_pos - rw_pos):.3f} m",        False, self._FG),
            ("",                                                         False, self._FG),
            ("── Robot root (env) ──────────",                           False, "#888899"),
            (f"[{rp[0]:+.3f}, {rp[1]:+.3f}, {rp[2]:+.3f}]",            False, "#ffcc00"),
        ]

        axt = self._ax_txt
        axt.cla()
        axt.set_facecolor(self._BG)
        axt.axis("off")
        y, dy = 0.97, 0.042
        for text, bold, color in lines:
            axt.text(0.04, y, text, transform=axt.transAxes,
                     color=color, fontsize=9 if bold else 8,
                     fontweight="bold" if bold else "normal",
                     fontfamily="monospace", va="top")
            y -= dy

        self._fig.canvas.draw()
        w, h = self._fig.canvas.get_width_height()
        buf = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        self._writer.append_data(buf[:, :, :3])

    def close(self) -> None:
        self._writer.close()
        self._plt.close(self._fig)


# =========================================================================
# Camera helpers (identical to eval_vla_sonic.py).
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

    # --- 1. Load parquet trajectory ------------------------------------
    streamer = ParquetActionStreamer(args.parquet)
    max_steps = min(args.max_steps_per_episode, streamer.n_frames)
    print(f"[parquet] running at most {max_steps} steps (parquet has {streamer.n_frames} frames)")

    # --- 2. Build env --------------------------------------------------
    env_cfg = parse_env_cfg(
        args.task, device="cuda:0", num_envs=args.num_envs, enable_cameras=True,
    )

    # --- 2a. Physics overrides to match MuJoCo training environment --------
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

    # --- 2b. Physics substep rate -------------------------------------------
    # 500 Hz (dt=2ms, 10 substeps per 20ms control step) matches MuJoCo's
    # integration rate and is required for stable contact dynamics.
    env_cfg.sim.dt = 1.0 / 500.0
    env_cfg.sim.decimation = 10
    print("[physics] substep rate: 500 Hz (dt=2 ms, decimation=10)")

    # --- 2c. Articulation solver iterations ---------------------------------
    try:
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        print("[physics] articulation solver pos_iter overridden to 8")
    except AttributeError:
        print("[physics] could not override articulation solver pos_iter")

    _inject_cameras(env_cfg)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    print(f"[env] {args.task}  action_space={env.action_space}")
    print(f"[render] has_rtx_sensors={env.unwrapped.sim.has_rtx_sensors()}")

    # --- 3. Build SONIC wrappers --------------------------------------
    print(f"[utm] encoder={args.encoder_onnx}")
    print(f"[utm] decoder={args.decoder_onnx}")
    utm     = UtmWrapper(args.encoder_onnx, args.decoder_onnx)
    planner = PlannerWrapper(args.planner_onnx)

    # --- 4. Joint-order permutation -----------------------------------
    robot = env.unwrapped.scene["robot"]
    isaac_to_utm_perm = build_isaac_to_utm_perm(list(robot.data.joint_names))

    # --- 5. History buffer --------------------------------------------
    history = HistoryBuffer()

    # --- 6. Video writers — created AFTER Isaac Lab initializes to avoid
    #         matplotlib thread conflicts with Omniverse during env.reset().
    writers: dict[str, VideoWriter] = {}
    vla_vis_writer: VLAVisWriter | None = None
    prefix: Path | None = Path(args.record_video) if args.record_video else None

    # --- 7. Rollout ---------------------------------------------------
    # Pump the Omniverse application event loop before the first env.reset().
    #
    # Isaac Lab compiles PhysX / USD / render shaders on first use and needs
    # the app event loop to be ticked while that happens.  eval_vla_sonic.py
    # gets this for free: loading Gr00tPolicy takes several minutes, during
    # which _APP.update() is called implicitly by torch / the Kit framework.
    # Without a VLA we must do it explicitly, otherwise env.reset() blocks
    # waiting for compilation that never progresses.
    print("[sim] pumping Omniverse event loop to let physics initialise...")
    for _i in range(60):
        _APP.update()
        time.sleep(0.5)
    print("[sim] ready")

    action_space_dim = env.action_space.shape[-1]
    zero_action      = torch.zeros((args.num_envs, action_space_dim),
                                   device="cuda:0", dtype=torch.float32)
    prev_utm_body_29 = np.zeros(29, dtype=np.float32)
    total_successes  = 0

    planner_context: np.ndarray | None = None
    cached_planner_out = None
    cached_planner_inputs = None
    cached_out_50hz: np.ndarray | None = None

    for ep in range(args.num_episodes):
        print(f"\n[episode {ep}]")
        obs, info = env.reset()
        env.step(zero_action)   # warm-up camera buffer
        _APP.update()           # flush warm-up render to annotators
        _APP.update()
        history.reset()
        planner_step       = 0
        cached_planner_out = None
        cached_planner_inputs = None
        cached_out_50hz    = None
        # Bootstrap context from actual robot state — matches C++ InitializeContext.
        _q_b  = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        _qs_b = _gather_with_mask(_q_b, isaac_to_utm_perm)
        _qm_b = _qs_b[MUJOCO_TO_ISAACLAB]
        _rp_b = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        _rq_b = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
        planner_context = _make_robot_planner_context(_rp_b, _rq_b, _qm_b)

        # Pre-fill decoder history with 10 frames of initial robot state.
        # Without pre-seeding, the decoder's zero history has no gait phase signal,
        # causing symmetric bilateral motion (both feet lifting simultaneously).
        _scale_inv_h = 1.0 / G1_ACTION_SCALE_SONIC
        _qd0 = _gather_with_mask(
            robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32),
            isaac_to_utm_perm,
        )
        _av0 = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
        _grav0 = R.from_quat(quat_wxyz_to_xyzw(_rq_b)).inv().apply(
            np.array([0.0, 0.0, -1.0], dtype=np.float32)
        ).astype(np.float32)
        _mq0 = np.concatenate([_rp_b, _rq_b, _qm_b]).astype(np.float32)
        _q0_delta = _qs_b - G1_DEFAULT_ANGLES_SONIC
        for _ in range(10):
            history.push(
                joint_pos=_q0_delta,
                joint_vel=_qd0,
                last_action=_q0_delta * _scale_inv_h,
                base_ang_vel=_av0,
                gravity_dir=_grav0,
                mujoco_qpos=_mq0,
            )
        print(f"[episode {ep}] decoder history pre-filled from initial robot state (10 frames)")

        # Open video writers on first episode after Isaac Lab is fully up.
        if ep == 0 and prefix is not None and not writers:
            scene_keys = list(env.unwrapped.scene.keys()) if hasattr(env.unwrapped.scene, "keys") else []
            print(f"[video] scene entities: {scene_keys}")
            if "camera" in scene_keys:
                writers["camera"] = VideoWriter(
                    prefix.with_name(prefix.name + "_third_person.mp4"), args.video_fps)
            else:
                print("[video] 'camera' missing — skipping third_person.mp4")
            if "camera_robot" in scene_keys:
                writers["camera_robot"] = VideoWriter(
                    prefix.with_name(prefix.name + "_ego.mp4"), args.video_fps)
            else:
                print("[video] 'camera_robot' missing — skipping ego.mp4")
            for w in writers.values():
                print(f"[video] writing {w.path}")
            vla_vis_writer = VLAVisWriter(
                prefix.with_name(prefix.name + "_parquet_skeleton.mp4"), args.video_fps)
            print(f"[video] writing {vla_vis_writer.path}")
        prev_utm_body_29 = np.zeros(29, dtype=np.float32)

        parquet_chunk: dict | None = None
        chunk_step = 0

        for step in range(max_steps):
            # 7b. Refresh action chunk from parquet every chunk_size steps.
            if parquet_chunk is None or chunk_step >= args.chunk_size:
                parquet_chunk = streamer.get_chunk(step, args.chunk_size)
                chunk_step = 0
                if step == 0:
                    print("\n[parquet @ step 0] chunk keys and shapes:")
                    for k, v in sorted(parquet_chunk.items()):
                        print(f"  {k}: {np.asarray(v).shape}  "
                              f"t=0 → {np.asarray(v)[0, 0].round(4).tolist()}")

            t_idx = chunk_step

            # 7b. Push current env state into history.
            q_isaac      = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            qd_isaac     = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
            q_sonic      = _gather_with_mask(q_isaac,  isaac_to_utm_perm)
            qd_sonic     = _gather_with_mask(qd_isaac, isaac_to_utm_perm)
            q_mujoco     = q_sonic[MUJOCO_TO_ISAACLAB]
            root_pos_w   = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
            root_quat_w  = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
            root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
            gravity_body = R.from_quat(quat_wxyz_to_xyzw(root_quat_w)).inv().apply(
                np.array([0.0, 0.0, -1.0], dtype=np.float32)
            ).astype(np.float32)
            mujoco_qpos = np.concatenate([root_pos_w, root_quat_w, q_mujoco]).astype(np.float32)

            history.push(
                joint_pos=q_sonic - G1_DEFAULT_ANGLES_SONIC,
                joint_vel=qd_sonic,
                last_action=prev_utm_body_29,
                base_ang_vel=root_ang_vel_b,
                gravity_dir=gravity_body,
                mujoco_qpos=mujoco_qpos,
            )

            # 7c. Run kinematic planner at 10 Hz (every _REPLAN_STEPS env steps).
            # At replan time, extract new context from the previous 50 Hz buffer at
            # frames [_CTX_START_50HZ + n*_CTX_SPACING_50HZ for n=0..3], matching
            # C++ UpdateContextFromMotion with gen_frame_=7 and spacing=50/30.
            if planner_step == 0:
                if cached_out_50hz is not None:
                    ctx_frames = []
                    for n in range(4):
                        t_f = _CTX_START_50HZ + n * _CTX_SPACING_50HZ
                        ctx_frames.append(_interp_planner_frame(cached_out_50hz[0], t_f))
                    planner_context = np.stack(ctx_frames)[np.newaxis]  # (1, 4, 36)
                # kinematic.py formula: pidx_f = step * (parquet_hz / ctrl_hz).
                # _PARQUET_HZ == _CTRL_HZ == 50 Hz so pidx_f = step (integer, alpha=0),
                # but the lerp structure is kept explicit to match kinematic.py exactly.
                _pidx_f = step * (_PARQUET_HZ / _CTRL_HZ)
                _replan_chunk = streamer.get_lerp_frame(_pidx_f)
                planner_inputs = build_planner_inputs(
                    vla_action=_replan_chunk,
                    context_mujoco_qpos=planner_context,
                    t_index=0,
                    clip_negative_speed=False,
                )
                # Derive mode using the same priority chain as eval_parquet_kinematic.py:
                # explicit planner_mode column → run flag → speed_to_mode(planner_speed).
                # target_vel is NOT overridden; build_planner_inputs reads planner_speed
                # from the parquet chunk and sets it correctly.
                _p0 = min(int(_pidx_f), streamer.n_frames - 1)
                planner_inputs.mode = np.array([_get_planner_mode(streamer, _p0)], dtype=np.int64)
                cached_planner_out    = planner.run(**planner_inputs.as_kwargs())
                cached_planner_inputs = planner_inputs
                cached_out_50hz       = _resample_planner_to_50hz(cached_planner_out.mujoco_qpos)

                if step == 0:
                    out_first = cached_planner_out.mujoco_qpos[0, 0]
                    print("\n[PLANNER @ step 0] inputs:")
                    print(f"  target_vel         = {planner_inputs.target_vel.tolist()}")
                    print(f"  mode               = {planner_inputs.mode.tolist()}")
                    print(f"  movement_direction = {planner_inputs.movement_direction[0].round(4).tolist()}")
                    print(f"  facing_direction   = {planner_inputs.facing_direction[0].round(4).tolist()}")
                    print(f"  height             = {planner_inputs.height.tolist()}")
                    print("[PLANNER @ step 0] outputs:")
                    print(f"  out[0] root_pos  = {out_first[0:3].round(4).tolist()}")
                    print(f"  out[0] root_quat = {out_first[3:7].round(4).tolist()}")
                    print(f"  out[0] legs[:12] = {out_first[7:19].round(4).tolist()}")

            # 7d. Extract anchor + lower-body trajectory from 50 Hz buffer.
            # planner_step is the current 50 Hz frame within the replan cycle (0–4).
            _t50       = cached_out_50hz.shape[1]
            _frames_50 = cached_out_50hz[0]
            _cur       = min(planner_step, _t50 - 1)
            anchor_pos_w     = _frames_50[_cur, PLANNER_ROOT_POS_SLICE].copy()
            anchor_quat_wxyz = _frames_50[_cur, PLANNER_ROOT_QUAT_SLICE].copy()
            _R_robot  = R.from_quat(quat_wxyz_to_xyzw(root_quat_w))
            _R_anchor = R.from_quat(quat_wxyz_to_xyzw(anchor_quat_wxyz))
            _R_rel_mat = (_R_robot.inv() * _R_anchor).as_matrix().astype(np.float32)
            # SONIC rot6d convention: rot6d[0:3]=R[:,0], rot6d[3:6]=R[:,1] (columns, not rows).
            # flatten("F") = column-major = col0 then col1. flatten("C") would be wrong.
            anchor_rot6d = _R_rel_mat[:, :2].flatten("F").astype(np.float32)
            # Encoder lookahead: 10 frames spaced _ENC_STEP_50HZ=5 apart (= 100 ms/step).
            # Matches C++ GatherMotionJointPositionsMultiFrame with step_size=5.
            _lb_idx    = [min(planner_step + k * _ENC_STEP_50HZ, _t50 - 1) for k in range(_ENC_FRAMES)]
            _lb_frames = _frames_50[_lb_idx]                                               # (10, 36)
            lb_pos = _lb_frames[:, LOWER_BODY_QPOS_INDICES_MUJOCO_ORDER].astype(np.float32)  # (10, 12)
            # Velocities: forward differences on dense 50 Hz buffer, then select sparse frames.
            # Matches C++: vel[f] = (pos[f+1] - pos[f]) * 50;  last frame copies second-to-last.
            _lb_dense     = _frames_50[:, LOWER_BODY_QPOS_INDICES_MUJOCO_ORDER].astype(np.float32)  # (T_50, 12)
            _lb_vel_dense = np.empty_like(_lb_dense)
            _lb_vel_dense[:-1] = (_lb_dense[1:] - _lb_dense[:-1]) * _CTRL_HZ
            _lb_vel_dense[-1]  = _lb_vel_dense[-2]
            lb_vel = _lb_vel_dense[_lb_idx]                                                          # (10, 12)

            # 7e. VR 3-point from parquet chunk — transform world → anchor-local.
            # The parquet stores raw VR positions and orientations in world frame.
            # The encoder expects them expressed in the planner's anchor frame
            # (i.e., relative to the predicted pelvis position/orientation).
            vr_pos_world, vr_rot6d_world = extract_vr_3pt(parquet_chunk, t_index=t_idx)

            _R_anchor_inv = _R_anchor.inv()
            # Positions: subtract anchor origin, rotate into anchor frame.
            vr_pts_world = vr_pos_world.reshape(3, 3)                          # (3, xyz)
            vr_pos_anchor_local = (
                _R_anchor_inv.apply(vr_pts_world - anchor_pos_w)
                .reshape(-1).astype(np.float32)                                # (9,)
            )
            # Orientations: R_local = R_anchor^{-1} * R_world, re-encoded as rot6d.
            _R_anchor_inv_mat = _R_anchor_inv.as_matrix().astype(np.float64)
            _vr_rot6d_local = np.empty((3, 6), dtype=np.float32)
            for _i in range(3):
                _r = vr_rot6d_world[_i * 6 : _i * 6 + 6].astype(np.float64)
                _c0 = _r[:3] / (np.linalg.norm(_r[:3]) + 1e-10)
                _c1_raw = _r[3:]
                _c1 = _c1_raw - np.dot(_c1_raw, _c0) * _c0
                _c1 /= (np.linalg.norm(_c1) + 1e-10)
                _R_world_i = np.stack([_c0, _c1, np.cross(_c0, _c1)], axis=1)
                _R_local_i = _R_anchor_inv_mat @ _R_world_i
                _vr_rot6d_local[_i] = _R_local_i[:, :2].flatten("F")
            vr_rot6d_anchor_local = _vr_rot6d_local.reshape(-1)                # (18,)

            # 7e-vis. Skeleton video frame (uses world-frame positions for display).
            if vla_vis_writer is not None:
                vla_vis_writer.write(
                    vr_pos=vr_pos_world,
                    vr_rot6d=vr_rot6d_world,
                    root_pos=root_pos_w,
                    planner_mode=int(np.atleast_1d(cached_planner_inputs.mode).flat[0]),
                    planner_speed=float(np.atleast_1d(cached_planner_inputs.target_vel).flat[0]),
                    planner_height=float(np.atleast_1d(cached_planner_inputs.height).flat[0]),
                    planner_facing=np.asarray(cached_planner_inputs.facing_direction[0], dtype=np.float32),
                    planner_movement=np.asarray(cached_planner_inputs.movement_direction[0], dtype=np.float32),
                    step=step,
                )

            # 7f. Encoder → token → decoder → body_29.
            enc_obs = build_encoder_obs(
                anchor_pos_world=anchor_pos_w,
                anchor_quat_wxyz=anchor_quat_wxyz,
                anchor_rot6d=anchor_rot6d,
                lower_body_positions_future=lb_pos,
                lower_body_velocities_future=lb_vel,
                vr_3pt_position_anchor_local=vr_pos_anchor_local,
                vr_3pt_rot6d=vr_rot6d_anchor_local,
            )
            token = utm.run_encoder({"obs_dict": enc_obs}).reshape(-1)

            dec_hist = history.decoder_history()
            dec_obs  = build_decoder_obs(token_state=token, **dec_hist.as_kwargs())
            utm_body_29 = utm.run_decoder({"obs_dict": dec_obs}).reshape(-1)

            # 7g. Assemble env action (body from UTM, fingers from parquet).
            env_action_np = utm_plus_vla_to_env_action(
                utm_body_29_sonic=utm_body_29,
                vla_action=parquet_chunk,
                t_index=t_idx,
            )
            if step < 3:
                print(f"[step {step}] utm_body_29[:15]       = {utm_body_29[:15].round(3).tolist()}")
                print(f"[step {step}]   env body[:12] (legs) = {env_action_np[:12].round(3).tolist()}")
                print(f"[step {step}]   env left  fingers    = {env_action_np[27:34].round(3).tolist()}")
                print(f"[step {step}]   env right fingers    = {env_action_np[34:41].round(3).tolist()}")

            env_action = torch.as_tensor(
                env_action_np[None, :], device="cuda:0", dtype=torch.float32)

            # 7h. Step.
            q_pre = robot.data.joint_pos[0].detach().cpu().numpy() if step < 5 else None
            obs, rew, term, trunc, info = env.step(env_action)
            if step < 5:
                q_post = robot.data.joint_pos[0].detach().cpu().numpy()
                delta = np.abs(q_post - q_pre)
                print(f"[step {step}] joint_pos max_delta={delta.max():.6f}  "
                      f"mean_delta={delta.mean():.6f}  rew={float(rew[0]):.4f}")

            # 7i. Flush RTX render pipeline before reading cameras.
            # eval_vla_sonic.py gets this flush for free from VLA inference time;
            # here we must be explicit so the annotators deliver the current frame.
            _APP.update()
            _APP.update()

            # 7j. Video frames.
            if writers:
                _prev_frame = getattr(main, "_prev_cam_frame", {})
                for key, w in writers.items():
                    frame = _read_camera_rgb(env, key)
                    if frame is not None:
                        if step < 5 and key in _prev_frame:
                            diff = int(np.abs(frame.astype(np.int32) - _prev_frame[key].astype(np.int32)).max())
                            print(f"[step {step}] camera '{key}' max_pixel_diff_from_prev={diff}")
                        _prev_frame[key] = frame.copy()
                        w.write(frame)
                main._prev_cam_frame = _prev_frame

            prev_utm_body_29 = utm_body_29
            planner_step = (planner_step + 1) % _REPLAN_STEPS
            chunk_step += 1

            if bool(term[0] if term.ndim > 0 else term):
                print(f"[step {step}] terminated  rew={float(rew[0]):.4f}")
                break

        n_succ = int(getattr(env.unwrapped, "n_successes", 0))
        total_successes = n_succ
        print(f"[episode {ep}] ended at step {step + 1}; cumulative successes={n_succ}")

    for w in writers.values():
        w.close()
    if vla_vis_writer is not None:
        vla_vis_writer.close()

    env.close()
    _APP.close()

    print(f"\n[eval] {total_successes}/{args.num_episodes} successes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
