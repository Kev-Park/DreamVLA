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
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="Multiply planner target_vel by this factor. The planner takes larger "
                             "steps than the RL policy at the same speed command, causing excess body "
                             "swing. Scaling to 0.5-0.7 reduces step amplitude. Waypoints compensate "
                             "for the lower commanded speed to preserve final reach distance.")
    parser.add_argument("--gravity", type=float, default=-9.81,
                        help="Z-component of gravity (m/s²). Default -9.81 matches both UTM training "
                             "(gear_sonic/Isaac Lab, no override) and MuJoCo deployment (scene_29dof.xml, "
                             "no <option gravity> tag). The -7.5 assumption in eval_sonic_control.py was wrong.")
    parser.add_argument("--static-friction", type=float, default=1.0,
                        help="Terrain + robot body static friction coefficient. Default 1.0 matches UTM "
                             "training (gear_sonic). Use 0.5 to match MuJoCo deployment scene friction.")
    parser.add_argument("--dynamic-friction", type=float, default=1.0,
                        help="Terrain + robot body dynamic friction coefficient. Default 1.0 matches UTM "
                             "training (gear_sonic). Use 0.5 to match MuJoCo deployment scene friction.")
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
_REPLAN_STEPS_30HZ    = round(_PLANNER_HZ / _REPLAN_HZ) # 3 — one replan cycle at native 30 Hz


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

# Indices into observation.state (and motion.reference_qpos joint slice) that select
# the 12 lower-body joints in the order the encoder expects (MUJOCO grouped: left leg
# first, then right leg). gear_sonic's robot_model.joint_names is MUJOCO-grouped — see
# features_sonic_vla.py::_get_joint_group_slices which assumes each joint group occupies
# a CONTIGUOUS range. So left_leg = [0..5], right_leg = [6..11], lower body = [0..11].
# No permutation needed — the SONIC-interleaved order (MUJOCO_TO_ISAACLAB) only applies
# to the UTM decoder's output, not to the encoder's input.
LOWER_BODY_OBS_STATE_INDICES = np.arange(12, dtype=np.int64)


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
    "teleop.planner_movement":      ("planner_movement",   np.float32),  # (3,)
    "teleop.planner_facing":        ("planner_facing",     np.float32),  # (3,)
    "teleop.planner_speed":         ("planner_speed",      np.float32),  # (1,)
    "teleop.planner_height":        ("planner_height",     np.float32),  # (1,)
    "teleop.planner_mode":          ("planner_mode",       np.int32),    # (1,) derived in converter
    "teleop.vr_3pt_position":       ("vr_3pt_position",    np.float32),  # (9,) world-frame in new schema
    "teleop.vr_3pt_orientation":    ("vr_3pt_orientation", np.float32),  # (18,) world-frame rot6d
    "teleop.left_hand_joints":      ("left_hand_joints",   np.float32),  # (7,)
    "teleop.right_hand_joints":     ("right_hand_joints",  np.float32),  # (7,)
    # Canonical root orientation (wxyz float64) — used for spawn heading extraction.
    "observation.root_orientation": ("root_orientation",   np.float64),  # (4,)
    # Absolute robot position in the collection world frame.
    "teleop.root_pos_w":            ("root_pos_w",         np.float32),  # (3,)
    # Full 29-dim joint state in SONIC/UTM order — used to construct the encoder's
    # lower-body lookahead trajectory directly from the recorded parquet (Q1 fix).
    "observation.state":            ("obs_state",          np.float32),  # (29,)
}

# Optional columns — loaded when present, ignored when absent.
_PARQUET_OPT_MAP: dict[str, tuple[str, type]] = {
    "teleop.run":           ("run",           np.float32),  # legacy compat
    # Object (mustard bottle) world pose at each frame.  Used after env.reset() to place
    # the object at the exact collection position rather than relying on random motion_ids.
    "teleop.object_pos_w":  ("object_pos_w",  np.float32),
    "teleop.object_quat_w": ("object_quat_w", np.float32),
    # Auxiliary planner-bypass column written by the upstream converter when source HDF5
    # contains motion-library reference data. Layout: [root_pos(3), root_quat_wxyz(4),
    # joints_in_gear_sonic_order(29)] = 36 floats. When present and non-trivial, the eval
    # loop reads encoder lookahead from this instead of running planner_sonic.onnx.
    "motion.reference_qpos": ("reference_qpos", np.float32),
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
    env_cfg.sim.gravity = (0.0, 0.0, args.gravity)
    print(f"[physics] gravity overridden to {env_cfg.sim.gravity}")
    try:
        env_cfg.scene.terrain.physics_material.static_friction = args.static_friction
        env_cfg.scene.terrain.physics_material.dynamic_friction = args.dynamic_friction
        print(f"[physics] terrain friction overridden to static={args.static_friction} dynamic={args.dynamic_friction}")
    except AttributeError:
        print("[physics] terrain.physics_material not found — skipping friction override")
    try:
        env_cfg.events.physics_material.params["static_friction_range"] = (args.static_friction, args.static_friction)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (args.dynamic_friction, args.dynamic_friction)
        print(f"[physics] robot body friction overridden to static={args.static_friction} dynamic={args.dynamic_friction}")
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
    env = gym.make(args.task, cfg=env_cfg)
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

    _has_root_pos_w = "root_pos_w" in streamer._arrays

    # Check once whether parquet has object pose data.
    _parquet_has_object_pose = (
        "object_pos_w" in streamer._arrays and "object_quat_w" in streamer._arrays
    )
    if _parquet_has_object_pose:
        _obj_quat0 = streamer._arrays["object_quat_w"][0].astype(np.float32)  # (4,) wxyz
        _obj_pos0  = streamer._arrays["object_pos_w"][0].astype(np.float32)   # (3,) world XYZ
        print(f"[object] parquet has pose data  pos={_obj_pos0.round(4).tolist()}  "
              f"quat={_obj_quat0.round(4).tolist()}")
    else:
        _obj_quat0 = None
        _obj_pos0  = None
        print("[object] no object_pos_w in parquet — bottle position uses env motion_ids (random)")

    # The bottle and kitchen are fixed-world-frame assets: the kitchen is always at
    # (2.04, 1.0, 0.0) and the bottle rests on its counter.  The correct eval bottle
    # position is therefore the collection's ABSOLUTE world position (object_pos_w[0]),
    # not a robot-relative offset.  A robot-relative offset would drift whenever the
    # eval robot spawns at a different XY than the collection robot, causing the bottle
    # to float in mid-air rather than rest on the counter.
    #
    # For the same reason the robot itself is restored to the collection's exact spawn
    # position (root_pos_w[0] + collection heading): this ensures the kitchen, the
    # bottle, and the VR arm targets all land in the correct world geometry.
    if _has_root_pos_w:
        _collection_robot_pos0 = streamer._arrays["root_pos_w"][0].astype(np.float32)
        print(f"[spawn] collection robot spawn: {_collection_robot_pos0.round(4).tolist()}")
    else:
        _collection_robot_pos0 = None
        print("[spawn] no root_pos_w in parquet — robot spawn XY uses env motion_id")

    for ep in range(args.num_episodes):
        print(f"\n[episode {ep}]")
        obs, info = env.reset()

        # Place bottle at the collection's absolute world position.
        # Must happen before warm-up env.step() so the position is propagated.
        if _parquet_has_object_pose:
            _obj_asset = env.unwrapped.scene["object"]
            _obj_pose_t = torch.tensor(
                np.concatenate([_obj_pos0, _obj_quat0])[None, :],
                device="cuda:0", dtype=torch.float32,
            )
            _obj_asset.write_root_pose_to_sim(
                _obj_pose_t, env_ids=torch.tensor([0], device="cuda:0")
            )
            _obj_asset.write_root_velocity_to_sim(
                torch.zeros((1, 6), device="cuda:0"),
                env_ids=torch.tensor([0], device="cuda:0"),
            )
            print(f"[object] position set to {_obj_pos0.round(4).tolist()}")

        # Restore the robot's exact collection spawn: absolute XY position + collection heading.
        # The env assigns a random motion_id at reset, which may place the robot at a different
        # world XY than the collection.  This matters for two reasons:
        #   1. The kitchen and bottle are world-fixed assets; a mismatched robot XY puts the
        #      kitchen counter at the wrong relative position, making the bottle float or clip.
        #   2. The VR 3-point anchor frame is robot-relative; an XY offset that also shifts the
        #      heading would make arm targets land in the wrong world direction.
        # Fix: restore both the collection's world XY/Z (from root_pos_w[0]) and heading yaw
        # (from observation.root_orientation[0]).
        _root_quat0 = streamer._arrays["root_orientation"][0].astype(np.float32)  # wxyz at frame 0
        _yaw_collection = float(R.from_quat(quat_wxyz_to_xyzw(_root_quat0)).as_euler("zyx")[0])
        _qc_xyzw = R.from_euler("z", _yaw_collection).as_quat()  # (x,y,z,w) scipy convention
        _qc_wxyz = np.array([_qc_xyzw[3], _qc_xyzw[0], _qc_xyzw[1], _qc_xyzw[2]], dtype=np.float32)
        if _collection_robot_pos0 is not None:
            _spawn_pos_w = _collection_robot_pos0          # exact collection XY + Z
        else:
            _spawn_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        robot.write_root_pose_to_sim(
            torch.tensor(np.concatenate([_spawn_pos_w, _qc_wxyz])[None, :],
                         device="cuda:0", dtype=torch.float32),
            env_ids=torch.tensor([0], device="cuda:0"),
        )
        robot.write_root_velocity_to_sim(
            torch.zeros((1, 6), device="cuda:0"), env_ids=torch.tensor([0], device="cuda:0")
        )
        print(f"[spawn] robot restored to collection pos={_spawn_pos_w.round(4).tolist()} "
              f"yaw={np.degrees(_yaw_collection):.1f}°")

        env.step(zero_action)   # warm-up camera buffer + propagates object pose and heading overrides
        _APP.update()           # flush warm-up render to annotators
        _APP.update()
        history.reset()
        planner_step       = 0
        cached_planner_out = None
        cached_planner_inputs = None
        # Bootstrap context from actual robot state — matches C++ InitializeContext.
        _q_b  = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        _qs_b = _gather_with_mask(_q_b, isaac_to_utm_perm)
        _qm_b = _qs_b[MUJOCO_TO_ISAACLAB]
        _rp_b = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        _rq_b = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
        planner_context = _make_robot_planner_context(_rp_b, _rq_b, _qm_b)
        # The planner is seeded with identity quaternion (see _make_robot_planner_context),
        # so its output quaternions are in a local frame where the robot starts facing +X.
        # To bring planner anchor_quat into world frame before computing motion_anchor_orientation,
        # we rotate by the robot's full initial world orientation (not yaw-only).
        _R_robot_init = R.from_quat(quat_wxyz_to_xyzw(_rq_b))

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

        for step in range(max_steps):
            # Read current parquet frame for VR and finger data (planner drives locomotion;
            # one parquet frame per control step, no VLA chunk buffering needed).
            parquet_frame = streamer.get_lerp_frame(float(step))
            if step == 0:
                print("\n[parquet @ step 0] initial frame values:")
                for k, v in sorted(parquet_frame.items()):
                    print(f"  {k}: {np.asarray(v)[0, 0].round(4).tolist()}")

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
            # Context rolling uses native 30 Hz planner output frames [3:7], matching
            # kinematic.py: one replan cycle = _REPLAN_STEPS_30HZ=3 frames at 30 Hz.
            if planner_step == 0:
                if cached_planner_out is not None:
                    n_pred = cached_planner_out.mujoco_qpos.shape[1]
                    if n_pred > 0:
                        cs = min(_REPLAN_STEPS_30HZ, n_pred - 1)
                        ctx_raw = cached_planner_out.mujoco_qpos[0, cs : cs + 4]
                        if ctx_raw.shape[0] < 4:
                            ctx_raw = np.concatenate(
                                [ctx_raw, np.tile(ctx_raw[-1:], (4 - ctx_raw.shape[0], 1))], axis=0
                            )
                        planner_context = ctx_raw[np.newaxis]  # (1, 4, 36)
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
                # build_planner_inputs zeroes movement_direction when speed_to_mode returns 0
                # (IDLE). If the mode override makes the mode non-zero (e.g., explicit
                # planner_mode column has SLOW_WALK at low speed), movement_direction would
                # remain zero and the planner falls back to facing_direction → robot walks
                # straight. Fix: restore the actual parquet movement vector (horizontal,
                # unit-normalised) so the planner receives the true lateral offset.
                if int(planner_inputs.mode.flat[0]) != 0:
                    _raw_mvmt = np.asarray(_replan_chunk["planner_movement"], dtype=np.float32)[0, 0]  # (3,)
                    _mvmt_xy = np.array([_raw_mvmt[0], _raw_mvmt[1], 0.0], dtype=np.float32)
                    _mvmt_norm = float(np.linalg.norm(_mvmt_xy[:2]))
                    if _mvmt_norm > 1e-6:
                        _mvmt_xy[:2] /= _mvmt_norm
                        planner_inputs.movement_direction = _mvmt_xy.reshape(1, 3)
                    else:
                        planner_inputs.movement_direction = planner_inputs.facing_direction.copy()
                if args.speed_scale != 1.0:
                    planner_inputs.target_vel = (planner_inputs.target_vel * args.speed_scale).astype(np.float32)
                cached_planner_out = planner.run(**planner_inputs.as_kwargs())
                cached_planner_inputs = planner_inputs

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

            # 7d. Extract anchor + lower-body trajectory.
            #
            # Two sources are supported:
            #
            # (A) PLANNER PATH (default fallback):
            #   The encoder's motion_*_lowerbody_10frame_step5 slots expect the PLANNER's
            #   predicted future trajectory — verified against gear_sonic_deploy/g1_deploy_onnx_ref.cpp:
            #   GatherMotionJointPositionsMultiFrame reads from current_motion_->JointPositions(),
            #   and in teleop mode current_motion_ points to planner_motion_ (populated by
            #   planner_sonic.onnx output each control cycle).
            #   - 10 frames at indices [0, 5, ..., 45] from 30Hz planner output (167ms/step)
            #
            # (B) REFERENCE-MOTION BYPASS (when motion.reference_qpos is present in parquet):
            #   Read the kinematic intent the WBC was tracking directly from parquet at
            #   indices [0, 5, ..., 45] at 50Hz (100ms/step, matching upstream C++). Skips
            #   the np.gradient + speed_to_mode reconstruction → planner_sonic.onnx step,
            #   feeding the encoder a less-lossy version of the same trajectory shape.
            _ref_qpos_arr = streamer._arrays.get("reference_qpos")
            _use_ref_bypass = (
                _ref_qpos_arr is not None
                and float(np.abs(_ref_qpos_arr[0]).sum()) > 1e-6
            )
            if _use_ref_bypass:
                # (B) Reference-motion bypass — same final shape as the planner path.
                # Column layout: [root_pos(3), root_quat_wxyz(4), joints_gs(num_joints)].
                # num_joints is robot_model.joint_names length (body + fingers, typically
                # 29 + 14 = 43). Body joints are the first 29 entries in gear_sonic order,
                # so slicing [7:7+29] extracts body joints regardless of finger count.
                #
                # Temporal stride: 30Hz stride-5 = 167ms per encoder frame, 1.5s total
                # lookahead — matches the planner-path (cached_planner_out.mujoco_qpos is
                # native 30Hz), which has been empirically verified to produce correct
                # gait cadence. The parquet is at 50Hz so the indices are rounded from
                # t·(50/30): [0, 8, 17, 25, 33, 42, 50, 58, 67, 75]. Max per-frame timing
                # error ≈ 13ms — well below the 167ms stride. (Earlier attempt at 50Hz
                # stride-5 = 100ms produced walking but with wrong footstep count.)
                _PARQUET_LB_LOOKAHEAD_IDX_50HZ = np.array(
                    [0, 8, 17, 25, 33, 42, 50, 58, 67, 75], dtype=np.int64
                )
                _PARQUET_LB_STEP_DT = 5.0 / 30.0  # 167ms — matches planner-path stride
                _N_ref = _ref_qpos_arr.shape[0]
                _enc_idx_pq = np.clip(step + _PARQUET_LB_LOOKAHEAD_IDX_50HZ, 0, _N_ref - 1)
                _ref_window = _ref_qpos_arr[_enc_idx_pq]                 # (10, 7+num_joints)
                _joints_gs = _ref_window[:, 7:7 + 29]                    # (10, 29) gear_sonic body
                lb_pos = _joints_gs[:, LOWER_BODY_OBS_STATE_INDICES].astype(np.float32)  # MJ order
                lb_vel = np.gradient(lb_pos, _PARQUET_LB_STEP_DT, axis=0).astype(np.float32)
                # Anchor: current frame's reference pose, already in world frame.
                _anchor_idx = int(np.clip(step, 0, _N_ref - 1))
                anchor_pos_w     = _ref_qpos_arr[_anchor_idx, 0:3].astype(np.float32)
                anchor_quat_wxyz = _ref_qpos_arr[_anchor_idx, 3:7].astype(np.float32)
                _R_robot   = R.from_quat(quat_wxyz_to_xyzw(root_quat_w))
                _R_anchor  = R.from_quat(quat_wxyz_to_xyzw(anchor_quat_wxyz))
                _R_rel_mat = (_R_robot.inv() * _R_anchor).as_matrix().astype(np.float32)
                anchor_rot6d = _R_rel_mat[:, :2].flatten("C").astype(np.float32)
                if step == 0:
                    print("[REF-BYPASS @ step 0] using motion.reference_qpos (no planner ONNX for encoder)")
                    print(f"  anchor_pos_w  = {anchor_pos_w.round(4).tolist()}")
                    print(f"  anchor_quat   = {anchor_quat_wxyz.round(4).tolist()}")
            else:
                # (A) Planner path — preserved as fallback for parquets without the auxiliary column.
                _qpos_30hz   = cached_planner_out.mujoco_qpos[0]             # (N, 36) native 30Hz
                anchor_pos_w     = _qpos_30hz[0, PLANNER_ROOT_POS_SLICE].copy()
                anchor_quat_wxyz = _qpos_30hz[0, PLANNER_ROOT_QUAT_SLICE].copy()
                _R_robot      = R.from_quat(quat_wxyz_to_xyzw(root_quat_w))
                _anchor_world = _R_robot_init * R.from_quat(quat_wxyz_to_xyzw(anchor_quat_wxyz))
                _R_rel_mat    = (_R_robot.inv() * _anchor_world).as_matrix().astype(np.float32)
                anchor_rot6d  = _R_rel_mat[:, :2].flatten("C").astype(np.float32)
                _enc_lb_idx = list(range(0, 50, 5))
                _n30  = _qpos_30hz.shape[0]
                _need = _enc_lb_idx[-1] + 1
                if _n30 < _need:
                    _qpos_30hz = np.concatenate(
                        [_qpos_30hz, np.tile(_qpos_30hz[-1:], (_need - _n30, 1))], axis=0
                    )
                _lb_all  = _qpos_30hz[:, LOWER_BODY_QPOS_INDICES_MUJOCO_ORDER].astype(np.float32)
                _vel_all = np.gradient(_lb_all, 1.0 / _PLANNER_HZ, axis=0).astype(np.float32)
                lb_pos = _lb_all[_enc_lb_idx]
                lb_vel = _vel_all[_enc_lb_idx]

            # 7e. VR 3-point: parquet stores pelvis-local positions/orientations.
            # collect_pick_cam.py:284-292 applies subtract_frame_transforms() before HDF5
            # storage, and convert_isaac_hdf5_to_lerobot.py:_apply_local_offset() passes
            # through (it adds local-frame offsets to the already-pelvis-relative SE3).
            # build_encoder_obs (planner_to_utm.py:158-163) expects pelvis-local — a
            # world→anchor transform here is a double-transform → instant ragdoll.
            vr_pos_anchor_local, vr_rot6d_anchor_local = extract_vr_3pt(parquet_frame, t_index=0)

            # 7e-vis. Skeleton video frame.
            if vla_vis_writer is not None:
                vla_vis_writer.write(
                    vr_pos=vr_pos_anchor_local,
                    vr_rot6d=vr_rot6d_anchor_local,
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

            if step == 0:
                print("\n[ENC @ step 0] encoder inputs:")
                print(f"  anchor_rot6d            = {anchor_rot6d.round(4).tolist()}")
                print(f"  lb_pos[0] (now)         = {lb_pos[0].round(4).tolist()}")
                print(f"  lb_pos[9] (far)         = {lb_pos[9].round(4).tolist()}")
                print(f"  lb_vel[0]               = {lb_vel[0].round(4).tolist()}")
                print(f"  vr_pos_anchor_local      = {vr_pos_anchor_local.round(4).tolist()}")
                print(f"  vr_rot6d_anchor_local[0] = {vr_rot6d_anchor_local[0].round(4).tolist()}")
                print(f"  vr_rot6d_anchor_local[1] = {vr_rot6d_anchor_local[1].round(4).tolist()}")
                print(f"  vr_rot6d_anchor_local[2] = {vr_rot6d_anchor_local[2].round(4).tolist()}")
                print("[ENC @ step 0] encoder output:")
                print(f"  token_norm              = {np.linalg.norm(token):.4f}")
                print(f"  token[:8]               = {token[:8].round(4).tolist()}")
                print(f"  token[56:64]            = {token[56:64].round(4).tolist()}")

            dec_hist = history.decoder_history()
            dec_obs  = build_decoder_obs(token_state=token, **dec_hist.as_kwargs())
            utm_body_29 = utm.run_decoder({"obs_dict": dec_obs}).reshape(-1)

            # 7g. Assemble env action (body from UTM, fingers from parquet).
            env_action_np = utm_plus_vla_to_env_action(
                utm_body_29_sonic=utm_body_29,
                vla_action=parquet_frame,
                t_index=0,
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
