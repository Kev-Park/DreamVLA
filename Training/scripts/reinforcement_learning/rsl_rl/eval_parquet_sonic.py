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
from vla_sonic.action_assembler import MUJOCO_TO_ISAACLAB  # noqa: E402
from vla_sonic.frame_transforms import quat_wxyz_to_xyzw  # noqa: E402


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
ENCODER_FUTURE_FRAME_INDICES = list(range(0, 50, 5))
PLANNER_OUTPUT_FPS = 30.0


def extract_anchor_pose(mujoco_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame0 = mujoco_qpos[0, 0]
    return (
        np.asarray(frame0[PLANNER_ROOT_POS_SLICE],  dtype=np.float32).copy(),
        np.asarray(frame0[PLANNER_ROOT_QUAT_SLICE], dtype=np.float32).copy(),
    )


def extract_lower_body_future(mujoco_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(mujoco_qpos[0], dtype=np.float32)
    need = max(ENCODER_FUTURE_FRAME_INDICES) + 1
    if qpos.shape[0] < need:
        pad = np.repeat(qpos[-1:], need - qpos.shape[0], axis=0)
        qpos = np.concatenate([qpos, pad], axis=0)
    lb_all  = qpos[:, LOWER_BODY_QPOS_INDICES_MUJOCO_ORDER]
    vel_all = np.gradient(lb_all, 1.0 / PLANNER_OUTPUT_FPS, axis=0).astype(np.float32)
    return (
        lb_all[ENCODER_FUTURE_FRAME_INDICES].astype(np.float32),
        vel_all[ENCODER_FUTURE_FRAME_INDICES].astype(np.float32),
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
            return val.astype(dtype) if isinstance(val, np.ndarray) else np.array(val, dtype=dtype)

        missing = [col for col in _PARQUET_COL_MAP if col not in df.columns]
        if missing:
            raise RuntimeError(f"Parquet missing required columns: {missing}")

        for col, (key, dtype) in _PARQUET_COL_MAP.items():
            rows = [_cell(df[col].iloc[i], dtype) for i in range(self.n_frames)]
            self._arrays[key] = np.stack(rows)  # (N, D)

        task = str(df["task"].iloc[0]) if "task" in df.columns else ""
        print(f"[parquet] {self.n_frames} frames  task='{task}'  path={path.name}")

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


# =========================================================================
# Video writers.
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
    _inject_cameras(env_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    print(f"[env] {args.task}  action_space={env.action_space}")

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
    action_space_dim = env.action_space.shape[-1]
    zero_action      = torch.zeros((args.num_envs, action_space_dim),
                                   device="cuda:0", dtype=torch.float32)
    prev_utm_body_29 = np.zeros(29, dtype=np.float32)
    total_successes  = 0

    for ep in range(args.num_episodes):
        print(f"\n[episode {ep}]")
        obs, info = env.reset()
        env.step(zero_action)   # warm-up camera buffer
        history.reset()

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
            # 7a. Refresh action chunk from parquet every chunk_size steps.
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
            from scipy.spatial.transform import Rotation as R
            gravity_body = R.from_quat(quat_wxyz_to_xyzw(root_quat_w)).inv().apply(
                np.array([0.0, 0.0, -1.0], dtype=np.float32)
            ).astype(np.float32)
            mujoco_qpos = np.concatenate([root_pos_w, root_quat_w, q_mujoco]).astype(np.float32)

            history.push(
                joint_pos=q_sonic,
                joint_vel=qd_sonic,
                last_action=prev_utm_body_29,
                base_ang_vel=root_ang_vel_b,
                gravity_dir=gravity_body,
                mujoco_qpos=mujoco_qpos,
            )

            # 7c. Run kinematic planner on parquet commands.
            planner_inputs = build_planner_inputs(
                vla_action=parquet_chunk,
                context_mujoco_qpos=history.planner_context(),
                t_index=t_idx,
            )
            planner_out = planner.run(**planner_inputs.as_kwargs())

            if step == 0:
                ctx_last  = planner_inputs.context_mujoco_qpos[0, -1]
                out_first = planner_out.mujoco_qpos[0, 0]
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
            anchor_pos_w, anchor_quat_wxyz = extract_anchor_pose(planner_out.mujoco_qpos)
            _R_robot  = R.from_quat(quat_wxyz_to_xyzw(root_quat_w))
            _R_anchor = R.from_quat(quat_wxyz_to_xyzw(anchor_quat_wxyz))
            _R_rel_mat = (_R_robot.inv() * _R_anchor).as_matrix().astype(np.float32)
            anchor_rot6d = _R_rel_mat[:, :2].flatten("C").astype(np.float32)
            lb_pos, lb_vel = extract_lower_body_future(planner_out.mujoco_qpos)

            # 7e. VR 3-point from parquet chunk.
            vr_pos_world, vr_rot6d = extract_vr_3pt(parquet_chunk, t_index=t_idx)

            # 7e-vis. Skeleton video frame.
            if vla_vis_writer is not None:
                vla_vis_writer.write(
                    vr_pos=vr_pos_world,
                    vr_rot6d=vr_rot6d,
                    root_pos=root_pos_w,
                    planner_mode=int(np.atleast_1d(planner_inputs.mode).flat[0]),
                    planner_speed=float(np.atleast_1d(planner_inputs.target_vel).flat[0]),
                    planner_height=float(np.atleast_1d(planner_inputs.height).flat[0]),
                    planner_facing=np.asarray(planner_inputs.facing_direction[0], dtype=np.float32),
                    planner_movement=np.asarray(planner_inputs.movement_direction[0], dtype=np.float32),
                    step=step,
                )

            # 7f. Encoder → token → decoder → body_29.
            enc_obs = build_encoder_obs(
                anchor_pos_world=anchor_pos_w,
                anchor_quat_wxyz=anchor_quat_wxyz,
                anchor_rot6d=anchor_rot6d,
                lower_body_positions_future=lb_pos,
                lower_body_velocities_future=lb_vel,
                vr_3pt_position_anchor_local=vr_pos_world,
                vr_3pt_rot6d=vr_rot6d,
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
            obs, rew, term, trunc, info = env.step(env_action)

            # 7i. Video frames.
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
