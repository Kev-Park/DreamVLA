"""Replay a MotionRecorder session (planner_motion or streamed) in viser.

Reads the CSV files written by MotionRecorder::WriteFrame from a
gear_sonic_deploy recording session and animates the G1 skeleton via
ViserUrdf on port 8082.

Recording format (one directory per session):
    joint_pos.csv   — (N, 29) joint angles in IsaacLab order
    body_pos.csv    — (N, B*3) body link world positions; body_0 = pelvis
                      B=1 for planner_motion / streamed recordings (root only)
    body_quat.csv   — (N, B*4) body link world quaternions wxyz; body_0 = pelvis
    joint_vel.csv   — (N, 29)  (not used for visualisation)

Joint order conversion:
    The C++ recorder stores joints in IsaacLab order using the
    mujoco_to_isaaclab permutation from policy_parameters.hpp:
        JointPositions[il_idx] = qpos[7 + mujoco_to_isaaclab[il_idx]]
    The inverse (numpy argsort) recovers MuJoCo order for the URDF.

Usage:
    python replay_recorded_motion.py \\
        --motion-dir /home/dvij/kevin/GR00T-WholeBodyControl/gear_sonic_deploy/\\
reference/recorded_motion/20260424/planner_motion_175248
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

# ============================================================
# Joint / URDF definitions  (matches eval_parquet_kinematic.py)
# ============================================================

LOWER_BODY_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
_NUM_LOWER = len(LOWER_BODY_JOINT_NAMES)  # 12

_DEFAULT_URDF = str(
    _HERE / "../../../../Sim2Real/resources/robots/g1_description/g1_29dof.urdf"
)

# mujoco_to_isaaclab from policy_parameters.hpp.
# CSV column il_idx holds MuJoCo joint mujoco_to_isaaclab[il_idx].
# np.argsort gives the inverse: for MuJoCo joint m, CSV column = _IL_TO_MJ[m].
_MJ_TO_IL = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
], dtype=np.int32)
_IL_TO_MJ = np.argsort(_MJ_TO_IL)  # CSV column index for each MuJoCo joint

# Lower-body MuJoCo joint indices 0-11 → CSV column indices.
_LB_CSV_COLS = _IL_TO_MJ[:_NUM_LOWER]

_MUJOCO_STANDING_Z = 0.78874

_DEFAULT_MOTION_DIR = (
    "/home/dvij/kevin/GR00T-WholeBodyControl/gear_sonic_deploy"
    "/reference/recorded_motion/20260424/planner_motion_175248"
)

# ============================================================
# CSV loading
# ============================================================

def load_motion(motion_dir: Path) -> dict[str, np.ndarray]:
    """Load joint_pos, body_pos, body_quat from a MotionRecorder session dir."""

    def _read(name: str) -> np.ndarray:
        p = motion_dir / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        df = pd.read_csv(p, header=0)
        return df.values.astype(np.float32)

    joint_pos  = _read("joint_pos")   # (N, 29) IsaacLab order
    body_pos   = _read("body_pos")    # (N, 14*3)
    body_quat  = _read("body_quat")   # (N, 14*4)

    n = joint_pos.shape[0]
    # C++ MotionRecorder writes only 1 body (root/pelvis) — shape (N, 3) and (N, 4).
    # Multi-body pkl-converted recordings have shape (N, 14*3) and (N, 14*4).
    assert body_pos.shape[0]  == n and body_pos.shape[1]  % 3 == 0, f"body_pos shape {body_pos.shape}"
    assert body_quat.shape[0] == n and body_quat.shape[1] % 4 == 0, f"body_quat shape {body_quat.shape}"

    # Lower-body angles in MuJoCo order.
    lb_angles = joint_pos[:, _LB_CSV_COLS]          # (N, 12)

    # Pelvis pose: body_0 = first 3 / first 4 columns.
    root_pos  = body_pos[:, 0:3]                     # (N, 3)  xyz world
    root_wxyz = body_quat[:, 0:4]                    # (N, 4)  wxyz world

    print(
        f"[load] {n} frames  "
        f"lb_range=[{lb_angles.min():.3f}, {lb_angles.max():.3f}]  "
        f"root_z=[{root_pos[:,2].min():.3f}, {root_pos[:,2].max():.3f}]"
    )

    return {
        "lb_angles": lb_angles,
        "root_pos":  root_pos,
        "root_wxyz": root_wxyz,
        "n_frames":  n,
    }


# ============================================================
# Viser visualisation
# ============================================================

def visualise(
    data: dict[str, np.ndarray],
    urdf_path: str,
    fps: float = 50.0,
    motion_dir: Path | None = None,
) -> None:
    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=8082)
    server.scene.world_axes.visible = True
    print("[viser] server running on http://localhost:8082")

    urdf_vis = ViserUrdf(server, Path(urdf_path), root_node_name="/robot")

    n_frames  = data["n_frames"]
    lb_angles = data["lb_angles"]
    root_wxyz = data["root_wxyz"]
    dt        = 1.0 / fps

    _PINNED_POS  = np.array([0.0, 0.0, _MUJOCO_STANDING_Z])
    _PINNED_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])

    with server.gui.add_folder("Playback"):
        gui_playing = server.gui.add_checkbox("Playing", initial_value=True)
        gui_slider  = server.gui.add_slider(
            "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
        )

    with server.gui.add_folder("Frame Info"):
        gui_info_md = server.gui.add_markdown("loading…")

    label = f"  dir: {motion_dir.name}" if motion_dir else ""
    print(f"[viser] {n_frames} frames @ {fps:.0f} Hz{label} — Ctrl-C to exit …")

    def _apply_frame(idx: int) -> None:
        angles = lb_angles[idx]
        wxyz   = root_wxyz[idx].astype(float)

        server.scene.add_frame(
            "/robot",
            position=_PINNED_POS,
            wxyz=_PINNED_WXYZ,
            show_axes=False,
        )
        urdf_vis.update_cfg(
            {name: float(angles[i]) for i, name in enumerate(LOWER_BODY_JOINT_NAMES)}
        )

        t_sec = idx / fps
        gui_info_md.content = (
            f"**frame** {idx}/{n_frames-1}  \n"
            f"**time**  {t_sec:.3f} s  \n"
            f"**root_wxyz** [{wxyz[0]:+.3f}, {wxyz[1]:+.3f}, {wxyz[2]:+.3f}, {wxyz[3]:+.3f}]  \n"
            f"**lb_range**  [{angles.min():+.3f}, {angles.max():+.3f}]"
        )

    _apply_frame(0)

    frame_idx = 0
    try:
        while True:
            if gui_playing.value:
                frame_idx = (frame_idx + 1) % n_frames
                gui_slider.value = frame_idx
                _apply_frame(frame_idx)
            else:
                new_idx = int(gui_slider.value)
                if new_idx != frame_idx:
                    frame_idx = new_idx
                    _apply_frame(frame_idx)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a gear_sonic_deploy MotionRecorder session in viser"
    )
    parser.add_argument(
        "--motion-dir",
        type=Path,
        default=Path(_DEFAULT_MOTION_DIR),
        help="Path to the MotionRecorder session directory (contains joint_pos.csv etc.).",
    )
    parser.add_argument(
        "--urdf",
        default=_DEFAULT_URDF,
        help="Path to g1_29dof.urdf.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Playback frame rate (default 50, matching the 50 Hz recording rate).",
    )
    args = parser.parse_args()

    if not args.motion_dir.exists():
        raise SystemExit(f"motion-dir not found: {args.motion_dir}")

    data = load_motion(args.motion_dir)
    visualise(data, urdf_path=args.urdf, fps=args.fps, motion_dir=args.motion_dir)


if __name__ == "__main__":
    main()
