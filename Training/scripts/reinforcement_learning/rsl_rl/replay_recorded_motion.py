"""Replay a MotionRecorder session (planner_motion or streamed) in viser.

Reads the CSV files written by MotionRecorder::WriteFrame from a
gear_sonic_deploy recording session and animates the G1 skeleton via
ViserUrdf on port 8082.

Recording format (one directory per session):
    joint_pos.csv     — (N, 29) joint angles in IsaacLab column order
    body_pos.csv      — (N, 3)  root/pelvis world position xyz
                        (C++ records 1 body; B=14 pkl-converted recordings give N×42)
    body_quat.csv     — (N, 4)  root/pelvis world quaternion wxyz
                        (or N×56 for pkl-converted 14-body recordings)
    joint_vel.csv     — (N, 29) joint velocities  (not used for visualisation)
    body_lin_vel.csv  — (N, 3)  root linear velocity  (not used)
    body_ang_vel.csv  — (N, 3)  root angular velocity (not used)

Joint order:
    The C++ recorder stores joints in IsaacLab column order:
        CSV column il_idx  ←→  MuJoCo joint  _MJ_TO_IL[il_idx]
    Apply np.argsort to recover MuJoCo order for the URDF.

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
# Joint / URDF definitions
# ============================================================

# All 29 G1 joint names in MuJoCo order (matches g1_29dof.urdf).
ALL_JOINT_NAMES: list[str] = [
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
    "waist_yaw_joint",
    "waist_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
assert len(ALL_JOINT_NAMES) == 29

_DEFAULT_URDF = str(
    _HERE / "../../../../Sim2Real/resources/robots/g1_description/g1_29dof.urdf"
)

# mujoco_to_isaaclab permutation from policy_parameters.hpp.
# _MJ_TO_IL[il_idx] = MuJoCo joint stored in IsaacLab CSV column il_idx.
# _IL_TO_MJ[mj_idx] = IsaacLab CSV column that holds MuJoCo joint mj_idx.
_MJ_TO_IL = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
    4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
], dtype=np.int32)
_IL_TO_MJ = np.argsort(_MJ_TO_IL)  # shape (29,): CSV column for each MuJoCo joint

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
        arr = df.values.astype(np.float32)
        print(f"  [{name}.csv] shape={arr.shape}")
        return arr

    print(f"[load] reading {motion_dir.name} …")
    joint_pos  = _read("joint_pos")    # (N, 29) IsaacLab column order
    body_pos   = _read("body_pos")     # (N, 3)  root xyz world  [or (N, 42) pkl]
    body_quat  = _read("body_quat")    # (N, 4)  root wxyz world [or (N, 56) pkl]

    assert joint_pos.shape[1] == 29, \
        f"joint_pos expected 29 columns, got {joint_pos.shape[1]}"
    assert body_pos.shape[1] % 3 == 0, \
        f"body_pos expected multiple-of-3 columns, got shape {body_pos.shape}"
    assert body_quat.shape[1] % 4 == 0, \
        f"body_quat expected multiple-of-4 columns, got shape {body_quat.shape}"

    # All three may differ by ±1 frame if recording ended mid-write; use the minimum.
    n = min(joint_pos.shape[0], body_pos.shape[0], body_quat.shape[0])
    if not (joint_pos.shape[0] == body_pos.shape[0] == body_quat.shape[0]):
        print(
            f"  [warn] frame counts differ "
            f"(joint_pos={joint_pos.shape[0]}, body_pos={body_pos.shape[0]}, "
            f"body_quat={body_quat.shape[0]}) — using min={n}"
        )

    joint_pos = joint_pos[:n]
    body_pos  = body_pos[:n]
    body_quat = body_quat[:n]

    # Reorder from IsaacLab CSV columns to MuJoCo joint order for the URDF.
    # joint_pos_mj[t, mj_idx] = joint_pos[t, _IL_TO_MJ[mj_idx]]
    joint_pos_mj = joint_pos[:, _IL_TO_MJ]   # (N, 29) MuJoCo order

    # Root pose: first body = pelvis.
    root_pos  = body_pos[:, 0:3]   # (N, 3) xyz world
    root_wxyz = body_quat[:, 0:4]  # (N, 4) wxyz world  (w,x,y,z)

    print(
        f"[load] {n} frames  "
        f"joint_range=[{joint_pos_mj.min():.3f}, {joint_pos_mj.max():.3f}]  "
        f"root_z=[{root_pos[:,2].min():.3f}, {root_pos[:,2].max():.3f}]"
    )

    return {
        "joint_pos_mj": joint_pos_mj,
        "root_pos":     root_pos,
        "root_wxyz":    root_wxyz,
        "n_frames":     n,
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

    n_frames     = data["n_frames"]
    joint_pos_mj = data["joint_pos_mj"]
    root_pos     = data["root_pos"]
    root_wxyz    = data["root_wxyz"]
    dt           = 1.0 / fps

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
        pos  = root_pos[idx].astype(float)
        wxyz = root_wxyz[idx].astype(float)
        angles = joint_pos_mj[idx]

        # Apply root translation and orientation from the recording.
        server.scene.add_frame(
            "/robot",
            position=pos,
            wxyz=wxyz,
            show_axes=False,
        )

        # Drive all 29 joints.
        urdf_vis.update_cfg(
            {name: float(angles[i]) for i, name in enumerate(ALL_JOINT_NAMES)}
        )

        t_sec = idx / fps
        gui_info_md.content = (
            f"**frame** {idx}/{n_frames-1}  \n"
            f"**time**  {t_sec:.3f} s  \n"
            f"**root_pos** [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]  \n"
            f"**root_wxyz** [{wxyz[0]:+.3f}, {wxyz[1]:+.3f}, {wxyz[2]:+.3f}, {wxyz[3]:+.3f}]"
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
