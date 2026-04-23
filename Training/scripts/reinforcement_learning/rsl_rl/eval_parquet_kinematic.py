"""Parquet-driven kinematic planner eval — standalone viser joint visualiser.

Reads planner command columns from a LeRobot parquet file, runs the SONIC
kinematic planner ONNX, then animates the 12 lower-body joint angles as spheres
in a viser scene on port 8082.

No Isaac Lab or simulator required — purely planner ONNX + viser.

Lower-body joint order (MuJoCo model order, qpos indices 7–18):
  [0]  left_hip_pitch_joint     [6]  right_hip_pitch_joint
  [1]  left_hip_roll_joint      [7]  right_hip_roll_joint
  [2]  left_hip_yaw_joint       [8]  right_hip_yaw_joint
  [3]  left_knee_joint          [9]  right_knee_joint
  [4]  left_ankle_pitch_joint  [10]  right_ankle_pitch_joint
  [5]  left_ankle_roll_joint   [11]  right_ankle_roll_joint

Usage:
    python eval_parquet_kinematic.py \\
        --parquet /path/to/episode_000000.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

# ============================================================
# Joint definitions
# ============================================================

LOWER_BODY_JOINT_NAMES: list[str] = [
    "left_hip_pitch_joint",    # mujoco qpos[7]
    "left_hip_roll_joint",     # qpos[8]
    "left_hip_yaw_joint",      # qpos[9]
    "left_knee_joint",         # qpos[10]
    "left_ankle_pitch_joint",  # qpos[11]
    "left_ankle_roll_joint",   # qpos[12]
    "right_hip_pitch_joint",   # qpos[13]
    "right_hip_roll_joint",    # qpos[14]
    "right_hip_yaw_joint",     # qpos[15]
    "right_knee_joint",        # qpos[16]
    "right_ankle_pitch_joint", # qpos[17]
    "right_ankle_roll_joint",  # qpos[18]
]
NUM_JOINTS = len(LOWER_BODY_JOINT_NAMES)  # 12

# Slice into the 36-D mujoco qpos for the 12 lower-body joints.
# MuJoCo body-tree order: left-leg-all-6 then right-leg-all-6 (see action_assembler.py).
_LB_QPOS_SLICE = slice(7, 19)

# Left leg = blue, right leg = orange (RGB 0-255)
_JOINT_COLORS: list[tuple[int, int, int]] = (
    [(51, 102, 230)] * 6    # left leg
    + [(230, 127, 25)] * 6  # right leg
)

# Fixed anatomical layout for spheres:
#   x: left column = -0.4,  right column = +0.4
#   z: hip at top (2.5), ankle at bottom (0.5),  0.4 m per level
#   y: updated each frame to encode joint angle (radians)
_BASE_XZ = np.array(
    [[-0.4, 2.5 - i * 0.4] for i in range(6)]   # left
    + [[+0.4, 2.5 - i * 0.4] for i in range(6)], # right
    dtype=np.float32,
)  # (12, 2)  columns are [x, z]


# ============================================================
# Parquet loading
# ============================================================

_REQUIRED_COLS: dict[str, tuple[str, type]] = {
    "teleop.planner_movement": ("planner_movement", np.float32),  # (3,)
    "teleop.planner_facing":   ("planner_facing",   np.float32),  # (3,)
    "teleop.planner_speed":    ("planner_speed",    np.float32),  # (1,)
    "teleop.planner_height":   ("planner_height",   np.float32),  # (1,)
}
_OPTIONAL_COLS: dict[str, tuple[str, type]] = {
    "teleop.planner_mode": ("planner_mode", np.int64),    # explicit mode int
    "teleop.run":          ("run",          np.float32),  # boolean RUN override
}


def _cell(val, dtype) -> np.ndarray:
    arr = val.astype(dtype) if isinstance(val, np.ndarray) else np.array(val, dtype=dtype)
    return np.atleast_1d(arr)


def load_parquet(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """Load planner command columns from a LeRobot parquet file.

    Returns (arrays, n_frames) where each array has shape (N, D).
    """
    df = pd.read_parquet(path)
    n = len(df)

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Parquet missing required columns: {missing}")

    arrays: dict[str, np.ndarray] = {}

    for col, (key, dtype) in _REQUIRED_COLS.items():
        arrays[key] = np.stack([_cell(df[col].iloc[i], dtype) for i in range(n)])

    for col, (key, dtype) in _OPTIONAL_COLS.items():
        if col in df.columns:
            arrays[key] = np.stack([_cell(df[col].iloc[i], dtype) for i in range(n)])

    task = str(df["task"].iloc[0]) if "task" in df.columns else ""
    print(f"[parquet] {n} frames  task='{task}'  path={path.name}")
    return arrays, n


# ============================================================
# Planner mode helpers
# ============================================================

def _get_mode(arrays: dict[str, np.ndarray], frame: int) -> int:
    """Derive planner mode integer for a given frame.

    Priority:
      1. teleop.planner_mode column (explicit)
      2. teleop.run flag  →  mode 3 (RUN) when True
      3. speed_to_mode(planner_speed)
    """
    n = arrays["planner_movement"].shape[0]
    f = min(frame, n - 1)

    if "planner_mode" in arrays:
        return int(arrays["planner_mode"][f].flat[0])

    if "run" in arrays and float(arrays["run"][f].flat[0]) > 0.5:
        return 3  # RUN

    import sys
    sys.path.insert(0, str(_HERE))
    from vla_sonic.frame_transforms import speed_to_mode
    speed = max(0.0, float(arrays["planner_speed"][f].flat[0]))
    return speed_to_mode(speed)


# ============================================================
# Planner inference
# ============================================================

def run_planner(
    parquet_arrays: dict[str, np.ndarray],
    n_frames: int,
    planner_onnx: str,
) -> np.ndarray:
    """Run the kinematic planner for every parquet frame.

    Returns joint_angles of shape (n_frames, 12): the lower-body joint angles
    (radians, MuJoCo order) from the planner's first predicted frame.
    """
    import sys
    sys.path.insert(0, str(_HERE))
    from vla_sonic import PlannerWrapper, build_planner_inputs

    planner = PlannerWrapper(planner_onnx)
    print(f"[planner] loaded {planner_onnx}")

    # Bootstrap context: valid identity quaternion at indices 3-6, zeros elsewhere.
    context = np.zeros((1, 4, 36), dtype=np.float32)
    context[0, :, 3] = 1.0  # w component of root quaternion (w, x, y, z)

    joint_angles = np.zeros((n_frames, NUM_JOINTS), dtype=np.float32)
    zero_frame_count = 0

    for frame in range(n_frames):
        n = parquet_arrays["planner_movement"].shape[0]
        f = min(frame, n - 1)

        # Build fake vla_chunk shaped (1, 1, D) for build_planner_inputs.
        chunk = {
            key: parquet_arrays[key][f][np.newaxis, np.newaxis]
            for key in ("planner_movement", "planner_facing", "planner_speed", "planner_height")
        }

        inputs = build_planner_inputs(
            vla_action=chunk,
            context_mujoco_qpos=context,
            t_index=0,
            batch_index=0,
        )
        # Override mode with column-derived or run-flag value.
        inputs.mode = np.array([_get_mode(parquet_arrays, frame)], dtype=np.int64)

        out = planner.run(**inputs.as_kwargs())

        if out.num_pred_frames > 0:
            first_qpos = out.mujoco_qpos[0, 0]           # (36,)
            joint_angles[frame] = first_qpos[_LB_QPOS_SLICE]

            # Roll context: use up to 4 frames from planner output.
            n_ctx = min(out.num_pred_frames, 4)
            new_ctx = out.mujoco_qpos[0, :n_ctx]          # (n_ctx, 36)
            if n_ctx < 4:
                pad = np.tile(new_ctx[-1:], (4 - n_ctx, 1))
                new_ctx = np.concatenate([new_ctx, pad], axis=0)
            context = new_ctx[np.newaxis]                  # (1, 4, 36)
        else:
            zero_frame_count += 1

        if frame % 100 == 0:
            angles_now = joint_angles[frame]
            print(f"[planner] frame {frame}/{n_frames}  mode={inputs.mode[0]}  "
                  f"num_pred_frames={out.num_pred_frames}  "
                  f"lb_joints_range=[{angles_now.min():.3f}, {angles_now.max():.3f}]")

    if zero_frame_count:
        print(f"[planner] WARNING: {zero_frame_count}/{n_frames} frames returned num_pred_frames=0")
    print(f"[planner] done — joint_angles range [{joint_angles.min():.3f}, {joint_angles.max():.3f}]")
    return joint_angles


# ============================================================
# Viser visualisation
# ============================================================

def _sphere_position(joint_idx: int, angle_rad: float) -> tuple[float, float, float]:
    """Return (x, y, z) for a joint sphere. y encodes the joint angle (radians)."""
    x, z = float(_BASE_XZ[joint_idx, 0]), float(_BASE_XZ[joint_idx, 1])
    return (x, angle_rad, z)


def visualise(joint_angles: np.ndarray, fps: float = 30.0) -> None:
    """Animate 12 lower-body joint-angle spheres in a viser scene on port 8082.

    Each sphere's y position encodes the joint angle in radians; x and z give
    the fixed anatomical column layout (left/right × hip-to-ankle).
    """
    import viser

    server = viser.ViserServer(port=8082)
    server.scene.world_axes.visible = True

    print("[viser] server running on http://localhost:8082")
    print("[viser] open a browser to view the visualisation")

    n_frames = joint_angles.shape[0]

    # ── frame counter label ──────────────────────────────────────────────────
    frame_label = server.scene.add_label("/info/frame", text="frame 0", position=(0.0, 0.0, 3.0))

    # ── create spheres once ──────────────────────────────────────────────────
    handles = []
    for i, name in enumerate(LOWER_BODY_JOINT_NAMES):
        r, g, b = _JOINT_COLORS[i]
        h = server.scene.add_icosphere(
            f"/joints/{name}",
            radius=0.07,
            color=(r, g, b),
            position=_sphere_position(i, float(joint_angles[0, i])),
        )
        handles.append(h)

    dt = 1.0 / fps
    print(f"[viser] looping {n_frames} frames at {fps:.0f} fps — Ctrl-C to exit …")

    try:
        while True:
            for frame_idx in range(n_frames):
                angles = joint_angles[frame_idx]
                for i, h in enumerate(handles):
                    h.position = _sphere_position(i, float(angles[i]))
                frame_label.text = f"frame {frame_idx + 1}/{n_frames}"
                time.sleep(dt)
    except KeyboardInterrupt:
        pass


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kinematic planner eval — viser lower-body joint visualiser (no sim required)"
    )
    parser.add_argument(
        "--parquet", required=True, type=Path,
        help="Path to a LeRobot episode_*.parquet file.",
    )
    parser.add_argument(
        "--planner-onnx",
        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
        help="Path to planner_sonic.onnx (default: GR00T-WholeBodyControl install).",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Playback frame rate in Hz (default 30).",
    )
    args = parser.parse_args()

    parquet_arrays, n_frames = load_parquet(args.parquet)
    joint_angles = run_planner(parquet_arrays, n_frames, args.planner_onnx)
    visualise(joint_angles, fps=args.fps)


if __name__ == "__main__":
    main()
