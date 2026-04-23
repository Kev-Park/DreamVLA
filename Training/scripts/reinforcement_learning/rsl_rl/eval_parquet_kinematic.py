"""Parquet-driven kinematic planner eval — G1 lower-body FK visualiser.

Reads planner command columns from a LeRobot parquet file, runs the SONIC
kinematic planner ONNX, then animates the G1's lower-body skeleton in a
viser scene on port 8082 using proper forward kinematics (pytorch_kinematics
+ g1_29dof.urdf).  Upper body is frozen at the URDF default pose.

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

# Default URDF path relative to this script.
# Script lives at: Training/scripts/reinforcement_learning/rsl_rl/
# URDF lives at:   Sim2Real/resources/robots/g1_description/g1_29dof.urdf
_DEFAULT_URDF = str(
    _HERE / "../../../../Sim2Real/resources/robots/g1_description/g1_29dof.urdf"
)

# G1 neutral standing pose in MuJoCo order (from policy_parameters.hpp::default_angles).
# qpos[7:36] should use these values, not zeros, to give the planner a valid context.
_MUJOCO_DEFAULT_ANGLES_29 = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # left  leg: hp, hr, hy, knee, ap, ar
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,   # right leg: hp, hr, hy, knee, ap, ar
     0.0,   0.0, 0.0,                        # waist: yaw, roll, pitch
     0.2,   0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left  arm: sp, sr, sy, elbow, wr, wp, wy
     0.2,  -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm: sp, sr, sy, elbow, wr, wp, wy
], dtype=np.float32)
_MUJOCO_STANDING_Z = 0.793  # typical G1 pelvis height in standing pose (metres)



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

    # Bootstrap context with the G1's neutral standing pose, not zeros.
    # Zero joints give the planner a completely straight-legged robot context which
    # produces degenerate output — the actual standing pose has hip_pitch=-0.312,
    # knee=0.669, ankle_pitch=-0.363 (from policy_parameters.hpp::default_angles).
    context = np.zeros((1, 4, 36), dtype=np.float32)
    context[0, :, 2] = _MUJOCO_STANDING_Z   # pelvis z height
    context[0, :, 3] = 1.0                  # quaternion w (identity orientation)
    context[0, :, 7:36] = _MUJOCO_DEFAULT_ANGLES_29

    joint_angles = np.zeros((n_frames, NUM_JOINTS), dtype=np.float32)
    planner_log = {
        "movement": np.zeros((n_frames, 3), dtype=np.float32),
        "facing":   np.zeros((n_frames, 3), dtype=np.float32),
        "speed":    np.zeros((n_frames,),   dtype=np.float32),
        "height":   np.zeros((n_frames,),   dtype=np.float32),
        "mode":     np.zeros((n_frames,),   dtype=np.int64),
    }
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

        # Log planner inputs for this frame.
        planner_log["movement"][frame] = inputs.movement_direction[0]
        planner_log["facing"][frame]   = inputs.facing_direction[0]
        planner_log["speed"][frame]    = float(inputs.target_vel[0])
        planner_log["height"][frame]   = float(inputs.height[0])
        planner_log["mode"][frame]     = int(inputs.mode[0])

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
    return joint_angles, planner_log


# ============================================================
# Viser visualisation
# ============================================================

_MODE_NAMES = {
    0: "IDLE", 1: "SLOW_WALK", 2: "WALK", 3: "RUN",
    4: "SQUAT", 5: "KNEELTWOLEGS", 6: "KNEELONELEG",
    7: "LYINGFACEDOWN", 8: "HANDCRAWLING", 9: "IDLEBOXING",
}


def visualise(
    joint_angles: np.ndarray,
    planner_log: dict,
    urdf_path: str,
    fps: float = 30.0,
) -> None:
    """Render the G1 with actual link meshes via ViserUrdf.

    Lower-body joints are animated from the planner output.
    Upper body stays frozen at the URDF default (zero) pose.
    """
    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=8082)
    server.scene.world_axes.visible = True
    print("[viser] server running on http://localhost:8082")

    urdf_vis = ViserUrdf(server, Path(urdf_path))

    n_frames = joint_angles.shape[0]
    dt = 1.0 / fps

    # ── GUI: playback controls ────────────────────────────────────────────────
    with server.gui.add_folder("Playback"):
        gui_playing = server.gui.add_checkbox("Playing", initial_value=True)
        gui_slider  = server.gui.add_slider(
            "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
        )

    # ── GUI: planner inputs (read-only markdown, updated each frame) ─────────
    with server.gui.add_folder("Planner Inputs"):
        gui_inputs_md = server.gui.add_markdown("loading…")

    print(f"[viser] {n_frames} frames at {fps:.0f} fps — Ctrl-C to exit …")

    def _apply_frame(idx: int) -> None:
        angles = joint_angles[idx]
        urdf_vis.update_cfg({
            name: float(angles[i])
            for i, name in enumerate(LOWER_BODY_JOINT_NAMES)
        })
        mv       = planner_log["movement"][idx]
        fac      = planner_log["facing"][idx]
        spd      = float(planner_log["speed"][idx])
        ht       = float(planner_log["height"][idx])
        mode_id  = int(planner_log["mode"][idx])
        gui_inputs_md.content = (
            f"**movement** [{mv[0]:+.2f}, {mv[1]:+.2f}, {mv[2]:+.2f}]  \n"
            f"**facing**   [{fac[0]:+.2f}, {fac[1]:+.2f}, {fac[2]:+.2f}]  \n"
            f"**speed**    {spd:.3f} m/s  \n"
            f"**height**   {ht:.3f} m  \n"
            f"**mode**     {mode_id} ({_MODE_NAMES.get(mode_id, '?')})"
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
        description="Kinematic planner eval — G1 lower-body FK visualiser (no sim required)"
    )
    parser.add_argument(
        "--parquet", required=True, type=Path,
        help="Path to a LeRobot episode_*.parquet file.",
    )
    parser.add_argument(
        "--planner-onnx",
        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
        help="Path to planner_sonic.onnx.",
    )
    parser.add_argument(
        "--urdf", default=_DEFAULT_URDF,
        help="Path to g1_29dof.urdf (default: Sim2Real/resources/robots/g1_description/).",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Playback frame rate in Hz (default 30).",
    )
    args = parser.parse_args()

    parquet_arrays, n_frames = load_parquet(args.parquet)
    joint_angles, planner_log = run_planner(parquet_arrays, n_frames, args.planner_onnx)
    visualise(joint_angles, planner_log, urdf_path=args.urdf, fps=args.fps)


if __name__ == "__main__":
    main()
