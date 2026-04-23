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

# Links to render per leg, in proximal→distal order.
# pelvis is the shared root rendered as a single neutral sphere.
_LEFT_LINKS  = ["left_hip_pitch_link",  "left_hip_roll_link",  "left_hip_yaw_link",
                "left_knee_link",        "left_ankle_pitch_link", "left_ankle_roll_link"]
_RIGHT_LINKS = ["right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
                "right_knee_link",       "right_ankle_pitch_link", "right_ankle_roll_link"]

# RGB colours
_COL_PELVIS = (200, 200, 200)
_COL_LEFT   = (51,  102, 230)   # blue
_COL_RIGHT  = (230, 127,  25)   # orange


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
# Forward kinematics
# ============================================================

def build_fk_chains(urdf_path: str):
    """Load two serial FK chains (left/right leg) from the G1 URDF.

    Returns (left_chain, right_chain) as pytorch_kinematics SerialChain objects
    rooted at `pelvis`, ending at the respective ankle_roll_link.
    """
    import pytorch_kinematics as pk

    with open(urdf_path) as f:
        urdf_str = f.read()

    left_chain = pk.build_serial_chain_from_urdf(
        urdf_str, "left_ankle_roll_link", root_link_name="pelvis"
    )
    right_chain = pk.build_serial_chain_from_urdf(
        urdf_str, "right_ankle_roll_link", root_link_name="pelvis"
    )
    print(f"[fk] left  joints: {left_chain.get_joint_parameter_names()}")
    print(f"[fk] right joints: {right_chain.get_joint_parameter_names()}")
    return left_chain, right_chain


def fk_positions(left_chain, right_chain, joint_angles_frame: np.ndarray) -> dict[str, np.ndarray]:
    """Run FK for one frame and return {link_name: (3,) position} for all leg links.

    joint_angles_frame: (12,) in MuJoCo lower-body order
      [0:6]  left  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
      [6:12] right hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    """
    import torch

    left_th  = torch.tensor(joint_angles_frame[:6],  dtype=torch.float32).unsqueeze(0)  # (1, 6)
    right_th = torch.tensor(joint_angles_frame[6:12], dtype=torch.float32).unsqueeze(0)  # (1, 6)

    left_fk  = left_chain.forward_kinematics(left_th,  end_only=False)
    right_fk = right_chain.forward_kinematics(right_th, end_only=False)

    positions: dict[str, np.ndarray] = {}
    for name, tf in {**left_fk, **right_fk}.items():
        mat = tf.get_matrix()  # (1, 4, 4)
        positions[name] = mat[0, :3, 3].detach().numpy()
    return positions


# ============================================================
# Viser visualisation
# ============================================================

def visualise(joint_angles: np.ndarray, urdf_path: str, fps: float = 30.0) -> None:
    """Animate the G1 lower-body skeleton using FK in a viser scene on port 8082.

    Upper body is absent — only pelvis + both legs are rendered.
    Pelvis is fixed at the URDF origin; only leg joints are animated.
    """
    import viser

    left_chain, right_chain = build_fk_chains(urdf_path)

    server = viser.ViserServer(port=8082)
    server.scene.world_axes.visible = True
    print("[viser] server running on http://localhost:8082")

    n_frames = joint_angles.shape[0]

    # Compute initial positions for sphere creation
    pos0 = fk_positions(left_chain, right_chain, joint_angles[0])

    # ── pelvis (fixed at origin) ─────────────────────────────────────────────
    server.scene.add_icosphere(
        "/skeleton/pelvis", radius=0.06, color=_COL_PELVIS, position=(0.0, 0.0, 0.0)
    )

    # ── per-link sphere handles ──────────────────────────────────────────────
    link_handles: dict[str, object] = {}
    for name in _LEFT_LINKS:
        p = pos0.get(name, np.zeros(3))
        link_handles[name] = server.scene.add_icosphere(
            f"/skeleton/{name}", radius=0.05, color=_COL_LEFT, position=tuple(p)
        )
    for name in _RIGHT_LINKS:
        p = pos0.get(name, np.zeros(3))
        link_handles[name] = server.scene.add_icosphere(
            f"/skeleton/{name}", radius=0.05, color=_COL_RIGHT, position=tuple(p)
        )

    # ── kinematic-chain line strips ──────────────────────────────────────────
    def _chain_pts(names: list[str], pos: dict[str, np.ndarray]) -> np.ndarray:
        """[pelvis(0,0,0)] + named links → (N+1, 3) point array."""
        pts = [np.zeros(3)] + [pos.get(n, np.zeros(3)) for n in names]
        return np.array(pts, dtype=np.float32)

    left_line  = server.scene.add_line_strip(
        "/skeleton/left_leg",  points=_chain_pts(_LEFT_LINKS,  pos0),
        colors=np.array([_COL_LEFT]  * (len(_LEFT_LINKS)  + 1), dtype=np.uint8), line_width=3.0,
    )
    right_line = server.scene.add_line_strip(
        "/skeleton/right_leg", points=_chain_pts(_RIGHT_LINKS, pos0),
        colors=np.array([_COL_RIGHT] * (len(_RIGHT_LINKS) + 1), dtype=np.uint8), line_width=3.0,
    )

    # ── frame counter ────────────────────────────────────────────────────────
    frame_label = server.scene.add_label("/info/frame", text="frame 0", position=(0.0, 0.0, 1.5))

    dt = 1.0 / fps
    print(f"[viser] looping {n_frames} frames at {fps:.0f} fps — Ctrl-C to exit …")

    try:
        while True:
            for frame_idx in range(n_frames):
                pos = fk_positions(left_chain, right_chain, joint_angles[frame_idx])

                for name, h in link_handles.items():
                    p = pos.get(name, np.zeros(3))
                    h.position = tuple(p)

                left_line.points  = _chain_pts(_LEFT_LINKS,  pos)
                right_line.points = _chain_pts(_RIGHT_LINKS, pos)
                frame_label.text  = f"frame {frame_idx + 1}/{n_frames}"
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
    joint_angles = run_planner(parquet_arrays, n_frames, args.planner_onnx)
    visualise(joint_angles, urdf_path=args.urdf, fps=args.fps)


if __name__ == "__main__":
    main()
