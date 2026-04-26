"""Parquet-driven kinematic planner eval — G1 lower-body FK visualiser.

Reads planner command columns from a LeRobot parquet file, runs the SONIC
kinematic planner ONNX, then animates the G1's skeleton in a viser scene on
port 8082 using ViserUrdf + g1_29dof.urdf.  Upper body is frozen at the URDF
default pose.

No Isaac Lab or simulator required — purely planner ONNX + viser.

Note: the g1_29dof.urdf omits the ±10° Y body-frame rotations that the MuJoCo
XML applies to hip_roll/knee links, so individual joint orientations are
approximate compared to the planner's training model.

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
_MUJOCO_STANDING_Z = 0.78874  # matches PlannerConfig::default_height in localmotion_kplanner.hpp


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
    planner_onnx: str,
) -> np.ndarray:
    """Run the kinematic planner at 30 Hz output rate, replanning at 10 Hz (C++ rate).

    C++ replans at 10 Hz (every 5 control steps at 50 Hz = every 3 planner frames at
    30 Hz). Between replans the cached trajectory is played through frame-by-frame.

    Context rolling matches UpdateContextFromMotion exactly:
        gen_time = (control_frame + look_ahead_steps) / 50.0
        context[n] = output_at(gen_time + n/30.0)  for n=0..3
    At 10 Hz replan this evaluates to frames [3,4,5,6] of the previous 30 Hz output
    (0.1s of elapsed time = 3 frames at 30 Hz, then 33 ms spacing).

    Returns (full_qpos, planner_log) where full_qpos has shape (n_steps, 36).
    """
    import sys
    sys.path.insert(0, str(_HERE))
    from vla_sonic import PlannerWrapper, build_planner_inputs

    planner = PlannerWrapper(planner_onnx)
    print(f"[planner] loaded {planner_onnx}")

    _PLANNER_HZ   = 30.0
    _PARQUET_HZ   = 50.0
    _REPLAN_HZ    = 10.0
    _REPLAN_STEPS = round(_PLANNER_HZ / _REPLAN_HZ)  # 3: one replan every 0.1 s at 30 Hz

    n_parquet = parquet_arrays["planner_movement"].shape[0]
    n_steps   = round(n_parquet * _PLANNER_HZ / _PARQUET_HZ)
    print(f"[planner] {n_parquet} parquet frames @ {_PARQUET_HZ:.0f} Hz "
          f"→ {n_steps} steps @ {_PLANNER_HZ:.0f} Hz, replanning @ {_REPLAN_HZ:.0f} Hz")

    # Bootstrap: 4 identical standing frames (matches C++ InitializeContext with
    # the robot in its default neutral pose at startup).
    context = np.zeros((1, 4, 36), dtype=np.float32)
    context[0, :, 2]    = _MUJOCO_STANDING_Z
    context[0, :, 3]    = 1.0
    context[0, :, 7:36] = _MUJOCO_DEFAULT_ANGLES_29

    full_qpos   = np.zeros((n_steps, 36), dtype=np.float32)
    planner_log = {
        "movement": np.zeros((n_steps, 3), dtype=np.float32),
        "facing":   np.zeros((n_steps, 3), dtype=np.float32),
        "speed":    np.zeros((n_steps,),   dtype=np.float32),
        "height":   np.zeros((n_steps,),   dtype=np.float32),
        "mode":     np.zeros((n_steps,),   dtype=np.int64),
    }
    zero_frame_count = 0

    cached_out  = None  # last planner output; reused for _REPLAN_STEPS steps
    cached_log  = None  # planner inputs for this cycle (used for in-between log entries)

    for step in range(n_steps):
        traj_step = step % _REPLAN_STEPS  # position within current replan cycle: 0,1,2

        if traj_step == 0:
            # ── Advance context to current position before replanning ──────────
            # C++ UpdateContextFromMotion: gen_time = (control_frame+2)/50,
            # context[n] = output_at(gen_time + n/30). After _REPLAN_STEPS steps
            # (0.1s) this evaluates to frames [3,4,5,6] of the previous output.
            if cached_out is not None and cached_out.num_pred_frames > 0:
                n_pred = cached_out.num_pred_frames
                cs     = min(_REPLAN_STEPS, n_pred - 1)
                ctx_4  = cached_out.mujoco_qpos[0, cs : cs + 4]
                if ctx_4.shape[0] < 4:
                    pad   = np.tile(ctx_4[-1:], (4 - ctx_4.shape[0], 1))
                    ctx_4 = np.concatenate([ctx_4, pad])
                context = ctx_4[np.newaxis]

            # ── Map 30 Hz step → parquet index and build inputs ───────────────
            pidx_f = step * (_PARQUET_HZ / _PLANNER_HZ)
            p0     = min(int(pidx_f), n_parquet - 1)
            p1     = min(p0 + 1, n_parquet - 1)
            alpha  = pidx_f - int(pidx_f)

            def _lerp(key: str) -> np.ndarray:
                a = parquet_arrays[key][p0].astype(np.float32)
                b = parquet_arrays[key][p1].astype(np.float32)
                return (1.0 - alpha) * a + alpha * b

            chunk = {
                key: _lerp(key)[np.newaxis, np.newaxis]
                for key in ("planner_movement", "planner_facing",
                            "planner_speed", "planner_height")
            }
            inputs = build_planner_inputs(
                vla_action=chunk,
                context_mujoco_qpos=context,
                t_index=0, batch_index=0, clip_negative_speed=False,
            )
            inputs.mode = np.array([_get_mode(parquet_arrays, p0)], dtype=np.int64)

            out = planner.run(**inputs.as_kwargs())
            if out.num_pred_frames == 0:
                zero_frame_count += 1
            cached_out = out
            cached_log = {
                "movement": inputs.movement_direction[0].copy(),
                "facing":   inputs.facing_direction[0].copy(),
                "speed":    float(inputs.target_vel[0]),
                "height":   float(inputs.height[0]),
                "mode":     int(inputs.mode[0]),
            }

        # ── Record log (same command held for the full replan cycle) ─────────
        if cached_log is not None:
            planner_log["movement"][step] = cached_log["movement"]
            planner_log["facing"][step]   = cached_log["facing"]
            planner_log["speed"][step]    = cached_log["speed"]
            planner_log["height"][step]   = cached_log["height"]
            planner_log["mode"][step]     = cached_log["mode"]

        # ── Use frame traj_step from the cached trajectory ───────────────────
        if cached_out is not None and cached_out.num_pred_frames > 0:
            frame_idx       = min(traj_step, cached_out.num_pred_frames - 1)
            full_qpos[step] = cached_out.mujoco_qpos[0, frame_idx]

        if step % 30 == 0:
            lb_now    = full_qpos[step][7:7+NUM_JOINTS]
            mode_now  = cached_log["mode"] if cached_log else 0
            n_pred_now = cached_out.num_pred_frames if cached_out else 0
            print(f"[planner] step {step}/{n_steps}  mode={mode_now}  "
                  f"num_pred_frames={n_pred_now}  "
                  f"lb_joints_range=[{lb_now.min():.3f}, {lb_now.max():.3f}]")

    if zero_frame_count:
        print(f"[planner] WARNING: {zero_frame_count}/{n_steps} steps returned num_pred_frames=0")
    lb = full_qpos[:, 7:7+NUM_JOINTS]
    print(f"[planner] done — joint_angles range [{lb.min():.3f}, {lb.max():.3f}]")
    return full_qpos, planner_log


# ============================================================
# Viser visualisation
# ============================================================

_MODE_NAMES = {
    0: "IDLE", 1: "SLOW_WALK", 2: "WALK", 3: "RUN",
    4: "SQUAT", 5: "KNEELTWOLEGS", 6: "KNEELONELEG",
    7: "LYINGFACEDOWN", 8: "HANDCRAWLING", 9: "IDLEBOXING",
}


def visualise(
    full_qpos: np.ndarray,
    planner_log: dict,
    urdf_path: str,
    fps: float = 30.0,
) -> None:
    """Render via ViserUrdf on http://localhost:8082.

    Note: the g1_29dof.urdf omits the ±10° Y body-frame rotations that the
    MuJoCo XML applies to hip_roll/knee links, so individual joint orientations
    are approximate. Use visualise_mujoco for accurate FK.
    """
    import viser
    from viser.extras import ViserUrdf

    server = viser.ViserServer(port=8082)
    server.scene.world_axes.visible = True
    print("[viser] server running on http://localhost:8082")

    urdf_vis = ViserUrdf(server, Path(urdf_path), root_node_name="/robot")

    n_frames = full_qpos.shape[0]
    dt = 1.0 / fps

    with server.gui.add_folder("Playback"):
        gui_playing = server.gui.add_checkbox("Playing", initial_value=True)
        gui_slider  = server.gui.add_slider(
            "Frame", min=0, max=n_frames - 1, step=1, initial_value=0
        )

    with server.gui.add_folder("Planner Inputs"):
        gui_inputs_md = server.gui.add_markdown("loading…")

    print(f"[viser] {n_frames} frames at {fps:.0f} fps — Ctrl-C to exit …")

    _DIAG_FRAMES = 5

    _PINNED_POS  = np.array([0.0, 0.0, _MUJOCO_STANDING_Z])
    _PINNED_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

    def _apply_frame(idx: int) -> None:
        qpos   = full_qpos[idx]
        angles = qpos[7:7+NUM_JOINTS]

        if idx < _DIAG_FRAMES:
            root_wxyz = qpos[3:7]
            print(
                f"[diag frame {idx:3d}]  predicted_xy=[{qpos[0]:+.4f},{qpos[1]:+.4f}]"
                f"  z={qpos[2]:+.4f}"
                f"  wxyz=[{root_wxyz[0]:+.4f},{root_wxyz[1]:+.4f},{root_wxyz[2]:+.4f},{root_wxyz[3]:+.4f}]"
                f"  lb=[{angles.min():+.3f}…{angles.max():+.3f}]"
            )

        server.scene.add_frame("/robot", position=_PINNED_POS, wxyz=_PINNED_WXYZ, show_axes=False)
        urdf_vis.update_cfg({
            name: float(angles[i]) for i, name in enumerate(LOWER_BODY_JOINT_NAMES)
        })
        mv      = planner_log["movement"][idx]
        fac     = planner_log["facing"][idx]
        spd     = float(planner_log["speed"][idx])
        ht      = float(planner_log["height"][idx])
        mode_id = int(planner_log["mode"][idx])
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

    parquet_arrays, _ = load_parquet(args.parquet)
    full_qpos, planner_log = run_planner(parquet_arrays, args.planner_onnx)
    visualise(full_qpos, planner_log, urdf_path=args.urdf, fps=args.fps)


if __name__ == "__main__":
    main()
