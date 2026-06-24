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
    parser.add_argument("--physics-preset", type=str, default="deploy", choices=["training", "deploy"],
                        help="Physics substep preset (both → 50 Hz control). 'deploy' (default) = "
                             "500 Hz/decimation-10, matches the real G1's 500 Hz motor rate and gives the "
                             "crispest live-feel motion. 'training' = 200 Hz/decimation-4, matches the "
                             "gear_sonic training sim substep (can feel sluggish/labored).")
    parser.add_argument("--waist-dof", type=int, default=29, choices=[27, 29],
                        help="Actuated body DOF. 29 (default, strict fidelity) actuates waist_roll/pitch "
                             "to match SONIC's 29-DOF training articulation — requires the 29-DOF+hands USD "
                             "(build via make_g1_29dof_with_hands.py). 27 = legacy welded-waist asset.")
    parser.add_argument("--robot-usd-29", type=str, default=None,
                        help="Explicit path to the 29-DOF+hands USD. Default: derive from the env's 27-DOF "
                             "asset filename (g1_27dof_..._white -> g1_29dof_..._white).")
    parser.add_argument("--kinematic-replay", action="store_true", default=False,
                        help="Pure kinematic playback of the --bypass-source trajectory: each step writes "
                             "the reference root+joints straight to the robot (no encoder/decoder, no physics "
                             "integration), then renders/records. Starts at --history-warmup-frames (same F0 "
                             "as the physics rollout) so the clip aligns frame-for-frame for comparison. Shows "
                             "the ideal body trajectory; does NOT exhibit contact with the bottle.")
    parser.add_argument("--history-warmup-frames", type=int, default=10,
                        help="Start the control loop this many parquet frames in (default 10), using the "
                             "PRECEDING frames as the decoder's causal proprioception history. The dataset "
                             "is skip-started (frame 0 is already mid-motion), so the robot is spawned in the "
                             "MOVING state at this frame (joint + root velocities from the parquet), not from "
                             "rest. Must be >= 10 to fill the 10-frame history.")
    parser.add_argument("--bypass-stride-hz", type=int, default=50, choices=[30, 50],
                        help="Reference-motion bypass: encoder lookahead temporal stride. "
                             "50 (default) = 50Hz stride-5 = 100ms per encoder frame, 0.9s lookahead — "
                             "matches paper Table 3 (δ_b = 0.1s) and upstream C++ ResampleGeneratedSequence50Hz. "
                             "30 = 30Hz stride-5 = 167ms per encoder frame, 1.5s lookahead — matches the "
                             "planner-path's native rate (cached_planner_out.mujoco_qpos at 30Hz). Use 30 if "
                             "100ms produces wrong cadence and 167ms (the planner-path's effective rate) "
                             "tracks the recorded trajectory better.")
    parser.add_argument("--bypass-source", type=str, default="executed",
                        choices=["executed", "reference"],
                        help="Source for the G1-mode body trajectory — ALL executed or ALL reference, never "
                             "spliced. 'executed' (default) reads observation.state + "
                             "observation.root_orientation (the robot's ACTUAL recorded state, internally "
                             "consistent with teleop.vr_3pt_*). 'reference' reads motion.reference_qpos "
                             "(root + joints) — the IDEAL kinematic trajectory the WBC tracked; smoother but "
                             "may diverge from executed under tracking error. The robot spawn + decoder "
                             "history are taken from the SAME source so the initial state matches what the "
                             "encoder/decoder track.")
    parser.add_argument("--velocity-source", type=str, default="parquet_50hz",
                        choices=["lookahead_gradient", "parquet_50hz"],
                        help="How to compute body_vel_future. "
                             "'parquet_50hz' (default) pre-computes velocities once per episode via "
                             "np.gradient at the parquet's native 50Hz (dt=20ms), then samples at the "
                             "encoder's lookahead indices — sharper, lower-noise velocities since the "
                             "finite-difference window is 40ms central instead of 200ms. "
                             "'lookahead_gradient' uses np.gradient on the sparse 100ms-stride lookahead "
                             "samples (original behavior, 200ms central diff windows).")
    parser.add_argument("--robot-spawn-offset-x", type=float, default=0.0,
                        help="Manual additive offset (meters) to robot spawn X position. Use the GRASP "
                             "printout's bottle-vs-hand offset to tune.")
    parser.add_argument("--robot-spawn-offset-y", type=float, default=0.0,
                        help="Manual additive offset (meters) to robot spawn Y position.")
    parser.add_argument("--robot-spawn-offset-z", type=float, default=0.0,
                        help="Manual additive offset (meters) to robot spawn Z position. Use with caution "
                             "(physics may reject extreme values).")
    parser.add_argument("--grasp-closure-threshold", type=float, default=0.5,
                        help="Mean right-hand finger joint angle (rad) above which a grasp is considered "
                             "executed. Default 0.5. Prints bottle↔hand offset at the first crossing of "
                             "this threshold each episode.")
    parser.add_argument("--right-hand-closure-scale", type=float, default=1.0,
                        help="Multiplicative scale applied to right-hand finger commands before they "
                             "reach the WBC. The recorded teleop.right_hand_joints column is the EXECUTED "
                             "finger state from the collection — if physical contact with the bottle "
                             "limited closure to e.g. 0.5 rad of magnitude, re-commanding that value in "
                             "eval reproduces the same loose grip. Values >1.0 amplify finger commands "
                             "proportionally (sign-preserving, since thumb closes positive but "
                             "index/middle close negative on G1). Try 1.3–2.0 to compensate for "
                             "under-gripping. No clamping is applied — WBC handles out-of-range commands.")
    parser.add_argument("--left-hand-closure-scale", type=float, default=1.0,
                        help="Same as --right-hand-closure-scale but for left hand.")
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
    build_g1_encoder_obs,
    build_planner_inputs,
    utm_plus_vla_to_env_action,
)
from scipy.spatial.transform import Rotation as R  # noqa: E402
from vla_sonic.action_assembler import (  # noqa: E402
    G1_ACTION_SCALE_SONIC,
    G1_DEFAULT_ANGLES_SONIC,
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    build_sonic29_to_env_perm,
    utm_plus_vla_to_env_action_dyn,
)
from vla_sonic.robot_29dof import apply_29dof_waist_override  # noqa: E402
from vla_sonic.frame_transforms import quat_wxyz_to_xyzw  # noqa: E402
from vla_sonic.physics_overrides import PHYSICS_PRESETS  # noqa: E402
from vla_sonic.planner_to_utm import ENCODER_TOTAL_DIM, DECODER_TOTAL_DIM  # noqa: E402


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

# Indices that select the 29 BODY joints (excluding fingers) from gear_sonic's 43-joint
# observation.state vector. gear_sonic's joint layout per features_sonic_vla.py's
# _JOINT_GROUPS_FOR_STATE list order is INTERLEAVED with hands between arms:
#   [left_leg(0-5), right_leg(6-11), waist(12-14), left_arm(15-21),
#    LEFT_HAND(22-28), right_arm(29-35), RIGHT_HAND(36-42)] = 43 total
# So body joints (legs+waist+arms = 29) live at indices [0..21] + [29..35] — NOT
# contiguous. Naive slicing [7:7+29] of motion.reference_qpos would pick up left_hand
# fingers as fake "right arm" joints and miss the real right_arm entirely; using these
# explicit indices instead delivers the 29-joint MUJOCO-grouped body vector that the
# G1-mode encoder's motion_joint_positions_10frame_step5 slot was trained on.
BODY_INDICES_IN_GEAR_SONIC = np.concatenate([
    np.arange(0, 22),   # legs (6+6) + waist (3) + left_arm (7) = 22
    np.arange(29, 36),  # right_arm (7)
]).astype(np.int64)
assert BODY_INDICES_IN_GEAR_SONIC.shape[0] == 29, "Body joint count mismatch"


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
    "teleop.vr_3pt_position":       ("vr_3pt_position",    np.float32),  # (9,) pelvis-local — _apply_local_offset in converter preserves local frame
    "teleop.vr_3pt_orientation":    ("vr_3pt_orientation", np.float32),  # (18,) pelvis-local rot6d
    "teleop.left_hand_joints":      ("left_hand_joints",   np.float32),  # (7,)
    "teleop.right_hand_joints":     ("right_hand_joints",  np.float32),  # (7,)
    # Canonical root orientation (wxyz float64) — used for spawn heading extraction.
    "observation.root_orientation": ("root_orientation",   np.float64),  # (4,)
    # Absolute robot position in the collection world frame.
    "teleop.root_pos_w":            ("root_pos_w",         np.float32),  # (3,)
    # Full 43-dim gear_sonic joint state (body 29 + left_hand 7 + right_hand 7) — used by
    # the G1-mode bypass to feed the encoder's full-body lookahead (BODY_INDICES_IN_GEAR_SONIC
    # extracts the 29 body joints; naive [:29] silently includes left_hand and misses right_arm).
    "observation.state":            ("obs_state",          np.float32),  # (43,) gear_sonic order
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
    # Both presets resolve to 50 Hz control. 'training' (200 Hz/dec-4) matches
    # gear_sonic training exactly and is the default; 'deploy' (500 Hz/dec-10) is a
    # finer substep that earlier A/B tests found produced cleaner/less-labored motion.
    _sim_dt, _decim = PHYSICS_PRESETS[args.physics_preset]
    env_cfg.sim.dt = _sim_dt
    env_cfg.sim.decimation = _decim
    print(f"[physics] preset='{args.physics_preset}': dt={_sim_dt:.5f}s, decimation={_decim} "
          f"→ {1.0 / (_sim_dt * _decim):.0f} Hz control")

    # --- 2c. Articulation solver iterations ---------------------------------
    try:
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        print("[physics] articulation solver pos_iter overridden to 8")
    except AttributeError:
        print("[physics] could not override articulation solver pos_iter")

    # --- 2d. 29-DOF articulation (strict fidelity: actuated waist roll/pitch) ---
    # SONIC trains on 29 DOF; promote the env to match (swaps USD, adds waist
    # actuator, extends the body action term 27->29). _body_action_joint_names is the
    # env's body-action joint order used to build the SONIC-29 -> env permutation.
    _body_action_joint_names = None
    if args.waist_dof == 29:
        _body_action_joint_names = apply_29dof_waist_override(env_cfg, usd_path=args.robot_usd_29)
    else:
        print("[29dof] --waist-dof 27: legacy welded-waist asset (waist roll/pitch fixed)")

    # Disable the red grab-location marker (env.goal_marker). It's drawn by the target_ref
    # obs terms from env.motion_lib at a RANDOM motion_id's grab point — misleading in
    # eval_parquet (the robot is driven by the parquet, not motion_lib). Setting
    # visualize_markers=False parks it below the floor; the obs tensor is unchanged. Mirrors
    # G1PickCamBinaryFingersEnvCfg (which the ContinuousFingers task does not inherit).
    _marker_off = 0
    for _term_name in ("target_ref_curr", "target_ref_next", "target_ref_next_next"):
        _term = getattr(env_cfg.observations.policy, _term_name, None)
        if _term is not None and isinstance(getattr(_term, "params", None), dict):
            _term.params = {**_term.params, "visualize_markers": False}
            _marker_off += 1
    print(f"[marker] disabled red goal_marker on {_marker_off} target_ref obs term(s)")

    # Kinematic replay: disable fall-terminations so a transient zero-action step can't
    # trip a DoneTerm and auto-reset the env while we're force-posing the robot each frame.
    if args.kinematic_replay:
        _disabled = []
        for _t in ("base_contact", "torso_below_threshold", "torso_angle_below_threshold"):
            if getattr(env_cfg.terminations, _t, None) is not None:
                setattr(env_cfg.terminations, _t, None)
                _disabled.append(_t)
        print(f"[kinematic] disabled terminations {_disabled} (kept time_out)")

    _inject_cameras(env_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    print(f"[env] {args.task}  action_space={env.action_space}")
    print(f"[render] has_rtx_sensors={env.unwrapped.sim.has_rtx_sensors()}")

    # --- 3. Build SONIC wrappers --------------------------------------
    print(f"[utm] encoder={args.encoder_onnx}")
    print(f"[utm] decoder={args.decoder_onnx}")
    utm     = UtmWrapper(args.encoder_onnx, args.decoder_onnx)
    # Fail fast if the downloaded ONNX doesn't match our obs layout (catches a
    # no-z 1751 encoder or a 4-frame-history 436 decoder before the rollout starts).
    _enc_dim = utm.encoder_inputs[0].shape[-1]
    _dec_dim = utm.decoder_inputs[0].shape[-1]
    if _enc_dim != ENCODER_TOTAL_DIM or _dec_dim != DECODER_TOTAL_DIM:
        raise RuntimeError(
            f"ONNX/layout mismatch: encoder input {_enc_dim} (expected {ENCODER_TOTAL_DIM}), "
            f"decoder input {_dec_dim} (expected {DECODER_TOTAL_DIM}). The encoder may be the "
            f"no-z (1751) release or the decoder a 4-frame-history (436) variant — re-download "
            f"the matching ONNX or update the layout in planner_to_utm.py."
        )
    print(f"[utm] ONNX dims OK: encoder={_enc_dim}, decoder={_dec_dim}")
    planner = PlannerWrapper(args.planner_onnx)

    # --- 4. Joint-order permutation -----------------------------------
    robot = env.unwrapped.scene["robot"]
    isaac_to_utm_perm = build_isaac_to_utm_perm(list(robot.data.joint_names))

    # SONIC-29 -> env body-action permutation (29-DOF strict-fidelity path). The body
    # action vector is ordered by the joint_pos action term's joint_names (preserve_order
    # =True), captured as _body_action_joint_names. Name-matched, so it adapts to the order.
    _body_perm = None
    if args.waist_dof == 29:
        _body_perm = build_sonic29_to_env_perm(_body_action_joint_names)
        _n_body = len(_body_perm)
        print(f"[29dof] SONIC-29 -> env body perm built ({_n_body} body joints, "
              f"action dim = {_n_body} body + 14 fingers = {_n_body + 14})")

    # --- 4b. Lookups for grasp detection -------------------------------
    # Right wrist body for hand-vs-bottle offset; right-hand finger joints for closure check.
    _right_wrist_ids, _ = robot.find_bodies(["right_wrist_yaw_link"])
    if len(_right_wrist_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one 'right_wrist_yaw_link' body match, got {_right_wrist_ids}"
        )
    _right_wrist_idx = int(_right_wrist_ids[0])
    _right_hand_joint_names = [
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    ]
    _name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    _right_hand_joint_ids = [_name_to_idx[n] for n in _right_hand_joint_names if n in _name_to_idx]
    if not _right_hand_joint_ids:
        print("[grasp] WARNING: no right-hand finger joints found by name — grasp detection disabled.")

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
            _spawn_pos_w = _collection_robot_pos0.copy()   # exact collection XY + Z
        else:
            _spawn_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
        # Apply CLI offsets for manual tuning of robot starting location.
        _spawn_offset = np.array(
            [_ARGS.robot_spawn_offset_x, _ARGS.robot_spawn_offset_y, _ARGS.robot_spawn_offset_z],
            dtype=np.float32,
        )
        _spawn_pos_w = _spawn_pos_w + _spawn_offset
        robot.write_root_pose_to_sim(
            torch.tensor(np.concatenate([_spawn_pos_w, _qc_wxyz])[None, :],
                         device="cuda:0", dtype=torch.float32),
            env_ids=torch.tensor([0], device="cuda:0"),
        )
        robot.write_root_velocity_to_sim(
            torch.zeros((1, 6), device="cuda:0"), env_ids=torch.tensor([0], device="cuda:0")
        )
        _offset_msg = ""
        if float(np.abs(_spawn_offset).sum()) > 1e-9:
            _offset_msg = f" (+ CLI offset {_spawn_offset.round(4).tolist()})"
        print(f"[spawn] robot restored to collection pos={_spawn_pos_w.round(4).tolist()} "
              f"yaw={np.degrees(_yaw_collection):.1f}°{_offset_msg}")

        # Diagnostic: compare REFERENCE frame-0 robot pose vs EXECUTED frame-0.
        # The robot spawn uses EXECUTED (teleop.root_pos_w + observation.root_orientation),
        # but the G1-mode encoder reads the REFERENCE trajectory (motion.reference_qpos).
        # If reference[0] ≠ executed[0], the robot starts at one location but the encoder's
        # trajectory is anchored to another — meaning the final root position (and therefore
        # arm reach to the bottle) will be offset by (executed[0] − reference[0]).
        if "reference_qpos" in streamer._arrays:
            _ref_qp_arr = streamer._arrays["reference_qpos"]
            _ref_pos0  = _ref_qp_arr[0, 0:3].astype(np.float32)
            _ref_quat0 = _ref_qp_arr[0, 3:7].astype(np.float32)
            _exec_pos0  = streamer._arrays["root_pos_w"][0].astype(np.float32)
            _exec_quat0 = streamer._arrays["root_orientation"][0].astype(np.float32)
            _pos_gap  = _exec_pos0 - _ref_pos0
            _pos_gap_mag = float(np.linalg.norm(_pos_gap))
            _ref_yaw  = float(R.from_quat(quat_wxyz_to_xyzw(_ref_quat0)).as_euler("zyx")[0])
            _exec_yaw = float(R.from_quat(quat_wxyz_to_xyzw(_exec_quat0)).as_euler("zyx")[0])
            _yaw_gap_deg = float(np.degrees(_exec_yaw - _ref_yaw))
            print("[spawn-cmp] frame 0 reference vs executed pose:")
            print(f"  reference root_pos = {_ref_pos0.round(4).tolist()}  yaw = {np.degrees(_ref_yaw):.2f}°")
            print(f"  executed  root_pos = {_exec_pos0.round(4).tolist()}  yaw = {np.degrees(_exec_yaw):.2f}°")
            print(f"  executed - reference = {_pos_gap.round(4).tolist()}  |Δ|={_pos_gap_mag:.4f}m"
                  f"  Δyaw={_yaw_gap_deg:+.2f}°")
            if _pos_gap_mag < 0.005 and abs(_yaw_gap_deg) < 0.5:
                print("  → frame-0 spawn matches reference (≤5mm, ≤0.5°). Bypass=reference is safe.")
            else:
                print(f"  → NONZERO gap. If using --bypass-source reference, robot will end up offset "
                      f"by ~{_pos_gap.round(4).tolist()}m from where the reference would have grabbed. "
                      f"Object spawn is also from executed, so object→robot relative geometry stays "
                      f"consistent with the recording. Use --bypass-source executed for full consistency.")

        # Per-episode grasp-event tracking. Fires once when right-hand mean finger angle
        # first crosses --grasp-closure-threshold (default 0.5 rad) — prints bottle vs
        # right-wrist coordinate offset so you can manually tune --robot-spawn-offset-*.
        _grasp_detected_this_episode = False

        env.step(zero_action)   # warm-up camera buffer + propagates object pose and heading overrides
        _APP.update()           # flush warm-up render to annotators
        _APP.update()

        # ---- Initialise the robot in the parquet's MOVING state at the warmup frame ----
        # The dataset is skip-started: frame 0 is already mid-motion (~0.8 m/s, ~8 rad/s
        # joint speed). Spawning from rest (zero root velocity + a near-static joint pose)
        # makes the robot lurch to catch the moving reference and never settles into a gait.
        # Set the FULL state (root pose+velocity, joint pos+velocity) from parquet frame
        # _F0, velocities by 50 Hz finite difference. Done AFTER the zero-action warm-up step
        # so that step (which commands a zero pose) doesn't snap the spawn. Source = executed
        # (post-WBC) recording, matching --bypass-source executed.
        _F0  = max(int(args.history_warmup_frames), 1)
        _fps = _PARQUET_HZ
        # Source the spawn + history from the SAME trajectory the encoder/decoder track
        # (--bypass-source), so the initial state matches what's being tracked. ALL-executed
        # or ALL-reference, never spliced.
        if args.bypass_source == "reference":
            _ref_qp       = streamer._arrays["reference_qpos"]   # (N, 50) root(3)+quat(4)+43 joints
            _joints_src   = _ref_qp[:, 7:]                       # (N, 43) gear_sonic joints
            _root_pos_arr = _ref_qp[:, 0:3]                      # (N, 3)
            _root_orn_arr = _ref_qp[:, 3:7]                      # (N, 4) wxyz
        else:  # executed
            _joints_src   = streamer._arrays["obs_state"]        # (N, 43) gear_sonic
            _root_pos_arr = streamer._arrays["root_pos_w"]       # (N, 3)
            _root_orn_arr = streamer._arrays["root_orientation"] # (N, 4) wxyz

        def _sonic_body29(f: int) -> np.ndarray:
            """source joints[f] (gear_sonic 43) -> SONIC-IsaacLab 29 body joints (abs radians)."""
            s = _joints_src[int(np.clip(f, 0, streamer.n_frames - 1))].astype(np.float32)
            return s[BODY_INDICES_IN_GEAR_SONIC][ISAACLAB_TO_MUJOCO]

        _q_sonic_f0  = _sonic_body29(_F0)
        _qd_sonic_f0 = (_sonic_body29(_F0) - _sonic_body29(_F0 - 1)) * _fps
        _jp = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32).copy()
        _jv = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32).copy()
        for _s in range(29):
            _idx = int(isaac_to_utm_perm[_s])
            if _idx >= 0:
                _jp[_idx] = _q_sonic_f0[_s]
                _jv[_idx] = _qd_sonic_f0[_s]
        _eid = torch.tensor([0], device="cuda:0")
        robot.write_joint_state_to_sim(
            torch.tensor(_jp[None, :], device="cuda:0", dtype=torch.float32),
            torch.tensor(_jv[None, :], device="cuda:0", dtype=torch.float32),
            env_ids=_eid,
        )
        _rp_f0 = _root_pos_arr[_F0].astype(np.float32) + np.array(
            [_ARGS.robot_spawn_offset_x, _ARGS.robot_spawn_offset_y, _ARGS.robot_spawn_offset_z],
            dtype=np.float32,
        )
        _rq_f0 = _root_orn_arr[_F0].astype(np.float32)          # wxyz, full (keeps roll/pitch)
        robot.write_root_pose_to_sim(
            torch.tensor(np.concatenate([_rp_f0, _rq_f0])[None, :], device="cuda:0", dtype=torch.float32),
            env_ids=_eid,
        )
        # Root velocity (WORLD frame): linear from pos finite-diff, angular from quat delta.
        _lin_w = (_root_pos_arr[_F0] - _root_pos_arr[_F0 - 1]).astype(np.float32) * _fps
        _Rprev = R.from_quat(quat_wxyz_to_xyzw(_root_orn_arr[_F0 - 1].astype(np.float32)))
        _Rcur  = R.from_quat(quat_wxyz_to_xyzw(_rq_f0))
        _ang_w = ((_Rcur * _Rprev.inv()).as_rotvec().astype(np.float32)) * _fps
        robot.write_root_velocity_to_sim(
            torch.tensor(np.concatenate([_lin_w, _ang_w])[None, :], device="cuda:0", dtype=torch.float32),
            env_ids=_eid,
        )
        env.unwrapped.sim.forward()   # propagate the written state into robot.data before the loop
        print(f"[spawn] moving-state init @ parquet frame {_F0}: pos={_rp_f0.round(3).tolist()} "
              f"|lin_vel|={float(np.linalg.norm(_lin_w)):.2f} m/s  "
              f"|joint_vel|max={float(np.abs(_qd_sonic_f0).max()):.2f} rad/s")

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

        # Pre-fill decoder history with the 10 parquet frames PRECEDING the start frame _F0
        # (genuine causal MOVING history) instead of a frozen frame-0 repeat. Positions
        # evolve consistently with velocities, giving the decoder a real gait phase. The old
        # frozen-repeat seeding told the decoder the robot was standing still, which (with the
        # from-rest spawn) caused the spawn lurch and the uncoordinated, settle-to-stand gait.
        _scale_inv_h = 1.0 / G1_ACTION_SCALE_SONIC
        for _hf in range(_F0 - 10, _F0):
            _hc   = int(np.clip(_hf, 0, streamer.n_frames - 1))
            _q_s  = _sonic_body29(_hc)
            _qd_s = (_sonic_body29(_hc) - _sonic_body29(_hc - 1)) * _fps
            _rqh  = _root_orn_arr[_hc].astype(np.float32)
            _rph  = _root_pos_arr[_hc].astype(np.float32)
            _Rh   = R.from_quat(quat_wxyz_to_xyzw(_rqh))
            _Rhp  = R.from_quat(quat_wxyz_to_xyzw(
                _root_orn_arr[int(np.clip(_hc - 1, 0, streamer.n_frames - 1))].astype(np.float32)))
            _angb = _Rh.inv().apply((_Rh * _Rhp.inv()).as_rotvec() * _fps).astype(np.float32)
            _grav = _Rh.inv().apply(np.array([0.0, 0.0, -1.0], dtype=np.float32)).astype(np.float32)
            _qdlt = _q_s - G1_DEFAULT_ANGLES_SONIC
            history.push(
                joint_pos=_qdlt,
                joint_vel=_qd_s,
                last_action=_qdlt * _scale_inv_h,
                base_ang_vel=_angb,
                gravity_dir=_grav,
                mujoco_qpos=np.concatenate([_rph, _rqh, _q_s[MUJOCO_TO_ISAACLAB]]).astype(np.float32),
            )
        print(f"[episode {ep}] decoder history seeded from parquet frames {_F0-10}..{_F0-1} (moving)")

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
        # Seed last-action from the frame just before the start so the first decode's
        # last_action history is consistent with the moving seed (not a zero frame).
        prev_utm_body_29 = ((_sonic_body29(_F0 - 1) - G1_DEFAULT_ANGLES_SONIC) * _scale_inv_h).astype(np.float32)

        # Start the control loop at _F0: frames [_F0-10 .. _F0-1] are the decoder's seeded
        # causal history (above), and the robot was spawned in frame-_F0's moving state.
        for step in range(_F0, max_steps):
            # Read current parquet frame for VR and finger data (planner drives locomotion;
            # one parquet frame per control step, no VLA chunk buffering needed).
            parquet_frame = streamer.get_lerp_frame(float(step))

            # ---- KINEMATIC REPLAY: force the robot onto the reference frame, no decoder/physics ----
            if args.kinematic_replay:
                # advance the env framework (sensors/render timing); physics result is discarded.
                env.step(zero_action)
                # overwrite the robot to the EXACT reference pose for this frame.
                _qref = _sonic_body29(step)
                _jpk = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32).copy()
                for _sk in range(29):
                    _ixk = int(isaac_to_utm_perm[_sk])
                    if _ixk >= 0:
                        _jpk[_ixk] = _qref[_sk]
                _rpk = _root_pos_arr[int(np.clip(step, 0, streamer.n_frames - 1))].astype(np.float32) + np.array(
                    [_ARGS.robot_spawn_offset_x, _ARGS.robot_spawn_offset_y, _ARGS.robot_spawn_offset_z],
                    dtype=np.float32)
                _rqk = _root_orn_arr[int(np.clip(step, 0, streamer.n_frames - 1))].astype(np.float32)
                _eidk = torch.tensor([0], device="cuda:0")
                robot.write_joint_state_to_sim(
                    torch.tensor(_jpk[None, :], device="cuda:0", dtype=torch.float32),
                    torch.zeros((1, robot.data.joint_vel.shape[1]), device="cuda:0", dtype=torch.float32),
                    env_ids=_eidk)
                robot.write_root_pose_to_sim(
                    torch.tensor(np.concatenate([_rpk, _rqk])[None, :], device="cuda:0", dtype=torch.float32),
                    env_ids=_eidk)
                robot.write_root_velocity_to_sim(
                    torch.zeros((1, 6), device="cuda:0", dtype=torch.float32), env_ids=_eidk)
                env.unwrapped.sim.forward()
                _APP.update(); _APP.update()
                if writers:
                    for _key, _w in writers.items():
                        _frame = _read_camera_rgb(env, _key)
                        if _frame is not None:
                            _w.write(_frame)
                if step < _F0 + 3:
                    print(f"[kinematic step {step}] robot set to reference pose")
                continue

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

            # 7d. Extract anchor + body trajectory for the encoder.
            #
            # Two paths are supported, switching on whether the parquet carries the
            # auxiliary motion.reference_qpos column:
            #
            # (A) PLANNER PATH (TELEOP mode, default fallback):
            #   Encoder mode 1 (teleop). lb_pos/lb_vel are taken from planner_sonic.onnx
            #   output, anchor is planner frame 0 (in planner-local frame → world via
            #   _R_robot_init). 30 Hz stride-5 = 167 ms per encoder frame, 1.5 s lookahead.
            #   See gear_sonic_deploy/g1_deploy_onnx_ref.cpp:GatherMotionJointPositionsMultiFrame
            #   for the upstream behavior this mirrors.
            #
            # (B) REFERENCE-MOTION BYPASS (G1 mode):
            #   Encoder mode 0 (g1). Full-body 29-joint future trajectory + per-frame
            #   anchor rot6d are read directly from motion.reference_qpos. No planner,
            #   no VR slots, no lossy command reconstruction. This is the canonical
            #   encoder mode for playing back recorded full-body motions per
            #   gear_sonic_deploy/g1_deploy_onnx_ref.cpp:2384 (loaded motions default
            #   to encode_mode=0). Stride matches the planner path (167 ms) for
            #   empirical consistency — sampled at parquet indices
            #   [0, 8, 17, 25, 33, 42, 50, 58, 67, 75] (rounded t·(50/30)).
            _ref_qpos_arr = streamer._arrays.get("reference_qpos")
            _use_ref_bypass = (
                _ref_qpos_arr is not None
                and float(np.abs(_ref_qpos_arr[0]).sum()) > 1e-6
            )
            # body_pos_future / body_vel_future / anchor_rot6d_future are only filled
            # in the bypass branch — used by build_g1_encoder_obs below.
            body_pos_future = None
            body_vel_future = None
            anchor_rot6d_future = None
            if _use_ref_bypass:
                # Bypass-stride options (CLI --bypass-stride-hz, default 50):
                #   50Hz stride-5 → 100ms per encoder frame, 0.9s lookahead, indices [0,5,...,45].
                #       Matches paper Table 3 (δ_b = 0.1s) and upstream C++ resampling.
                #   30Hz-equivalent stride-5 → 167ms per encoder frame, 1.5s lookahead,
                #       rounded 50Hz indices [0,8,17,25,33,42,50,58,67,75]. Matches the
                #       planner-path's native rate (cached_planner_out is 30Hz).
                if _ARGS.bypass_stride_hz == 50:
                    _PARQUET_LB_LOOKAHEAD_IDX_50HZ = np.arange(0, 50, 5, dtype=np.int64)
                    _PARQUET_LB_STEP_DT = 5.0 / 50.0  # 100 ms
                else:  # 30Hz
                    _PARQUET_LB_LOOKAHEAD_IDX_50HZ = np.array(
                        [0, 8, 17, 25, 33, 42, 50, 58, 67, 75], dtype=np.int64
                    )
                    _PARQUET_LB_STEP_DT = 5.0 / 30.0  # 167 ms
                # Pick the data source — both are in gear_sonic 43-joint order. EXECUTED
                # mode reads observation.state (the recorded robot state, post-WBC) and
                # observation.root_orientation; REFERENCE mode reads motion.reference_qpos
                # (the kinematic intent, pre-WBC). Executed gives a fully-internally-consistent
                # view of "what actually happened in the recording" at the cost of some
                # WBC-tracking noise; reference gives smoother kinematic intent but diverges
                # from the actual robot state under tracking error.
                _obs_state_arr_local = streamer._arrays["obs_state"]            # (N, 43) gear_sonic order
                _root_orn_arr_local  = streamer._arrays["root_orientation"]     # (N, 4) wxyz
                _N_src = _ref_qpos_arr.shape[0]
                _enc_idx_pq = np.clip(step + _PARQUET_LB_LOOKAHEAD_IDX_50HZ, 0, _N_src - 1)
                # Pick / cache source position array (full 43-joint gear_sonic).
                # Cache via streamer attribute so the mixed-mode .copy() + override only
                # happens once per episode (not once per env step).
                # ALL-executed or ALL-reference — never spliced.
                _src_pos_attr = f"_cached_src_pos_{_ARGS.bypass_source}"
                if not hasattr(streamer, _src_pos_attr):
                    if _ARGS.bypass_source == "reference":
                        _val = _ref_qpos_arr[:, 7:].astype(np.float32)          # reference joints
                    else:  # executed
                        _val = _obs_state_arr_local.astype(np.float32)          # executed joints
                    setattr(streamer, _src_pos_attr, _val)
                _src_pos_full = getattr(streamer, _src_pos_attr)
                # Root quat source for the anchor — from the SAME source as the joints.
                if _ARGS.bypass_source == "reference":
                    _root_quat_full = _ref_qpos_arr[:, 3:7]                     # (N, 4) wxyz
                else:  # executed
                    _root_quat_full = _root_orn_arr_local                       # (N, 4)
                # Pre-compute velocities at the parquet's native 50Hz (dt=20ms central diff)
                # once per source — sharper than np.gradient on the sparse 100ms-stride
                # lookahead. Cached on the streamer keyed by the source name.
                # When velocity-source == "lookahead_gradient" we skip the precompute and
                # use the original sparse-lookahead np.gradient downstream.
                if _ARGS.velocity_source == "parquet_50hz":
                    _vel_cache_attr = f"_cached_vel_{_ARGS.bypass_source}"
                    if not hasattr(streamer, _vel_cache_attr):
                        setattr(
                            streamer, _vel_cache_attr,
                            np.gradient(_src_pos_full, 1.0 / 50.0, axis=0).astype(np.float32),
                        )
                    _src_vel_full = getattr(streamer, _vel_cache_attr)
                # Window the source at the lookahead indices.
                _joints_window     = _src_pos_full[_enc_idx_pq]                 # (10, 43)
                _root_quat_window  = _root_quat_full[_enc_idx_pq]               # (10, 4)
                # Full-body 29-joint future for G1 mode's motion_joint_positions_10frame_step5 slot.
                # BODY_INDICES_IN_GEAR_SONIC (non-contiguous: skips left_hand fingers at gear_sonic
                # indices 22-28). Naive [:, 0:29] would silently include left_hand fingers and
                # exclude right_arm. Result is in MUJOCO-grouped order (gear_sonic.joint_names body slice).
                body_pos_future_mj = _joints_window[:, BODY_INDICES_IN_GEAR_SONIC].astype(np.float32)  # (10, 29)
                if _ARGS.velocity_source == "parquet_50hz":
                    # Sample the precomputed 50Hz velocities at the same lookahead indices.
                    body_vel_future_mj = (
                        _src_vel_full[_enc_idx_pq][:, BODY_INDICES_IN_GEAR_SONIC].astype(np.float32)
                    )
                else:
                    # Fallback: noisy 200ms-window finite differences on the sparse lookahead.
                    body_vel_future_mj = np.gradient(
                        body_pos_future_mj, _PARQUET_LB_STEP_DT, axis=0
                    ).astype(np.float32)
                # G1 encoder slot motion_joint_positions_10frame_step5 expects 29 joints in
                # SONIC-IsaacLab interleaved order (LHP, RHP, WY, LHR, RHR, WR, LHY, RHY, WP,
                # LK, RK, LSP, RSP, LAP, RAP, LSR, RSR, LAR, RAR, LSY, RSY, LE, RE, LWR, RWR,
                # LWP, RWP, LWY, RWY) — verified against
                # gear_sonic_deploy/.../policy_parameters.hpp:92.
                # ISAACLAB_TO_MUJOCO[i] = MJ index of the i-th IsaacLab joint.
                body_pos_future = body_pos_future_mj[:, ISAACLAB_TO_MUJOCO].astype(np.float32)  # (10, 29) IsaacLab order
                body_vel_future = body_vel_future_mj[:, ISAACLAB_TO_MUJOCO].astype(np.float32)
                # Per-frame anchor rot6d (relative to current robot's world orientation).
                # ROW-major flatten — matches what TELEOP mode's single-frame anchor uses
                # and is the format the encoder was empirically trained on.
                _R_robot = R.from_quat(quat_wxyz_to_xyzw(root_quat_w))
                anchor_rot6d_future = np.zeros((10, 6), dtype=np.float32)
                for _k in range(10):
                    _R_anchor_k = R.from_quat(quat_wxyz_to_xyzw(
                        _root_quat_window[_k].astype(np.float32)))
                    _R_rel_k = (_R_robot.inv() * _R_anchor_k).as_matrix().astype(np.float32)
                    anchor_rot6d_future[_k] = _R_rel_k[:, :2].flatten("C")
                # Single-frame mirrors of the above — kept for debug prints and so the
                # visualization writer / VR-passthrough block doesn't need a separate path.
                # Use the MUJOCO-ordered version (body_pos_future_mj) for lb_pos since
                # LOWER_BODY_OBS_STATE_INDICES = arange(12) selects the first-12 MUJOCO body
                # joints (left leg + right leg). Slicing the IsaacLab-permuted body_pos_future
                # would give a scrambled "lower body" since IsaacLab interleaves L/R per joint.
                lb_pos = body_pos_future_mj[:, LOWER_BODY_OBS_STATE_INDICES]                # (10, 12) MJ order
                lb_vel = body_vel_future_mj[:, LOWER_BODY_OBS_STATE_INDICES]
                _anchor_idx = int(np.clip(step, 0, _N_src - 1))
                # Single-frame anchor for vis/debug — matches the selected source.
                # anchor_pos_w is not actually fed to the encoder (only used for visualization);
                # both sources have a valid root position to draw from.
                if _ARGS.bypass_source == "reference":
                    anchor_pos_w     = _ref_qpos_arr[_anchor_idx, 0:3].astype(np.float32)
                    anchor_quat_wxyz = _ref_qpos_arr[_anchor_idx, 3:7].astype(np.float32)
                else:  # executed
                    anchor_pos_w     = streamer._arrays["root_pos_w"][_anchor_idx].astype(np.float32)
                    anchor_quat_wxyz = _root_orn_arr_local[_anchor_idx].astype(np.float32)
                anchor_rot6d     = anchor_rot6d_future[0]
                if step == 0:
                    _stride_ms = int(round(_PARQUET_LB_STEP_DT * 1000))
                    _lookahead_ms = int(round(_PARQUET_LB_STEP_DT * 9 * 1000))
                    _source_desc = {
                        "executed":  "observation.state + observation.root_orientation (post-WBC, internally consistent)",
                        "reference": "motion.reference_qpos (pre-WBC kinematic intent)",
                        "mixed":     "lower body from reference, upper body from observation.state, anchor from reference",
                    }[_ARGS.bypass_source]
                    print(f"[REF-BYPASS @ step 0] G1 mode + {_ARGS.bypass_source.upper()}-derived "
                          "body_pos_future/body_vel_future/anchor_rot6d_future")
                    print(f"  source              = {_ARGS.bypass_source} ({_source_desc})")
                    print(f"  stride              = {_stride_ms}ms (--bypass-stride-hz {_ARGS.bypass_stride_hz}), "
                          f"{_lookahead_ms}ms total lookahead")
                    print(f"  lookahead indices   = {_PARQUET_LB_LOOKAHEAD_IDX_50HZ.tolist()}")
                    print(f"  anchor_pos_w        = {anchor_pos_w.round(4).tolist()}")
                    print(f"  anchor_quat         = {anchor_quat_wxyz.round(4).tolist()}")
                    print(f"  anchor_rot6d[0]     = {anchor_rot6d_future[0].round(4).tolist()}")
                    print(f"  anchor_rot6d[9]     = {anchor_rot6d_future[9].round(4).tolist()}")
                    print(f"  body_pos_future[0]  (MUJOCO) = {body_pos_future_mj[0].round(4).tolist()}")
                    print(f"  body_pos_future[0]  (ISAAC)  = {body_pos_future[0].round(4).tolist()}")
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
            #
            # Bypass active: G1 mode (robot motion encoder φ_r per SONIC paper). The encoder
            # consumes 29-joint full-body future trajectory in IsaacLab-interleaved order.
            # body_pos_future / body_vel_future were permuted from MUJOCO-grouped above.
            # No bypass: TELEOP mode (hybrid encoder φ_b) using planner-derived lb_pos/lb_vel.
            if _use_ref_bypass:
                enc_obs = build_g1_encoder_obs(
                    body_positions_future=body_pos_future,
                    body_velocities_future=body_vel_future,
                    anchor_rot6d_future=anchor_rot6d_future,
                )
            else:
                enc_obs = build_encoder_obs(
                    anchor_pos_world=anchor_pos_w,
                    anchor_quat_wxyz=anchor_quat_wxyz,
                    anchor_rot6d=anchor_rot6d,
                    lower_body_positions_future=lb_pos,
                    lower_body_velocities_future=lb_vel,
                    vr_3pt_position_anchor_local=vr_pos_anchor_local,
                    vr_3pt_rot6d=vr_rot6d_anchor_local,
                )
            # Encoder ONNX bakes in full FSQ, so this token is already on the FSQ
            # lattice — feed it straight to the decoder, NO snap (see vla_sonic/fsq.py).
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
            if _body_perm is not None:
                # 29-DOF: name-matched body perm, no waist drop. Action = body_29 + 14 fingers.
                env_action_np = utm_plus_vla_to_env_action_dyn(
                    utm_body_29_sonic=utm_body_29,
                    vla_action=parquet_frame,
                    sonic_to_env_body_perm=_body_perm,
                    t_index=0,
                )
            else:
                # 27-DOF legacy: waist roll/pitch dropped.
                env_action_np = utm_plus_vla_to_env_action(
                    utm_body_29_sonic=utm_body_29,
                    vla_action=parquet_frame,
                    t_index=0,
                )
            _n_body_act = len(env_action_np) - 14  # 27 or 29 — finger slices below adapt

            # 7g-grip. Finger command post-processing for grip-strength tuning.
            # teleop.{left,right}_hand_joints in the parquet are EXECUTED finger angles
            # from the recording (joint_pos at collect-time), not commanded angles —
            # if collection-time contact with the bottle limited closure, replaying those
            # values reproduces the same loose grip. The --{left,right}-hand-closure-scale
            # flags multiply the finger commands; multiplication is sign-preserving so it
            # closes both the positive-direction-closing thumb joints and the
            # negative-direction-closing index/middle joints proportionally. No clamping
            # is applied — the WBC's internal joint-limit handling takes over for any
            # commands beyond physical range.
            _lh0, _rh0 = _n_body_act, _n_body_act + 7  # finger offsets adapt to 27/29 body
            if _ARGS.left_hand_closure_scale != 1.0:
                env_action_np[_lh0:_lh0 + 7] *= _ARGS.left_hand_closure_scale
            if _ARGS.right_hand_closure_scale != 1.0:
                env_action_np[_rh0:_rh0 + 7] *= _ARGS.right_hand_closure_scale

            if step < 3:
                print(f"[step {step}] utm_body_29[:15]       = {utm_body_29[:15].round(3).tolist()}")
                print(f"[step {step}]   env body[:12] (legs) = {env_action_np[:12].round(3).tolist()}")
                print(f"[step {step}]   env left  fingers    = {env_action_np[_lh0:_lh0 + 7].round(3).tolist()}")
                print(f"[step {step}]   env right fingers    = {env_action_np[_rh0:_rh0 + 7].round(3).tolist()}")

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

            # 7h-grasp. Detect first right-hand closure and print bottle↔hand offset.
            # Triggers once per episode on the rising edge of mean right-finger angle
            # crossing --grasp-closure-threshold. Use the printed XYZ offset to manually
            # tune --robot-spawn-offset-{x,y,z} so the next run's robot reaches the
            # bottle in time.
            if not _grasp_detected_this_episode and _right_hand_joint_ids:
                _rh_angles = robot.data.joint_pos[0, _right_hand_joint_ids].detach().cpu().numpy()
                _rh_mean = float(_rh_angles.mean())
                if _rh_mean > _ARGS.grasp_closure_threshold:
                    _grasp_detected_this_episode = True
                    _bottle_pos = (env.unwrapped.scene["object"].data.root_pos_w[0]
                                   .detach().cpu().numpy().astype(np.float32))
                    _wrist_pos = (robot.data.body_pos_w[0, _right_wrist_idx]
                                  .detach().cpu().numpy().astype(np.float32))
                    _offset_bw = _bottle_pos - _wrist_pos
                    _dist = float(np.linalg.norm(_offset_bw))
                    print(f"\n[GRASP @ step {step}] right hand closed "
                          f"(mean finger angle = {_rh_mean:.3f} rad, "
                          f"threshold = {_ARGS.grasp_closure_threshold:.3f})")
                    print(f"  right fingers    = {_rh_angles.round(3).tolist()}  "
                          f"(thumb_0/1/2, index_0/1, middle_0/1)")
                    print(f"  bottle pos       = {_bottle_pos.round(4).tolist()}  (world XYZ, m)")
                    print(f"  right wrist pos  = {_wrist_pos.round(4).tolist()}  (world XYZ, m)")
                    print(f"  bottle - wrist   = {_offset_bw.round(4).tolist()}  "
                          f"(world XYZ offset; +X is forward, +Y is left, +Z is up)")
                    print(f"  euclidean dist   = {_dist:.4f} m")
                    print(f"  → to close gap, try --robot-spawn-offset-x {_offset_bw[0]:+.3f} "
                          f"--robot-spawn-offset-y {_offset_bw[1]:+.3f}")
                    if _rh_mean < 1.0:
                        print(f"  → if grip is loose, try --right-hand-closure-scale "
                              f"{max(1.0, 1.2/_rh_mean):.2f} or --hand-closure-latch\n")
                    else:
                        print("")

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
