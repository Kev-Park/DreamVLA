"""VLA + SONIC closed-loop *statistical* eval in Isaac Lab.

Runs the VLA→SONIC closed loop for a configurable number of episodes and reports
height-lift statistics — the VLA analog of ``eval_sonic_adapter.py``. No video is
written; see ``play_vla_sonic.py`` for the qualitative recording sibling (the two
share the exact same closed-loop wiring so the videos reflect what is measured here).

Pipeline per env step (inside the chunk, see --chunk-size):

    env obs ─▶ ObsToPolicyAdapter ─▶ Gr00tPolicy.get_action ─▶ action_dict
                                                                     │
           action_dict (first chunk step only) ──▶ PlannerWrapper ───▶ mujoco_qpos
                                                                     │
           anchor_pose + lb_trajectory + vr_3pt  ──▶ build_encoder_obs
                                                                     │
                                                            UtmWrapper.run_encoder → token
                                                                     │
                                       token + HistoryBuffer ──▶ build_decoder_obs
                                                                     │
                                                            UtmWrapper.run_decoder → body_29
                                                                     │
                body_29 + vla_fingers ──▶ utm_plus_vla_to_env_action → env_action_41
                                                                     │
                                                              env.step(env_action_41)

Reports (matching eval_sonic_adapter.py):
  1. ``Episodes with any lift`` — per-episode discrete success rate (the bottle
     cleared the lift threshold during the closed/grasp phase).
  2. ``Mean lift fraction`` — over lifted episodes, fraction of closed-phase steps
     the bottle stayed lifted (grasp-retention quality).
  3. ``Cumulative lift-steps`` — env.n_successes.sum() (the reward's own counter).
  4. ``Termination breakdown`` — time_out vs terminated.

NOTE: unlike eval_sonic_adapter.py, this eval CANNOT run thousands of parallel
envs — the VLA needs the ego camera every step (camera VRAM caps us at num_envs=1)
and the whole vla_sonic pipeline is single-env (numpy, index [0]). Episodes are
therefore run sequentially; --num-episodes is a sequential count, not parallel.

Run:

    cd WBCBenchmark/Training && python3 scripts/reinforcement_learning/rsl_rl/eval_vla_sonic.py \\
        --vla-checkpoint /home/dvij/kevin/checkpoints/run-01 \\
        --num-episodes 50
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
# Phase 1: Isaac Lab AppLauncher MUST come first (before any gym/torch that
# might touch omniverse).
# =========================================================================

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLA+SONIC closed-loop statistical eval")
    parser.add_argument("--task", default="Isaac-Motion-Tracking-Pick-Cam-ContFingers-v0")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Keep at 1 to avoid camera OOM (the VLA pipeline is single-env).")
    parser.add_argument("--num-episodes", type=int, default=20,
                        help="Number of episodes to run sequentially and aggregate stats over.")
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Execute first N of the VLA's 16 predicted steps before replanning.")
    parser.add_argument("--vla-checkpoint", required=True,
                        help="Path to a gear_sonic fine-tuned GR00T checkpoint dir (the "
                             "'new_embodiment' VLA that emits vr_3pt / motion_token). The "
                             "base nvidia/GR00T-N1.7-3B model does NOT work here — it lacks "
                             "the gear_sonic action heads, so a fine-tuned checkpoint is "
                             "mandatory. There is no released pretrained SONIC-token VLA.")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--language", default="pick up the mustard bottle")
    # Default paths assume DreamVLA/ and GR00T-WholeBodyControl/ are sibling repos,
    # and you run this script from DreamVLA/Training/. Override if your layout differs.
    parser.add_argument("--encoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx")
    parser.add_argument("--decoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")
    parser.add_argument("--planner-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx")
    parser.add_argument("--lift-thres", type=float, default=0.95,
                        help="Bottle z (m) above which a frame counts as 'lifted'. Matches the "
                             "env's object_above height_thres (object rests at 0.9; 0.95 = a "
                             "5 cm pickup).")
    parser.add_argument("--seed", type=int, default=0)
    # AppLauncher args get appended below.
    return parser


_parser = _parse_cli()

# Lazy import AppLauncher so --help works even without isaaclab installed.
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(_parser)
_ARGS = _parser.parse_args()

# Cameras always needed for the VLA's ego view — override if not set.
_ARGS.enable_cameras = True

app_launcher = AppLauncher(_ARGS)
_APP = app_launcher.app


# =========================================================================
# Phase 2: Heavy imports (safe now that AppLauncher is up).
# =========================================================================

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401  # registers tasks
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim import PinholeCameraCfg  # noqa: E402

# Ensure vla_sonic package is importable from its parent dir.
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
from vla_sonic.obs_to_policy import ObsAdapterConfig, ObsToPolicyAdapter  # noqa: E402
from vla_sonic.simple_robot_model import SimpleG1RobotModel  # noqa: E402

from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402


# =========================================================================
# Joint-order helpers: Isaac's robot → UTM's 29-DoF MuJoCo order.
# =========================================================================

# SONIC-IsaacLab 29-DoF joint order — the order the UTM ONNX models see on
# BOTH input (history joint positions/velocities/last_actions) and output
# (decoder body action). Reconstructed from
# GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/
#   policy_parameters.hpp:100 (isaaclab_to_mujoco = MuJoCo order in IsaacLab index).
# This interleaves left/right pairs at each kinematic level — very different
# from the obvious "left leg then right leg" DEFAULT_DOF_ANGLES ordering.
UTM_29_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "right_hip_pitch_joint",      # 1
    "waist_yaw_joint",            # 2
    "left_hip_roll_joint",        # 3
    "right_hip_roll_joint",       # 4
    "waist_roll_joint",           # 5  <-- drop when mapping to env
    "left_hip_yaw_joint",         # 6
    "right_hip_yaw_joint",        # 7
    "waist_pitch_joint",          # 8  <-- drop when mapping to env
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
    """Return (29,) array of Isaac indices s.t. ``isaac_q[perm] == utm_q``.

    Entries are -1 for UTM joints that don't exist on the Isaac robot
    (expected: waist_roll/pitch are absent from the env's 27-DoF G1). Callers
    must mask and zero-fill those entries via ``_gather_with_mask``.
    """
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
    """Apply the permutation, filling ``perm < 0`` positions with 0."""
    out = np.zeros(perm.shape[0], dtype=np.float32)
    valid = perm >= 0
    out[valid] = isaac_values[perm[valid]]
    return out


# =========================================================================
# Planner output decomposition.
# =========================================================================

# Planner output layout per frame (mujoco_qpos, 36-D):
#   [0:3]   root_pos  (world frame)
#   [3:7]   root_quat (wxyz scalar-first)
#   [7:36]  29 joints in UTM_29_JOINT_NAMES order
PLANNER_ROOT_POS_SLICE = slice(0, 3)
PLANNER_ROOT_QUAT_SLICE = slice(3, 7)
PLANNER_JOINTS_SLICE = slice(7, 36)

# Lower body = legs (12 joints). The encoder's
# ``motion_joint_positions_lowerbody_10frame_step5`` slot expects values in
# **MuJoCo order** — left-leg-all-6 (pitch, roll, yaw, knee, apitch, aroll)
# then right-leg-all-6 — per policy_parameters.hpp:93:
#   lower_body_joint_mujoco_order_in_mujoco_index = {0,1,2,3,4,5,6,7,8,9,10,11}
# The planner's mujoco_qpos is already in MuJoCo order; just slice the first
# 12 joint slots after the 7-element root prefix.
LOWER_BODY_QPOS_INDICES_MUJOCO_ORDER = np.array(
    [7 + i for i in range(12)], dtype=np.int64,
)

# Encoder expects 10 future frames at step 5 (frames [0, 5, 10, ..., 45]).
ENCODER_FUTURE_FRAME_INDICES = list(range(0, 50, 5))

# Planner raw output frame rate (before resampling, per planner_onnx.md).
PLANNER_OUTPUT_FPS = 30.0


def extract_anchor_pose(mujoco_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (anchor_pos_world, anchor_quat_wxyz) from planner frame 0.

    The caller must compute anchor_rot6d itself in C++ row-wise format:
        R_mat[:, :2].flatten('C') — identity → [1,0,0,1,0,0].
    Do NOT compute rot6d here; gear_sonic col-major format [col0,col1] differs
    at positions [3,4] and would feed wrong data to the encoder.
    """
    frame0 = mujoco_qpos[0, 0]  # (36,)
    anchor_pos = np.asarray(frame0[PLANNER_ROOT_POS_SLICE], dtype=np.float32).copy()
    anchor_quat = np.asarray(frame0[PLANNER_ROOT_QUAT_SLICE], dtype=np.float32).copy()
    return anchor_pos, anchor_quat


def extract_lower_body_future(mujoco_qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions (10,12), velocities (10,12)) at the encoder's subsample grid.

    The 12 joints are in MuJoCo order (left-leg-all-6 then right-leg-all-6):
    [L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll,
     R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_ankle_pitch, R_ankle_roll]
    This matches the encoder's training convention (policy_parameters.hpp:93).
    """
    qpos = np.asarray(mujoco_qpos[0], dtype=np.float32)  # (N, 36)
    n_frames = qpos.shape[0]
    need = max(ENCODER_FUTURE_FRAME_INDICES) + 1
    if n_frames < need:
        # Pad by repeating the last frame so the slicer doesn't error.
        pad = np.repeat(qpos[-1:], need - n_frames, axis=0)
        qpos = np.concatenate([qpos, pad], axis=0)
    # Gather the 12 lower-body joints in SONIC-IsaacLab interleaved order.
    lb_all = qpos[:, LOWER_BODY_QPOS_INDICES_MUJOCO_ORDER]  # (N, 12) MuJoCo order
    # Velocities at 30 Hz: central differences over the full trajectory, then subsample.
    vel_all = np.gradient(lb_all, 1.0 / PLANNER_OUTPUT_FPS, axis=0).astype(np.float32)
    pos = lb_all[ENCODER_FUTURE_FRAME_INDICES].astype(np.float32)  # (10, 12)
    vel = vel_all[ENCODER_FUTURE_FRAME_INDICES].astype(np.float32)  # (10, 12)
    return pos, vel


# =========================================================================
# VR 3-point extraction from the VLA action dict.
# =========================================================================

def extract_vr_3pt(
    vla_action: dict, *, t_index: int = 0, batch_index: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vr_3pt_position_world (9,), vr_3pt_rot6d (18,)) for one VLA step.

    The VLA emits these keys as (B, T, D) per ``gear_sonic_config.py``:
        vr_3pt_position    : (B, T, 9)   world-frame xyz per point
        vr_3pt_orientation : (B, T, 18)  rot6d per point (left_wrist, right_wrist, torso)
    """
    pos = np.asarray(vla_action["vr_3pt_position"], dtype=np.float32)
    orn = np.asarray(vla_action["vr_3pt_orientation"], dtype=np.float32)
    if pos.ndim != 3 or pos.shape[-1] != 9:
        raise ValueError(f"vr_3pt_position must be (B,T,9); got {pos.shape}")
    if orn.ndim != 3 or orn.shape[-1] != 18:
        raise ValueError(f"vr_3pt_orientation must be (B,T,18); got {orn.shape}")
    return pos[batch_index, t_index].copy(), orn[batch_index, t_index].copy()


# =========================================================================
# Camera injection — the env cfg's __post_init__ ran inside parse_env_cfg
# before we had a handle to flip `enable_cameras_for_collection`, so we wire
# the ego camera (which the VLA reads) directly into the scene config here.
# Only the ego d435 is injected — no third-person camera, since this is a
# headless statistical eval (no video). Matches collect_pick_cam.py:683-699.
# =========================================================================

def _inject_ego_camera(env_cfg) -> None:
    """Force-set the robot-mounted `camera_robot` ego camera on the scene cfg.

    Overwriting unconditionally because SceneCfg dataclasses often declare this
    field as an annotation even when the env cfg's ``__post_init__`` did not
    assign a value — ``hasattr`` returns True for annotated-but-unset.
    """
    env_cfg.scene.camera_robot = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/Camera_robot",
        spawn=PinholeCameraCfg(
            focal_length=7.6,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.01, 100.0),
        ),
        data_types=["rgb"],
        height=480, width=640,
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.36),
            rot=(0.568, 0.421, -0.421, -0.568),
            convention="opengl",
        ),
    )


# =========================================================================
# Main rollout.
# =========================================================================

def main() -> int:
    args = _ARGS

    # Deterministic motion draw (reproducible across runs).
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[eval_vla_sonic] seed = {args.seed}")

    # --- 1. Build env ---------------------------------------------------
    env_cfg = parse_env_cfg(
        args.task,
        device="cuda:0",
        num_envs=args.num_envs,
        enable_cameras=True,
    )
    env_cfg.seed = args.seed
    # Inject only the ego camera the VLA needs (no third-person camera — headless).
    _inject_ego_camera(env_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    print(f"[env] {args.task}  action_space={env.action_space}")

    # --- 2. Build VLA policy -------------------------------------------
    print(f"[vla] loading {args.vla_checkpoint}")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.vla_checkpoint,
        device="cuda:0",
    )

    # --- 3. Build SONIC wrappers ---------------------------------------
    print(f"[utm] encoder={args.encoder_onnx}")
    print(f"[utm] decoder={args.decoder_onnx}")
    utm = UtmWrapper(args.encoder_onnx, args.decoder_onnx)

    print(f"[planner] {args.planner_onnx}")
    planner = PlannerWrapper(args.planner_onnx)

    # --- 4. Obs adapter & joint-order perm -----------------------------
    robot_model = SimpleG1RobotModel.build()
    obs_adapter = ObsToPolicyAdapter(
        env,
        ObsAdapterConfig(
            language_instruction=args.language,
            robot_model=robot_model,
            camera_scene_key="camera_robot",
        ),
    )
    robot = env.unwrapped.scene["robot"]
    isaac_to_utm_perm = build_isaac_to_utm_perm(list(robot.data.joint_names))

    # The pick reward's ``object_above_threshold`` only increments its success
    # counter when ``hasattr(env, "n_successes")`` AND num_envs < 1001. Without
    # this init the counter never exists and cumulative lift-steps stays 0.
    env.unwrapped.n_successes = torch.zeros(env.num_envs, device="cuda:0", dtype=torch.float32)

    # --- 5. History buffer for decoder + planner context ---------------
    history = HistoryBuffer()

    # --- 6. Rollout setup ----------------------------------------------
    action_space_dim = env.action_space.shape[-1]
    zero_action = torch.zeros((args.num_envs, action_space_dim), device="cuda:0", dtype=torch.float32)
    lift_thres = args.lift_thres

    # Aggregate stats across episodes (single env → scalar bookkeeping).
    completed_episodes = 0
    completed_any_lift = 0
    sum_lift_fraction_over_lifted_episodes = 0.0
    completed_episodes_with_any_lift = 0
    termination_counts = {"time_out": 0, "terminated": 0}
    episode_lengths: list[int] = []

    print(f"[eval_vla_sonic] starting eval: num_episodes={args.num_episodes}, "
          f"max_steps_per_episode={args.max_steps_per_episode}, lift_thres={lift_thres}")
    t_start = time.time()

    for ep in range(args.num_episodes):
        print(f"\n[episode {ep}]")
        obs, info = env.reset()
        # Isaac Lab camera sensors are populated on env.step(), not env.reset().
        # One silent warm-up step fills the camera buffer so obs_adapter() can
        # read ego-view pixels without blocking on an empty output dict. Pump the
        # Omniverse event loop afterwards to flush the RTX render pipeline so the
        # first ego frame the VLA sees is current.
        env.step(zero_action)
        _APP.update()
        _APP.update()
        history.reset()
        prev_utm_body_29 = np.zeros(29, dtype=np.float32)

        # Per-episode lift trackers.
        had_any_lift = False
        closed_steps = 0
        lift_steps = 0

        vla_chunk: dict | None = None
        chunk_step = 0
        was_time_out = False
        step = 0

        for step in range(args.max_steps_per_episode):
            # 7a. Build VLA obs + refresh action chunk every `chunk_size` steps.
            if vla_chunk is None or chunk_step >= args.chunk_size:
                vla_obs = obs_adapter()
                vla_out = policy.get_action(vla_obs)
                # Gr00tPolicy.get_action returns either a dict or (dict, ...) tuple.
                vla_chunk = vla_out[0] if isinstance(vla_out, tuple) else vla_out
                chunk_step = 0
                if ep == 0 and step == 0:
                    print("\n[VLA @ ep0 step0] action-dict dump (t=0 slice, batch=0):")
                    for k in sorted(vla_chunk.keys()):
                        arr = np.asarray(vla_chunk[k])
                        slice_ = arr[0, 0] if arr.ndim == 3 else arr.reshape(-1)
                        print(f"  {k} [shape {tuple(arr.shape)}] = {slice_.round(4).tolist()}")

            # 7b. Extract this chunk-step's VLA slice.
            t_idx = chunk_step

            # 7c. Push current env state into history BEFORE planner so context is up to date.
            q_isaac = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            qd_isaac = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
            # History buffer expects joint state in SONIC-IsaacLab order (what the
            # UTM decoder was trained with). UTM_29_JOINT_NAMES is in SONIC order,
            # so isaac_to_utm_perm produces SONIC-ordered values.
            q_sonic = _gather_with_mask(q_isaac, isaac_to_utm_perm)
            qd_sonic = _gather_with_mask(qd_isaac, isaac_to_utm_perm)
            # The planner's context_mujoco_qpos needs MuJoCo order (per its name).
            q_mujoco = q_sonic[MUJOCO_TO_ISAACLAB]
            root_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
            root_quat_w = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)  # wxyz
            root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
            # Projected gravity in body frame.
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

            # 7d. Run planner using this chunk-step's VLA planner commands.
            planner_inputs = build_planner_inputs(
                vla_action=vla_chunk,
                context_mujoco_qpos=history.planner_context(),
                t_index=t_idx,
            )
            planner_out = planner.run(**planner_inputs.as_kwargs())

            # 7e. Extract anchor + lower-body trajectory from planner output.
            anchor_pos_w, anchor_quat_wxyz = extract_anchor_pose(planner_out.mujoco_qpos)
            # motion_anchor_orientation: relative rotation (robot_base_inv × planner_frame0)
            # as first-2-columns of the rotation matrix, flattened ROW-WISE
            # (identity → [1, 0, 0, 1, 0, 0]).
            _R_robot = R.from_quat(quat_wxyz_to_xyzw(root_quat_w))
            _R_anchor = R.from_quat(quat_wxyz_to_xyzw(anchor_quat_wxyz))
            _R_rel_mat = (_R_robot.inv() * _R_anchor).as_matrix().astype(np.float32)
            anchor_rot6d = _R_rel_mat[:, :2].flatten('C').astype(np.float32)  # C++ row-wise
            lb_pos, lb_vel = extract_lower_body_future(planner_out.mujoco_qpos)

            # 7f. VR 3-point from VLA (world frame; encoder builder does the anchor-local transform).
            vr_pos_world, vr_rot6d = extract_vr_3pt(vla_chunk, t_index=t_idx)

            # 7g. Build encoder obs → run encoder → build decoder obs → run decoder.
            enc_obs = build_encoder_obs(
                anchor_pos_world=anchor_pos_w,
                anchor_quat_wxyz=anchor_quat_wxyz,
                anchor_rot6d=anchor_rot6d,
                lower_body_positions_future=lb_pos,
                lower_body_velocities_future=lb_vel,
                # VLA output is already pelvis-local (matches converter's
                # subtract_frame_transforms step); no further transform needed.
                vr_3pt_position_anchor_local=vr_pos_world,
                vr_3pt_rot6d=vr_rot6d,
            )
            token = utm.run_encoder({"obs_dict": enc_obs}).reshape(-1)  # (64,)

            dec_hist = history.decoder_history()
            dec_obs = build_decoder_obs(
                token_state=token,
                **dec_hist.as_kwargs(),
            )
            utm_body_29 = utm.run_decoder({"obs_dict": dec_obs}).reshape(-1)  # (29,)

            # 7h. Assemble env action.
            env_action_np = utm_plus_vla_to_env_action(
                utm_body_29_sonic=utm_body_29,
                vla_action=vla_chunk,
                t_index=t_idx,
            )  # (41,)
            env_action = torch.as_tensor(env_action_np[None, :], device="cuda:0", dtype=torch.float32)

            # 7i. Step.
            obs, rew, term, trunc, info = env.step(env_action)
            # Flush render so the next chunk's obs_adapter() reads a fresh ego frame
            # (the VLA must not act on a one-step-stale image).
            _APP.update()
            _APP.update()

            # 7j. Lift bookkeeping — read bottle z + is_closed AFTER the step.
            bottle_z = float(env.unwrapped.scene["object"].data.root_pos_w[0, 2].item())
            motion_times = (
                env.unwrapped.episode_length_buf * env.unwrapped.step_dt
                + env.unwrapped.start_motion_times.clone().detach().to(
                    device="cuda:0", dtype=torch.float32)
            )
            motion_res = env.unwrapped.motion_lib.get_motion_state(
                env.unwrapped.motion_ids, motion_times)
            is_closed = bool(motion_res["is_closed"][0].item() > 0.5)
            lifted = (bottle_z > lift_thres) and is_closed
            if is_closed:
                closed_steps += 1
            if lifted:
                lift_steps += 1
                had_any_lift = True

            prev_utm_body_29 = utm_body_29
            chunk_step += 1

            # End on termination OR truncation (time-out). The env auto-resets
            # done envs on the NEXT step, so break here to keep one episode clean.
            term_flag = bool(term[0] if hasattr(term, "ndim") and term.ndim > 0 else term)
            trunc_flag = bool(trunc[0] if hasattr(trunc, "ndim") and trunc.ndim > 0 else trunc)
            if term_flag or trunc_flag:
                was_time_out = trunc_flag and not term_flag
                break

        # --- episode bookkeeping ---
        completed_episodes += 1
        episode_lengths.append(step + 1)
        if had_any_lift:
            completed_any_lift += 1
            if closed_steps > 0:
                sum_lift_fraction_over_lifted_episodes += lift_steps / closed_steps
                completed_episodes_with_any_lift += 1
        if was_time_out:
            termination_counts["time_out"] += 1
        else:
            termination_counts["terminated"] += 1

        n_succ_so_far = float(env.unwrapped.n_successes.sum().item())
        any_lift_rate = completed_any_lift / max(completed_episodes, 1)
        print(f"[episode {ep}] ended at step {step+1}  any_lift={had_any_lift}  "
              f"closed_steps={closed_steps}  lift_steps={lift_steps}  "
              f"(running any_lift_rate={any_lift_rate:.3f}, cumulative_lift_steps={n_succ_so_far:.0f})")

    elapsed = time.time() - t_start

    # --- final report (mirrors eval_sonic_adapter.py) ------------------
    n_succ_total = float(env.unwrapped.n_successes.sum().item())
    any_lift_rate = completed_any_lift / max(completed_episodes, 1)
    mean_lift_fraction = (
        sum_lift_fraction_over_lifted_episodes / max(completed_episodes_with_any_lift, 1)
        if completed_episodes_with_any_lift > 0 else 0.0
    )
    mean_ep_len = sum(episode_lengths) / max(len(episode_lengths), 1)

    print("\n" + "=" * 60)
    print("                  VLA + SONIC EVAL SUMMARY")
    print("=" * 60)
    print(f"  VLA checkpoint:             {args.vla_checkpoint}")
    print(f"  Task:                       {args.task}")
    print(f"  num_envs:                   {args.num_envs}")
    print(f"  chunk_size:                 {args.chunk_size}")
    print(f"  lift_thres:                 {lift_thres} m")
    print(f"  Completed episodes:         {completed_episodes}")
    print(f"  Mean episode length:        {mean_ep_len:.1f} steps")
    print(f"  Wall time:                  {elapsed:.1f}s")
    print(f"")
    print(f"  Episodes with any lift:     {completed_any_lift} / {completed_episodes} "
          f"= {100*any_lift_rate:.2f}%")
    print(f"  Mean lift fraction          {100*mean_lift_fraction:.2f}%")
    print(f"    (over episodes with lift; closed_phase_lift_steps / closed_phase_steps)")
    print(f"  Cumulative lift-steps       {n_succ_total:.0f}")
    print(f"    (env.n_successes.sum() — uses the env reward's height_thres)")
    print(f"")
    print(f"  Termination breakdown (of completed episodes):")
    for k, v in termination_counts.items():
        if v > 0:
            print(f"    {k:30s} {v}  ({100*v/max(completed_episodes,1):.1f}%)")
    print("=" * 60)

    env.close()
    _APP.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
