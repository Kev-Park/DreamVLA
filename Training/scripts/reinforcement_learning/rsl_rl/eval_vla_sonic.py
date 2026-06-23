"""VLA (unitree_g1_sonic) + SONIC decoder closed-loop *statistical* eval in Isaac Lab.

Runs a ``unitree_g1_sonic`` VLA in closed loop for a configurable number of
episodes and reports height-lift statistics. VLA analog of ``eval_sonic_adapter.py``.
No video is written; see ``play_vla_sonic.py`` for the recording sibling (the two
share the exact same closed-loop wiring).

This VLA embodiment predicts the SONIC latent token DIRECTLY (``motion_token``,
64-D) plus the finger joints, so the pipeline is short — no kinematic planner,
no encoder, no vr_3pt teleop stage:

    env obs ─▶ ObsToPolicyAdapter ─▶ Gr00tPolicy.get_action ─▶ action_dict
                                                                     │
                          action_dict["motion_token"] (64-D)  ───────┤
                                                                     │
                          token + HistoryBuffer ──▶ build_decoder_obs
                                                                     │
                                              UtmWrapper.run_decoder → body_29
                                                                     │
        body_29 + action_dict["{left,right}_hand_joints"] ──▶ utm_plus_vla_to_env_action → env_action_41
                                                                     │
                                                              env.step(env_action_41)

(The older vr_3pt → planner → encoder → token path was for the ``new_embodiment``
VLA formulation; a ``unitree_g1_sonic`` VLA outputs the token itself, replacing
that whole front half.)

Reports (matching eval_sonic_adapter.py):
  1. ``Episodes with any lift`` — per-episode discrete success rate.
  2. ``Mean lift fraction`` — over lifted episodes, fraction of closed-phase steps lifted.
  3. ``Cumulative lift-steps`` — env.n_successes.sum().
  4. ``Termination breakdown`` — time_out vs terminated.

NOTE: cameras are required (the VLA needs the ego view) and the pipeline is
single-env (numpy, index [0]), so episodes run sequentially; --num-episodes is a
sequential count, not parallel.

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

# This VLA embodiment predicts the SONIC token directly. Its tag is baked into
# the checkpoint as ``unitree_g1_sonic`` (NOT ``new_embodiment`` — that was the
# older vr_3pt formulation). Overridable via --embodiment-tag.
DEFAULT_EMBODIMENT_TAG = "unitree_g1_sonic"


# =========================================================================
# Phase 1: Isaac Lab AppLauncher MUST come first (before any gym/torch that
# might touch omniverse).
# =========================================================================

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLA(unitree_g1_sonic)+SONIC closed-loop statistical eval")
    parser.add_argument("--task", default="Isaac-Motion-Tracking-Pick-Cam-ContFingers-v0")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Keep at 1 to avoid camera OOM (the VLA pipeline is single-env).")
    parser.add_argument("--num-episodes", type=int, default=20,
                        help="Number of episodes to run sequentially and aggregate stats over.")
    parser.add_argument("--max-steps-per-episode", type=int, default=500)
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Execute first N of the VLA's predicted steps before replanning.")
    parser.add_argument("--vla-checkpoint", required=True,
                        help="Path to the unitree_g1_sonic fine-tuned GR00T checkpoint dir "
                             "(the VLA that emits motion_token + hand joints).")
    parser.add_argument("--embodiment-tag", default=DEFAULT_EMBODIMENT_TAG)
    parser.add_argument("--language", default="pick up the mustard bottle")
    # Default paths assume DreamVLA/ and GR00T-WholeBodyControl/ are sibling repos,
    # and you run this script from DreamVLA/Training/. Override if your layout differs.
    # Only the decoder is used; the encoder is loaded by UtmWrapper but never run
    # (this VLA replaces the encoder by predicting the token directly).
    parser.add_argument("--encoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx",
                        help="Loaded by UtmWrapper for consistency but NOT used in this pipeline.")
    parser.add_argument("--decoder-onnx",
                        default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx")
    parser.add_argument("--lift-thres", type=float, default=0.95,
                        help="Bottle z (m) above which a frame counts as 'lifted'. Matches the "
                             "env's object_above height_thres (object rests at 0.9; 0.95 = 5 cm).")
    parser.add_argument("--no-fsq-snap", dest="fsq_snap", action="store_false", default=True,
                        help="Disable snapping the VLA's continuous motion_token onto the FSQ "
                             "lattice (32 levels) before the decoder. Snapping is ON by default "
                             "because the decoder was trained on on-grid tokens; pass this to A/B "
                             "test whether the snap helps or hurts your checkpoint.")
    parser.add_argument("--skip-start-frames", type=int, default=None,
                        help="Start episodes N frames into the motion (e.g. 20 skips the "
                             "refinement's interpolate-to-initial-pose prepend; keeps the walk). "
                             "MUST match the value used to COLLECT the training data, or the VLA "
                             "starts out-of-distribution.")
    parser.add_argument("--start-pregrab-margin", type=float, default=None,
                        help="Start episodes this many seconds before the grab (drops the prepend "
                             "AND the walk approach). MUST match the training collection setting "
                             "for an in-distribution eval.")
    parser.add_argument("--raw-visuals", action="store_true", default=False,
                        help="Skip matching the ego view to the collection scene. By default the "
                             "red grab marker, kitchen glass-bottle prop, and ground plane are "
                             "hidden so the VLA sees the same clean view it trained on.")
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
from isaaclab.managers import EventTermCfg  # noqa: E402

# Ensure vla_sonic package is importable from its parent dir.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from vla_sonic import (  # noqa: E402
    HistoryBuffer,
    UtmWrapper,
    build_decoder_obs,
    utm_plus_vla_to_env_action,
)
from vla_sonic.action_assembler import G1_ACTION_SCALE_SONIC, G1_DEFAULT_ANGLES_SONIC  # noqa: E402
from vla_sonic.obs_to_policy import ObsAdapterConfig, ObsToPolicyAdapter  # noqa: E402
from vla_sonic.physics_overrides import apply_sonic_physics_overrides  # noqa: E402
from vla_sonic.simple_robot_model import SimpleG1RobotModel  # noqa: E402

from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402


# =========================================================================
# Joint-order helpers: Isaac's robot → UTM's 29-DoF SONIC-IsaacLab order.
# The decoder's proprio history (joint pos/vel/last-action) is in SONIC order.
# =========================================================================

# SONIC-IsaacLab 29-DoF joint order — the order the UTM decoder sees on its
# history inputs and emits on its body output. Reconstructed from
# policy_parameters.hpp:100 (isaaclab_to_mujoco). Interleaves left/right pairs.
UTM_29_JOINT_NAMES = [
    "left_hip_pitch_joint",       # 0
    "right_hip_pitch_joint",      # 1
    "waist_yaw_joint",            # 2
    "left_hip_roll_joint",        # 3
    "right_hip_roll_joint",       # 4
    "waist_roll_joint",           # 5  <-- absent on env's 27-DoF G1 (zero-filled)
    "left_hip_yaw_joint",         # 6
    "right_hip_yaw_joint",        # 7
    "waist_pitch_joint",          # 8  <-- absent on env's 27-DoF G1 (zero-filled)
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

    Entries are -1 for UTM joints absent on the Isaac robot (waist_roll/pitch on
    the 27-DoF G1). Callers must mask + zero-fill via ``_gather_with_mask``.
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
# VLA SONIC-token extraction.
# =========================================================================

def extract_motion_token(
    vla_action: dict, *, t_index: int = 0, batch_index: int = 0
) -> np.ndarray:
    """Return the (64,) SONIC latent token for one VLA step.

    The ``unitree_g1_sonic`` VLA emits ``motion_token`` as (B, T, 64) — the same
    64-D latent the SONIC encoder would otherwise produce. Fed straight into the
    decoder's ``token_state`` slot.
    """
    if "motion_token" not in vla_action:
        raise KeyError(
            f"vla_action has no 'motion_token'; got {sorted(vla_action.keys())}. "
            f"Is this a unitree_g1_sonic checkpoint? (--embodiment-tag)"
        )
    tok = np.asarray(vla_action["motion_token"], dtype=np.float32)
    if tok.ndim != 3 or tok.shape[-1] != 64:
        raise ValueError(f"motion_token must be (B,T,64); got {tok.shape}")
    return tok[batch_index, t_index].copy()


def fsq_snap_token(token: np.ndarray) -> np.ndarray:
    """Snap a continuous token onto SONIC's FSQ lattice (32 levels, step 1/16).

    The SONIC decoder was trained on FSQ-quantized tokens — exact grid points
    k/16 in [-1, 15/16]. The encoder ONNX emits these directly; a VLA regresses a
    continuous APPROXIMATION of them. Snapping recovers the on-grid values the
    decoder expects (identical to the lattice snap in token_adapter_wrapper.py).
    """
    half_width = 16.0  # 32 FSQ levels → half_width = 32 // 2
    return np.clip(
        np.round(token * half_width) / half_width,
        -1.0, (half_width - 1.0) / half_width,
    ).astype(np.float32)


# =========================================================================
# Camera injection — inject only the ego camera the VLA reads (headless eval,
# no third-person camera). Matches collect_pick_cam.py:683-699.
# =========================================================================

def _inject_ego_camera(env_cfg) -> None:
    """Force-set the robot-mounted `camera_robot` ego camera on the scene cfg."""
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
# Match the ego view to the collection (training) scene.
#
# The VLA's training data was collected by collect_sonic_adapter.py, whose env
# hides the red grab marker, the kitchen USD's decorative glass-bottle prop, and
# the ground plane from the ego view. Leaving those visible in eval is perception
# OOD for the vision-conditioned VLA, so we replicate that clean view here.
# =========================================================================

def _hide_eval_clutter(env, env_ids=None) -> None:
    """Startup event: hide the ground plane + the kitchen USD's glass-bottle prop.

    Toggles USD visibility only — colliders/physics untouched. The mustard manipuland
    (``/env_*/Object``) is never hidden.
    """
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    hidden = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName().lower()
        is_ground = path == "/World/ground" or path.startswith("/World/ground/")
        # Decorative glass bottle inside the kitchen USD ("/Kitchen"); never the
        # mustard manipuland under "/Object".
        is_kitchen_bottle = ("/Kitchen" in path) and ("/Object" not in path) and ("bottle" in name)
        if is_ground or is_kitchen_bottle:
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden += 1
    print(f"[match-visuals] hid {hidden} clutter prim(s) (ground plane + kitchen glass bottle)")


def _match_collection_visuals(env_cfg) -> None:
    """Disable the red grab marker and schedule the clutter-hide startup event."""
    # The red grab marker (env.goal_marker) is drawn by the target_ref obs terms when
    # visualize_markers=True. Disabling parks it below the floor (matches the collection env).
    pol = getattr(env_cfg.observations, "policy", None)
    if pol is not None:
        for _t in ("target_ref_curr", "target_ref_next", "target_ref_next_next"):
            term = getattr(pol, _t, None)
            if term is not None and isinstance(getattr(term, "params", None), dict):
                term.params = {**term.params, "visualize_markers": False}
    # Ground plane + kitchen glass bottle: hidden at startup (visibility only).
    env_cfg.events.match_visuals_hide = EventTermCfg(func=_hide_eval_clutter, mode="startup")
    print("[match-visuals] red grab marker disabled; ground + glass-bottle hide scheduled "
          "(matches collect_sonic_adapter.py ego view)")


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
    # Match the SONIC decoder's training-time physics (500 Hz substep, fixed
    # friction, self-collisions, solver iters). The decoder emits ABSOLUTE joint
    # targets calibrated for this regime; without it the env's default randomized
    # friction / coarser substep / self-collisions-off make the robot unstable.
    # Same call both working SONIC scripts use (eval/play_sonic_adapter.py).
    apply_sonic_physics_overrides(env_cfg)
    _inject_ego_camera(env_cfg)
    # The env injects a 1920x2560 third-person `camera` under enable_cameras=True; eval
    # never uses it (no video) — drop it to save VRAM/render time (keeps only the ego cam).
    if getattr(env_cfg.scene, "camera", None) is not None:
        env_cfg.scene.camera = None
    # Episode-start offset — MUST match the training-data collection (collect_sonic_adapter.py
    # has the same flags). If the data was collected near the grab (skip-start / pregrab-margin)
    # but eval starts at the full motion beginning, the VLA faces a walk-up it never trained on
    # → out-of-distribution start → falls / poor grasps / premature terminations.
    if args.start_pregrab_margin is not None:
        env_cfg.motion_start_pregrab_margin_s = args.start_pregrab_margin
        print(f"[eval_vla_sonic] start_pregrab_margin = {args.start_pregrab_margin}s")
    if args.skip_start_frames is not None:
        env_cfg.motion_skip_start_frames = args.skip_start_frames
        print(f"[eval_vla_sonic] skip_start_frames = {args.skip_start_frames}")
    # Match the collection (training) ego view: hide red grab marker / glass bottle / ground.
    if not args.raw_visuals:
        _match_collection_visuals(env_cfg)
    # render_mode="rgb_array" activates the RTX render product so the camera annotators
    # actually receive frames. Without it camera_robot.data.output["rgb"] comes back
    # empty (1-D) and the obs adapter's permute fails. Matches the working camera scripts
    # (collect_sonic_adapter.py / play_sonic_adapter.py).
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    print(f"[env] {args.task}  action_space={env.action_space}")
    # Pump the Omniverse event loop so RTX shaders/materials/texture streaming finish
    # loading before the first camera read (early frames otherwise render against
    # partially-loaded assets). Mirrors the 60-pump warm-up in the camera scripts.
    for _ in range(60):
        _APP.update()

    # --- 2. Build VLA policy -------------------------------------------
    print(f"[vla] loading {args.vla_checkpoint}  (embodiment={args.embodiment_tag})")
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.vla_checkpoint,
        device="cuda:0",
    )

    # --- 3. Build SONIC decoder (encoder loaded but unused) ------------
    print(f"[utm] decoder={args.decoder_onnx}")
    utm = UtmWrapper(args.encoder_onnx, args.decoder_onnx)

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
    env.unwrapped.n_successes = torch.zeros(env.unwrapped.num_envs, device="cuda:0", dtype=torch.float32)

    # --- 5. Decoder proprio-history buffer -----------------------------
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
        # Camera sensors populate on env.step(), not env.reset(). Warm-up step +
        # render flush so the first ego frame the VLA sees is current.
        env.step(zero_action)
        _APP.update()
        _APP.update()
        history.reset()
        prev_utm_body_29 = None  # set to (q-default)/scale on frame 0, then decoder output

        # Per-episode lift trackers.
        had_any_lift = False
        closed_steps = 0
        lift_steps = 0
        # reset_object_state drops the bottle from z=1.0 each episode; it settles to its
        # rest over the first frames, so it STARTS above lift_thres. Only count a lift once
        # the bottle has been seen at/below the threshold (i.e. reached rest) — otherwise the
        # reset-drop transient, OR a resting height already above thres (e.g. the kitchen
        # counter), registers as a spurious lift (lift_steps == closed_steps every episode).
        # Same gate as collect_sonic_adapter.py.
        object_settled = False

        vla_chunk: dict | None = None
        chunk_step = 0
        was_time_out = False
        step = 0

        for step in range(args.max_steps_per_episode):
            # 7a. Build VLA obs + refresh action chunk every `chunk_size` steps.
            if vla_chunk is None or chunk_step >= args.chunk_size:
                vla_obs = obs_adapter()
                vla_out = policy.get_action(vla_obs)
                vla_chunk = vla_out[0] if isinstance(vla_out, tuple) else vla_out
                chunk_step = 0
                if ep == 0 and step == 0:
                    print("\n[VLA @ ep0 step0] action-dict dump (t=0 slice, batch=0):")
                    for k in sorted(vla_chunk.keys()):
                        arr = np.asarray(vla_chunk[k])
                        slice_ = arr[0, 0] if arr.ndim == 3 else arr.reshape(-1)
                        prev = slice_.reshape(-1)[:8]
                        print(f"  {k} [shape {tuple(arr.shape)}] = {prev.round(4).tolist()}"
                              f"{' ...' if slice_.size > 8 else ''}")

            t_idx = chunk_step

            # 7b. Push current env state into the decoder history. Convention
            # matches the validated token_action_wrapper.py: joint positions are
            # RELATIVE to the SONIC default standing pose, gravity is IsaacLab's
            # body-frame projected gravity (NOT recomputed), and last_action is the
            # previous decoder output (seeded on frame 0 with the latent that
            # reproduces the current pose, q-default/scale). Feeding raw absolute
            # joint positions here is off-distribution for the decoder → instability.
            q_isaac = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
            qd_isaac = robot.data.joint_vel[0].detach().cpu().numpy().astype(np.float32)
            q_sonic = _gather_with_mask(q_isaac, isaac_to_utm_perm)
            qd_sonic = _gather_with_mask(qd_isaac, isaac_to_utm_perm)
            jp_sonic = (q_sonic - G1_DEFAULT_ANGLES_SONIC).astype(np.float32)
            gravity_body = robot.data.projected_gravity_b[0].detach().cpu().numpy().astype(np.float32)
            root_ang_vel_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy().astype(np.float32)
            if prev_utm_body_29 is None:
                last_action = (jp_sonic / G1_ACTION_SCALE_SONIC).astype(np.float32)
            else:
                last_action = prev_utm_body_29
            history.push(
                joint_pos=jp_sonic,
                joint_vel=qd_sonic,
                last_action=last_action,
                base_ang_vel=root_ang_vel_b,
                gravity_dir=gravity_body,
                mujoco_qpos=np.zeros(36, dtype=np.float32),  # unused (no planner)
            )

            # 7c. Token comes straight from the VLA (no planner/encoder).
            token = extract_motion_token(vla_chunk, t_index=t_idx)  # (64,)
            if ep == 0 and step == 0 and float(np.abs(token).max()) < 1e-6:
                print("\n[WARN] motion_token is ALL ZERO — the VLA is emitting a null SONIC "
                      "token, so the robot is NOT VLA-controlled (the decoder runs on a constant "
                      "zero token → nominal gait). Likely cause: action.motion_token was "
                      "zero-filled in the training dataset (the populator in PARQUET_POPULATE_PLAN.md "
                      "was never run) or its normalization stats are degenerate. Any lift/success "
                      "numbers below are meaningless until this is fixed.\n")
            if args.fsq_snap:
                token = fsq_snap_token(token)

            # 7d. Decoder: token + proprio history → 29-D SONIC body action.
            dec_hist = history.decoder_history()
            dec_obs = build_decoder_obs(token_state=token, **dec_hist.as_kwargs())
            utm_body_29 = utm.run_decoder({"obs_dict": dec_obs}).reshape(-1)  # (29,)

            # 7e. Assemble env action (body + VLA fingers).
            env_action_np = utm_plus_vla_to_env_action(
                utm_body_29_sonic=utm_body_29,
                vla_action=vla_chunk,
                t_index=t_idx,
            )  # (41,)
            env_action = torch.as_tensor(env_action_np[None, :], device="cuda:0", dtype=torch.float32)

            # 7f. Step.
            obs, rew, term, trunc, info = env.step(env_action)
            _APP.update()
            _APP.update()

            # 7g. Lift bookkeeping — read bottle z + is_closed AFTER the step.
            bottle_z = float(env.unwrapped.scene["object"].data.root_pos_w[0, 2].item())
            motion_times = (
                env.unwrapped.episode_length_buf * env.unwrapped.step_dt
                + env.unwrapped.start_motion_times.clone().detach().to(
                    device="cuda:0", dtype=torch.float32)
            )
            motion_res = env.unwrapped.motion_lib.get_motion_state(
                env.unwrapped.motion_ids, motion_times)
            is_closed = bool(motion_res["is_closed"][0].item() > 0.5)
            if bottle_z <= lift_thres:
                object_settled = True
            lifted = object_settled and (bottle_z > lift_thres) and is_closed
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
              f"settled={object_settled}  closed_steps={closed_steps}  lift_steps={lift_steps}  "
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
    print(f"  Embodiment tag:             {args.embodiment_tag}")
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
