"""VLA (unitree_g1_sonic) + SONIC decoder closed-loop *statistical* eval in Isaac Lab.

Runs a ``unitree_g1_sonic`` VLA in closed loop for N episodes (sweeping the reference motion
library) and reports the PHYSICAL-HOLD success used by eval_sonic_adapter.py. No video is
written; see ``play_vla_sonic.py`` for a recording sibling.

This VLA predicts the SONIC latent token DIRECTLY (``motion_token``, 64-D) plus the finger
joints. The token is fed through the SAME frozen decoder + proprio-history wrapper the residual
policy was trained and collected with (``TokenActionDecoderVecEnvWrapper``), so the only thing
that changes between data collection and this eval is WHO produces the token:

    env obs ─▶ ObsToPolicyAdapter ─▶ Gr00tPolicy.get_action ─▶ action_dict
                        motion_token (64) + right_hand_joints (7 -> binary scalar)
                                        │
                     TokenActionDecoderVecEnvWrapper.step([token | finger])
                       FSQ snap ─▶ [token | 10-frame history] ─▶ frozen decoder (.pt or ONNX)
                       ─▶ 29 body targets + binary finger slots ─▶ env.step

Metrics (all reference-independent, computed from the sim object and the sim right palm):
  1. [PHYS] held  — object >= 2 cm / 5 cm above its rest height AND within --phys-radius of the
     sim palm for >= --phys-steps consecutive steps.
  2. touched / toppled / max lift, post-grab window, per-term termination breakdown.
Requires cameras (the VLA reads the ego view); the env is the Cam-* variant of the training env.
"""

from __future__ import annotations

import argparse

from vla_sonic.repo_paths import gear_sonic_deploy  # sibling-repo ONNX defaults (worktree-safe)
import builtins
import os
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
    parser.add_argument("--task", default="Isaac-Motion-Tracking-Pick-Cam-HOI-v0",
                        help="Kitchen-visual + ego-camera variant of the HOI pick env (identical physics/"
                             "obs/actions/rewards/terminations to the residual training env, and the env "
                             "collect_sonic_adapter.py records in).")
    parser.add_argument("--sonic-pt", type=str, default=None,
                        help="Directory of the native SONIC .pt checkpoint whose DECODER turns the VLA's "
                             "motion_token into joint targets. MUST be the model the training data's "
                             "tokens were recorded with (collect_sonic_adapter.py --sonic-pt). Overrides "
                             "--decoder-onnx.")
    parser.add_argument("--ref-motions-path", type=str, default=None,
                        help="Override the env's ref_motions_path (dir of reference .pkl files) -- the set the "
                             "training data was collected on.")
    parser.add_argument("--waist-dof", type=int, default=29, choices=[27, 29],
                        help="29 = waist-actuated 29-DOF articulation (current pipeline).")
    parser.add_argument("--sweep-motions", dest="sweep_motions", action="store_true", default=True,
                        help="Episode k runs reference motion k mod N (deterministic coverage of the library).")
    parser.add_argument("--random-motions", dest="sweep_motions", action="store_false",
                        help="Draw a random motion per episode instead of sweeping.")
    parser.add_argument("--finger-close-thres", type=float, default=0.6,
                        help="Binary-fingers env: the VLA's 7 predicted right-hand joint angles are mapped "
                             "to CLOSE when mean|q| exceeds this (rad); closed pose mean|q| ~ 1.27, open = 0.")
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
    parser.add_argument("--decoder-onnx",
                        default=gear_sonic_deploy("policy/release/model_decoder.onnx"))
    parser.add_argument("--lift-thres", type=float, default=0.95,
                        help="LEGACY diagnostic only (absolute bottle z). The headline metric is the "
                             "physical hold below.")
    parser.add_argument("--phys-radius", type=float, default=0.15,
                        help="[PHYS] object must be within this (m) of the SIM right palm ...")
    parser.add_argument("--phys-steps", type=int, default=25,
                        help="... for >= this many consecutive steps, while lifted >= 2 cm / 5 cm above rest.")
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

from vla_sonic.obs_to_policy import ObsAdapterConfig, ObsToPolicyAdapter  # noqa: E402
from vla_sonic.physics_overrides import apply_sonic_physics_overrides  # noqa: E402
from vla_sonic.simple_robot_model import SimpleG1RobotModel  # noqa: E402
from vla_sonic.token_action_wrapper import TokenActionDecoderVecEnvWrapper, load_frozen_decoder  # noqa: E402
from vla_sonic.sonic_pt import load_sonic_pt  # noqa: E402
from vla_sonic.robot_29dof import apply_29dof_waist_override  # noqa: E402

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
    if args.ref_motions_path is not None:
        env_cfg.ref_motions_path = args.ref_motions_path
        print(f"[eval_vla_sonic] ref_motions_path override -> {args.ref_motions_path}")
    if args.waist_dof == 29:
        apply_29dof_waist_override(env_cfg)
    if os.environ.get("HS_EVAL_NO_EE_TERM", "0") == "1" and getattr(env_cfg.terminations, "ee_body_pos", None) is not None:
        env_cfg.terminations.ee_body_pos = None
        print("[eval_vla_sonic] HS_EVAL_NO_EE_TERM=1: ee_body_pos termination removed")
    # Use the env's own ego camera when it defines one (the Cam-* envs: 1280x960, identical to
    # the collection render, downsized by ObsToPolicyAdapter exactly as the converter did);
    # inject the 640x480 fallback only for envs without a camera.
    if getattr(env_cfg.scene, "camera_robot", None) is None:
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
    # The Cam-* envs already carry the collection env's own visual hooks (green box, ground
    # plane, glass bottle hidden; grab marker off); adding the eval's copy is redundant.
    if not args.raw_visuals and "-Cam-" not in args.task:
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

    # --- 3. SONIC decoder: the same frozen decoder + proprio-history wrapper the residual
    # policy was trained/collected with (TokenActionDecoderVecEnvWrapper). The VLA's token
    # replaces "encoder base token + residual"; everything downstream is byte-identical:
    # FSQ snap -> 994-D decoder obs (token | 10-frame history) -> 29 body targets, plus the
    # binary right-hand slot. This is the wiring that produced the training data.
    if args.sonic_pt:
        _enc_unused, decoder = load_sonic_pt(args.sonic_pt, "cuda:0")
        print(f"[sonic] native .pt decoder from {args.sonic_pt}")
    else:
        decoder = load_frozen_decoder(args.decoder_onnx, "cuda:0")
        print(f"[sonic] ONNX decoder {args.decoder_onnx} (v1.0 latent space -- only valid if the "
              f"training tokens were recorded with the ONNX model)")
    env = TokenActionDecoderVecEnvWrapper(env, decoder, "cuda:0", clip_actions=None)
    unw = env.unwrapped
    total_motions = int(unw.total_motions)

    # --- 4. Obs adapter (reads env.unwrapped.scene: state + ego camera) --------------
    robot_model = SimpleG1RobotModel.build()
    obs_adapter = ObsToPolicyAdapter(
        env,
        ObsAdapterConfig(
            language_instruction=args.language,
            robot_model=robot_model,
            camera_scene_key="camera_robot",
        ),
    )
    robot = unw.scene["robot"]
    _rw_bid = robot.find_bodies("right_wrist_yaw_link")[0][0]
    _tm = unw.termination_manager
    term_names = list(_tm.active_terms)
    # measured right-finger joints (7) -- to check the hand physically closes when commanded
    _rf_jids = robot.find_joints("right_hand_.*")[0]

    def _finger_scalar(vla_chunk: dict, t_idx: int) -> float:
        """VLA right_hand_joints (7) -> binary env scalar: <0 closes, >=0 opens."""
        rh = np.asarray(vla_chunk["right_hand_joints"], dtype=np.float32)
        q = rh[0, t_idx] if rh.ndim == 3 else rh.reshape(-1)
        return -1.0 if float(np.abs(q).mean()) > args.finger_close_thres else 1.0

    # --- 5. Rollout ------------------------------------------------------------------
    lift_thres = args.lift_thres
    PHYS_DZS = (0.02, 0.05)
    stats = {
        "episodes": 0, "phys_held": {dz: 0 for dz in PHYS_DZS}, "touched": 0, "toppled": 0,
        "legacy_any_lift": 0, "term": {n: 0 for n in term_names}, "term_other": 0,
        "ep_len": [], "max_lift": [], "post_grab": [],
    }
    print(f"[eval_vla_sonic] starting eval: num_episodes={args.num_episodes}, "
          f"max_steps_per_episode={args.max_steps_per_episode}, motions={total_motions}, "
          f"phys: lift>=2/5cm & obj within {args.phys_radius} m of sim palm for >={args.phys_steps} steps")
    t_start = time.time()
    for ep in range(args.num_episodes):
        mid = ep % total_motions if args.sweep_motions else None
        unw._forced_motion_id = mid
        print(f"\n[episode {ep}] motion_id={mid if mid is not None else 'random'}")
        with torch.inference_mode():
            obs, _ = env.reset()                      # wrapper.reset() also seeds the decoder history
        _APP.update(); _APP.update()

        vla_chunk = None; chunk_step = 0
        g_pred = []; g_cmd_close = []; g_meas = []                       # per-step grasp diagnostics
        obj_rest_z = None; phys_run = {dz: 0 for dz in PHYS_DZS}; phys_ok = {dz: False for dz in PHYS_DZS}
        max_lift = 0.0; toppled = False; touched = False; legacy_lift = False; object_settled = False
        grab_step = -1; fired = []
        step = 0
        for step in range(args.max_steps_per_episode):
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
                        print(f"  {k} [shape {tuple(arr.shape)}] = {slice_.reshape(-1)[:8].round(4).tolist()}"
                              f"{' ...' if slice_.size > 8 else ''}")
            t_idx = chunk_step
            token = extract_motion_token(vla_chunk, t_index=t_idx)                # (64,) continuous
            if ep == 0 and step == 0 and float(np.abs(token).max()) < 1e-6:
                print("\n[WARN] motion_token is ALL ZERO -- the VLA is emitting a null SONIC token; "
                      "the robot is NOT VLA-controlled. Check action.motion_token in the dataset.\n")
            latent = torch.zeros((1, 65), device="cuda:0", dtype=torch.float32)
            latent[0, :64] = torch.as_tensor(token, device="cuda:0")
            latent[0, 64] = _finger_scalar(vla_chunk, t_idx)
            _rh = np.asarray(vla_chunk["right_hand_joints"], dtype=np.float32)
            g_pred.append(float(np.abs(_rh[0, t_idx] if _rh.ndim == 3 else _rh.reshape(-1)).mean()))
            g_cmd_close.append(bool(latent[0, 64].item() < 0))
            with torch.inference_mode():
                obs, _rew, dones, _extras = env.step(latent)                   # FSQ snap + decoder inside
            g_meas.append(float(robot.data.joint_pos[0, _rf_jids].abs().mean().item()))
            _APP.update(); _APP.update()

            # ---- metrics (env-local frame), read AFTER the step ----
            org = unw.scene.env_origins[0]
            obj_p = unw.scene["object"].data.root_pos_w[0] - org
            oq = unw.scene["object"].data.root_quat_w[0]
            hq = robot.data.body_quat_w[0, _rw_bid]
            w_, x_, y_, z_ = hq[0], hq[1], hq[2], hq[3]
            hand_x = torch.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ + w_ * z_), 2 * (x_ * z_ - w_ * y_)])
            palm = (robot.data.body_pos_w[0, _rw_bid] - org) + 0.12 * hand_x
            d_palm = float(torch.norm(palm - obj_p).item())
            done_now = bool(torch.as_tensor(dones).reshape(-1)[0].item())
            if not done_now:                       # after a done the env has already reset -> skip
                if obj_rest_z is None or step <= 50:          # running min over the first 1 s (settle)
                    obj_rest_z = float(obj_p[2].item()) if obj_rest_z is None else min(obj_rest_z, float(obj_p[2].item()))
                dz = float(obj_p[2].item()) - obj_rest_z
                max_lift = max(max_lift, dz)
                for dzt in PHYS_DZS:
                    phys_run[dzt] = phys_run[dzt] + 1 if (d_palm < args.phys_radius and dz >= dzt) else 0
                    if phys_run[dzt] >= args.phys_steps:
                        phys_ok[dzt] = True
                if d_palm < 0.12:
                    touched = True
                if float((1.0 - 2.0 * (oq[1] ** 2 + oq[2] ** 2)).item()) < 0.7071:
                    toppled = True
                bottle_z = float(unw.scene["object"].data.root_pos_w[0, 2].item())
                mt = unw.episode_length_buf * unw.step_dt + unw.start_motion_times.to("cuda:0", dtype=torch.float32)
                is_closed = bool(unw.motion_lib.get_motion_state(unw.motion_ids, mt)["is_closed"][0].item() > 0.5)
                if is_closed and grab_step < 0:
                    grab_step = step
                if bottle_z <= lift_thres:
                    object_settled = True
                if object_settled and bottle_z > lift_thres and is_closed:
                    legacy_lift = True
            chunk_step += 1
            if done_now:
                try:
                    fired = [n for n in term_names if bool(_tm.get_term(n)[0].item())]
                except Exception:
                    fired = []
                break

        # ---- episode bookkeeping ----
        stats["episodes"] += 1
        stats["ep_len"].append(step + 1)
        stats["max_lift"].append(max_lift)
        if grab_step >= 0:
            stats["post_grab"].append(step + 1 - grab_step)
        for dzt in PHYS_DZS:
            stats["phys_held"][dzt] += int(phys_ok[dzt])
        stats["touched"] += int(touched); stats["toppled"] += int(toppled); stats["legacy_any_lift"] += int(legacy_lift)
        if fired:
            for n in fired:
                stats["term"][n] += 1
        else:
            stats["term_other"] += 1
        print(f"[episode {ep}] ended at step {step+1} ({','.join(fired) if fired else 'max_steps'})  "
              f"phys_held(2cm/5cm)={phys_ok[0.02]}/{phys_ok[0.05]}  max_lift={max_lift*100:.1f}cm  "
              f"touched={touched} toppled={toppled}  running held5={stats['phys_held'][0.05]}/{stats['episodes']}")
        if g_pred:
            _gp = np.array(g_pred); _gc = np.array(g_cmd_close); _gm = np.array(g_meas)
            _first = int(np.argmax(_gc)) if _gc.any() else -1
            print(f"[episode {ep}][grasp] VLA right-hand mean|q|: max {_gp.max():.2f}  median {np.median(_gp):.2f}  "
                  f"steps>0.4: {int((_gp > 0.4).sum())}  >0.6(=close cmd): {int(_gc.sum())}/{len(_gc)}  first close step: {_first}  "
                  f"ref grab step: {grab_step}  |  measured fingers mean|q| max {_gm.max():.2f}"
                  + (f", while commanded closed: mean {_gm[_gc].mean():.2f}" if _gc.any() else ""))

    elapsed = time.time() - t_start
    ce = max(stats["episodes"], 1)
    ml = np.array(stats["max_lift"]) if stats["max_lift"] else np.zeros(1)
    print("\n" + "=" * 60)
    print("                  VLA + SONIC EVAL SUMMARY")
    print("=" * 60)
    print(f"  VLA checkpoint:             {args.vla_checkpoint}")
    print(f"  Embodiment tag:             {args.embodiment_tag}")
    print(f"  Task:                       {args.task}")
    print(f"  SONIC decoder:              {args.sonic_pt or args.decoder_onnx}")
    print(f"  chunk_size:                 {args.chunk_size}")
    print(f"  Completed episodes:         {stats['episodes']}   (wall {elapsed:.1f}s)")
    print(f"  Mean episode length:        {np.mean(stats['ep_len']):.1f} steps")
    if stats["post_grab"]:
        print(f"  Post-grab window:           median {np.median(stats['post_grab']):.0f} steps "
              f"({len(stats['post_grab'])} episodes reached the grab)")
    for dzt in PHYS_DZS:
        print(f"  [PHYS] held (lift>={100*dzt:.0f}cm & obj within {args.phys_radius} m of sim palm "
              f">={args.phys_steps} steps): {stats['phys_held'][dzt]} / {ce} = {100*stats['phys_held'][dzt]/ce:.2f}%")
    print(f"  [PHYS] max lift above rest: median {np.median(ml)*100:.1f} cm  mean {ml.mean()*100:.1f} cm")
    print(f"  Object touched (<0.12 m):   {stats['touched']} / {ce} = {100*stats['touched']/ce:.2f}%")
    print(f"  Object toppled (>45 deg):   {stats['toppled']} / {ce} = {100*stats['toppled']/ce:.2f}%")
    print(f"  legacy any-lift (z>{lift_thres}): {stats['legacy_any_lift']} / {ce}")
    print(f"  Termination breakdown:")
    for n, v in stats["term"].items():
        print(f"    term {n:<22} {v:5d}  ({100*v/ce:.1f}%)")
    print(f"    max_steps / other      {stats['term_other']:5d}  ({100*stats['term_other']/ce:.1f}%)")
    print("=" * 60)
    env.close()
    _APP.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
