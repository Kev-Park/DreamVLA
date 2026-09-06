# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless multi-env eval for a SONIC residual-ADAPTER checkpoint — no cameras, no video.

Adapter analog of ``eval_play_sonic.py``: same statistical reporting, but wired for the
frozen-encoder + residual-adapter pipeline (``train_sonic_adapter.py`` /
``play_sonic_adapter.py``):

    frozen G1 encoder → base token ─┬─▶ obs ─▶ adapter policy ─▶ residual ─┐
                                    └──────────────────────────── (add) ◀──┘
                                                  │
                                            FSQ → frozen decoder → env body action

Reports:
  1. ``Episodes with any lift`` — per-episode discrete success rate (bottle cleared the
     lift threshold during the closed phase). The headline "did it pick up the bottle".
  2. ``Mean lift fraction`` — over lifted episodes, fraction of closed-phase steps lifted
     (grasp retention quality).
  3. ``Cumulative lift-steps`` — env.n_successes.sum() (the reward's own success counter;
     only fires at num_envs<1001, so num_envs is capped).
  4. ``Termination breakdown`` — time_out vs other.

``--zero-residual`` evaluates the frozen-SONIC base (no checkpoint needed) as a baseline.
Deterministic actor mean via ``get_inference_policy``; fixed seed for reproducible motions.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import builtins
from functools import partial

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

print = partial(builtins.print, flush=True)

parser = argparse.ArgumentParser(description="Headless multi-env eval for a SONIC residual-adapter checkpoint.")
parser.add_argument("--num_envs", type=int, default=1000,
                    help="Number of parallel envs. Capped to 1000 (n_successes only updates "
                         "when env.num_envs < 1001; see object_above_threshold).")
parser.add_argument("--num_episodes", type=int, default=2000,
                    help="Stop once at least this many episodes have completed across all envs.")
parser.add_argument("--max_steps", type=int, default=10000,
                    help="Hard cap on step count, regardless of episode completion.")
parser.add_argument("--task", type=str, default="Isaac-Motion-Tracking-Pick-BinaryFingers-v0",
                    help="Task name. Default matches train_sonic_adapter.py.")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for deterministic motion selection across runs.")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--path", type=str, default=None,
                    help="Explicit checkpoint path (overrides auto-discovery).")
parser.add_argument("--sonic-decoder-onnx", type=str,
                    default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx",
                    help="Path to the frozen SONIC decoder ONNX (must match training).")
parser.add_argument("--sonic-pt", type=str, default=None,
                    help="Directory of a native SONIC .pt checkpoint (groot-era); overrides ONNX.")
parser.add_argument("--sonic-encoder-onnx", type=str,
                    default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx",
                    help="Path to the frozen SONIC encoder ONNX (must match training).")
parser.add_argument("--residual-scale", type=float, default=0.3,
                    help="Residual bound — MUST match the value used by train_sonic_adapter.py.")
parser.add_argument("--residual-transform", type=str, default="additive",
                    choices=["additive", "multiplicative", "multiplicative_free", "unclamped"],
                    help="Token residual transform — MUST match train_sonic_adapter.py for this checkpoint.")
parser.add_argument("--zero-residual", action="store_true", default=False,
                    help="Evaluate the frozen-SONIC base (residual=0, fingers open) as a "
                         "baseline. No checkpoint required.")
parser.add_argument("--lift-thres", type=float, default=0.95,
                    help="Bottle z (m) above which a frame counts as 'lifted'. Matches the "
                         "env's object_above height_thres (object rests at 0.9; 0.95 = a "
                         "5 cm pickup, below the observed success apex ~0.976).")
parser.add_argument("--skip-start-frames", type=int, default=None,
                    help="Skip the refinement's interpolate-to-initial-pose prepend (pass 22). The env "
                         "resets 10 frames later so the decoder-history seed lands on real motion.")
parser.add_argument("--waist-dof", type=int, default=27, choices=[27, 29],
                    help="Body DOF. 29 actuates waist_roll/pitch to match SONIC training. 27 = welded.")
parser.add_argument("--ref-motions-path", type=str, default=None,
                    help="Override the env's ref_motions_path (dir of reference .pkl files). Point at "
                         "the 29-DOF holosoma dataset, e.g. ../TrajGen/sample/Holosoma_Pick_29_full.")
parser.add_argument("--start-pregrab-margin", type=float, default=None,
                    help="Start episodes this many seconds before the grab (drops prepend + walk).")
# append RSL-RL cli args (gives --checkpoint, --load_run, etc.)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# NEVER enable cameras — headless eval.
args_cli.enable_cameras = False

# Cap num_envs so n_successes fires (gated at <1001 in reward func).
if args_cli.num_envs >= 1001:
    print(f"[eval_sonic_adapter] num_envs={args_cli.num_envs} >= 1001 — capping to 1000 so "
          f"env.n_successes counter fires (gate is num_envs<1001 in object_above_threshold).")
    args_cli.num_envs = 1000

# launch omniverse app (headless)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import math
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# SONIC adapter wiring (same as train_sonic_adapter.py / play_sonic_adapter.py)
from vla_sonic.token_action_wrapper import load_frozen_decoder
from vla_sonic.token_adapter_wrapper import TokenAdapterVecEnvWrapper, load_frozen_encoder
from vla_sonic.physics_overrides import apply_sonic_physics_overrides
from vla_sonic.robot_29dof import apply_29dof_waist_override
from vla_sonic.adapter_actor_critic import AdapterActorCritic

# Register the custom ActorCritic with rsl_rl (eval(class_name) resolves in on_policy_runner globals).
import rsl_rl.modules
import rsl_rl.runners.on_policy_runner as _rsl_rl_opr
rsl_rl.modules.AdapterActorCritic = AdapterActorCritic
_rsl_rl_opr.AdapterActorCritic = AdapterActorCritic


def main():
    # Deterministic motion draw (reproducible across runs).
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    print(f"[eval_sonic_adapter] seed = {args_cli.seed}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        enable_cameras=False,
    )
    env_cfg.seed = args_cli.seed
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # ---- mirror train_sonic_adapter.py's agent overrides EXACTLY ----
    agent_cfg.experiment_name = f"{agent_cfg.experiment_name}_sonic_adapter"
    agent_cfg.policy.class_name = "AdapterActorCritic"
    agent_cfg.policy.actor_hidden_dims = [256, 128]
    agent_cfg.policy.critic_hidden_dims = [512, 256, 256]

    # Match SONIC decoder's training-time physics.
    apply_sonic_physics_overrides(env_cfg)

    # 29-DOF strict-fidelity articulation (actuated waist roll/pitch) to match SONIC training.
    if args_cli.waist_dof == 29:
        apply_29dof_waist_override(env_cfg)

    # Point eval at an explicit reference dataset (e.g. the 29-DOF holosoma set).
    if args_cli.ref_motions_path is not None:
        env_cfg.ref_motions_path = args_cli.ref_motions_path
        print(f"[eval_sonic_adapter] ref_motions_path override -> {args_cli.ref_motions_path}")

    # Optional episode-start offsets (same flags as train/play).
    if args_cli.start_pregrab_margin is not None:
        env_cfg.motion_start_pregrab_margin_s = args_cli.start_pregrab_margin
        print(f"[eval_sonic_adapter] start_pregrab_margin = {args_cli.start_pregrab_margin}s")
    if args_cli.skip_start_frames is not None:
        # +10: reset 10 frames later so the decoder-history seed lands on real motion.
        env_cfg.motion_skip_start_frames = args_cli.skip_start_frames + 10
        print(f"[eval_sonic_adapter] skip_start_frames = {args_cli.skip_start_frames} "
              f"(+10 warmup -> reset at frame {args_cli.skip_start_frames + 10})")

    # ---- resolve checkpoint (skipped in --zero-residual mode) ----
    resume_path = None
    if not args_cli.zero_residual:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        elif args_cli.path:
            resume_path = args_cli.path
        else:
            # Auto-pick the newest TIMESTAMPED run dir that actually has checkpoints. The default
            # get_checkpoint_path globs load_run=".*" and sorts alphabetically, so non-training dirs
            # like `zero_shot`/`reference` (which sort after `2026-*` and have no model_*.pt) shadow
            # the real run and crash. Prefer the most-recently-modified `20*/` dir with a model_*.pt.
            import glob as _glob
            _runs = sorted(
                (d for d in _glob.glob(os.path.join(log_root_path, "20*"))
                 if os.path.isdir(d) and _glob.glob(os.path.join(d, "model_*.pt"))),
                key=os.path.getmtime,
            )
            if _runs:
                resume_path = get_checkpoint_path(log_root_path, os.path.basename(_runs[-1]), agent_cfg.load_checkpoint)
                print(f"[eval_sonic_adapter] auto-selected newest run with checkpoints: {os.path.basename(_runs[-1])}")
            else:
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[env] action_space (pre-wrapper) = {env.action_space}")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ---- frozen encoder + decoder + adapter wrapper (same wiring as training) ----
    device = agent_cfg.device
    print(f"[eval_sonic_adapter] loading frozen SONIC decoder ONNX: {args_cli.sonic_decoder_onnx}")
    if args_cli.sonic_pt:
        from vla_sonic.sonic_pt import load_sonic_pt
        encoder_pt, decoder_pt = load_sonic_pt(args_cli.sonic_pt, device)
        decoder = decoder_pt
    else:
        decoder = load_frozen_decoder(args_cli.sonic_decoder_onnx, device)
    print(f"[eval_sonic_adapter] loading frozen SONIC encoder ONNX: {args_cli.sonic_encoder_onnx}")
    encoder = encoder_pt if args_cli.sonic_pt else load_frozen_encoder(args_cli.sonic_encoder_onnx, device)
    env = TokenAdapterVecEnvWrapper(
        env, decoder, encoder, device,
        residual_scale=args_cli.residual_scale,
        residual_transform=args_cli.residual_transform,
        clip_actions=None,
        pt_mode=bool(args_cli.sonic_pt),
    )

    # ---- initialize env.n_successes (gated on existence in object_above_threshold) ----
    env.unwrapped.n_successes = torch.zeros(env.num_envs, device=device, dtype=torch.float32)

    # ---- policy: trained adapter, or the zero-residual baseline ----
    if args_cli.zero_residual:
        print("[eval_sonic_adapter] ZERO-RESIDUAL baseline: residual=0, fingers open (frozen SONIC).")

        def policy(obs):
            return torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float32)
    else:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
        ppo_runner.load(resume_path)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ---- per-env episode trackers ----
    num_envs = env.num_envs
    target_episodes = args_cli.num_episodes
    max_steps = args_cli.max_steps
    lift_thres = args_cli.lift_thres

    per_env_had_any_lift = torch.zeros(num_envs, device=device, dtype=torch.bool)
    per_env_closed_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
    per_env_lift_steps = torch.zeros(num_envs, device=device, dtype=torch.long)

    completed_episodes = 0
    completed_any_lift = 0
    sum_lift_fraction_over_lifted_episodes = 0.0
    completed_episodes_with_any_lift = 0
    termination_counts = {"time_out": 0, "other": 0}

    # ---- REWORK success metric: object HELD near the synthesized object ref during the grasp ----
    # (the height-based lift metric is miscalibrated under REWORK — the object rests at the grounded
    # reference height, not 0.90). "held" = sim object within HELD_TOL of the synth ref (rest->palm)
    # while is_closed. An episode succeeds if it held for a fraction of the closed phase >= HELD_FRAC.
    REWORK = os.environ.get("HS_REWORK", "0") == "1"
    HELD_TOL = float(os.environ.get("HS_EVAL_HELD_TOL", "0.10"))   # m
    HELD_FRAC = float(os.environ.get("HS_EVAL_HELD_FRAC", "0.5"))  # fraction of closed phase
    HAND_FWD = float(os.environ.get("HS_REWORK_HAND_FWD", "1.5"))
    per_env_held_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
    completed_held = 0
    # ---- failure classification (touched / toppled) ----
    TOUCH_TOL = float(os.environ.get('HS_EVAL_TOUCH_TOL', '0.12'))   # m: robot hand within this of object = touched
    TOUCH_OFFX = float(os.environ.get('HS_EVAL_TOUCH_OFFX', '0.12')) # m: wrist->grasp forward projection
    TOPPLE_DEG = float(os.environ.get('HS_EVAL_TOPPLE_DEG', '45.0')) # object up-axis tilt beyond this = toppled
    _TOPPLE_COS = math.cos(math.radians(TOPPLE_DEG))
    per_env_touched = torch.zeros(num_envs, device=device, dtype=torch.bool)
    per_env_toppled = torch.zeros(num_envs, device=device, dtype=torch.bool)
    completed_touched = 0; completed_toppled = 0; completed_touched_not_held = 0; completed_never_touched = 0
    try:
        _hand_bid = env.unwrapped.scene['robot'].find_bodies('right_wrist_yaw_link')[0][0]
    except Exception:
        _hand_bid = None
    # Motion-tracking-only envs (Isaac-Motion-Tracking-MotionOnly-v0) have NO manipuland.
    # Every object-based metric (lift / held / touched / toppled and the FAILCLASS classes
    # built on them) is undefined there; we still report terminations + root error.
    try:
        _ = env.unwrapped.scene["object"]
        HAS_OBJECT = True
    except Exception:
        HAS_OBJECT = False
    if not HAS_OBJECT:
        print("  [eval] no manipuland in this env -> object metrics skipped; "
              "reporting termination breakdown + root tracking error only")

    just_reset_mask = torch.ones(num_envs, device=device, dtype=torch.bool)
    # ---- per-clip (per motion_id) attribution ----
    _NM = int(env.unwrapped.total_motions)
    per_env_motion_id = env.unwrapped.motion_ids.clone()
    m_ep = torch.zeros(_NM, dtype=torch.long)
    m_top = torch.zeros(_NM, dtype=torch.long)
    m_touch = torch.zeros(_NM, dtype=torch.long)
    m_held = torch.zeros(_NM, dtype=torch.long)
    m_never = torch.zeros(_NM, dtype=torch.long)

    # ---- HS_EVAL_FAILCLASS=1: per-episode failure taxonomy ----
    # Classifies every episode into ONE mode (priority order) so the failure distribution can
    # arbitrate "fix references" vs "change rewards":
    #   HELD                 success (existing criterion)
    #   KNOCKED_PRE_REACH    object toppled BEFORE the palm ever got within TOUCH_TOL
    #                        (wrist/forearm/body knock — approach-geometry failure)
    #   KNOCKDOWN_ON_ARRIVAL toppled within KNOCK_WIN steps of first palm contact
    #   DROPPED_LATE         toppled well after arrival (hold/carry failure)
    #   ROOT_AHEAD / ROOT_OFF at grab time the root was > ROOT_TOL off the reference root
    #                        (ahead = stumbled/overshot forward) and the hand never reached
    #                        (locomotion failure — root off-location puts the arm off-position)
    #   ARM_OFF              root ON location at grab, hand still never reached (arm/reach failure)
    #   EARLY_END            episode ended before the grab phase was ever entered
    #   TOUCHED_NOT_HELD     reached, nothing toppled, still not held (grasp-actuation ceiling)
    FAILCLASS = os.environ.get("HS_EVAL_FAILCLASS", "0") == "1"
    ROOT_TOL = float(os.environ.get("HS_EVAL_ROOT_TOL", "0.25"))    # m, xy root error at grab
    ROOT_AHEAD_X = float(os.environ.get("HS_EVAL_ROOT_AHEAD_X", "0.15"))  # m, forward overshoot split
    KNOCK_WIN = int(os.environ.get("HS_EVAL_KNOCK_WIN", "25"))      # steps (~0.5 s) after first touch
    _CLASSES = ["HELD", "KNOCKED_PRE_REACH", "KNOCKDOWN_ON_ARRIVAL", "DROPPED_LATE",
                "ROOT_AHEAD", "ROOT_OFF", "ARM_OFF", "EARLY_END", "TOUCHED_NOT_HELD"]
    fc_counts = {c: 0 for c in _CLASSES}
    fc_root_err_sum = {c: 0.0 for c in _CLASSES}
    m_class = torch.zeros(_NM, len(_CLASSES), dtype=torch.long)
    per_env_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
    per_env_first_touch = torch.full((num_envs,), -1, device=device, dtype=torch.long)
    per_env_first_topple = torch.full((num_envs,), -1, device=device, dtype=torch.long)
    per_env_grab_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)
    per_env_root_err_grab = torch.full((num_envs,), -1.0, device=device)
    per_env_root_fwd_grab = torch.zeros(num_envs, device=device)

    obs_out = env.get_observations()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out

    print(f"[eval_sonic_adapter] starting eval: num_envs={num_envs}, target_episodes={target_episodes}, "
          f"max_steps={max_steps}, lift_thres={lift_thres}")
    t_start = time.time()
    step_count = 0

    # ---- HS_EVAL_STABILITY=1: static stability of the SIMULATED robot, per step ----
    # Every stability number elsewhere in this project is computed on the reference kinematics.
    # This measures whether the robot ACTUALLY tracked a balanced state: CoM ground projection
    # inside the convex hull of the planted feet's contact points. Batched over all envs.
    STAB = os.environ.get("HS_EVAL_STABILITY", "0") == "1"
    _stab_sum = _stab_n = 0
    _stab_marg = []
    if STAB:
        _rb = env.unwrapped.scene["robot"]
        _names = list(_rb.data.body_names)
        try:
            _mass = _rb.root_physx_view.get_masses()[0].to(device)
        except Exception:
            _mass = _rb.data.default_mass[0].to(device)
        _feet_i = [_names.index(f) for f in ("left_ankle_roll_link", "right_ankle_roll_link")
                   if f in _names]
        _FCP = torch.tensor([[-0.05, 0.025, -0.03], [-0.05, -0.025, -0.03],
                             [0.12, 0.030, -0.03], [0.12, -0.030, -0.03]],
                            dtype=torch.float32, device=device)
        print(f"[stability] SIM-robot readout ON: {len(_names)} bodies, "
              f"{float(_mass.sum()):.2f} kg, feet={len(_feet_i)}")

    def _convex_hull_np(pts):
        pts = np.unique(pts, axis=0)
        if len(pts) < 3: return pts
        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        def half(A):
            h = []
            for q in A:
                while len(h) >= 2 and np.cross(h[-1]-h[-2], q-h[-2]) <= 0: h.pop()
                h.append(q)
            return h
        return np.array(half(pts)[:-1] + half(pts[::-1])[:-1])

    def _margin_np(pt, H):
        ins, dmin = True, 1e9
        for i in range(len(H)):
            a, b = H[i], H[(i+1) % len(H)]
            e = b - a; L = float(np.linalg.norm(e))
            if L < 1e-9: continue
            if np.cross(e, pt-a) < 0: ins = False
            t = float(np.clip(np.dot(pt-a, e)/(L*L), 0, 1))
            dmin = min(dmin, float(np.linalg.norm(pt-(a+t*e))))
        return dmin if ins else -dmin

    def _sim_stability():
        """(N,) signed margin of the sim robot's CoM w.r.t. its planted-foot polygon."""
        pos = _rb.data.body_pos_w - env.unwrapped.scene.env_origins[:, None, :]
        quat = _rb.data.body_quat_w
        com = (_mass[None, :, None] * pos).sum(1) / _mass.sum()
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        R = torch.stack([
            torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
            torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
            torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)
        P = []
        for bi in _feet_i:
            P.append(pos[:, bi, None, :] + torch.einsum("nij,kj->nki", R[:, bi], _FCP))
        P = torch.cat(P, 1)                                   # (N, 4*nfeet, 3)
        floor = P[..., 2].min(1, keepdim=True).values
        planted = P[..., 2] < floor + 0.03                    # (N, K)
        out = torch.full((pos.shape[0],), float("nan"), device=device)
        Pc = P.cpu().numpy(); pl = planted.cpu().numpy(); cm = com.cpu().numpy()
        for n in range(Pc.shape[0]):
            pts = Pc[n][pl[n]][:, :2]
            if len(pts) < 3: continue
            H = _convex_hull_np(pts)
            if len(H) < 3: continue
            out[n] = _margin_np(cm[n, :2], H)
        return out

    while completed_episodes < target_episodes and step_count < max_steps:
        per_env_motion_id.copy_(env.unwrapped.motion_ids)
        with torch.inference_mode():
            actions = policy(obs).clone()
            obs, _, dones, extras = env.step(actions)
        if STAB:
            with torch.inference_mode():
                _m = _sim_stability()
            _ok = ~torch.isnan(_m)
            if _ok.any():
                _stab_sum += int((_m[_ok] > 0).sum()); _stab_n += int(_ok.sum())
                _stab_marg.append(_m[_ok].cpu().numpy())

        # Read bottle z + is_closed AFTER step.
        bottle_z = (env.unwrapped.scene["object"].data.root_pos_w[:, 2] if HAS_OBJECT
                    else torch.zeros(num_envs, device=device))
        motion_times = (
            env.unwrapped.episode_length_buf * env.unwrapped.step_dt
            + env.unwrapped.start_motion_times.clone().detach().to(device=device, dtype=torch.float32)
        )
        motion_res = env.unwrapped.motion_lib.get_motion_state(env.unwrapped.motion_ids, motion_times)
        is_closed = motion_res["is_closed"].bool()
        lifted = (bottle_z > lift_thres) & is_closed

        held = torch.zeros(num_envs, device=device, dtype=torch.bool)
        if REWORK and HAS_OBJECT:
            gk = motion_res["global_keypts"]                                         # (N,39,3) env-local
            palm = gk[:, -1, :] + HAND_FWD * (gk[:, -1, :] - gk[:, -2, :])            # forward-projected palm
            if "object_poses" in motion_res:
                rest = motion_res["object_poses"][:, :3]
            else:
                # Legacy pkls (original DreamControl motions) carry no object trajectory; the
                # static grab_pos(+offsets) IS the object rest. Matches the reward-side fallback
                # in motion_tracking_pick_env._synth_object_ref_pos — without this the whole
                # branch was skipped and HELD was structurally 0 for every episode.
                rest = motion_res["grab_pos"] + motion_res["offsets"]
            synth_ref = torch.where(is_closed.unsqueeze(-1), palm, rest)             # (N,3) env-local
            obj_local = env.unwrapped.scene["object"].data.root_pos_w - env.unwrapped.scene.env_origins
            held = (torch.norm(obj_local - synth_ref, dim=1) < HELD_TOL) & is_closed

        # ---- touched / toppled (per-step) ----
        toppled_now = torch.zeros(num_envs, device=device, dtype=torch.bool)
        touched_now = torch.zeros(num_envs, device=device, dtype=torch.bool)
        if HAS_OBJECT:
            obj_pos_w = env.unwrapped.scene['object'].data.root_pos_w - env.unwrapped.scene.env_origins
            oq = env.unwrapped.scene['object'].data.root_quat_w                   # wxyz
            up_z = 1.0 - 2.0 * (oq[:, 1] ** 2 + oq[:, 2] ** 2)                      # object up-axis z
            toppled_now = up_z < _TOPPLE_COS
        if HAS_OBJECT and _hand_bid is not None:
            robot = env.unwrapped.scene['robot']
            hq = robot.data.body_quat_w[:, _hand_bid, :]                            # wxyz
            wv, xv, yv, zv = hq[:, 0], hq[:, 1], hq[:, 2], hq[:, 3]
            ax = torch.stack([1 - 2*(yv*yv+zv*zv), 2*(xv*yv+wv*zv), 2*(xv*zv-wv*yv)], dim=1)  # hand world x-axis
            palm = (robot.data.body_pos_w[:, _hand_bid, :] - env.unwrapped.scene.env_origins) + ax * TOUCH_OFFX
            touched_now = torch.norm(palm - obj_pos_w, dim=1) < TOUCH_TOL
        valid = ~just_reset_mask
        if valid.any():
            per_env_had_any_lift |= lifted & valid
            per_env_closed_steps += (is_closed & valid).long()
            per_env_lift_steps += (lifted & valid).long()
            per_env_held_steps += (held & valid).long()
            per_env_touched |= touched_now & valid
            per_env_toppled |= toppled_now & valid
            if FAILCLASS:
                per_env_steps += valid.long()
                _new_touch = touched_now & valid & (per_env_first_touch < 0)
                per_env_first_touch[_new_touch] = per_env_steps[_new_touch]
                _new_topple = toppled_now & valid & (per_env_first_topple < 0)
                per_env_first_topple[_new_topple] = per_env_steps[_new_topple]
                _cross = is_closed & valid & (per_env_grab_step < 0)
                if _cross.any():
                    per_env_grab_step[_cross] = per_env_steps[_cross]
                    _rt = env.unwrapped.scene["robot"].data.root_pos_w - env.unwrapped.scene.env_origins
                    _rr = motion_res["root_pos"]
                    per_env_root_err_grab[_cross] = torch.norm((_rt - _rr)[_cross, :2], dim=1)
                    per_env_root_fwd_grab[_cross] = (_rt - _rr)[_cross, 0]

        dones_bool = dones.bool() if dones.dtype != torch.bool else dones
        if dones_bool.any():
            done_idxs = torch.where(dones_bool)[0]
            time_outs = extras.get("time_outs", torch.zeros_like(dones_bool))
            time_out_mask = (
                time_outs[done_idxs].bool() if isinstance(time_outs, torch.Tensor)
                else torch.zeros(len(done_idxs), dtype=torch.bool)
            )
            for k, idx in enumerate(done_idxs.tolist()):
                completed_episodes += 1
                if bool(per_env_had_any_lift[idx].item()):
                    completed_any_lift += 1
                    cs = int(per_env_closed_steps[idx].item())
                    ls = int(per_env_lift_steps[idx].item())
                    if cs > 0:
                        sum_lift_fraction_over_lifted_episodes += ls / cs
                        completed_episodes_with_any_lift += 1
                if REWORK:
                    cs = int(per_env_closed_steps[idx].item())
                    hs = int(per_env_held_steps[idx].item())
                    if cs > 0 and (hs / cs) >= HELD_FRAC:
                        completed_held += 1
                _touched = bool(per_env_touched[idx].item())
                _toppled = bool(per_env_toppled[idx].item())
                _held_ok = REWORK and int(per_env_closed_steps[idx].item()) > 0 and (int(per_env_held_steps[idx].item()) / max(int(per_env_closed_steps[idx].item()),1)) >= HELD_FRAC
                if _touched: completed_touched += 1
                if _toppled: completed_toppled += 1
                if not _held_ok:
                    if _touched: completed_touched_not_held += 1
                    else: completed_never_touched += 1
                _mid = int(per_env_motion_id[idx].item())
                if 0 <= _mid < _NM:
                    m_ep[_mid] += 1
                    if _toppled: m_top[_mid] += 1
                    if _touched: m_touch[_mid] += 1
                    if _held_ok: m_held[_mid] += 1
                    if (not _held_ok) and (not _touched): m_never[_mid] += 1
                if FAILCLASS:
                    _ft = int(per_env_first_touch[idx].item())
                    _fp = int(per_env_first_topple[idx].item())
                    _gs = int(per_env_grab_step[idx].item())
                    _re = float(per_env_root_err_grab[idx].item())
                    _fw = float(per_env_root_fwd_grab[idx].item())
                    if _held_ok:
                        _cls = "HELD"
                    elif _toppled and (_ft < 0 or _fp < _ft):
                        _cls = "KNOCKED_PRE_REACH"
                    elif _toppled and _fp <= _ft + KNOCK_WIN:
                        _cls = "KNOCKDOWN_ON_ARRIVAL"
                    elif _toppled:
                        _cls = "DROPPED_LATE"
                    elif _ft < 0:
                        if _gs < 0:
                            _cls = "EARLY_END"
                        elif _re > ROOT_TOL:
                            _cls = "ROOT_AHEAD" if _fw > ROOT_AHEAD_X else "ROOT_OFF"
                        else:
                            _cls = "ARM_OFF"
                    else:
                        _cls = "TOUCHED_NOT_HELD"
                    fc_counts[_cls] += 1
                    if _re >= 0:
                        fc_root_err_sum[_cls] += _re
                    if 0 <= _mid < _NM:
                        m_class[_mid, _CLASSES.index(_cls)] += 1
                if k < len(time_out_mask) and bool(time_out_mask[k].item()):
                    termination_counts["time_out"] += 1
                else:
                    termination_counts["other"] += 1

            per_env_had_any_lift[done_idxs] = False
            per_env_closed_steps[done_idxs] = 0
            per_env_lift_steps[done_idxs] = 0
            per_env_held_steps[done_idxs] = 0
            per_env_touched[done_idxs] = False
            per_env_toppled[done_idxs] = False
            if FAILCLASS:
                per_env_steps[done_idxs] = 0
                per_env_first_touch[done_idxs] = -1
                per_env_first_topple[done_idxs] = -1
                per_env_grab_step[done_idxs] = -1
                per_env_root_err_grab[done_idxs] = -1.0
                per_env_root_fwd_grab[done_idxs] = 0.0

        just_reset_mask = dones_bool
        step_count += 1

        if step_count % 200 == 0:
            elapsed = time.time() - t_start
            rate = step_count / max(elapsed, 1e-6)
            any_lift_rate = completed_any_lift / max(completed_episodes, 1)
            n_succ = float(env.unwrapped.n_successes.sum().item())
            print(f"[step {step_count}] episodes={completed_episodes}/{target_episodes}  "
                  f"any_lift_rate={any_lift_rate:.3f}  cumulative_lift_steps={n_succ:.0f}  "
                  f"({rate:.0f} steps/s)")

    elapsed = time.time() - t_start
    print(f"\n[eval_sonic_adapter] DONE — {completed_episodes} episodes in {elapsed:.1f}s "
          f"({step_count} steps, {step_count/max(elapsed,1e-6):.0f} steps/s)")

    # ---- final report ----
    n_succ_total = float(env.unwrapped.n_successes.sum().item())
    any_lift_rate = completed_any_lift / max(completed_episodes, 1)
    mean_lift_fraction = (
        sum_lift_fraction_over_lifted_episodes / max(completed_episodes_with_any_lift, 1)
        if completed_episodes_with_any_lift > 0 else 0.0
    )

    print("\n" + "=" * 60)
    print("                 SONIC ADAPTER EVAL SUMMARY")
    print("=" * 60)
    print(f"  Mode:                       {'ZERO-RESIDUAL (base)' if args_cli.zero_residual else 'trained adapter'}")
    print(f"  Checkpoint:                 {resume_path if resume_path else '(none)'}")
    print(f"  residual_scale:             {args_cli.residual_scale}")
    print(f"  num_envs:                   {num_envs}")
    print(f"  lift_thres:                 {lift_thres} m")
    print(f"  Completed episodes:         {completed_episodes}")
    print(f"  Total steps:                {step_count}")
    print(f"")
    print(f"  Episodes with any lift:     {completed_any_lift} / {completed_episodes} "
          f"= {100*any_lift_rate:.2f}%")
    print(f"  Mean lift fraction          {100*mean_lift_fraction:.2f}%")
    print(f"    (over episodes with lift; closed_phase_lift_steps / closed_phase_steps)")
    print(f"  Cumulative lift-steps       {n_succ_total:.0f}")
    if REWORK:
        held_rate = completed_held / max(completed_episodes, 1)
        print(f"")
        print(f"  [REWORK] Object HELD success: {completed_held} / {completed_episodes} "
              f"= {100*held_rate:.2f}%")
        print(f"    (object within {HELD_TOL} m of the synth obj ref for >= {HELD_FRAC:.0%} of the closed phase)")
        ce = max(completed_episodes, 1)
        print(f"")
        print(f"  Failure/outcome classification:")
        print(f"    Object touched:           {completed_touched} / {completed_episodes} = {100*completed_touched/ce:.2f}%  (hand reached < {TOUCH_TOL} m)")
        print(f"    -> HELD (success):        {completed_held} = {100*completed_held/ce:.2f}%")
        print(f"    -> touched, not held:     {completed_touched_not_held} = {100*completed_touched_not_held/ce:.2f}%")
        print(f"    Never touched object:     {completed_never_touched} = {100*completed_never_touched/ce:.2f}%")
        print(f"    Object toppled:           {completed_toppled} / {completed_episodes} = {100*completed_toppled/ce:.2f}%  (tilt > {TOPPLE_DEG:.0f} deg)")
    print(f"    (env.n_successes.sum() — uses the env reward's height_thres)")
    if FAILCLASS:
        print(f"")
        print(f"  [FAILCLASS] per-episode failure taxonomy (root_tol={ROOT_TOL} m, knock_win={KNOCK_WIN} steps):")
        for c in _CLASSES:
            n = fc_counts[c]
            if n == 0:
                continue
            mre = fc_root_err_sum[c] / n if n else 0.0
            print(f"    {c:22s} {n:5d}  ({100*n/max(completed_episodes,1):5.2f}%)   mean_root_err@grab={mre:.3f} m")
        print(f"    per-motion dominant failure (top 12 by episode count):")
        _order = torch.argsort(m_ep, descending=True)[:12]
        for _i in _order.tolist():
            if m_ep[_i] == 0:
                continue
            _dom = int(m_class[_i].argmax())
            _fn = ""
            try:
                _fn = os.path.basename(env.unwrapped.motion_lib._motion_data_load[_i])
            except Exception:
                pass
            print(f"      motion {_i:3d} {_fn:14s} eps={int(m_ep[_i]):3d}  dominant={_CLASSES[_dom]:22s} ({int(m_class[_i,_dom])})")
    print(f"")
    if os.environ.get("HS_EVAL_STABILITY", "0") == "1" and _stab_n > 0:
        _M = np.concatenate(_stab_marg)
        print(f"")
        print(f"  SIM-ROBOT static stability (CoM in planted-foot polygon, all stepped frames):")
        print(f"    stable          {100.0*_stab_sum/_stab_n:6.1f}%   ({_stab_sum}/{_stab_n} frames)")
        print(f"    margin  mean    {_M.mean():+7.4f} m")
        print(f"    margin  median  {np.median(_M):+7.4f} m")
        print(f"    margin  p05     {np.percentile(_M, 5):+7.4f} m")
        print(f"    margin  min     {_M.min():+7.4f} m")

    print(f"  Termination breakdown (of completed episodes):")
    for k, v in termination_counts.items():
        if v > 0:
            print(f"    {k:30s} {v}  ({100*v/max(completed_episodes,1):.1f}%)")
    print("=" * 60)

    # ---- per-clip ranking ----
    try:
        ml = env.unwrapped.motion_lib
        files = None
        for _a in ('_motion_files', 'motion_files', '_motion_data_load'):
            _v = getattr(ml, _a, None)
            if _v is not None and len(_v) == _NM:
                files = [str(x).split('/')[-1] for x in _v]; break
    except Exception:
        files = None
    rows = []
    for i in range(_NM):
        ep = int(m_ep[i].item())
        if ep == 0:
            continue
        top = int(m_top[i].item()); hel = int(m_held[i].item())
        tou = int(m_touch[i].item()); nev = int(m_never[i].item())
        rows.append((i, ep, 100.0*top/ep, 100.0*hel/ep, 100.0*tou/ep, 100.0*nev/ep,
                     files[i] if files else ''))
    # rank: worst first by topple%, then by low held%
    rows.sort(key=lambda r: (-r[2], r[3]))
    hdr = '  motion_id  ep   topple%  held%  touch%  never%  file'
    lines = ['PER-CLIP OUTCOME RANKING (worst topple first)', hdr]
    for (i, ep, tp, hl, tc, nv, fn) in rows:
        lines.append('  %8d %4d  %6.1f  %5.1f  %6.1f  %6.1f  %s' % (i, ep, tp, hl, tc, nv, fn))
    out = '\n'.join(lines)
    print('\n' + out)
    open('/tmp/_perclip_eval.txt', 'w').write(out + '\n')
    # compact machine-readable: worst-by-topple and worst-by-not-held (top 8 each)
    by_top = [r[0] for r in rows if r[1] >= 5][:8]
    by_nh = [r[0] for r in sorted(rows, key=lambda r: (r[3], -r[2])) if r[1] >= 5][:8]
    print('WORST_BY_TOPPLE=' + ','.join(map(str, by_top)))
    print('WORST_BY_NOTHELD=' + ','.join(map(str, by_nh)))
    open('/tmp/_perclip_worst.txt', 'w').write('WORST_BY_TOPPLE=%s\nWORST_BY_NOTHELD=%s\n' % (
        ','.join(map(str, by_top)), ','.join(map(str, by_nh))))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
