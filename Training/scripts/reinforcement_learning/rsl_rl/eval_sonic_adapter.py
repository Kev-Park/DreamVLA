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
parser.add_argument("--sonic-encoder-onnx", type=str,
                    default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx",
                    help="Path to the frozen SONIC encoder ONNX (must match training).")
parser.add_argument("--residual-scale", type=float, default=0.3,
                    help="Residual bound — MUST match the value used by train_sonic_adapter.py.")
parser.add_argument("--zero-residual", action="store_true", default=False,
                    help="Evaluate the frozen-SONIC base (residual=0, fingers open) as a "
                         "baseline. No checkpoint required.")
parser.add_argument("--lift-thres", type=float, default=0.97,
                    help="Bottle z (m) above which a frame counts as 'lifted'. Match the "
                         "env's object_above height_thres (object rests at 0.9, success "
                         "apex ~0.976, so 0.97 = a genuine pickup).")
parser.add_argument("--skip-start-frames", type=int, default=None,
                    help="Skip the refinement's 20-frame interpolate-to-initial-pose prepend "
                         "(pass 20). Keeps the walk.")
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

    # Optional episode-start offsets (same flags as train/play).
    if args_cli.start_pregrab_margin is not None:
        env_cfg.motion_start_pregrab_margin_s = args_cli.start_pregrab_margin
        print(f"[eval_sonic_adapter] start_pregrab_margin = {args_cli.start_pregrab_margin}s")
    if args_cli.skip_start_frames is not None:
        env_cfg.motion_skip_start_frames = args_cli.skip_start_frames
        print(f"[eval_sonic_adapter] skip_start_frames = {args_cli.skip_start_frames}")

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
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[env] action_space (pre-wrapper) = {env.action_space}")
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # ---- frozen encoder + decoder + adapter wrapper (same wiring as training) ----
    device = agent_cfg.device
    print(f"[eval_sonic_adapter] loading frozen SONIC decoder ONNX: {args_cli.sonic_decoder_onnx}")
    decoder = load_frozen_decoder(args_cli.sonic_decoder_onnx, device)
    print(f"[eval_sonic_adapter] loading frozen SONIC encoder ONNX: {args_cli.sonic_encoder_onnx}")
    encoder = load_frozen_encoder(args_cli.sonic_encoder_onnx, device)
    env = TokenAdapterVecEnvWrapper(
        env, decoder, encoder, device,
        residual_scale=args_cli.residual_scale,
        clip_actions=None,
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

    just_reset_mask = torch.ones(num_envs, device=device, dtype=torch.bool)

    obs_out = env.get_observations()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out

    print(f"[eval_sonic_adapter] starting eval: num_envs={num_envs}, target_episodes={target_episodes}, "
          f"max_steps={max_steps}, lift_thres={lift_thres}")
    t_start = time.time()
    step_count = 0

    while completed_episodes < target_episodes and step_count < max_steps:
        with torch.inference_mode():
            actions = policy(obs).clone()
            obs, _, dones, extras = env.step(actions)

        # Read bottle z + is_closed AFTER step.
        bottle_z = env.unwrapped.scene["object"].data.root_pos_w[:, 2]
        motion_times = (
            env.unwrapped.episode_length_buf * env.unwrapped.step_dt
            + env.unwrapped.start_motion_times.clone().detach().to(device=device, dtype=torch.float32)
        )
        motion_res = env.unwrapped.motion_lib.get_motion_state(env.unwrapped.motion_ids, motion_times)
        is_closed = motion_res["is_closed"].bool()
        lifted = (bottle_z > lift_thres) & is_closed

        valid = ~just_reset_mask
        if valid.any():
            per_env_had_any_lift |= lifted & valid
            per_env_closed_steps += (is_closed & valid).long()
            per_env_lift_steps += (lifted & valid).long()

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
                if k < len(time_out_mask) and bool(time_out_mask[k].item()):
                    termination_counts["time_out"] += 1
                else:
                    termination_counts["other"] += 1

            per_env_had_any_lift[done_idxs] = False
            per_env_closed_steps[done_idxs] = 0
            per_env_lift_steps[done_idxs] = 0

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
    print(f"    (env.n_successes.sum() — uses the env reward's height_thres)")
    print(f"")
    print(f"  Termination breakdown (of completed episodes):")
    for k, v in termination_counts.items():
        if v > 0:
            print(f"    {k:30s} {v}  ({100*v/max(completed_episodes,1):.1f}%)")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
