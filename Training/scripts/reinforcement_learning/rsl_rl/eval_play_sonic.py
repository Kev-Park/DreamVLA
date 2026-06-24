# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless multi-env eval for a SONIC-encoder checkpoint — no cameras, no video.

Sister script to ``play_sonic.py``. Drops every camera/RTX-render code path so it
scales to large ``num_envs`` for statistical evaluation. Reports four metrics:

  1. ``n_successes`` (cumulative lift-step count, summed from the env reward function's
     internal accumulator at ``env.unwrapped.n_successes`` — only updates when
     ``num_envs < 1001``, so we cap automatically).
  2. ``episodes_with_any_lift`` — per-episode discrete success: did the bottle clear
     the 1.05 m threshold at any point during the closed phase? Reported as a rate.
  3. ``mean_lift_fraction`` — for episodes that had any lift, what fraction of the
     closed-phase steps were lifted. Captures grasp *quality*, not just initiation.
  4. ``termination breakdown`` — time_out vs base_contact vs torso_angle, so you can
     tell whether the policy is just running out the clock or actively failing.

Uses the deterministic actor mean via ``get_inference_policy`` — same as play_sonic.py.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import builtins
from functools import partial

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

print = partial(builtins.print, flush=True)

parser = argparse.ArgumentParser(description="Headless multi-env eval for a SONIC-encoder checkpoint.")
parser.add_argument("--num_envs", type=int, default=1000,
                    help="Number of parallel envs. Capped to 1000 because n_successes only "
                         "updates when env.num_envs < 1001 (see object_above_threshold).")
parser.add_argument("--num_episodes", type=int, default=2000,
                    help="Stop once at least this many episodes have completed across all envs.")
parser.add_argument("--max_steps", type=int, default=10000,
                    help="Hard cap on step count, regardless of episode completion.")
parser.add_argument("--task", type=str, default="Isaac-Motion-Tracking-Pick-BinaryFingers-v0",
                    help="Task name. Default matches train_sonic.py.")
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--path", type=str, default=None,
                    help="Explicit checkpoint path (overrides auto-discovery).")
parser.add_argument("--sonic-decoder-onnx", type=str,
                    default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx",
                    help="Path to the frozen SONIC decoder ONNX (must match training).")
parser.add_argument("--physics-preset", type=str, default="deploy", choices=["training", "deploy"],
                    help="Physics substep (both → 50 Hz control). 'deploy' (default) = 500 Hz/dec-10, "
                         "matches the real G1 motor rate / crispest live motion. 'training' = 200 Hz/dec-4 "
                         "(gear_sonic training substep; can feel sluggish).")
# append RSL-RL cli args (gives --checkpoint, --load_run, etc.)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# NEVER enable cameras — entire point of this script is headless eval.
args_cli.enable_cameras = False

# Cap num_envs so n_successes actually fires (gated at <1001 in reward func).
if args_cli.num_envs >= 1001:
    print(f"[eval_play_sonic] num_envs={args_cli.num_envs} >= 1001 — capping to 1000 so "
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

# SONIC token-action wrapper (same wiring as train_sonic.py / play_sonic.py)
from vla_sonic.token_action_wrapper import TokenActionDecoderVecEnvWrapper, load_frozen_decoder
from vla_sonic.physics_overrides import apply_sonic_physics_overrides
from vla_sonic.split_head_actor_critic import SplitHeadActorCritic

# Register the custom ActorCritic with rsl_rl (same pattern as train_sonic.py /
# play_sonic.py — eval() in rsl_rl runner resolves class_name in its own module globals).
import rsl_rl.modules
import rsl_rl.runners.on_policy_runner as _rsl_rl_opr
rsl_rl.modules.SplitHeadActorCritic = SplitHeadActorCritic
_rsl_rl_opr.SplitHeadActorCritic = SplitHeadActorCritic


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        enable_cameras=False,
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Match training-time split-head architecture so load_state_dict succeeds.
    agent_cfg.policy.class_name = "SplitHeadActorCritic"

    # SONIC physics substep (default 'deploy'=500 Hz, matches real-robot motor rate).
    apply_sonic_physics_overrides(env_cfg, preset=args_cli.physics_preset)

    # ---- resolve checkpoint ----
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

    # ---- wrap so the 64-D token policy drives the frozen decoder ----
    device = agent_cfg.device
    print(f"[eval_play_sonic] loading frozen SONIC decoder ONNX: {args_cli.sonic_decoder_onnx}")
    decoder = load_frozen_decoder(args_cli.sonic_decoder_onnx, device)
    env = TokenActionDecoderVecEnvWrapper(env, decoder, device, clip_actions=agent_cfg.clip_actions)

    # ---- initialize env.n_successes ----
    # object_above_threshold checks ``if env.num_envs < 1001 and hasattr(env, "n_successes")``,
    # so the counter only updates if it already exists on the env. Initialize it here.
    env.unwrapped.n_successes = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
    print(f"[eval_play_sonic] initialized env.n_successes (shape={env.unwrapped.n_successes.shape}, "
          f"dtype={env.unwrapped.n_successes.dtype})")

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    ppo_runner.load(resume_path)

    # DETERMINISTIC inference (actor mean, no Gaussian sampling).
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # ---- per-env episode trackers ----
    num_envs = env.num_envs
    target_episodes = args_cli.num_episodes
    max_steps = args_cli.max_steps

    # Episode-level discrete trackers (reset per env on episode end).
    per_env_had_any_lift = torch.zeros(num_envs, device=device, dtype=torch.bool)
    per_env_closed_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
    per_env_lift_steps = torch.zeros(num_envs, device=device, dtype=torch.long)

    # Aggregates over completed episodes.
    completed_episodes = 0
    completed_any_lift = 0
    sum_lift_fraction_over_lifted_episodes = 0.0
    completed_episodes_with_any_lift = 0
    termination_counts = {"time_out": 0, "base_contact": 0, "torso_angle_below_threshold": 0, "torso_below_threshold": 0, "other": 0}

    # Flag: skip tracking on the first step after a reset (the reset obs's bottle
    # state and is_closed flag belong to the *new* episode, not the one that just ended).
    # On step N+1 after a done at step N, we want to start tracking fresh.
    just_reset_mask = torch.ones(num_envs, device=device, dtype=torch.bool)  # all envs are "just reset" at the start

    obs_out = env.get_observations()
    obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out

    print(f"[eval_play_sonic] starting eval: num_envs={num_envs}, target_episodes={target_episodes}, "
          f"max_steps={max_steps}")
    t_start = time.time()
    step_count = 0

    while completed_episodes < target_episodes and step_count < max_steps:
        # 1. policy → token action → frozen-decoder → env.step
        with torch.inference_mode():
            actions = policy(obs).clone()
            obs, _, dones, extras = env.step(actions)

        # 2. read bottle z and is_closed for trackers (AFTER step, so this is the
        # state at the end of this transition).
        bottle_z = env.unwrapped.scene["object"].data.root_pos_w[:, 2]
        motion_times = (
            env.unwrapped.episode_length_buf * env.unwrapped.step_dt
            + env.unwrapped.start_motion_times.clone().detach().to(device=device, dtype=torch.float32)
        )
        motion_res = env.unwrapped.motion_lib.get_motion_state(env.unwrapped.motion_ids, motion_times)
        is_closed = motion_res["is_closed"].bool()
        lifted = (bottle_z > 1.05) & is_closed

        # 3. update trackers for envs that are NOT freshly reset this step.
        # (Just-reset envs have their bottle at spawn, is_closed=0 typically — would not
        # contaminate the counters, but we skip anyway for clarity.)
        valid = ~just_reset_mask
        if valid.any():
            per_env_had_any_lift |= lifted & valid
            per_env_closed_steps += (is_closed & valid).long()
            per_env_lift_steps += (lifted & valid).long()

        # 4. process episode completions.
        dones_bool = dones.bool() if dones.dtype != torch.bool else dones
        if dones_bool.any():
            done_idxs = torch.where(dones_bool)[0]

            # Pull termination reason from extras (mirrors training-time logging).
            time_outs = extras.get("time_outs", torch.zeros_like(dones_bool))
            time_out_mask = time_outs[done_idxs].bool() if isinstance(time_outs, torch.Tensor) else torch.zeros(len(done_idxs), dtype=torch.bool)

            for k, idx in enumerate(done_idxs.tolist()):
                completed_episodes += 1
                had_lift = bool(per_env_had_any_lift[idx].item())
                if had_lift:
                    completed_any_lift += 1
                    cs = int(per_env_closed_steps[idx].item())
                    ls = int(per_env_lift_steps[idx].item())
                    if cs > 0:
                        sum_lift_fraction_over_lifted_episodes += ls / cs
                        completed_episodes_with_any_lift += 1

                # Termination breakdown (rough: just time_out vs not, since extras only
                # carries time_outs cleanly; finer breakdown would need to dig into the
                # termination manager state).
                if k < len(time_out_mask) and bool(time_out_mask[k].item()):
                    termination_counts["time_out"] += 1
                else:
                    termination_counts["other"] += 1

            # Reset per-env trackers for completed episodes.
            per_env_had_any_lift[done_idxs] = False
            per_env_closed_steps[done_idxs] = 0
            per_env_lift_steps[done_idxs] = 0

        just_reset_mask = dones_bool  # the envs that just ended are "just reset" next step

        step_count += 1

        # progress prints every 200 steps
        if step_count % 200 == 0:
            elapsed = time.time() - t_start
            rate = step_count / max(elapsed, 1e-6)
            any_lift_rate = completed_any_lift / max(completed_episodes, 1)
            n_succ = float(env.unwrapped.n_successes.sum().item())
            print(f"[step {step_count}] episodes={completed_episodes}/{target_episodes}  "
                  f"any_lift_rate={any_lift_rate:.3f}  cumulative_lift_steps={n_succ:.0f}  "
                  f"({rate:.0f} steps/s)")

    elapsed = time.time() - t_start
    print(f"\n[eval_play_sonic] DONE — {completed_episodes} episodes in {elapsed:.1f}s "
          f"({step_count} steps, {step_count/max(elapsed,1e-6):.0f} steps/s)")

    # ---- final report ----
    n_succ_total = float(env.unwrapped.n_successes.sum().item())
    any_lift_rate = completed_any_lift / max(completed_episodes, 1)
    mean_lift_fraction = (
        sum_lift_fraction_over_lifted_episodes / max(completed_episodes_with_any_lift, 1)
        if completed_episodes_with_any_lift > 0 else 0.0
    )

    print("\n" + "=" * 60)
    print("                    SONIC EVAL SUMMARY")
    print("=" * 60)
    print(f"  Checkpoint:                 {resume_path}")
    print(f"  num_envs:                   {num_envs}")
    print(f"  Completed episodes:         {completed_episodes}")
    print(f"  Total steps:                {step_count}")
    print(f"")
    print(f"  Episodes with any lift:     {completed_any_lift} / {completed_episodes} "
          f"= {100*any_lift_rate:.2f}%")
    print(f"  Mean lift fraction          {100*mean_lift_fraction:.2f}%")
    print(f"    (over episodes with lift; closed_phase_lift_steps / closed_phase_steps)")
    print(f"  Cumulative lift-steps       {n_succ_total:.0f}")
    print(f"    (env.n_successes.sum() — sum across envs & all eval steps)")
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
