# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL-train a lightweight RESIDUAL ADAPTER between the frozen SONIC encoder and decoder.

Sister script to ``train_sonic.py``, implementing residual policy learning in token space:

    frozen G1 encoder ──▶ base token ──┐
                                       ├──▶ FSQ ──▶ frozen decoder ──▶ env body action
    adapter (policy) ──▶ residual ─────┘
    adapter (policy) ──▶ 1-D finger scalar ───────────────────────▶ env finger action

Differences vs train_sonic.py:
  - The frozen SONIC G1 encoder runs in the wrapper each step on the 1.0 s reference
    lookahead; its 64-D token is appended to the policy obs AND used as the base the
    policy's 64-D residual is added to (tanh-bounded by --residual-scale).
  - The policy is a small MLP (default [256, 128]) with a ZERO-INIT output layer —
    step-0 behavior is exact zero-shot SONIC playback; PPO learns a task delta.
  - Reference-tracking reward terms are scaled by --tracking-scale (default 1.0 —
    full tracking reward retained): the residual is rewarded for following the
    reference walk/reach/grasp. The loose residual bound alone does NOT pin the base
    motion, so dropping tracking (0.0) collapses the policy to standing. Survival,
    task (finger binary, wrist orientation, object lift) and smoothness regularizers
    stay active alongside tracking.
  - std anneal runs 0.3 → 0.15 (smaller than train_sonic's 0.5 → 0.2: exploration is
    around a competent base, large noise just fights the anchor).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a residual token adapter between frozen SONIC encoder/decoder.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Motion-Tracking-Pick-BinaryFingers-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=5000, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--sonic-decoder-onnx", type=str,
    default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx",
    help="Path to the frozen SONIC decoder ONNX (model_decoder.onnx).",
)
parser.add_argument(
    "--sonic-encoder-onnx", type=str,
    default="../../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx",
    help="Path to the frozen SONIC encoder ONNX (model_encoder.onnx).",
)
parser.add_argument(
    "--residual-scale", type=float, default=0.3,
    help="Hard bound on the token residual: token = base + scale*tanh(policy_out). "
         "FSQ token channels live in roughly [-1, 1], so 0.3 allows substantial local "
         "correction while structurally anchoring behavior to the frozen SONIC base.",
)
parser.add_argument(
    "--start-pregrab-margin", type=float, default=None,
    help="Seconds before each motion's grab frame to start episodes at, dropping BOTH the "
         "prepend slide and the walk. Default None = full motion. Isolates the grasp.",
)
parser.add_argument(
    "--skip-start-frames", type=int, default=None,
    help="Skip the refinement's interpolate-to-initial-pose prepend (the ~1 s constant-rate "
         "linear slide) while KEEPING the walk. Pass 22. The env actually resets 10 frames "
         "later (frame skip+10) so the decoder-history seed lands on real motion. Mutually "
         "exclusive with --start-pregrab-margin.",
)
parser.add_argument(
    "--waist-dof", type=int, default=27, choices=[27, 29],
    help="Body DOF. 29 actuates waist_roll/pitch (29-DOF + dex-hands USD) to match SONIC's "
         "training articulation; requires the asset from make_g1_29dof_with_hands.py. "
         "27 = legacy welded-waist asset.",
)
parser.add_argument(
    "--tracking-scale", type=float, default=1.0,
    help="Multiplier on the reference-tracking reward weights (joint/keypts/position/"
         "orientation tracking). Default 1.0 = full tracking reward retained, so the "
         "residual is rewarded for following the reference (walk + reach + grasp). "
         "Set 0.0 to drop tracking (task rewards only) — but the loose residual bound "
         "alone does NOT hold the base motion, so the policy collapses to standing.",
)
parser.add_argument(
    "--wrist-weight", type=float, default=None,
    help="Override target_orientation_error (wrist level+point) weight — a PENALTY, pass "
         "negative (e.g. -1.5). Default None = use the env cfg value.")
parser.add_argument(
    "--lift-weight", type=float, default=None,
    help="Override object_lift reward weight (e.g. 16.0). Default None = use env cfg value.")
parser.add_argument(
    "--body-track-weight", type=float, default=None,
    help="Override BOTH tracking_relative_body_pos and tracking_relative_body_ori weights "
         "(whole-body non-right-arm keypoint + link-orientation tracking). Applied AFTER "
         "--tracking-scale. Default None = use env cfg values.")
parser.add_argument(
    "--arm-pos-weight", type=float, default=None,
    help="Override tracking_right_arm_pos weight (global right-arm keypoint position tracking, "
         "shoulder->wrist_yaw). Applied AFTER --tracking-scale. Default None = use env cfg value.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

import omni
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# SONIC wrappers (imported after app launch; depend on onnx2torch + isaaclab_rl)
from vla_sonic.token_action_wrapper import load_frozen_decoder
from vla_sonic.token_adapter_wrapper import TokenAdapterVecEnvWrapper, load_frozen_encoder
from vla_sonic.physics_overrides import apply_sonic_physics_overrides
from vla_sonic.robot_29dof import apply_29dof_waist_override
from vla_sonic.adapter_actor_critic import AdapterActorCritic

# Register the custom ActorCritic with rsl_rl (eval(class_name) resolves in the
# on_policy_runner module's globals — same pattern as train_sonic.py).
import rsl_rl.modules
import rsl_rl.runners.on_policy_runner as _rsl_rl_opr
rsl_rl.modules.AdapterActorCritic = AdapterActorCritic
_rsl_rl_opr.AdapterActorCritic = AdapterActorCritic

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train a residual token adapter against the frozen SONIC encoder + decoder."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # Keep adapter runs in their own experiment directory (logs/rsl_rl/<exp>_sonic_adapter).
    agent_cfg.experiment_name = f"{agent_cfg.experiment_name}_sonic_adapter"

    # ---- adapter policy configuration ----
    # Small residual network: the task delta around a competent base needs conditional
    # logic (when/where to nudge the token, when to close the hand), not whole-body
    # control capacity. Two hidden layers keep PPO sample-efficient.
    agent_cfg.policy.class_name = "AdapterActorCritic"
    agent_cfg.policy.actor_hidden_dims = [256, 128]
    agent_cfg.policy.critic_hidden_dims = [512, 256, 256]
    agent_cfg.policy.init_noise_std = 0.3
    print(f"[train_sonic_adapter] policy = AdapterActorCritic, actor={agent_cfg.policy.actor_hidden_dims}, "
          f"critic={agent_cfg.policy.critic_hidden_dims}, init_noise_std={agent_cfg.policy.init_noise_std}")

    # ---- reward configuration ----
    # --tracking-scale multiplies the SONIC-matched whole-body tracking weights (anchor
    # pos/ori + relative-body pos + body linvel; exp Gaussian kernels). Default 1.0 keeps
    # them at the gear_sonic-trained values (0.5/0.5/1.0/1.0); set <1 to down-weight body
    # tracking. Survival, grasp/task (lift, finger, wrist) and smoothness terms stay untouched.
    ts = float(args_cli.tracking_scale)
    for term_name in ("tracking_anchor_pos", "tracking_anchor_ori",
                      "tracking_relative_body_pos", "tracking_relative_body_ori",
                      "tracking_body_linvel", "tracking_right_arm_pos"):
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None:
            old_w = term.weight
            term.weight = old_w * ts
            print(f"[train_sonic_adapter] rewards.{term_name}.weight: {old_w} -> {term.weight} "
                  f"(tracking_scale={ts})")

    # Direct per-term weight overrides (applied AFTER --tracking-scale so they win). Default
    # None = keep the env cfg value. Lets reward-weight sweeps run in parallel on separate GPUs
    # differing by FLAG instead of by an edited source file (no shared-file race).
    _overrides = []
    if args_cli.wrist_weight is not None:
        _overrides.append(("target_orientation_error", args_cli.wrist_weight))
    if args_cli.lift_weight is not None:
        _overrides.append(("object_lift", args_cli.lift_weight))
    if args_cli.body_track_weight is not None:
        _overrides.append(("tracking_relative_body_pos", args_cli.body_track_weight))
        _overrides.append(("tracking_relative_body_ori", args_cli.body_track_weight))
    if args_cli.arm_pos_weight is not None:
        _overrides.append(("tracking_right_arm_pos", args_cli.arm_pos_weight))
    for term_name, new_w in _overrides:
        term = getattr(env_cfg.rewards, term_name, None)
        if term is not None:
            old_w = term.weight
            term.weight = float(new_w)
            print(f"[train_sonic_adapter] rewards.{term_name}.weight: {old_w} -> {term.weight} (CLI override)")
        else:
            print(f"[train_sonic_adapter] WARN: rewards.{term_name} not found; weight override ignored")

    # One-line effective-config banner for run identification — grep 'EFFECTIVE CONFIG' in the log.
    def _w(n):
        t = getattr(env_cfg.rewards, n, None)
        return t.weight if t is not None else None
    _ap = getattr(env_cfg.rewards, "tracking_anchor_pos", None)
    _eps = _ap.params.get("eps") if _ap is not None else None
    print(f"[train_sonic_adapter] EFFECTIVE CONFIG: residual_scale={args_cli.residual_scale} "
          f"tracking_scale={ts} wrist={_w('target_orientation_error')} lift={_w('object_lift')} "
          f"body_pos={_w('tracking_relative_body_pos')} body_ori={_w('tracking_relative_body_ori')} "
          f"arm_pos={_w('tracking_right_arm_pos')} deadband_eps={_eps}")

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env_cfg.log_dir = log_dir
    env_cfg.enable_cameras = bool(args_cli.enable_cameras)
    if hasattr(env_cfg, "enable_cameras_for_collection"):
        env_cfg.enable_cameras_for_collection = bool(args_cli.enable_cameras)

    # Match the SONIC decoder's training-time physics.
    apply_sonic_physics_overrides(env_cfg)

    # 29-DOF strict-fidelity articulation (actuated waist roll/pitch) to match SONIC training.
    if args_cli.waist_dof == 29:
        apply_29dof_waist_override(env_cfg)

    # Optional: skip the foot-skating walk approach by starting episodes near the grab.
    if args_cli.start_pregrab_margin is not None:
        env_cfg.motion_start_pregrab_margin_s = args_cli.start_pregrab_margin
        print(f"[train_sonic_adapter] start_pregrab_margin = {args_cli.start_pregrab_margin}s "
              "(episodes begin near grab; walk approach skipped)")
    if args_cli.skip_start_frames is not None:
        # +10: reset 10 frames LATER so the decoder-history seed (the 10 frames PRECEDING the
        # reset) lands on REAL motion, not the skipped interpolation prepend. So
        # --skip-start-frames 22 -> robot resets at frame 32, history = frames 22..31.
        env_cfg.motion_skip_start_frames = args_cli.skip_start_frames + 10
        print(f"[train_sonic_adapter] skip_start_frames = {args_cli.skip_start_frames} "
              f"(+10 history warmup -> env resets at frame {args_cli.skip_start_frames + 10}; "
              "decoder history seeded from the 10 preceding real frames)")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ---- load BOTH frozen SONIC modules and wrap the env ----
    device = agent_cfg.device
    print(f"[train_sonic_adapter] loading frozen SONIC decoder ONNX: {args_cli.sonic_decoder_onnx}")
    decoder = load_frozen_decoder(args_cli.sonic_decoder_onnx, device)
    print(f"[train_sonic_adapter] loading frozen SONIC encoder ONNX: {args_cli.sonic_encoder_onnx}")
    encoder = load_frozen_encoder(args_cli.sonic_encoder_onnx, device)  # batch-dim-patched loader
    # clip_actions=None on purpose: the residual is tanh-bounded in the wrapper and the
    # body token is FSQ-bounded downstream — an outer clamp on (base + residual) could
    # clip the FROZEN BASE's contribution, which must never be distorted.
    env = TokenAdapterVecEnvWrapper(
        env, decoder, encoder, device,
        residual_scale=args_cli.residual_scale,
        clip_actions=None,
    )

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # Annealed std clamp (same machinery as train_sonic.py, smaller schedule: the adapter
    # explores around a competent base — large noise only fights the anchor).
    # Hold 0.3 for the first 20% of iterations, anneal to 0.15 by 80%, hold to the end.
    if agent_cfg.class_name == "OnPolicyRunner":
        import math
        _STD_MIN = 0.001
        _STD_MAX_START, _STD_MAX_END = 0.3, 0.15
        _total_iters = max(int(agent_cfg.max_iterations), 1)
        _anneal_start = int(0.2 * _total_iters)
        _anneal_end = int(0.8 * _total_iters)
        _update_counter = {"n": 0}
        _orig_update = runner.alg.update

        def _current_std_max() -> float:
            n = _update_counter["n"]
            if n <= _anneal_start:
                return _STD_MAX_START
            if n >= _anneal_end:
                return _STD_MAX_END
            frac = (n - _anneal_start) / max(_anneal_end - _anneal_start, 1)
            return _STD_MAX_START + frac * (_STD_MAX_END - _STD_MAX_START)

        def _update_with_std_clamp(*args, **kwargs):
            info = _orig_update(*args, **kwargs)
            _update_counter["n"] += 1
            std_max = _current_std_max()
            with torch.no_grad():
                policy = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor_critic", None)
                if policy is not None:
                    if hasattr(policy, "std") and isinstance(policy.std, torch.nn.Parameter):
                        policy.std.data.clamp_(min=_STD_MIN, max=std_max)
                    elif hasattr(policy, "log_std") and isinstance(policy.log_std, torch.nn.Parameter):
                        policy.log_std.data.clamp_(min=math.log(_STD_MIN), max=math.log(std_max))
            n = _update_counter["n"]
            if n in (_anneal_start, (_anneal_start + _anneal_end) // 2, _anneal_end):
                print(f"[train_sonic_adapter] std anneal: iter {n}/{_total_iters} → std_max = {std_max:.3f}")
            return info

        runner.alg.update = _update_with_std_clamp
        print(f"[train_sonic_adapter] installed ANNEALED std clamp: "
              f"hold {_STD_MAX_START} until iter {_anneal_start}, "
              f"anneal → {_STD_MAX_END} by iter {_anneal_end}, min={_STD_MIN}")

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
